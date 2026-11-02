#!/usr/bin/env python3
"""Measure one prepared XCAT study and save phase-zero axial slices.

Example: python evaluate.py --study DATA/static/gates_8 --mode static \
    --osem RUNS/osem --bsrem RUNS/bsrem --dip RUNS/static_td_dip_step003000.pt \
    --output RESULTS/static_8
The dynamic study uses the same static truth directory with --mode dynamic.
"""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reconstructions'))
from helpers import MultiResolutionDIPNet3D, MultiResolutionGeometry, NetworkSettings
from prepare_data import resample_density
import truth_masks

METRICS = ('cnr', 'wall_cov_percent', 'ssim', 'psnr_db')


def open_result(method, path, mode, device):
    path = Path(path)
    if method == 'OSEM':
        h = h5py.File(path / f'{mode}_osem.h5' if path.is_dir() else path, 'r')
        geo = [h.attrs['volume_shape_xyz'], h.attrs['volume_origin_xyz'], h.attrs['volume_spacing_xyz']]
        return h['reconstruction_packed'], geo, float(h.attrs['gain']), h
    if method == 'BSREM + NLM':
        directory = path if path.is_dir() else path.parent
        with np.load(directory / 'geometry.npz') as meta:
            geo = [meta['volume_shape_xyz'].copy(), meta['volume_origin_xyz'].copy(), meta['volume_spacing_xyz'].copy()]
            gain = float(meta['gain'])
        return np.load(directory / 'reconstruction.npy' if path.is_dir() else path, mmap_mode='r'), geo, gain, None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    geometry = MultiResolutionGeometry.from_dict(checkpoint['geometry'])
    model = MultiResolutionDIPNet3D(NetworkSettings(**checkpoint['network_settings']), geometry).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    latent = checkpoint['latents'].to(device)
    geo = [np.asarray(checkpoint[k]) for k in ('volume_shape_xyz', 'volume_origin_xyz', 'volume_spacing_xyz')]
    class Decoder:
        def __len__(self): return len(latent)
        def __getitem__(self, frame):
            with torch.inference_mode(): return model(latent[frame:frame + 1]).float().cpu().numpy().reshape(-1)
    return Decoder(), geo, float(checkpoint['gain']), None


def masks_for_study(study, manifest, geometry, cache):
    shapes, origins, steps = geometry
    if cache and cache.exists():
        with np.load(cache) as data:
            if not (np.allclose(data['volume_origin_mm'], origins[0]) and np.allclose(data['volume_spacing_mm'], steps[0])):
                raise ValueError('Mask geometry does not match reconstruction')
            result = tuple(data[k].copy() for k in ('wall_mask_xyz', 'cavity_mask_xyz', 'truth_roi_mbq_ml'))
            expected = (manifest['gate_count'], *shapes[0])
            if any(a.shape != expected for a in result):
                raise ValueError('Mask dimensions do not match the study')
            return result
    truth_masks.SHAPE = np.asarray(shapes[0], int)
    truth_masks.ORIGIN = np.asarray(origins[0], float)
    truth_masks.STEP = np.asarray(steps[0], float)
    activity, _, origin, spacing, _ = truth_masks.read_truth(study, 0)
    _, _, anchor, _ = truth_masks.chamber_masks(activity, origin, spacing)
    wall, cavity, truth = [], [], []
    for gate in range(manifest['gate_count']):
        w, c, t, _, _, _ = truth_masks.make_masks(study, gate, anchor)
        wall.append(w); cavity.append(c); truth.append(t)
        print(f"Truth masks: {gate + 1}/{manifest['gate_count']}", flush=True)
    result = np.asarray(wall), np.asarray(cavity), np.asarray(truth)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, wall_mask_xyz=result[0], cavity_mask_xyz=result[1],
                            truth_roi_mbq_ml=result[2], volume_origin_mm=origins[0],
                            volume_spacing_mm=steps[0], mask_definition_version='lv_uniform_core_v2')
    return result


def image_metrics(image, truth, wall, cavity, peak):
    a = image.astype(np.float64); b = truth.astype(np.float64)
    w, c = a[wall], a[cavity]
    wm, cm = w.mean(), c.mean()
    ws, cs = w.std(ddof=1), c.std(ddof=1)
    smooth = lambda x: gaussian_filter(x, sigma=1.5, truncate=3.5, mode='reflect')
    ux, uy = smooth(b), smooth(a)
    vx, vy, cov = smooth(b*b)-ux*ux, smooth(a*a)-uy*uy, smooth(a*b)-ux*uy
    c1, c2 = (.01*peak)**2, (.03*peak)**2
    ssim = ((2*ux*uy+c1)*(2*cov+c2))/((ux*ux+uy*uy+c1)*(vx+vy+c2))
    score = float(ssim[5:-5, 5:-5, 5:-5].mean())
    mse = float(np.mean((a-b)**2))
    return (abs(wm-cm)/np.hypot(ws,cs) if np.hypot(ws,cs)>0 else np.nan,
            100*ws/wm if wm>0 else np.nan, score,
            10*np.log10(peak**2/mse) if mse>0 else np.inf)


def axial_plane(packed, geometry, manifest, index):
    shapes, origins, steps = (np.asarray(x) for x in geometry)
    native_shape = np.asarray(manifest['image_shape_xyz']); native_step = np.asarray(manifest['voxel_size_mm'])
    full_shape = np.asarray(manifest['reconstruction_shape_xyz'])
    full_origin = -native_shape * native_step / 2
    full_step = native_shape * native_step / full_shape
    x = full_origin[0] + full_step[0] * (np.arange(full_shape[0]) + .5)
    y = full_origin[1] + full_step[1] * (np.arange(full_shape[1]) + .5)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    xyz = np.stack((xx, yy, np.full_like(xx, origins[0,2] + steps[0,2] * (index + .5))), axis=-1)
    plane = np.full(xx.shape, np.nan, np.float32)
    offsets = np.r_[0, np.cumsum(np.prod(shapes, axis=1))].astype(int)
    for k in [*range(1, len(shapes)), 0]:
        inside = np.all((xyz >= origins[k]-1e-5) & (xyz < origins[k] + shapes[k]*steps[k]-1e-5), axis=-1)
        indices = np.floor((xyz[inside]-origins[k])/steps[k]).astype(int)
        volume = packed[offsets[k]:offsets[k+1]].reshape(tuple(shapes[k]), order='F')
        plane[inside] = volume[tuple(indices.T)]
    if not np.isfinite(plane).all(): raise ValueError('Reconstruction geometry leaves holes in axial slice')
    return plane.T


def truth_packed(study, manifest, geometry):
    with np.load(study / manifest['truth_files'][0]) as volume:
        density = volume['data'] * np.float32(1e6 / np.prod(manifest['voxel_size_mm']))
    native_step = np.asarray(manifest['voxel_size_mm']); native_origin = -np.asarray(density.shape)*native_step/2
    return np.concatenate([resample_density(density, native_origin, native_step, shape, origin, step).ravel(order='F')
                           for shape, origin, step in zip(*geometry)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True, help='Prepared static gates_N directory containing truth')
    p.add_argument('--mode', choices=('static', 'dynamic'), required=True)
    p.add_argument('--osem', type=Path); p.add_argument('--bsrem', type=Path); p.add_argument('--dip', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--masks', type=Path, help='Read or write truth-mask cache (default: OUTPUT/truth_masks.npz)')
    p.add_argument('--device', default='cpu', help='Device for TD-DIP checkpoint inference')
    p.add_argument('--cycle', type=int, default=600, help='Cycle for dynamic phase-zero slice')
    p.add_argument('--slice', type=int, default=17, help='Zero-based axial index within fine ROI')
    args = p.parse_args()
    paths = [('OSEM', args.osem), ('BSREM + NLM', args.bsrem), ('TD-DIP', args.dip)]
    paths = [(m, path) for m, path in paths if path is not None]
    if not paths: p.error('Provide at least one reconstruction')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.study / 'study.json').read_text()); gates = int(manifest['gate_count'])
    sources, geometry, handles = [], None, []
    gain = float(json.loads((args.study.parent / 'gain_calibration.json').read_text())['gain'])
    for method, path in paths:
        source, current, source_gain, handle = open_result(method, path, args.mode, torch.device(args.device))
        if not np.isclose(source_gain, gain, rtol=1e-6):
            raise ValueError(f'{method} gain differs from the prepared study')
        current = [np.asarray(current[0], int), np.asarray(current[1], float), np.asarray(current[2], float)]
        if geometry is not None and any(not np.allclose(a,b) for a,b in zip(geometry,current)):
            raise ValueError(f'{method} geometry differs from other methods')
        geometry = current
        expected = gates if args.mode == 'static' else 1200 * gates
        if len(source) != expected: raise ValueError(f'{method} contains {len(source)} frames; expected {expected}')
        sources.append((method, source)); handles.append(handle)
    if not 0 <= args.slice < geometry[0][0,2]: p.error('Axial slice is outside ROI')
    if args.mode == 'dynamic' and not 0 <= args.cycle < 1200: p.error('Cycle must be 0..1199')
    wall, cavity, truth = masks_for_study(args.study, manifest, geometry, args.masks or args.output/'truth_masks.npz')
    truth *= 1000
    peak = 75 * 210 / 11_681_335 * 1e6 / np.prod(manifest['voxel_size_mm'])
    selected = 0 if args.mode == 'static' else args.cycle * gates
    slices = {}; summaries = []
    with gzip.open(args.output/'image_metrics_per_frame.csv.gz', 'wt', newline='') as f:
        writer = csv.writer(f); writer.writerow(['method','gate_count','frame_index','phase_index','cycle_index',*METRICS])
        for method, source in sources:
            results = []
            for frame in range(len(source)):
                packed = np.asarray(source[frame]).reshape(-1)
                roi = packed[:int(np.prod(geometry[0][0]))].reshape(tuple(geometry[0][0]), order='F')
                phase = frame % gates
                vals = image_metrics(roi, truth[phase], wall[phase], cavity[phase], peak)
                writer.writerow([method,gates,frame,phase,frame//gates,*vals]); results.append(vals)
                if frame == selected:
                    slices[f'{method}_roi'] = roi[:,:,args.slice].T.copy()
                    slices[f'{method}_full'] = axial_plane(packed, geometry, manifest, args.slice)
                if frame % 1000 == 0: print(f'{method}: {frame + 1}/{len(source)} frames', flush=True)
            results = np.asarray(results)
            for index, metric in enumerate(METRICS):
                v = results[:,index]; v = v[np.isfinite(v)]
                summaries.append(dict(method=method,gate_count=gates,metric=metric,n_valid=len(v),
                                      median=float(np.median(v)) if len(v) else float('nan'),
                                      p2_5=float(np.percentile(v,2.5)) if len(v) else float('nan'),
                                      p97_5=float(np.percentile(v,97.5)) if len(v) else float('nan')))
    packed = truth_packed(args.study, manifest, geometry)
    slices['Ground truth_roi'] = packed[:int(np.prod(geometry[0][0]))].reshape(tuple(geometry[0][0]), order='F')[:,:,args.slice].T
    slices['Ground truth_full'] = axial_plane(packed, geometry, manifest, args.slice)
    np.savez_compressed(args.output/'phase0_slices.npz', **slices)
    with (args.output/'image_metrics_summary.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    (args.output/'result.json').write_text(json.dumps(dict(mode=args.mode,gates=gates,cycle=args.cycle,
        axial_roi_index=args.slice,truth_peak_kbq_ml=peak,units='kBq/mL',methods=[m for m,_ in paths],
        masks='truth-derived LV wall and cavity cores', summary='median and 2.5/97.5 percentiles across frames'),indent=2)+'\n')
    for handle in handles:
        if handle is not None: handle.close()


if __name__ == '__main__': main()
