#!/usr/bin/env python3
"""Prepare matched conventional and cycle-resolved XCAT studies from SIMIND.

Each fine phase is sampled once. Its counts are split among the ten cardiac
cycles at each view and then accumulated into the requested cardiac gates.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.sparse import coo_matrix, csr_matrix

from helpers import open_study

HERE = Path(__file__).resolve().parent
PHASES, POSITIONS, CYCLES = 256, 120, 10
DETECTOR_SHAPE = (128, 128)
NATIVE_SHAPE = (256, 256, 306)
VOXEL_MM = 2.46
SOURCE_SCALE_MBQ = 210.0 / 11_681_335
WINDOWS = ('prompt', 'lower', 'upper')


def native_volume(path):
    volume = np.fromfile(path, dtype='<f4').reshape(306, 256, 256)
    return volume.transpose(1, 2, 0)[::-1, ::-1, ::-1].copy()


def phase_projection(simind_dir, phase, window):
    heads, radii = [], []
    for head in (0, 1):
        folder = simind_dir / f'phase_{phase:03d}' / f'head_{head}'
        heads.append(np.fromfile(folder / f'projection_tot_w{window}.a00', dtype='<f4')
                     .reshape(*DETECTOR_SHAPE, POSITIONS, order='F'))
        if window == 1:
            radii.append(np.loadtxt(folder / 'projection.cor', ndmin=2)[:, 0] * 10 + 35)
    views = np.stack(heads, axis=3).reshape(*DETECTOR_SHAPE, 2 * POSITIONS)
    return views, np.stack(radii, axis=1).ravel() if radii else None


def split_phase(counts, rng, row_parts, col_parts, value_parts):
    # A column is one head image in Fortran order; rows are position/cycle.
    by_position = counts.reshape(*DETECTOR_SHAPE, POSITIONS, 2).transpose(2, 3, 1, 0).reshape(POSITIONS, -1)
    flat = np.flatnonzero(by_position)
    remaining = by_position.ravel()[flat].astype(np.int64)
    position, column = np.divmod(flat, 2 * np.prod(DETECTOR_SHAPE))
    for cycle in range(CYCLES):
        draw = rng.binomial(remaining, 1 / (CYCLES - cycle)) if cycle < CYCLES - 1 else remaining
        keep = draw > 0
        row_parts.append((position[keep] * CYCLES + cycle).astype(np.int32))
        col_parts.append(column[keep].astype(np.int32))
        value_parts.append(draw[keep].astype(np.uint32))
        remaining = remaining - draw


def save_sparse(path, matrix):
    matrix.sum_duplicates()
    matrix.sort_indices()
    maximum = int(matrix.data.max(initial=0))
    dtype = np.uint8 if maximum <= 255 else np.uint16 if maximum <= 65535 else np.uint32
    np.savez_compressed(path, row_ptr=matrix.indptr.astype(np.uint32),
                        indices=matrix.indices.astype(np.uint16),
                        counts=matrix.data.astype(dtype), shape=np.asarray(matrix.shape, np.uint32))


def static_manifest(gates, seed):
    angles = np.stack((np.arange(POSITIONS) * 3,
                       (np.arange(POSITIONS) * 3 + 90) % 360), axis=1).ravel()
    return dict(gate_count=gates, voxel_size_mm=[VOXEL_MM] * 3,
        image_shape_xyz=list(NATIVE_SHAPE), reconstruction_shape_xyz=[128, 128, 156],
        physical_detector_axial_mm=500, physical_detector_transverse_mm=400,
        detector_pixel_mm=[3.5, 3.5], detector_shape=list(DETECTOR_SHAPE),
        collimator_length_mm=35, collimator_hole_radius_mm=.75, intrinsic_fwhm_mm=3.8,
        angles_deg=angles.tolist(), omega_angles_deg=(-angles).tolist(),
        head_ids_zero_based=np.tile([0, 1], POSITIONS).tolist(),
        exposure_s_per_gate_view=10 / gates, acquisition_wall_time_s=1200,
        positions_per_head=POSITIONS, heart_center_mm=[28.58, -33.80, 37.85],
        attenuation_files=[f'mu_{g:03d}.npz' for g in range(1, gates + 1)],
        truth_files=[f'truth_{g:03d}.npz' for g in range(1, gates + 1)],
        projection_files=[f'gate_{g:03d}.mat' for g in range(1, gates + 1)],
        truth_units='MBq per native voxel', attenuation_units='cm^-1',
        poisson_seed=seed, scatter_method='2 * (lower + upper)',
        gate_phase_intervals_s=[[g / gates, (g + 1) / gates] for g in range(gates)])


def prepare_gates(simind_dir, xcat_dir, static_dir, dynamic_dir, gates, seed):
    static_dir.mkdir(parents=True, exist_ok=False)
    if dynamic_dir is not None:
        dynamic_dir.mkdir(parents=True, exist_ok=False)
    per_gate_matrices = {name: [] for name in WINDOWS}
    radii = []
    phases_per_gate = PHASES // gates
    for gate in range(gates):
        counts = {name: np.zeros((*DETECTOR_SHAPE, 2 * POSITIONS), np.uint32) for name in WINDOWS}
        activity = np.zeros(NATIVE_SHAPE, np.float32)
        attenuation = np.zeros(NATIVE_SHAPE, np.float32)
        radius = np.zeros(2 * POSITIONS, np.float64)
        fragments = {name: ([], [], []) for name in WINDOWS}
        for phase in range(gate * phases_per_gate + 1, (gate + 1) * phases_per_gate + 1):
            rng = np.random.default_rng(np.random.SeedSequence([seed, phase]))
            for window, name in enumerate(WINDOWS, 1):
                expected, phase_radius = phase_projection(simind_dir, phase, window)
                if phase_radius is not None:
                    radius += phase_radius / phases_per_gate
                sampled = rng.poisson(np.maximum(expected, 0)).astype(np.uint32)
                counts[name] += sampled
                if dynamic_dir is not None:
                    split_rng = np.random.default_rng(np.random.SeedSequence([seed, phase, window, 10]))
                    split_phase(sampled, split_rng, *fragments[name])
            activity += native_volume(xcat_dir / f'cardiac256_act_{phase}.bin') * (SOURCE_SCALE_MBQ / phases_per_gate)
            attenuation += native_volume(xcat_dir / f'cardiac256_atn_{phase}.bin') / (0.246 * phases_per_gate)
        scatter = 2 * (counts['lower'].astype(np.float32) + counts['upper'].astype(np.float32))
        savemat(static_dir / f'gate_{gate + 1:03d}.mat',
                {**counts, 'scatter': scatter, 'radius_to_crystal_mm': radius.astype(np.float32)},
                do_compression=True)
        np.savez_compressed(static_dir / f'truth_{gate + 1:03d}.npz', data=activity)
        np.savez_compressed(static_dir / f'mu_{gate + 1:03d}.npz', data=attenuation)
        radii.append(radius)
        if dynamic_dir is not None:
            for name in WINDOWS:
                rows, columns, values = fragments[name]
                matrix = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                                    shape=(POSITIONS * CYCLES, 2 * np.prod(DETECTOR_SHAPE)), dtype=np.uint32).tocsr()
                per_gate_matrices[name].append(matrix)
        print(f'{gates} gates: prepared gate {gate + 1}/{gates}', flush=True)
    manifest = static_manifest(gates, seed)
    (static_dir / 'study.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if dynamic_dir is not None:
        prepare_dynamic(dynamic_dir, static_dir, manifest, np.asarray(radii), per_gate_matrices, seed)


def interleave_gates(matrices):
    gates = len(matrices)
    lengths = np.stack([np.diff(m.indptr) for m in matrices], axis=1).ravel()
    row_ptr = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    data = np.empty(row_ptr[-1], dtype=np.uint32)
    indices = np.empty(row_ptr[-1], dtype=np.int32)
    for cycle in range(POSITIONS * CYCLES):
        for gate, matrix in enumerate(matrices):
            start, end = matrix.indptr[cycle:cycle + 2]
            target = row_ptr[cycle * gates + gate]
            data[target:target + end - start] = matrix.data[start:end]
            indices[target:target + end - start] = matrix.indices[start:end]
    return csr_matrix((data, indices, row_ptr), shape=(POSITIONS * CYCLES * gates, matrices[0].shape[1]))


def prepare_dynamic(directory, static_dir, static, radii, matrices, seed):
    gates = static['gate_count']
    index = np.arange(POSITIONS * CYCLES * gates)
    cycle, phase = index // gates, index % gates
    x = (np.arange(DETECTOR_SHAPE[0]) - (DETECTOR_SHAPE[0] - 1) / 2) * 3.5
    z = (np.arange(DETECTOR_SHAPE[1]) - (DETECTOR_SHAPE[1] - 1) / 2) * 3.5
    mask = (abs(x[:, None]) <= 200) & (abs(z[None, :]) <= 250)
    np.savez_compressed(directory / 'timeline_geometry.npz', cycle_index=cycle.astype(np.uint16),
        gate_index=phase.astype(np.uint8), position_index=(cycle // CYCLES).astype(np.uint8),
        interval_start_s=cycle + phase / gates, interval_stop_s=cycle + (phase + 1) / gates,
        radius_to_crystal_mm=radii.reshape(gates, POSITIONS, 2).astype(np.float32),
        angles_deg=np.asarray(static['angles_deg']).reshape(POSITIONS, 2),
        omega_angles_deg=np.asarray(static['omega_angles_deg']).reshape(POSITIONS, 2),
        active_detector_mask=mask)
    for name in WINDOWS:
        save_sparse(directory / f'{name}.npz', interleave_gates(matrices[name]))
    relative = lambda path: os.path.relpath(path, directory)
    manifest = dict(format='xcat_dynamic_sparse_csr_v1', gate_count=gates,
        interval_count=int(index.size), acquisition_wall_time_s=1200,
        exposure_s_per_interval_head=1 / gates,
        source_static_study=relative(static_dir / 'study.json'),
        gain_calibration_file=relative(static_dir.parent / 'gain_calibration.json'),
        timeline_geometry_file='timeline_geometry.npz',
        count_files={name: f'{name}.npz' for name in WINDOWS},
        allocation_seed=seed, synthetic_timeline=True,
        row_order='cycle * gate_count + gate',
        scatter_method='2 * (lower + upper)', reconstruction_units='kBq/mL')
    (directory / 'study.json').write_text(json.dumps(manifest, indent=2) + '\n')


def resample_density(density, source_origin, source_spacing, shape, origin, spacing):
    """Conservatively average the native density over each projector voxel."""
    result = density
    for axis in range(3):
        rows, columns, weights = [], [], []
        for j in range(shape[axis]):
            lo = origin[axis] + j * spacing[axis]
            hi = lo + spacing[axis]
            first = max(0, int(np.floor((lo - source_origin[axis]) / source_spacing[axis])))
            last = min(density.shape[axis], int(np.ceil((hi - source_origin[axis]) / source_spacing[axis])))
            for i in range(first, last):
                width = min(hi, source_origin[axis] + (i + 1) * source_spacing[axis]) - max(lo, source_origin[axis] + i * source_spacing[axis])
                if width > 0:
                    rows.append(j)
                    columns.append(i)
                    weights.append(width / spacing[axis])
        matrix = csr_matrix((np.asarray(weights, np.float32), (rows, columns)),
                            shape=(shape[axis], density.shape[axis]))
        moved = np.moveaxis(result, axis, 0)
        averaged = matrix @ moved.reshape(moved.shape[0], -1)
        result = np.moveaxis(averaged.reshape((shape[axis], *moved.shape[1:])), 0, axis)
    return np.asarray(result, np.float32)


def calibrate_gain(static_dir, device):
    """Fit the study's full-FOV, eight-gate count-ratio gain on the GPU."""
    import torch
    import cupy

    device = torch.device(device)
    torch.cuda.set_device(device)
    cupy.cuda.Device(device.index or 0).use()
    study = open_study('static', static_dir, static_dir, 8, device, subsets=1,
                       rays=13, cache_size=1, full_fov=True, gain_override=1)
    native_spacing = np.full(3, VOXEL_MM)
    native_origin = -np.asarray(NATIVE_SHAPE) * native_spacing / 2
    total_prompt = total_scatter = total_projection = 0.
    for gate in range(8):
        options = study.projector(gate)
        shape = np.array([options.Nx, options.Ny, options.Nz], int).reshape(3)
        origin = np.array([options.bx, options.by, options.bz], float).reshape(3)
        spacing = np.array([options.dx, options.dy, options.dz], float).reshape(3)
        with np.load(static_dir / f'truth_{gate + 1:03d}.npz') as source:
            density = source['data'] * np.float32(1e6 / np.prod(native_spacing))
        truth = resample_density(density, native_origin, native_spacing, shape, origin, spacing)
        image = torch.as_tensor(truth.ravel(order='F').copy(), device=device)
        with torch.no_grad():
            projected = study.forward(gate, 0, image)
            prompt, scatter, mask = study.frame_data(gate, 0, device)
            total_projection += float(projected.sum(dtype=torch.float64))
            total_prompt += float((prompt * mask).sum(dtype=torch.float64))
            total_scatter += float((scatter * mask).sum(dtype=torch.float64))
        print(f'Calibrated reference gate {gate + 1}/8', flush=True)
    if total_prompt <= total_scatter or total_projection <= 0:
        raise ValueError('Cannot calibrate gain: nonpositive net or projected counts')
    result = dict(gain=(total_prompt - total_scatter) / total_projection,
                  reconstruction_units='kBq/mL', calibration_reference_gate_count=8,
                  calibration_full_fov=True, rays=13,
                  formula='(sum(prompt) - sum(scatter)) / sum(exposure * masked A(truth))',
                  prompt_counts=total_prompt, scatter_counts=total_scatter,
                  unit_gain_projected_truth_counts=total_projection)
    path = static_dir.parent / 'gain_calibration.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(f"Calibrated gain: {result['gain']:.8g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gates', type=int, choices=(8, 16, 32, 64, 128, 256), default=8)
    parser.add_argument('--simind-dir', type=Path, default=HERE.parent / 'simind' / 'runs')
    parser.add_argument('--xcat-dir', type=Path, default=HERE.parent / 'xcat' / 'output')
    parser.add_argument('--output-dir', type=Path,
                        help='Output directory (default: reconstructions/data/NGates)')
    parser.add_argument('--device', default='cuda:0', help='CUDA/ROCm device used for gain calibration')
    parser.add_argument('--seed', type=int, default=20261014)
    args = parser.parse_args()
    output = args.output_dir or HERE / 'data' / f'{args.gates}gates'
    static = output / 'static'
    dynamic = output / 'dynamic'
    reference = static / 'gates_8'
    if output.exists() and any(output.iterdir()):
        parser.error(f'Output directory is not empty: {output}')
    import torch
    import cupy
    from omegatomo.projector import proj
    if not torch.cuda.is_available():
        parser.error('A CUDA or ROCm GPU is required to calibrate the gain')
    if args.gates == 8:
        prepare_gates(args.simind_dir, args.xcat_dir, reference, dynamic / 'gates_8', 8, args.seed)
    else:
        prepare_gates(args.simind_dir, args.xcat_dir, reference, None, 8, args.seed)
        prepare_gates(args.simind_dir, args.xcat_dir, static / f'gates_{args.gates}',
                      dynamic / f'gates_{args.gates}', args.gates, args.seed)
    calibrate_gain(reference, args.device)


if __name__ == '__main__':
    main()
