#!/usr/bin/env python3
"""Calculate CNR, wall CoV and projection-domain Poisson KL for XCAT truth.

Example:
    python plotting/ground_truth_metrics.py --study reconstructions/data/8gates/static/gates_8 \
        --mode static --gates 8 --output results/static/gates_8/ground_truth_metrics

Dynamic input is a prepared dynamic gates_N directory. Its activity truth and
masks come from the linked static study, while KL uses each dynamic frame's
counts, exposure and detector position. KL requires the OMEGA GPU projector.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reconstructions'))
from helpers import GATE_COUNTS, open_study
from prepare_data import resample_density
from evaluate import image_metrics, masks_for_study


def packed_truth(study, gate):
    """Resample one native activity image onto all projector volumes, in kBq/mL."""
    manifest = study.static_manifest
    spacing = np.asarray(manifest['voxel_size_mm'], dtype=np.float64)
    with np.load(study.static_root / manifest['truth_files'][gate]) as source:
        activity = np.asarray(source['data'], dtype=np.float32)
    density = activity * np.float32(1e6 / np.prod(spacing))
    origin = -np.asarray(density.shape) * spacing / 2
    volumes = (resample_density(density, origin, spacing, shape, corner, step).ravel(order='F')
               for shape, corner, step in zip(study.volume_shape_xyz,
                                               study.volume_origin_xyz,
                                               study.volume_spacing_xyz))
    return np.concatenate(list(volumes))


def poisson_kl(expected, observed, mask):
    """Sum D_Poisson(observed || expected) over active detector bins."""
    valid = mask > 0
    mean = expected[valid].to(torch.float64)
    counts = observed[valid].to(torch.float64)
    if not torch.isfinite(mean).all() or not torch.isfinite(counts).all():
        raise ValueError('Non-finite expected or observed projection values')
    if torch.any(mean <= 0) or torch.any(counts < 0):
        raise ValueError('Expected counts must be positive and observed counts nonnegative')
    positive = counts > 0
    terms = mean - counts
    terms[positive] += counts[positive] * (counts[positive].log() - mean[positive].log())
    return float(terms.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True, help='Prepared static or dynamic gates_N directory')
    parser.add_argument('--mode', choices=('static', 'dynamic'), required=True)
    parser.add_argument('--gates', type=int, choices=GATE_COUNTS, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Directory for CSV and metadata')
    parser.add_argument('--device', default='cuda:0', help='CUDA/ROCm device for OMEGA')
    parser.add_argument('--masks', type=Path, help='Optional truth-mask cache from evaluate.py')
    parser.add_argument('--projector-cache-size', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    study = open_study(args.mode, args.study, args.output, args.gates, args.device,
                       subsets=1, rays=13, roi_shape=(32, 32, 32), coarse_scale=.25,
                       cache_size=args.projector_cache_size)
    geometry = (study.volume_shape_xyz, study.volume_origin_xyz, study.volume_spacing_xyz)
    wall, cavity, truth = masks_for_study(study.static_root, study.static_manifest, geometry, args.masks)
    truth = truth * 1000  # MBq/mL to kBq/mL, matching reconstruction values.
    peak = 75 * 210 / 11_681_335 * 1e6 / np.prod(study.static_manifest['voxel_size_mm'])
    image_scores = [image_metrics(image, image, w, c, peak)[:2]
                    for image, w, c in zip(truth, wall, cavity)]
    epsilon = float(study.options.epps)
    fields = ('frame_index', 'gate_index', 'cnr', 'wall_cov_percent', 'poisson_kl')
    metrics = []
    # Process by gate to limit projector creation; write results in study order.
    for gate in range(study.gate_count):
        image = torch.as_tensor(packed_truth(study, gate), dtype=torch.float32, device=study.device)
        cnr, cov = image_scores[gate]
        for frame in np.flatnonzero(study.gate_indices == gate):
            frame = int(frame)
            with torch.no_grad():
                counts, scatter, mask = study.frame_data(frame, 0, study.device)
                expected = study.forward(frame, 0, image) + scatter + epsilon
                kl = poisson_kl(expected, counts, mask)
            metrics.append(dict(frame_index=frame, gate_index=gate, cnr=cnr,
                                wall_cov_percent=cov, poisson_kl=kl))
        print(f'Ground-truth metrics: gate {gate + 1}/{study.gate_count}', flush=True)
    metrics.sort(key=lambda row: row['frame_index'])
    with (args.output / 'ground_truth_metrics_per_frame.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    summary = []
    for name in fields[2:]:
        values = np.asarray([row[name] for row in metrics], dtype=np.float64)
        values = values[np.isfinite(values)]
        summary.append(dict(metric=name, n_valid=len(values),
                            median=float(np.median(values)) if len(values) else float('nan'),
                            p2_5=float(np.percentile(values, 2.5)) if len(values) else float('nan'),
                            p97_5=float(np.percentile(values, 97.5)) if len(values) else float('nan')))
    with (args.output / 'ground_truth_metrics_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('metric', 'n_valid', 'median', 'p2_5', 'p97_5'))
        writer.writeheader()
        writer.writerows(summary)
    metadata = dict(mode=args.mode, gates=args.gates, frames=study.num_frames,
                    gain=study.gain, epsilon=epsilon,
                    cnr='absolute difference of wall and cavity means divided by the root sum of their sample variances',
                    wall_cov_percent='100 times sample standard deviation divided by mean within the truth wall mask',
                    poisson_kl='sum of Poisson KL D(observed prompt || projected truth + TEW scatter + epsilon) over active detector bins',
                    truth='gate-averaged XCAT activity resampled onto the reconstruction volumes',
                    kl_units='sum over detector bins, not normalized per bin')
    (args.output / 'ground_truth_metrics.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
