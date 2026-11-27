"""Median gate-wise SSIM scoring for reconstruction iteration selection."""

import json
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from prepare_data import resample_density


class MedianGateSSIMSelector:
    """Score a candidate reconstruction by median SSIM across cardiac gates.

    For dynamic studies, each gate's SSIM is first aggregated across all
    acquisition cycles with a median, then those gate medians are combined
    with a median. This keeps every cycle in the selection rule while giving
    each cardiac gate equal weight.
    """

    def __init__(self, study):
        self.study = study
        self.gates = int(study.gate_count)
        self.mode = 'dynamic' if study.dynamic else 'static'
        self.shape = tuple(int(i) for i in study.volume_shape_xyz[0])
        self.fine_size = int(np.prod(self.shape))
        manifest = study.static_manifest
        peak = 75 * 210 / 11_681_335 * 1e6 / np.prod(manifest['voxel_size_mm'])
        self.c1, self.c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
        self.truth = []
        self.truth_stats = []
        native_spacing = np.asarray(manifest['voxel_size_mm'], dtype=np.float64)
        native_origin = -np.asarray(manifest['image_shape_xyz'], dtype=np.float64) * native_spacing / 2
        origin = np.asarray(study.volume_origin_xyz[0], dtype=np.float64)
        spacing = np.asarray(study.volume_spacing_xyz[0], dtype=np.float64)
        for filename in manifest['truth_files']:
            with np.load(study.static_root / filename) as source:
                density = source['data'] * np.float32(1e6 / np.prod(native_spacing))
            truth = resample_density(density, native_origin, native_spacing,
                                     self.shape, origin, spacing)
            truth = truth.astype(np.float64, copy=False)
            self.truth.append(truth)
            smooth = self._smooth
            mean = smooth(truth)
            self.truth_stats.append((mean, smooth(truth * truth) - mean * mean))

    @staticmethod
    def _smooth(image):
        return gaussian_filter(image, sigma=1.5, truncate=3.5, mode='reflect')

    def _score_image(self, frame, packed):
        gate = int(self.study.gate_indices[frame])
        if hasattr(packed, 'detach'):
            packed = packed.detach().cpu().numpy()
        image = np.asarray(packed).reshape(-1)[:self.fine_size].reshape(self.shape, order='F').astype(np.float64)
        truth = self.truth[gate]
        ux, vx = self.truth_stats[gate]
        uy = self._smooth(image)
        vy = self._smooth(image * image) - uy * uy
        cov = self._smooth(truth * image) - ux * uy
        c1, c2 = self.c1, self.c2
        ssim = ((2 * ux * uy + c1) * (2 * cov + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
        return float(ssim[5:-5, 5:-5, 5:-5].mean())

    def score_frame(self, frame, packed):
        """Score one study frame, returning its 3D SSIM value."""
        return self._score_image(int(frame), packed)

    def aggregate(self, frame_scores):
        """Aggregate ``(frame_index, ssim)`` values by gate and then median."""
        per_gate = [[] for _ in range(self.gates)]
        seen = np.zeros(self.study.num_frames, dtype=bool)
        for frame, score in frame_scores:
            frame = int(frame)
            if frame < 0 or frame >= self.study.num_frames or seen[frame]:
                raise ValueError(f'Invalid or duplicate frame {frame} in SSIM candidate')
            seen[frame] = True
            per_gate[int(self.study.gate_indices[frame])].append(float(score))
        if not seen.all():
            raise ValueError('SSIM candidate does not contain every study frame')
        gate_medians = [float(np.median(values)) for values in per_gate]
        return float(np.median(gate_medians)), gate_medians

    def score(self, frame_images):
        """Score all frames, then aggregate cycle medians within each gate."""
        return self.aggregate((frame, self.score_frame(frame, packed))
                              for frame, packed in frame_images)


def write_selection(path, *, mode, gates, step, median_gate_ssim, gate_medians, history):
    """Write the selected optimizer step and its candidate score history."""
    Path(path).write_text(json.dumps({
        'mode': mode,
        'gates': int(gates),
        'best_visualization_step': int(step),
        'best_visualization_iteration': int(step),
        'metric': 'median_gate_ssim',
        'median_gate_ssim': float(median_gate_ssim),
        'per_gate_median_ssim': [float(value) for value in gate_medians],
        'dynamic_aggregation': 'median_ssim_across_cycles_per_gate_then_median_across_gates'
            if mode == 'dynamic' else 'ssim_per_gate_then_median_across_gates',
        'candidate_history': history,
    }, indent=2) + '\n')


def select_from_score_files(paths, selector):
    """Select the best shared iteration from per-rank frame SSIM CSV files."""
    from collections import defaultdict

    candidates = defaultdict(list)
    for path in paths:
        with Path(path).open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                candidates[int(row['step'])].append((int(row['frame_index']), float(row['ssim'])))
    history = []
    best = None
    for step in sorted(candidates):
        score, gate_medians = selector.aggregate(candidates[step])
        history.append({'step': step, 'median_gate_ssim': score,
                        'per_gate_median_ssim': gate_medians})
        if best is None or score > best[1]:
            best = (step, score, gate_medians)
    if best is None:
        raise ValueError('No DIP iteration SSIM candidates were recorded')
    return (*best, history)
