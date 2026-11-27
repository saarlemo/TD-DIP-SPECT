#!/usr/bin/env python3
"""Create ROI and full-field slice montages with per-image metrics.

Pass output directories created by ``evaluate.py`` for the same gating mode.
Each montage has gate counts as rows and reconstruction methods as columns.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHODS = (
    ('OSEM', 'OSEM'),
    ('BSREM', 'BSREM + NLM'),
    ('DIP', 'DIP'),
    ('TD-DIP', 'TD-DIP'),
    ('Ground truth', 'Ground truth'),
)


def save(fig, output, stem):
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{stem}.{extension}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def format_value(records, gate, method, metric, digits, suffix=''):
    value = records.get((gate, method, metric))
    if value is None or np.isnan(value):
        return '--'
    if np.isinf(value):
        return f'∞{suffix}'
    return f'{value:.{digits}f}{suffix}'


def panel_metrics(records, gate, method):
    psnr = format_value(records, gate, method, 'psnr_db', 2, ' dB')
    cnr = format_value(records, gate, method, 'cnr', 2)
    cov = format_value(records, gate, method, 'wall_cov_percent', 1, '%')
    values = [f'PSNR {psnr}']
    # SSIM is undefined as a useful quality measure for the reference itself.
    if method != 'Ground truth':
        values.append(f'SSIM {format_value(records, gate, method, "ssim", 3)}')
    return '   '.join(values) + f'\nCNR {cnr}   CoV {cov}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+', type=Path,
                        help='Per-gate output directories from evaluate.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--display-max', type=float, default=100,
                        help='Upper grayscale activity limit in kBq/mL')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records = {}
    panels = {}
    modes = set()
    for path in args.results:
        metadata = json.loads((path / 'result.json').read_text())
        gate = int(metadata['gates'])
        modes.add(metadata['mode'])
        if gate in panels:
            raise ValueError(f'Duplicate {gate}-gate result')
        with (path / 'image_metrics_summary.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                records[(gate, row['method'], row['metric'])] = float(row['median'])
        with np.load(path / 'phase0_slices.npz') as archive:
            panels[gate] = {key: archive[key].copy() for key in archive.files}

    if len(modes) != 1:
        raise ValueError('Plot static and dynamic studies separately')

    gate_counts = sorted(panels)
    mode = next(iter(modes))
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'figure.facecolor': 'white',
    })
    method_count = len(METHODS)

    for scope in ('roi', 'full'):
        fig, axes = plt.subplots(
            len(gate_counts), method_count,
            figsize=(2.35 * method_count, 2.75 * len(gate_counts)),
            squeeze=False,
        )
        fig.subplots_adjust(left=.065, right=.997, bottom=.025, top=.91,
                            wspace=.045, hspace=.35)
        for row, gate in enumerate(gate_counts):
            for col, (label, key_method) in enumerate(METHODS):
                ax = axes[row, col]
                key = f'{key_method}_{scope}'
                if key in panels[gate]:
                    image = np.rot90(np.flipud(panels[gate][key]), k=-1)
                    ax.imshow(image, cmap='gray', vmin=0, vmax=args.display_max,
                              interpolation='nearest')
                else:
                    ax.set_facecolor('black')
                    ax.text(.5, .5, 'Unavailable', color='white', ha='center',
                            va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    title = 'OSEM (EM)' if mode == 'dynamic' and label == 'OSEM' else label
                    ax.set_title(title, pad=8)
                if col == 0:
                    ax.set_ylabel(f'{gate} gates', labelpad=8)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if key in panels[gate]:
                    ax.text(.5, -.035, panel_metrics(records, gate, key_method),
                            ha='center', va='top', fontsize=7.4,
                            transform=ax.transAxes, clip_on=False, linespacing=1.25)
        save(fig, args.output, f'phase0_axial_{scope}')


if __name__ == '__main__':
    main()
