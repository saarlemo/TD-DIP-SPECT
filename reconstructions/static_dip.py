"""DIP for static ECG-gated XCAT studies."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist

from helpers import (
    MultiResolutionGeometry, NetworkSettings, frame_update_budgets,
    merge_ranked_dip_outputs, open_study, setup_workers, train_frame,
)
from ssim_selection import MedianGateSSIMSelector, select_from_score_files, write_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path, help='Prepared static-gated XCAT study')
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--gates', type=int, choices=(8, 16, 32), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=10000,
                        help='Equivalent shared TD-DIP update budget')
    parser.add_argument('--selection-interval', type=int, default=100,
                        help='Per-frame updates between SSIM visualization candidates')
    parser.add_argument('--frame-start', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--frame-count', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--log-interval', type=int, default=0)
    parser.add_argument('--projector-cache-size', type=int, default=8)
    parser.add_argument('--fp32', action='store_true')
    args = parser.parse_args()

    if args.steps < 1 or args.selection_interval < 1:
        parser.error('--steps and --selection-interval must be positive')
    if not 1 <= args.batch_size <= args.gates:
        parser.error('--batch-size must fit within the number of gates')

    device, rank, world_size = setup_workers(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    study = open_study(
        'static', args.input_dir, args.output_dir, args.gates, device,
        subsets=1, rays=13, roi_shape=(32, 32, 32), coarse_scale=0.25,
        cache_size=args.projector_cache_size,
    )
    torch.manual_seed(67)
    geometry = MultiResolutionGeometry.from_options(study.options)
    settings = NetworkSettings(manifold_dim=8, output_activation='softplus')
    frame_updates = frame_update_budgets(study.num_frames, args.steps * args.batch_size)
    if args.frame_start != 0 or (args.frame_count is not None and args.frame_count != study.num_frames):
        parser.error('SSIM visualization selection requires every study gate')
    frame_start = args.frame_start
    frame_stop = (study.num_frames if args.frame_count is None
                  else frame_start + args.frame_count)
    if frame_start < 0 or frame_stop > study.num_frames or frame_start >= frame_stop:
        parser.error(f'Frame range must be a non-empty subset of 0..{study.num_frames - 1}')
    selected_frames = range(frame_start, frame_stop)
    if world_size > len(selected_frames):
        parser.error('Number of workers exceeds selected frames')
    rank_frames = list(selected_frames)[rank::world_size]
    min_updates = int(frame_updates.min())
    selection_steps = set(range(args.selection_interval, min_updates + 1, args.selection_interval))
    selection_steps.add(min_updates)
    selector = MedianGateSSIMSelector(study)
    output = args.output_dir / 'static_dip.h5'
    metrics_path = args.output_dir / 'static_dip_frame_kl.csv'
    rank_output = (args.output_dir / f'.static_dip.rank{rank:03d}.h5'
                   if world_size > 1 else output)
    rank_metrics_path = (args.output_dir / f'.static_dip_frame_kl.rank{rank:03d}.csv'
                         if world_size > 1 else metrics_path)
    rank_ssim_path = args.output_dir / f'.static_dip_frame_ssim.rank{rank:03d}.csv'
    shapes = np.asarray(study.volume_shape_xyz, dtype=np.int64)
    origins = np.asarray(study.volume_origin_xyz, dtype=np.float64)
    spacings = np.asarray(study.volume_spacing_xyz, dtype=np.float64)
    with h5py.File(rank_output, 'w') as result:
        images = result.create_dataset(
            'reconstruction_packed', shape=(len(rank_frames), study.image_size), dtype=np.float32,
            chunks=(1, study.image_size), compression='gzip', compression_opts=1, shuffle=True,
        )
        result.create_dataset('frame_indices', data=np.asarray(rank_frames, dtype=np.int64))
        for name, value in (('volume_shape_xyz', shapes), ('volume_origin_xyz', origins),
                            ('volume_spacing_xyz', spacings)):
            result.attrs[name] = value
        result.attrs['gain'] = float(study.gain)
        result.attrs['gates'] = int(args.gates)
        result.attrs['mode'] = 'static'
        result.attrs['frame_start'] = int(frame_start)
        result.attrs['frame_count'] = int(len(rank_frames))
        result.attrs['td_dip_update_budget'] = int(args.steps)
        result.attrs['td_dip_batch_size'] = int(args.batch_size)
        result.attrs['updates_per_frame_min'] = int(frame_updates.min())
        result.attrs['updates_per_frame_max'] = int(frame_updates.max())
        result.attrs['total_optimizer_iterations'] = int(frame_updates.sum())
        result.attrs['rank_frame_evaluations'] = int(sum(frame_updates[frame] for frame in rank_frames))
        result.attrs['seed'] = 67
        result.attrs['network_settings_json'] = json.dumps(asdict(settings))
        result.attrs['geometry_json'] = json.dumps(geometry.to_dict())
        result.attrs['frame_order'] = 'gate order from study.json'

        with rank_metrics_path.open('w', newline='', encoding='utf-8') as stream, \
                rank_ssim_path.open('w', newline='', encoding='utf-8') as ssim_stream:
            writer = csv.DictWriter(
                stream, fieldnames=('frame_index', 'gate_index', 'midpoint_phase',
                                    'seed', 'updates', 'first_update_kl', 'final_kl')
            )
            ssim_writer = csv.DictWriter(ssim_stream, fieldnames=('frame_index', 'step', 'ssim'))
            writer.writeheader()
            ssim_writer.writeheader()
            for output_index, gate in enumerate(rank_frames):
                def record_ssim(frame, step, packed):
                    ssim_writer.writerow({'frame_index': frame, 'step': step,
                                          'ssim': selector.score_frame(frame, packed)})
                packed, first_kl, final_kl = train_frame(
                    study, gate, geometry, settings, device=device,
                    steps=int(frame_updates[gate]),
                    learning_rate=0.002,
                    epsilon=float(study.options.epps), seed=67,
                    mixed_precision=not args.fp32, log_interval=args.log_interval,
                    selection_steps=selection_steps, selection_callback=record_ssim,
                )
                images[output_index] = packed
                writer.writerow({
                    'frame_index': gate,
                    'gate_index': int(study.gate_indices[gate]),
                    'midpoint_phase': float(study.midpoint_times[gate]),
                    'seed': 67 + gate,
                    'updates': int(frame_updates[gate]),
                    'first_update_kl': first_kl,
                    'final_kl': final_kl,
                })
                if output_index % 100 == 0 or output_index + 1 == len(rank_frames):
                    print(f'static DIP rank {rank}: {output_index + 1}/{len(rank_frames)} gates', flush=True)

    shard_paths = [args.output_dir / f'.static_dip.rank{worker:03d}.h5'
                   for worker in range(world_size)]
    metrics_shard_paths = [args.output_dir / f'.static_dip_frame_kl.rank{worker:03d}.csv'
                           for worker in range(world_size)]
    ssim_shard_paths = [args.output_dir / f'.static_dip_frame_ssim.rank{worker:03d}.csv'
                        for worker in range(world_size)]
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        if world_size > 1:
            merge_ranked_dip_outputs(
                output, metrics_path, shard_paths, metrics_shard_paths,
                frame_start, frame_stop,
                int(frame_updates[frame_start:frame_stop].sum()),
                ('frame_index', 'gate_index', 'midpoint_phase', 'seed', 'updates',
                 'first_update_kl', 'final_kl'),
            )
        best_step, best_score, gate_medians, history = select_from_score_files(
            ssim_shard_paths, selector)
        write_selection(args.output_dir / 'selection.json', mode='static', gates=args.gates,
                        step=best_step, median_gate_ssim=best_score,
                        gate_medians=gate_medians, history=history)
    if world_size > 1:
        selected = torch.tensor([best_step if rank == 0 else 0], dtype=torch.int64, device=device)
        score_value = torch.tensor([best_score if rank == 0 else 0.0], dtype=torch.float64, device=device)
        dist.broadcast(selected, src=0)
        dist.broadcast(score_value, src=0)
        best_step, best_score = int(selected.item()), float(score_value.item())

    viz_rank_path = (args.output_dir / f'.static_dip_best_visualization.rank{rank:03d}.h5'
                     if world_size > 1 else args.output_dir / 'best_visualization.h5')
    with h5py.File(rank_output, 'r') as source, h5py.File(viz_rank_path, 'w') as visual:
        for name, value in source.attrs.items():
            visual.attrs[name] = value
        visual.attrs['iterations_per_frame'] = int(best_step)
        visual.attrs['best_visualization_iteration'] = int(best_step)
        visual.attrs['median_gate_ssim'] = float(best_score)
        visual.attrs['visualization_metric'] = 'median_gate_ssim'
        visual.create_dataset('frame_indices', data=np.asarray(rank_frames, dtype=np.int64))
        selected_images = visual.create_dataset(
            'reconstruction_packed', (len(rank_frames), study.image_size), dtype=np.float32,
            chunks=(1, study.image_size), compression='gzip', compression_opts=1, shuffle=True,
        )
        for output_index, frame in enumerate(rank_frames):
            packed, _, _ = train_frame(
                study, frame, geometry, settings, device=device,
                steps=best_step, learning_rate=0.002,
                epsilon=float(study.options.epps), seed=67,
                mixed_precision=not args.fp32, log_interval=0,
            )
            selected_images[output_index] = packed

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        best_path = args.output_dir / 'best_visualization.h5'
        if world_size > 1:
            with h5py.File(output, 'r') as source, h5py.File(best_path, 'w') as visual:
                for name, value in source.attrs.items():
                    visual.attrs[name] = value
                visual.attrs['iterations_per_frame'] = int(best_step)
                visual.attrs['best_visualization_iteration'] = int(best_step)
                visual.attrs['median_gate_ssim'] = float(best_score)
                visual.attrs['visualization_metric'] = 'median_gate_ssim'
                visual.create_dataset('frame_indices', data=np.arange(frame_start, frame_stop))
                selected_images = visual.create_dataset(
                    'reconstruction_packed', (len(selected_frames), study.image_size), dtype=np.float32,
                    chunks=(1, study.image_size), compression='gzip', compression_opts=1, shuffle=True,
                )
                for shard in viz_rank_path.parent.glob('.static_dip_best_visualization.rank*.h5'):
                    with h5py.File(shard, 'r') as current:
                        for row, frame in enumerate(current['frame_indices'][:]):
                            selected_images[int(frame) - frame_start] = current['reconstruction_packed'][row]
        with h5py.File(output, 'r+') as final:
            final.attrs['best_visualization_iteration'] = int(best_step)
            final.attrs['median_gate_ssim'] = float(best_score)
            final.attrs['visualization_metric'] = 'median_gate_ssim'
        for path in [*ssim_shard_paths, *metrics_shard_paths]:
            if path.exists():
                path.unlink()
        for path in args.output_dir.glob('.static_dip_best_visualization.rank*.h5'):
            path.unlink()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(f'Saved {len(selected_frames)} DIP gates to {output}', flush=True)


if __name__ == '__main__':
    main()
