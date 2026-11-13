"""Independent non-time-dependent DIP for cycle-resolved XCAT studies."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path, help='Prepared cycle-resolved XCAT study')
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--gates', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=100000,
                        help='Equivalent shared TD-DIP update budget')
    parser.add_argument('--frame-start', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--frame-count', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--log-interval', type=int, default=0)
    parser.add_argument('--projector-cache-size', type=int, default=8)
    parser.add_argument('--fp32', action='store_true')
    args = parser.parse_args()

    device, rank, world_size = setup_workers(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    study = open_study(
        'dynamic', args.input_dir, args.output_dir, args.gates, device,
        subsets=1, rays=13, roi_shape=(32, 32, 32), coarse_scale=0.25,
        cache_size=args.projector_cache_size,
    )
    torch.manual_seed(67)
    geometry = MultiResolutionGeometry.from_options(study.options)
    settings = NetworkSettings(manifold_dim=8, output_activation='softplus')
    frame_updates = frame_update_budgets(study.num_frames, args.steps * args.batch_size)
    frame_start = args.frame_start
    frame_stop = (study.num_frames if args.frame_count is None
                  else frame_start + args.frame_count)
    if frame_start < 0 or frame_stop > study.num_frames or frame_start >= frame_stop:
        parser.error(f'Frame range must be a non-empty subset of 0..{study.num_frames - 1}')
    selected_frames = range(frame_start, frame_stop)
    if world_size > len(selected_frames):
        parser.error('Number of workers exceeds selected frames')
    training_frames = sorted(
        selected_frames, key=lambda frame: (int(study.gate_indices[frame]), frame)
    )
    rank_training = training_frames[rank::world_size]
    rank_frames = sorted(rank_training)
    output_index = {frame: index for index, frame in enumerate(rank_frames)}
    decay_steps = 2500

    output = args.output_dir / 'dynamic_dip.h5'
    metrics_path = args.output_dir / 'dynamic_dip_frame_kl.csv'
    rank_output = (args.output_dir / f'.dynamic_dip.rank{rank:03d}.h5'
                   if world_size > 1 else output)
    rank_metrics_path = (args.output_dir / f'.dynamic_dip_frame_kl.rank{rank:03d}.csv'
                         if world_size > 1 else metrics_path)
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
        result.attrs['mode'] = 'dynamic'
        result.attrs['frame_start'] = int(frame_start)
        result.attrs['frame_count'] = int(len(rank_frames))
        result.attrs['td_dip_update_budget'] = int(args.steps)
        result.attrs['td_dip_batch_size'] = int(args.batch_size)
        result.attrs['updates_per_frame_min'] = int(frame_updates[frame_start:frame_stop].min())
        result.attrs['updates_per_frame_max'] = int(frame_updates[frame_start:frame_stop].max())
        result.attrs['td_dip_total_frame_evaluations'] = int(frame_updates.sum())
        result.attrs['total_frame_evaluations'] = int(sum(frame_updates[frame] for frame in rank_frames))
        result.attrs['td_dip_decay_steps'] = int(decay_steps)
        result.attrs['seed'] = 67
        result.attrs['network_settings_json'] = json.dumps(asdict(settings))
        result.attrs['geometry_json'] = json.dumps(geometry.to_dict())
        result.attrs['frame_order'] = 'study.json order; cycle-major then gate'

        with rank_metrics_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(
                stream, fieldnames=('frame_index', 'gate_index', 'midpoint_time_s',
                                    'seed', 'updates', 'first_update_kl', 'final_kl')
            )
            writer.writeheader()
            rows = []
            for index, frame in enumerate(rank_training):
                packed, first_kl, final_kl = train_frame(
                    study, frame, geometry, settings, device=device,
                    steps=int(frame_updates[frame]), batch_size=args.batch_size,
                    learning_rate=0.002, decay_steps=decay_steps, decay_factor=0.75,
                    epsilon=float(study.options.epps), seed=67,
                    mixed_precision=not args.fp32, log_interval=args.log_interval,
                )
                images[output_index[frame]] = packed
                rows.append({
                    'frame_index': frame,
                    'gate_index': int(study.gate_indices[frame]),
                    'midpoint_time_s': float(study.midpoint_times[frame]),
                    'seed': 67 + frame,
                    'updates': int(frame_updates[frame]),
                    'first_update_kl': first_kl,
                    'final_kl': final_kl,
                })
                if index % 100 == 0 or index + 1 == len(rank_training):
                    print(f'dynamic DIP rank {rank}: {index + 1}/{len(rank_training)} frames', flush=True)
            writer.writerows(sorted(rows, key=lambda row: row['frame_index']))

    if world_size > 1:
        dist.barrier()
        if rank == 0:
            shard_paths = [args.output_dir / f'.dynamic_dip.rank{worker:03d}.h5'
                           for worker in range(world_size)]
            metrics_shard_paths = [args.output_dir / f'.dynamic_dip_frame_kl.rank{worker:03d}.csv'
                                   for worker in range(world_size)]
            merge_ranked_dip_outputs(
                output, metrics_path, shard_paths, metrics_shard_paths,
                frame_start, frame_stop,
                int(frame_updates[frame_start:frame_stop].sum()),
                ('frame_index', 'gate_index', 'midpoint_time_s', 'seed', 'updates',
                 'first_update_kl', 'final_kl'),
            )
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(f'Saved {len(selected_frames)} independent DIP frames to {output}', flush=True)


if __name__ == '__main__':
    main()
