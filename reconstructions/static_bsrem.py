"""Static-gated BSREM with a current-image spatiotemporal NLM prior.

The study supplies frame_data(frame, subset, device), forward(frame, subset,
image), backward(frame, subset, projection), initial(frame, device), and the
num_frames and num_subsets attributes. The calibrated projectors include the
exposure, attenuation and detector mask. Packed images contain the fine ROI
first, followed by the exterior volumes.
"""
import torch


from helpers import _reconstruct, nlm_residual, volume_multiplier


from helpers import open_study


def reconstruct(study, images, shape_xyz, multiplier, device='cpu', iterations=40,
                beta=None, relaxation_a=None, epsilon=1e-5, bandwidth=60.,
                snapshot=None, checkpoint=None):
    """Reconstruct cardiac gates with periodic neighboring phases.

    Images and optional snapshot use frame-major packed storage; they are updated
    in place. A disk-backed snapshot can be supplied for larger studies.
    """
    gates = study.num_frames
    if beta is None:
        beta = 0.005 * 8 / gates
    if relaxation_a is None:
        relaxation_a = 0.1 * gates / 8
    fine_size = shape_xyz[0] * shape_xyz[1] * shape_xyz[2]
    if snapshot is None:
        snapshot = torch.empty((gates, fine_size), dtype=images.dtype, device=images.device)
    return _reconstruct(study, images, snapshot, shape_xyz, multiplier, beta,
                        relaxation_a, lambda i: ((i-1) % gates, (i+1) % gates),
                        iterations, torch.device(device), epsilon, bandwidth, checkpoint)


def main():
    """Load an XCAT study and save its packed BSREM reconstruction."""
    import argparse
    from pathlib import Path
    import tempfile
    import numpy as np
    from ssim_selection import MedianGateSSIMSelector, write_selection

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--gates', required=True, type=int, choices=(8, 16, 32))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--projector-cache-size', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--bandwidth', type=float, default=60.)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    study = open_study('static', args.input_dir, args.output_dir, args.gates, device,
                       subsets=12, rays=13, roi_shape=(32, 32, 32), coarse_scale=.25,
                       cache_size=args.projector_cache_size)
    shapes = study.volume_shape_xyz
    spacing = study.volume_spacing_xyz
    shape_xyz = tuple(int(i) for i in shapes[0])
    fine_size = int(np.prod(shape_xyz))
    multiplier = volume_multiplier(shapes, spacing, device)
    mapped = np.lib.format.open_memmap(args.output_dir/'reconstruction.npy', mode='w+',
                                       dtype=np.float32, shape=(study.num_frames, study.image_size))
    mapped[:] = 1.
    images = torch.from_numpy(mapped)
    selector = MedianGateSSIMSelector(study)
    history = []
    best_iteration, best_score, best_gate_medians = None, -float('inf'), None

    def checkpoint(iteration, result):
        nonlocal best_iteration, best_score, best_gate_medians
        median_ssim, gate_medians = selector.score(
            (frame, result[frame]) for frame in range(study.num_frames))
        history.append({'step': int(iteration), 'median_gate_ssim': median_ssim,
                        'per_gate_median_ssim': gate_medians})
        if median_ssim > best_score:
            best_iteration, best_score, best_gate_medians = int(iteration), median_ssim, gate_medians
        mapped.flush()

    # Disk-backed snapshot keeps the synchronous prior within single-GPU memory.
    with tempfile.TemporaryDirectory(prefix='bsrem_snapshot_', dir=args.output_dir) as temp:
        snapshot_file = np.memmap(Path(temp)/'snapshot.bin', mode='w+', dtype=np.float32,
                                  shape=(study.num_frames, fine_size))
        snapshot = torch.from_numpy(snapshot_file)
        reconstruct(study, images, shape_xyz, multiplier, device=device,
                    iterations=args.iterations, beta=args.beta, bandwidth=args.bandwidth,
                    snapshot=snapshot, checkpoint=checkpoint)
        del snapshot, snapshot_file
    mapped.flush()
    best_path = args.output_dir / 'best_visualization.npy'
    best_mapped = np.lib.format.open_memmap(best_path, mode='w+', dtype=np.float32,
                                           shape=(study.num_frames, study.image_size))
    if best_iteration == args.iterations:
        best_mapped[:] = mapped[:]
    else:
        best_mapped[:] = 1.
        best_images = torch.from_numpy(best_mapped)
        with tempfile.TemporaryDirectory(prefix='bsrem_visual_snapshot_', dir=args.output_dir) as temp:
            snapshot_file = np.memmap(Path(temp)/'snapshot.bin', mode='w+', dtype=np.float32,
                                      shape=(study.num_frames, fine_size))
            snapshot = torch.from_numpy(snapshot_file)
            reconstruct(study, best_images, shape_xyz, multiplier, device=device,
                        iterations=best_iteration, beta=args.beta, bandwidth=args.bandwidth,
                        snapshot=snapshot)
            del snapshot, snapshot_file
    best_mapped.flush()
    np.savez(args.output_dir/'geometry.npz', volume_shape_xyz=shapes,
             volume_spacing_xyz=spacing, volume_origin_xyz=study.volume_origin_xyz,
             mode='static', gates=args.gates, iterations=args.iterations,
             best_visualization_iteration=best_iteration,
             median_gate_ssim=best_score, visualization_metric='median_gate_ssim',
             bandwidth=args.bandwidth, beta=args.beta if args.beta is not None else 0.005 * 8 / args.gates,
             gain=study.gain)
    write_selection(args.output_dir/'selection.json', mode='static', gates=args.gates,
                    step=best_iteration, median_gate_ssim=best_score,
                    gate_medians=best_gate_medians, history=history)


if __name__ == '__main__':
    main()
