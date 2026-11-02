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


def reconstruct(study, images, shape_xyz, multiplier, device='cpu', iterations=20,
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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--gates', required=True, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--projector-cache-size', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--bandwidth', type=float, default=60.)
    parser.add_argument('--checkpoint-interval', type=int, default=0)
    args = parser.parse_args()

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
    np.savez(args.output_dir/'geometry.npz', volume_shape_xyz=shapes,
             volume_spacing_xyz=spacing, volume_origin_xyz=study.volume_origin_xyz,
             mode='static', gates=args.gates, iterations=args.iterations,
             bandwidth=args.bandwidth, beta=args.beta if args.beta is not None else 0.005 * 8 / args.gates,
             gain=study.gain)
    mapped = np.lib.format.open_memmap(args.output_dir/'reconstruction.npy', mode='w+',
                                       dtype=np.float32, shape=(study.num_frames, study.image_size))
    mapped[:] = 1.
    images = torch.from_numpy(mapped)

    def checkpoint(iteration, result):
        mapped.flush()
        if args.checkpoint_interval and iteration % args.checkpoint_interval == 0 and iteration != args.iterations:
            np.save(args.output_dir/f'iteration_{iteration:04d}.npy', result.numpy())

    # Disk-backed snapshot keeps the synchronous prior within single-GPU memory.
    with tempfile.TemporaryDirectory(prefix='bsrem_snapshot_', dir=args.output_dir) as temp:
        snapshot_file = np.memmap(Path(temp)/'snapshot.bin', mode='w+', dtype=np.float32,
                                  shape=(study.num_frames, fine_size))
        snapshot = torch.from_numpy(snapshot_file)
        kwargs = dict(device=device, iterations=args.iterations, beta=args.beta,
                      bandwidth=args.bandwidth, snapshot=snapshot, checkpoint=checkpoint)
        reconstruct(study, images, shape_xyz, multiplier, **kwargs)
        del snapshot, snapshot_file
    mapped.flush()


if __name__ == '__main__':
    main()
