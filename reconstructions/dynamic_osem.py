"""Independent single-subset EM reconstructions of cycle-resolved frames.

The study supplies ``num_frames``, ``initial(frame, device)``,
``frame_data(frame, subset, device)`` returning prompt, scatter and detector
mask, and paired ``forward(frame, subset, image)`` and
``backward(frame, subset, projection)`` methods. Its projector must include
the acquisition gain, exposure, attenuation and physical detector mask.
Images use OMEGA's packed multiresolution volume order.
"""

import torch


from helpers import open_study


def reconstruct(study, device, iterations=4, epsilon=1e-5, checkpoint=None):
    """Yield each cycle-phase frame and its reconstructed image on the CPU.

    The single subset comprises the two simultaneous head projections for the
    frame. Frames are processed individually to bound device memory.
    """
    for frame in range(study.num_frames):
        with torch.no_grad():
            image = study.initial(frame, device).clone()
            prompt, scatter, mask = study.frame_data(frame, 0, device)
            sensitivity = study.backward(frame, 0, mask).clamp_min(epsilon)
            for iteration in range(iterations):
                expected = (study.forward(frame, 0, image) + scatter).clamp_min(epsilon)
                correction = study.backward(frame, 0, mask * prompt / expected)
                image.mul_(correction / sensitivity).clamp_(min=epsilon)
                if checkpoint is not None:
                    checkpoint(frame, iteration + 1, image)
        yield frame, image.cpu()


def main():
    import argparse
    from pathlib import Path
    import h5py
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path, help='Prepared dynamic-gated XCAT study')
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--gates', type=int, required=True)
    parser.add_argument('--device', default='cuda:0', help='PyTorch CUDA/ROCm device')
    parser.add_argument('--projector-cache-size', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--checkpoint-interval', type=int, default=0,
                        help='Save intermediate iterations; 0 saves only the final image')
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # A 32^3 heart ROI at native resolution, with a fourfold coarser exterior;
    # 13 rays per dimension model the detector response.
    study = open_study('dynamic', args.input_dir, args.output_dir, args.gates,
                       device, subsets=1, rays=13, roi_shape=(32, 32, 32),
                       coarse_scale=0.25, cache_size=args.projector_cache_size)
    packed_size = study.initial(0, device).numel()
    output_path = args.output_dir / 'dynamic_osem.h5'
    with h5py.File(output_path, 'w') as output:
        images = output.create_dataset('reconstruction_packed',
            shape=(study.num_frames, packed_size), dtype='f4',
            chunks=(1, packed_size), compression='gzip', compression_opts=1)
        output.attrs['gates'] = args.gates
        output.attrs['iterations'] = args.iterations
        output.attrs['volume_shape_xyz'] = np.asarray(study.volume_shape_xyz)
        output.attrs['volume_spacing_xyz'] = np.asarray(study.volume_spacing_xyz)
        output.attrs['volume_origin_xyz'] = np.asarray(study.volume_origin_xyz)
        output.attrs['gain'] = study.gain

        def save_checkpoint(frame, iteration, image):
            if args.checkpoint_interval and iteration % args.checkpoint_interval == 0 and iteration < args.iterations:
                name = f'iteration_{iteration}'
                if name not in output:
                    output.create_dataset(name, shape=images.shape, dtype='f4',
                                          chunks=images.chunks, compression='gzip', compression_opts=1)
                output[name][frame] = image.detach().cpu().numpy().reshape(-1)

        for frame, image in reconstruct(study, device, iterations=args.iterations,
                                        epsilon=float(study.options.epps), checkpoint=save_checkpoint):
            images[frame] = image.numpy().reshape(-1)
            output.flush()


if __name__ == '__main__':
    main()
