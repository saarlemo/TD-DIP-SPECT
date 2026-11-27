"""Independent ordered-subsets EM reconstructions of the cardiac gates.

The study supplies ``num_frames``, ``num_subsets``, ``initial(frame, device)``,
``frame_data(frame, subset, device)`` returning prompt, scatter and detector
mask, and paired ``forward(frame, subset, image)`` and
``backward(frame, subset, projection)`` methods. The projector must include
the acquisition gain, exposure, attenuation and physical detector mask.
Images use OMEGA's packed multiresolution volume order.
"""

import torch


from helpers import open_study


def reconstruct(study, device, iterations=10, epsilon=1e-5, checkpoint=None):
    """Yield each gate and its reconstructed image on the CPU.

    Only one gate and its subset sensitivities are kept on the device at a
    time. The prepared study determines the ordering and number of subsets.
    """
    for gate in range(study.num_frames):
        with torch.no_grad():
            image = study.initial(gate, device).clone()
            measurements = [study.frame_data(gate, subset, device)
                            for subset in range(study.num_subsets)]
            sensitivities = [
                study.backward(gate, subset, data[2]).clamp_min(epsilon)
                for subset, data in enumerate(measurements)
            ]
            for iteration in range(iterations):
                for subset, (prompt, scatter, mask) in enumerate(measurements):
                    expected = (study.forward(gate, subset, image) + scatter).clamp_min(epsilon)
                    correction = study.backward(gate, subset, mask * prompt / expected)
                    image.mul_(correction / sensitivities[subset]).clamp_(min=epsilon)
                if checkpoint is not None:
                    checkpoint(gate, iteration + 1, image)
        yield gate, image.cpu()


def main():
    import argparse
    from pathlib import Path
    import h5py
    import numpy as np
    from ssim_selection import MedianGateSSIMSelector, write_selection

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path, help='Prepared static-gated XCAT study')
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--gates', type=int, choices=(8, 16, 32), required=True)
    parser.add_argument('--device', default='cuda:0', help='PyTorch CUDA/ROCm device')
    parser.add_argument('--projector-cache-size', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # A 32^3 heart ROI at native resolution, with a fourfold coarser exterior;
    # 13 rays per dimension model the detector response.
    study = open_study('static', args.input_dir, args.output_dir, args.gates,
                       device, subsets=12, rays=13, roi_shape=(32, 32, 32),
                       coarse_scale=0.25, cache_size=args.projector_cache_size)
    packed_size = study.initial(0, device).numel()
    selector = MedianGateSSIMSelector(study)
    candidate_scores = {iteration: [] for iteration in range(1, args.iterations + 1)}
    output_path = args.output_dir / 'static_osem.h5'
    with h5py.File(output_path, 'w') as output:
        images = output.create_dataset('reconstruction_packed',
            shape=(study.num_frames, packed_size), dtype='f4',
            chunks=(1, packed_size), compression='gzip', compression_opts=1)
        output.attrs['gates'] = args.gates
        output.attrs['iterations'] = args.iterations
        output.attrs['requested_iterations'] = args.iterations
        output.attrs['volume_shape_xyz'] = np.asarray(study.volume_shape_xyz)
        output.attrs['volume_spacing_xyz'] = np.asarray(study.volume_spacing_xyz)
        output.attrs['volume_origin_xyz'] = np.asarray(study.volume_origin_xyz)
        output.attrs['gain'] = study.gain

        def save_checkpoint(frame, iteration, image):
            candidate_scores[iteration].append((frame, selector.score_frame(frame, image)))

        for frame, image in reconstruct(study, device, iterations=args.iterations,
                                        epsilon=float(study.options.epps), checkpoint=save_checkpoint):
            images[frame] = image.numpy().reshape(-1)
            output.flush()

        history = []
        best_iteration, best_score, best_gate_medians = None, -float('inf'), None
        for iteration, scores in candidate_scores.items():
            median_ssim, gate_medians = selector.aggregate(scores)
            history.append({'step': iteration, 'median_gate_ssim': median_ssim,
                            'per_gate_median_ssim': gate_medians})
            if median_ssim > best_score:
                best_iteration, best_score, best_gate_medians = iteration, median_ssim, gate_medians
        output.attrs['best_visualization_iteration'] = int(best_iteration)
        output.attrs['median_gate_ssim'] = float(best_score)
        output.attrs['visualization_metric'] = 'median_gate_ssim'
        best_path = args.output_dir / 'best_visualization.h5'
        with h5py.File(best_path, 'w') as visual:
            for name, value in output.attrs.items():
                visual.attrs[name] = value
            visual.attrs['iterations'] = int(best_iteration)
            visual.attrs['best_visualization_iteration'] = int(best_iteration)
            visual.attrs['median_gate_ssim'] = float(best_score)
            visual.attrs['visualization_metric'] = 'median_gate_ssim'
            selected = visual.create_dataset('reconstruction_packed', images.shape, dtype='f4',
                                             chunks=images.chunks, compression='gzip', compression_opts=1)
            visual.create_dataset('frame_indices', data=np.arange(study.num_frames))
            for frame, image in reconstruct(study, device, iterations=best_iteration,
                                            epsilon=float(study.options.epps)):
                selected[frame] = image.numpy().reshape(-1)
        write_selection(args.output_dir / 'selection.json', mode='static', gates=args.gates,
                        step=best_iteration, median_gate_ssim=best_score,
                        gate_medians=best_gate_medians, history=history)


if __name__ == '__main__':
    main()
