"""Single-device TD-DIP for cycle-resolved ECG-gated projections.

Each study frame is one cardiac interval and subset 0 contains its simultaneous
head pair. ``frame_data(frame, 0, device)`` returns prompt, scatter and physical
bin mask; ``forward(frame, 0, image)`` and ``backward(frame, 0, projection)``
are the matched, gain- and exposure-calibrated projection and adjoint.
"""

from collections import deque

import torch


from helpers import open_study


from helpers import MultiResolutionDIPNet3D, MultiResolutionGeometry, NetworkSettings


def reconstruct(study, model, gate_indices, midpoint_times, number_of_gates, *,
                device, steps=100000, batch_size=8, latent_dim=8,
                learning_rate=0.002, decay_steps=2500, decay_factor=0.75,
                epsilon=1e-5, seed=0, mixed_precision=False, checkpoint=None):
    """Train a phase-and-time decoder; midpoint times can use any time unit.

    ``checkpoint``, if given, receives (step, model, latents, losses) after each
    update. Save the network and latents rather than all interval images.
    """
    device = torch.device(device)
    count = study.num_frames
    latent_generator = torch.Generator(device='cpu').manual_seed(seed)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    phase = torch.as_tensor(gate_indices, dtype=torch.float32)
    angle = 2 * torch.pi * phase / number_of_gates
    time = torch.as_tensor(midpoint_times, dtype=torch.float64)
    span = time.max() - time.min()
    time = (2 * (time - time.min()) / span - 1 if span > 0 else torch.zeros_like(time)).float()
    slack = torch.rand(latent_dim - 3, generator=latent_generator)
    latents = torch.cat((torch.cos(angle)[:, None], torch.sin(angle)[:, None],
                         time[:, None], slack.expand(count, -1)), dim=1).to(device)
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps,
                                                gamma=decay_factor)
    pending, losses = deque(), []
    for step in range(1, steps + 1):
        # Carry an incomplete sweep into the next batch, as in the original.
        while len(pending) < batch_size:
            pending.extend(torch.randperm(count, generator=generator).tolist())
        batch = [pending.popleft() for _ in range(batch_size)]
        optimizer.zero_grad(set_to_none=True)
        batch_loss = torch.zeros((), device=device)
        for frame in batch:
            prompt, scatter, mask = study.frame_data(frame, 0, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=mixed_precision):
                image = model(latents[frame:frame + 1]).float()
            with torch.no_grad():
                expected = study.forward(frame, 0, image.detach().reshape(-1)) + scatter + epsilon
                valid = mask > 0
                y, mean = prompt[valid], expected[valid]
                gradient = torch.zeros_like(expected)
                gradient[valid] = 1 - y / mean
                positive = y > 0
                loss = (mean - y).sum() + (y[positive] * (y[positive].log() - mean[positive].log())).sum()
                image_gradient = study.backward(frame, 0, gradient).reshape_as(image)
                batch_loss += loss
            image.backward(image_gradient / len(batch))
        optimizer.step()
        scheduler.step()
        losses.append(float(batch_loss / len(batch)))
        if checkpoint is not None:
            checkpoint(step, model, latents, losses)
    return {'latents': latents.detach(), 'losses': losses}


def main():
    import argparse
    from dataclasses import asdict
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path, help='Prepared cycle-resolved XCAT study')
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--gates', type=int, required=True)
    parser.add_argument('--device', default='cuda:0', help='PyTorch CUDA/ROCm device')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--projector-cache-size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--checkpoint-interval', type=int, default=5000)
    parser.add_argument('--fp32', action='store_true', help='Disable BF16 network autocasting')
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The projector uses a 32^3 heart ROI, 4x coarser exterior, 13 rays per
    # dimension and one simultaneous detector-head pair per cardiac interval.
    study = open_study('dynamic', args.input_dir, args.output_dir, args.gates,
                       device, subsets=1, rays=13, roi_shape=(32, 32, 32),
                       coarse_scale=0.25, cache_size=args.projector_cache_size)
    torch.manual_seed(0)
    geometry = MultiResolutionGeometry.from_options(study.options)
    settings = NetworkSettings(manifold_dim=8, output_activation='softplus')
    model = MultiResolutionDIPNet3D(settings, geometry).to(device)

    def save(step, network, latents, losses):
        if step % args.checkpoint_interval and step != args.steps:
            return
        payload = {'step': step, 'network_settings': asdict(settings),
                   'geometry': geometry.to_dict(),
                   'volume_shape_xyz': torch.as_tensor(study.volume_shape_xyz),
                   'volume_origin_xyz': torch.as_tensor(study.volume_origin_xyz),
                   'volume_spacing_xyz': torch.as_tensor(study.volume_spacing_xyz),
                   'gain': study.gain,
                   'model_state_dict': {k: v.detach().cpu() for k, v in network.state_dict().items()},
                   'latents': latents.detach().cpu(), 'losses': losses,
                   'gates': args.gates, 'gate_indices': torch.as_tensor(study.gate_indices),
                   'midpoint_times': torch.as_tensor(study.midpoint_times)}
        torch.save(payload, args.output_dir / f'dynamic_td_dip_step{step:06d}.pt')

    reconstruct(study, model, study.gate_indices, study.midpoint_times,
                args.gates, device=device, steps=args.steps, batch_size=args.batch_size,
                mixed_precision=not args.fp32, checkpoint=save,
                epsilon=float(study.options.epps))


if __name__ == '__main__':
    main()
