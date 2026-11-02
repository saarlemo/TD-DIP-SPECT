"""Single-device TD-DIP for conventional ECG-gated projections.

The study supplies a full-view subset 0 for each gate via
``frame_data(gate, 0, device) -> (prompt, scatter, mask)`` and matching
``forward(gate, 0, image)`` and ``backward(gate, 0, projection)`` methods.
The network maps one latent row to one packed multiresolution image. The
projector includes the acquisition exposure and calibration gain.
"""

import torch


from helpers import open_study


from helpers import MultiResolutionDIPNet3D, MultiResolutionGeometry, NetworkSettings


def reconstruct(study, model, *, device, steps=3000, batch_size=8,
                latent_dim=8, learning_rate=0.002, decay_steps=2500,
                decay_factor=0.75, epsilon=1e-5, seed=0, mixed_precision=False,
                checkpoint=None):
    """Train the circular-latent decoder and return its latents and losses.

    ``checkpoint``, if given, is called with (step, model, latents, losses)
    after each update. It can save the model without materializing all images.
    """
    device = torch.device(device)
    gates = study.num_frames
    if not 1 <= batch_size <= gates:
        raise ValueError('batch_size must fit within the number of gates')
    latent_generator = torch.Generator(device='cpu').manual_seed(seed)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    angle = 2 * torch.pi * torch.arange(gates, dtype=torch.float32) / gates
    slack = torch.rand(latent_dim - 2, generator=latent_generator)
    latents = torch.cat((torch.cos(angle)[:, None], torch.sin(angle)[:, None],
                         slack.expand(gates, -1)), dim=1).to(device)
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps,
                                                gamma=decay_factor)
    losses = []
    while len(losses) < steps:
        # The original workflow drops the incomplete batch at each gate sweep.
        order = torch.randperm(gates, generator=generator).tolist()
        for start in range(0, gates - batch_size + 1, batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device)
            for gate in batch:
                prompt, scatter, mask = study.frame_data(gate, 0, device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=mixed_precision):
                    image = model(latents[gate:gate + 1]).float()
                with torch.no_grad():
                    expected = study.forward(gate, 0, image.detach().reshape(-1)) + scatter + epsilon
                    valid = mask > 0
                    y, mean = prompt[valid], expected[valid]
                    gradient = torch.zeros_like(expected)
                    gradient[valid] = 1 - y / mean
                    positive = y > 0
                    loss = (mean - y).sum() + (y[positive] * (y[positive].log() - mean[positive].log())).sum()
                    image_gradient = study.backward(gate, 0, gradient).reshape_as(image)
                    batch_loss += loss
                image.backward(image_gradient / len(batch))
            optimizer.step()
            scheduler.step()
            losses.append(float(batch_loss / len(batch)))
            if checkpoint is not None:
                checkpoint(len(losses), model, latents, losses)
            if len(losses) == steps:
                break
    return {'latents': latents.detach(), 'losses': losses}


def main():
    import argparse
    from dataclasses import asdict
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path, help='Prepared static-gated XCAT study')
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--gates', type=int, required=True)
    parser.add_argument('--device', default='cuda:0', help='PyTorch CUDA/ROCm device')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--projector-cache-size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--checkpoint-interval', type=int, default=200)
    parser.add_argument('--fp32', action='store_true', help='Disable BF16 network autocasting')
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The projector uses a 32^3 high-resolution heart ROI with a 4x coarser
    # exterior, 13 rays per dimension and the full projection set per gate.
    study = open_study('static', args.input_dir, args.output_dir, args.gates,
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
                   'gates': args.gates}
        torch.save(payload, args.output_dir / f'static_td_dip_step{step:06d}.pt')

    result = reconstruct(study, model, device=device, steps=args.steps, batch_size=args.batch_size,
                         mixed_precision=not args.fp32, checkpoint=save,
                         epsilon=float(study.options.epps))
    model.eval()
    with torch.no_grad():
        images = [model(latent[None]).float().cpu() for latent in result['latents']]
    torch.save(torch.cat(images), args.output_dir / 'static_td_dip_images.pt')


if __name__ == '__main__':
    main()
