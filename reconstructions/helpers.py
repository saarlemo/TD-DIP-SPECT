"""Shared study, TD-DIP network and BSREM + NLM utilities."""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.ndimage import zoom
from scipy.sparse import csr_matrix


def _volume(path):
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path) as values:
            return values['data']
    if path.suffix == '.npy':
        return np.load(path)
    return loadmat(path)['attenuation_cm1']


def _configure_multiresolution(options, manifest, roi_shape, scale):
    from omegatomo.util.efovcorrection import CTEFOVCorrection

    full_shape = np.array([options.Nx, options.Ny, options.Nz], dtype=int)
    full_fov = np.array([options.FOVa_x, options.FOVa_y, options.axial_fov])
    spacing = full_fov / full_shape
    requested = np.asarray(manifest['heart_center_mm'])
    center = np.floor(requested / (spacing / scale) + .5) * (spacing / scale)
    options.Nx, options.Ny, options.Nz = roi_shape
    options.FOVa_x, options.FOVa_y, options.axial_fov = np.asarray(roi_shape) * spacing
    options.oOffsetX, options.oOffsetY, options.oOffsetZ = center
    options.useEFOV = options.useMultiResolutionVolumes = True
    options.multiResolutionScale = scale
    options.eFOVSize = full_fov
    options.eFOVShift = -center
    options.veritonHeartRequestedMm = requested
    options.veritonHeartCenterMm = center
    options.veritonFullCenterMm = np.zeros(3)
    CTEFOVCorrection(options)


class XCATStudy:
    def __init__(self, mode, input_dir, output_dir, gates, device, subsets,
                 rays=13, roi_shape=(32, 32, 32), coarse_scale=.25, cache_size=8,
                 full_fov=False, gain_override=None):
        self.cache_size = cache_size
        self.full_fov = full_fov
        self.mode = mode
        self.root = Path(input_dir).resolve()
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        import cupy
        cupy.cuda.Device(self.device.index or 0).use()
        self.gate_count = int(gates)
        self.num_subsets = int(subsets)
        self.rays, self.roi_shape, self.coarse_scale = rays, roi_shape, coarse_scale
        self.manifest = json.loads((self.root / 'study.json').read_text())
        if int(self.manifest['gate_count']) != self.gate_count:
            raise ValueError('Input gate count does not match the requested gate count')
        self.dynamic = mode == 'dynamic'
        if self.dynamic:
            self.static_root = (self.root / self.manifest['source_static_study']).resolve().parent
            calibration = self.root / self.manifest['gain_calibration_file']
            with np.load(self.root / self.manifest['timeline_geometry_file']) as source:
                self.timeline = {name: source[name].copy() for name in source.files}
            self.num_frames = int(self.manifest['interval_count'])
            self.gate_indices = self.timeline['gate_index'].astype(int)
            self.midpoint_times = (self.timeline['interval_start_s'] + self.timeline['interval_stop_s']) / 2
            self.counts = {}
            for window, name in self.manifest['count_files'].items():
                with np.load(self.root / name) as a:
                    self.counts[window] = csr_matrix((a['counts'], a['indices'].astype(np.int32),
                                                      a['row_ptr'].astype(np.int32)), shape=tuple(a['shape']))
            self.exposure = float(self.manifest['exposure_s_per_interval_head'])
        else:
            self.static_root = self.root
            calibration = self.root.parent / 'gain_calibration.json'
            self.num_frames = self.gate_count
            self.gate_indices = np.arange(self.gate_count)
            self.midpoint_times = (self.gate_indices + .5) / self.gate_count
        self.static_manifest = json.loads((self.static_root / 'study.json').read_text())
        self.gain = (float(json.loads(calibration.read_text())['gain'])
                     if gain_override is None else float(gain_override))
        self.exposure = (self.exposure if self.dynamic else
                         np.broadcast_to(np.asarray(self.static_manifest['exposure_s_per_gate_view']),
                                         (self.gate_count,)))
        m = self.static_manifest
        x = (np.arange(m['detector_shape'][0]) - (m['detector_shape'][0] - 1) / 2) * m['detector_pixel_mm'][0]
        z = (np.arange(m['detector_shape'][1]) - (m['detector_shape'][1] - 1) / 2) * m['detector_pixel_mm'][1]
        self.active_mask = ((abs(x[:, None]) <= m['physical_detector_transverse_mm'] / 2) &
                            (abs(z[None, :]) <= m['physical_detector_axial_mm'] / 2))
        self._projectors = OrderedDict()
        self._gate_data = None
        self._gate = None
        self.options = self.projector(0)
        self.volume_shape_xyz = np.column_stack((self.options.Nx, self.options.Ny, self.options.Nz)).astype(int)
        self.volume_spacing_xyz = np.column_stack((self.options.dx, self.options.dy, self.options.dz))
        self.volume_origin_xyz = np.column_stack((self.options.bx, self.options.by, self.options.bz))
        self.image_size = int(np.prod(self.volume_shape_xyz, axis=1).sum())

    def _data(self, gate):
        if self._gate != gate:
            self._gate_data = loadmat(self.static_root / self.static_manifest['projection_files'][gate],
                                      variable_names=['prompt', 'scatter', 'radius_to_crystal_mm'])
            self._gate = gate
        return self._gate_data

    def projector(self, frame):
        gate = int(self.gate_indices[frame])
        if gate in self._projectors:
            self._projectors.move_to_end(gate)
            return self._projectors[gate]
        from omegatomo.projector import proj
        m = self.static_manifest
        values = self._data(gate)
        views = np.arange(240)
        if self.dynamic:
            views = np.r_[np.arange(0, 240, 2), np.arange(1, 240, 2)]
        o = proj.projectorClass()
        o.SPECT = True
        o.Nx, o.Ny, o.Nz = m['reconstruction_shape_xyz']
        o.FOVa_x, o.FOVa_y, o.axial_fov = np.asarray(m['image_shape_xyz']) * np.asarray(m['voxel_size_mm'])
        if self.dynamic:
            o.SinM = np.zeros((*m['detector_shape'], len(views)), dtype=np.float32, order='F')
            radius = self.timeline['radius_to_crystal_mm'].reshape(self.gate_count, 240)[gate]
        else:
            o.SinM = np.asfortranarray(values['prompt'][:, :, views])
            radius = values['radius_to_crystal_mm'].reshape(-1)
        o.nRowsD, o.nColsD, o.nProjections = o.SinM.shape
        o.nHeads = 2
        o.angles = np.asarray(m['omega_angles_deg'], dtype=np.float32)[views]
        o.swivelAngles = o.angles + 180
        o.DetectorVector = np.asarray(m['head_ids_zero_based'], dtype=np.uint32)[views]
        o.radiusPerProj = np.asarray(radius, dtype=np.float32)[views].copy()
        o.dPitchX, o.dPitchY = m['detector_pixel_mm']
        o.colLxy = o.colLz = float(m['collimator_length_mm'])
        o.colR = float(m['collimator_hole_radius_mm'])
        o.colD = 0.
        o.iR = float(m['intrinsic_fwhm_mm'])
        o.colFxy = o.colFz = np.inf
        mu = _volume(self.static_root / m['attenuation_files'][gate])
        shape = np.asarray(m['reconstruction_shape_xyz'])
        if tuple(mu.shape) != tuple(shape):
            mu = zoom(mu, shape / mu.shape, order=1, mode='nearest', prefilter=False, grid_mode=True)
        o.vaimennus = np.asfortranarray(mu, dtype=np.float32)
        o.attenuation_correction = o.CT_attenuation = o.attIncm = True
        o.scatter_correction = o.normalization_correction = False
        o.corrections_during_reconstruction = True
        o.projector_type = 1
        o.useCUDA = o.useCuPy = o.useTorch = True
        o.deviceNum = self.device.index or 0
        o.useAF = o.useImages = o.useCPU = False
        o.useHIPFastAtomics = torch.version.hip is not None
        o.subsets = 120 if self.dynamic else self.num_subsets
        o.subsetType = 8 if self.dynamic else 9
        o.seed = -1 if self.dynamic else 20261005
        o.n_rays_transaxial = o.n_rays_axial = self.rays
        o.verbose = 0
        if not self.full_fov:
            _configure_multiresolution(o, m, self.roi_shape, self.coarse_scale)
        o.addProjector()
        o.initProj()
        self._projectors[gate] = o
        if len(self._projectors) > self.cache_size:
            self._projectors.popitem(last=False)
        return o

    def initial(self, frame, device):
        values = self.projector(frame).x0
        if isinstance(values, (list, tuple)):
            values = np.concatenate([np.asarray(v).ravel(order='F') for v in values])
        return torch.as_tensor(np.asarray(values).ravel(order='F'), dtype=torch.float32, device=device).clone()

    def frame_data(self, frame, subset, device):
        if self.dynamic:
            data = {k: torch.as_tensor(v.getrow(frame).toarray().ravel(), dtype=torch.float32, device=device)
                    for k, v in self.counts.items()}
            mask = torch.as_tensor(np.tile(self.active_mask.ravel(order='F'), 2), dtype=torch.float32, device=device)
            return data['prompt'] * mask, 2 * (data['lower'] + data['upper']) * mask, mask
        o = self.projector(frame)
        counts = np.asarray(o.nProjSubset).ravel()
        offsets = np.r_[0, np.cumsum(counts)]
        indices = np.asarray(o.index, dtype=int).ravel()[offsets[subset]:offsets[subset + 1]]
        values = self._data(frame)
        mask = np.tile(self.active_mask.ravel(order='F'), len(indices)).astype(np.float32)
        prompt = values['prompt'][:, :, indices].ravel(order='F') * mask
        scatter = values['scatter'][:, :, indices].ravel(order='F') * mask
        return tuple(torch.as_tensor(a, dtype=torch.float32, device=device) for a in (prompt, scatter, mask))

    def _selection(self, frame, subset):
        return int(self.timeline['position_index'][frame]) if self.dynamic else subset

    def _scale(self, frame):
        exposure = self.exposure if self.dynamic else self.exposure[frame]
        return self.gain * float(exposure)

    def _mask(self, length, device):
        return torch.as_tensor(np.tile(self.active_mask.ravel(order='F'),
                                       length // self.active_mask.size), device=device, dtype=torch.float32)

    def forward(self, frame, subset, image):
        o = self.projector(frame)
        sizes = np.prod(np.column_stack((o.Nx, o.Ny, o.Nz)), axis=1).tolist()
        volumes = list(torch.split(image.reshape(-1), sizes)) if not self.full_fov else image.reshape(-1)
        result = o.forwardProject(volumes, subset=self._selection(frame, subset))
        return result * self._mask(result.numel(), result.device) * self._scale(frame)

    def backward(self, frame, subset, projection):
        o = self.projector(frame)
        weighted = projection * self._mask(projection.numel(), projection.device) * self._scale(frame)
        result = o.backwardProject(weighted, subset=self._selection(frame, subset))
        return torch.cat([v.reshape(-1) for v in result]) if isinstance(result, (list, tuple)) else result.reshape(-1)


def open_study(mode, input_dir, output_dir, gates, device, subsets,
               rays=13, roi_shape=(32, 32, 32), coarse_scale=.25, cache_size=8,
               full_fov=False, gain_override=None):
    return XCATStudy(mode, input_dir, output_dir, gates, device, subsets, rays,
                     roi_shape, coarse_scale, cache_size, full_fov, gain_override)

@dataclass(frozen=True)
class NetworkSettings:
    latent_spatial_shape: tuple[int, int, int] = (8, 8, 8)
    manifold_dim: int = 2
    hidden_layer_size: int = 256
    mapping_depth: int = 0
    image_feature_channels: int = 32
    roi_feature_channels: int = 64
    conv_repeats_per_scale: int = 2
    input_channels: int = 1
    output_channels: int = 1
    need_conv_bias: bool = False
    upsample_factor: int | tuple[int, int, int] = 2
    upsample_mode: str = "nearest"
    decoder_batch_norm: bool = True
    output_activation: str | None = "relu"
    output_leaky_relu_slope: float = 0.01

    @property
    def mapped_dim(self) -> int:
        return self.input_channels * int(np.prod(self.latent_spatial_shape))


def _conv3d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias
    )


def _decoder_spatial_schedule(input_shape, output_shape, expansion_factor):
    input_shape, output_shape = tuple(input_shape), tuple(output_shape)
    if len(input_shape) != 3 or len(output_shape) != 3:
        raise ValueError("spatial shapes must have three dimensions")
    if any(dimension < 1 for dimension in input_shape + output_shape):
        raise ValueError("spatial dimensions must be positive")
    if any(output < input for input, output in zip(input_shape, output_shape)):
        raise ValueError("decoder output cannot be smaller than its input")
    if isinstance(expansion_factor, int):
        factors = (expansion_factor,) * 3
    elif isinstance(expansion_factor, tuple) and len(expansion_factor) == 3:
        factors = expansion_factor
    else:
        raise TypeError("upsample_factor must be an int or three-element tuple")
    if any(factor < 1 for factor in factors):
        raise ValueError("upsample_factor components must be positive")
    if any(
        factor == 1 and input < output
        for factor, input, output in zip(factors, input_shape, output_shape)
    ):
        raise ValueError("a growing axis needs an expansion factor greater than one")
    schedule, current = [input_shape], input_shape
    while current != output_shape:
        next_stage = tuple(
            current_axis
            if current_axis >= output_axis
            else min(current_axis * factor, output_axis)
            for current_axis, output_axis, factor in zip(current, output_shape, factors)
        )
        if next_stage == current:
            raise ValueError("upsample_factor cannot reach the requested output shape")
        schedule.append(next_stage)
        current = next_stage
    return tuple(schedule)


class MappingNet3D(nn.Module):
    def __init__(self, settings: NetworkSettings) -> None:
        super().__init__()
        layers = [nn.Linear(settings.manifold_dim, settings.hidden_layer_size), nn.ReLU(inplace=True)]
        for _ in range(settings.mapping_depth):
            layers.extend(
                [
                    nn.Linear(settings.hidden_layer_size, settings.hidden_layer_size),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Linear(settings.hidden_layer_size, settings.mapped_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class CenterCrop3D(nn.Module):
    def __init__(self, target_shape) -> None:
        super().__init__()
        self.target_shape = tuple(target_shape)

    def forward(self, x):
        _, _, depth, height, width = x.shape
        if any(
            target > actual
            for target, actual in zip(self.target_shape, (depth, height, width))
        ):
            raise ValueError("target_shape cannot be larger than the decoder output")
        depth_start = (depth - self.target_shape[0]) // 2
        height_start = (height - self.target_shape[1]) // 2
        width_start = (width - self.target_shape[2]) // 2
        return x[
            :,
            :,
            depth_start : depth_start + self.target_shape[0],
            height_start : height_start + self.target_shape[1],
            width_start : width_start + self.target_shape[2],
        ]


class Net3D(nn.Module):
    def __init__(self, settings: NetworkSettings, target_shape, *, initialize_features=True) -> None:
        super().__init__()
        if settings.upsample_mode not in ("trilinear", "nearest"):
            raise ValueError("upsample_mode must be 'trilinear' or 'nearest'")

        def norm_layer(channels):
            if not settings.decoder_batch_norm:
                return nn.Identity()
            return nn.BatchNorm3d(channels, affine=True, track_running_stats=False)

        feature_channels = settings.image_feature_channels
        layers = []
        if initialize_features:
            layers = [
                _conv3d(settings.input_channels, feature_channels, 3, settings.need_conv_bias),
                norm_layer(feature_channels),
                nn.ReLU(inplace=True),
            ]
            for _ in range(settings.conv_repeats_per_scale):
                layers.extend(
                    [
                        _conv3d(feature_channels, feature_channels, 3, settings.need_conv_bias),
                        norm_layer(feature_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
        current_channels = feature_channels if initialize_features else settings.input_channels
        schedule = _decoder_spatial_schedule(
            settings.latent_spatial_shape, target_shape, settings.upsample_factor
        )
        for input_shape, output_shape in zip(schedule[:-1], schedule[1:]):
            scale = tuple(output // input for input, output in zip(input_shape, output_shape))
            exact_expansion = all(
                value > 1 and output == input * value
                for input, output, value in zip(input_shape, output_shape, scale)
            )
            upsample_kwargs = (
                {"scale_factor": scale, "mode": settings.upsample_mode}
                if exact_expansion
                else {"size": output_shape, "mode": settings.upsample_mode}
            )
            if settings.upsample_mode == "trilinear":
                upsample_kwargs["align_corners"] = False
            layers.extend(
                [
                    nn.Upsample(**upsample_kwargs),
                    _conv3d(current_channels, feature_channels, 3, settings.need_conv_bias),
                    norm_layer(feature_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            current_channels = feature_channels
            for _ in range(settings.conv_repeats_per_scale):
                layers.extend(
                    [
                        _conv3d(feature_channels, feature_channels, 3, settings.need_conv_bias),
                        norm_layer(feature_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
        layers.append(_conv3d(feature_channels, settings.output_channels, 3, settings.need_conv_bias))
        self.net = nn.Sequential(*layers)
        self.crop = CenterCrop3D(target_shape)

    def forward(self, z):
        return self.crop(self.net(z))


@dataclass(frozen=True)
class MultiResolutionGeometry:
    """Initialized OMEGA geometry, expressed in PyTorch's (z, y, x) order."""

    low_shape_zyx: tuple[int, int, int]
    main_shape_zyx: tuple[int, int, int]
    main_low_shape_zyx: tuple[int, int, int]
    main_start_zyx: tuple[int, int, int]
    exterior_starts_zyx: tuple[tuple[int, int, int], ...]
    exterior_shapes_zyx: tuple[tuple[int, int, int], ...]

    @classmethod
    def from_options(cls, options):
        sizes = np.column_stack((options.Nx, options.Ny, options.Nz)).astype(np.int64)
        scale = float(options.multiResolutionScale)
        if not 0 < scale < 1:
            raise ValueError("multiResolutionScale must be between zero and one (0.25 for 4x)")
        main = np.floor(sizes[0] * scale + 0.5).astype(np.int64)
        if np.any(main < 1) or not np.allclose(main / scale, sizes[0]):
            raise ValueError("main volume dimensions must align with the low-resolution grid")
        if len(sizes) == 7:
            before = np.array([sizes[3, 0], sizes[5, 1], sizes[1, 2]])
            after = np.array([sizes[4, 0], sizes[6, 1], sizes[2, 2]])
            starts = [(before[0], before[1], 0), (before[0], before[1], before[2]+main[2]),
                      (0, 0, 0), (before[0]+main[0], 0, 0),
                      (before[0], 0, 0), (before[0], before[1]+main[1], 0)]
        elif len(sizes) == 5:
            before = np.array([sizes[1, 0], sizes[3, 1], 0])
            after = np.array([sizes[2, 0], sizes[4, 1], 0])
            starts = [(0, 0, 0), (before[0]+main[0], 0, 0),
                      (before[0], 0, 0), (before[0], before[1]+main[1], 0)]
        elif len(sizes) == 3:
            before = np.array([0, 0, sizes[1, 2]])
            after = np.array([0, 0, sizes[2, 2]])
            starts = [(0, 0, 0), (0, 0, before[2]+main[2])]
        else:
            raise ValueError("initialized multi-resolution geometry needs 3, 5, or 7 volumes")
        low = np.maximum(before + main + after, sizes[1:].max(axis=0))
        reverse = lambda values: tuple(int(value) for value in values[::-1])
        return cls(reverse(low), reverse(sizes[0]), reverse(main), reverse(before),
                   tuple(reverse(start) for start in starts),
                   tuple(reverse(size) for size in sizes[1:]))

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        for name in values:
            values[name] = tuple(tuple(item) for item in values[name]) if name.startswith("exterior_") else tuple(values[name])
        return cls(**values)

    @property
    def main_slices_zyx(self):
        return tuple(slice(start, start + size) for start, size in
                     zip(self.main_start_zyx, self.main_low_shape_zyx))

    @property
    def packed_size(self):
        return int(np.prod(self.main_shape_zyx) +
                   sum(np.prod(shape) for shape in self.exterior_shapes_zyx))

    def pack(self, high, low):
        """Main first, then OMEGA exterior slabs; x varies fastest in each block."""
        if tuple(high.shape[2:]) != self.main_shape_zyx or tuple(low.shape[2:]) != self.low_shape_zyx:
            raise ValueError("image shapes do not match multi-resolution geometry")
        blocks = [high.flatten(start_dim=1)]
        for start, shape in zip(self.exterior_starts_zyx, self.exterior_shapes_zyx):
            slices = tuple(slice(a, a+n) for a, n in zip(start, shape))
            block = low[(slice(None), slice(None), *slices)]
            if tuple(block.shape[2:]) != shape:
                raise ValueError("exterior slab lies outside the low-resolution grid")
            blocks.append(block.flatten(start_dim=1))
        return torch.cat(blocks, dim=1)


class MultiResolutionDIPNet3D(nn.Module):
    """Decode full-FOV features, crop the heart ROI, then refine that ROI.

    The full-FOV intensity head supplies only the disjoint low-resolution
    exterior slabs. The high-resolution head supplies the main volume. Both
    heads share the low-resolution feature decoder and mapping network.
    """

    def __init__(self, settings, options, output_mask=None):
        super().__init__()
        from dataclasses import replace

        if settings.output_channels != 1:
            raise ValueError("the OMEGA packed image requires output_channels=1")
        self.settings = settings
        self.geometry = options if isinstance(options, MultiResolutionGeometry) else MultiResolutionGeometry.from_options(options)
        self.mapping_net = MappingNet3D(settings)
        low_decoder = Net3D(settings, self.geometry.low_shape_zyx)
        # Keep the full feature tensor at the low-resolution grid for refinement.
        self.low_decoder = nn.Sequential(*list(low_decoder.net.children())[:-1])
        self.low_head = list(low_decoder.net.children())[-1]
        high_settings = replace(settings, latent_spatial_shape=self.geometry.main_low_shape_zyx,
                                input_channels=settings.image_feature_channels,
                                image_feature_channels=settings.roi_feature_channels)
        # Continue the existing feature pyramid after cropping. Do not insert
        # another input/stem block at the crop's resolution.
        self.high_decoder = Net3D(high_settings, self.geometry.main_shape_zyx,
                                  initialize_features=False)
        if output_mask is None:
            output_mask = torch.ones(self.geometry.packed_size)
        if output_mask.numel() != self.geometry.packed_size:
            raise ValueError("output_mask must contain one value per packed voxel")
        self.register_buffer("output_mask", output_mask.reshape(1, -1))

    def _activate(self, x):
        activation = self.settings.output_activation
        if activation == "relu":
            return F.relu(x)
        if activation == "softplus":
            return F.softplus(x)
        if activation == "leakyrelu":
            return F.leaky_relu(x, negative_slope=self.settings.output_leaky_relu_slope)
        if activation is not None:
            raise ValueError("output_activation must be 'relu', 'softplus', 'leakyrelu', or None")
        return x

    def forward_images(self, z):
        flattened = z.flatten(start_dim=1)
        if flattened.shape[1] != self.settings.manifold_dim:
            raise ValueError("manifold samples must have manifold_dim coordinates")
        mapped = self.mapping_net(flattened).reshape(z.shape[0], self.settings.input_channels,
                                                    *self.settings.latent_spatial_shape)
        features = self.low_decoder(mapped)
        low = self._activate(self.low_head(features))
        heart_features = features[(slice(None), slice(None), *self.geometry.main_slices_zyx)]
        high = self._activate(self.high_decoder(heart_features))
        return high, low

    def forward(self, z):
        high, low = self.forward_images(z)
        return self.geometry.pack(high, low) * self.output_mask

_SOURCE = r'''
extern "C" __global__ void st_nlm(const float* images, float* output,
 const float* gaussian, int nx, int ny, int nz, int nframes,
 int radius, int patch, float h2, float eps, float beta) {
 int i = blockDim.x * blockIdx.x + threadIdx.x;
 int count = nx*ny*nz;
 if (i >= count) return;
 int x=i%nx, y=(i/nx)%ny, z=i/(nx*ny);
 float numerator=0.0f, denominator=eps;
 float center=images[i];
 for (int f=0; f<nframes; ++f) {
  const float* other=images+f*count;
  for (int dx=-radius; dx<=radius; ++dx)
   for (int dy=-radius; dy<=radius; ++dy)
    for (int dz=-radius; dz<=radius; ++dz) {
     if (f==0 && dx==0 && dy==0 && dz==0) continue;
     float distance=0.0f;
     int gi=0;
     for (int pz=-patch; pz<=patch; ++pz)
      for (int py=-patch; py<=patch; ++py)
       for (int px=-patch; px<=patch; ++px, ++gi) {
        int ax=min(max(x+px,0),nx-1);
        int ay=min(max(y+py,0),ny-1);
        int az=min(max(z+pz,0),nz-1);
        int bx=min(max(x+dx+px,0),nx-1);
        int by=min(max(y+dy+py,0),ny-1);
        int bz=min(max(z+dz+pz,0),nz-1);
        float d=images[ax+nx*(ay+ny*az)]-other[bx+nx*(by+ny*bz)];
        distance += gaussian[gi]*d*d;
       }
     float weight=expf(-distance/h2);
     int bx=min(max(x+dx,0),nx-1);
     int by=min(max(y+dy,0),ny-1);
     int bz=min(max(z+dz,0),nz-1);
     numerator += weight*(center-other[bx+nx*(by+ny*bz)]);
     denominator += weight;
    }
 }
 output[i]=beta*numerator/denominator;
}
'''


_NLM_KERNELS = {}


def nlm_residual(roi, neighbors, shape_xyz, beta, bandwidth=60., epsilon=1e-5,
                 search_radius=1, patch_radius=1, gaussian_std=1.):
    """OMEGA-style normalized NLM residual, using Fortran-flattened XYZ ROIs."""
    if roi.device.type == 'cuda':
        # CuPy launches the same fused CUDA/HIP kernel as the original campaign.
        import cupy as cp
        import numpy as np
        device_index = roi.device.index or 0
        key = (device_index, patch_radius, gaussian_std)
        with cp.cuda.Device(device_index):
            stream = cp.cuda.ExternalStream(torch.cuda.current_stream(roi.device).cuda_stream)
            with stream:
                if key not in _NLM_KERNELS:
                    module = cp.RawModule(code=_SOURCE)
                    axis = np.arange(-patch_radius, patch_radius+1, dtype=np.float32)
                    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
                    gaussian = np.exp(-(x*x+y*y+z*z)/(2*gaussian_std**2)).astype(np.float32)
                    _NLM_KERNELS[key] = (module, module.get_function('st_nlm'),
                                         cp.asarray(gaussian.ravel(order='F')))
                _, kernel, gaussian = _NLM_KERNELS[key]
                packed = torch.stack((roi, *neighbors)).contiguous()
                result = torch.empty_like(roi)
                ints = tuple(np.int32(v) for v in (*shape_xyz, len(neighbors)+1,
                                                    search_radius, patch_radius))
                args = (cp.from_dlpack(packed), cp.from_dlpack(result), gaussian, *ints,
                        np.float32(bandwidth**2), np.float32(epsilon), np.float32(beta))
                kernel(((roi.numel()+127)//128,), (128,), args)
                return result
    nx, ny, nz = shape_xyz
    radius, patch = search_radius, patch_radius
    halo = radius + patch
    shape = (1, 1, nz, ny, nx)
    volumes = [F.pad(x.reshape(shape), (halo,) * 6, mode='replicate')
               for x in (roi, *neighbors)]

    def patch_field(volume, dx, dy, dz):
        z, y, x = (halo - patch + dz, halo - patch + dy, halo - patch + dx)
        return volume[:, :, z:z+nz+2*patch, y:y+ny+2*patch, x:x+nx+2*patch]

    target = patch_field(volumes[0], 0, 0, 0)
    center = roi.reshape(shape)
    offsets = torch.arange(-patch, patch+1, device=roi.device, dtype=roi.dtype)
    z, y, x = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
    gaussian = torch.exp(-(x*x+y*y+z*z)/(2*gaussian_std**2))[None, None]
    numerator = torch.zeros_like(center)
    denominator = torch.full_like(center, epsilon)
    for frame, volume in enumerate(volumes):
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                for dz in range(-radius, radius+1):
                    if frame == 0 and dx == dy == dz == 0:
                        continue
                    candidate = patch_field(volume, dx, dy, dz)
                    distance = F.conv3d((target-candidate).square(), gaussian)
                    weight = torch.exp(-distance / bandwidth**2)
                    candidate_center = candidate[:, :, patch:patch+nz,
                                                 patch:patch+ny, patch:patch+nx]
                    numerator.add_(weight * (center-candidate_center))
                    denominator.add_(weight)
    return (beta * numerator / denominator).reshape(-1)


def volume_multiplier(shapes_xyz, spacing_xyz, device):
    """Campaign volume preconditioner: fine ROI factor 32, exterior by volume."""
    shapes = torch.as_tensor(shapes_xyz, dtype=torch.long, device=device)
    spacing = torch.as_tensor(spacing_xyz, dtype=torch.float32, device=device)
    voxel_volumes = spacing.prod(dim=1)
    factors = voxel_volumes.max() / voxel_volumes
    factors[0] = 32.
    return torch.repeat_interleave(factors, shapes.prod(dim=1))


@torch.no_grad()
def _reconstruct(study, images, snapshot, shape_xyz, multiplier, beta,
                 relaxation_a, neighbors, iterations, device, epsilon, bandwidth, checkpoint=None,
                 frame_order=None):
    """Update all likelihoods, then apply the prior to an immutable snapshot.

    ``images`` and ``snapshot`` may be CPU tensors backed by memory-mapped files.
    Only one packed image and up to three fine ROIs reside on the compute device.
    """
    fine_size = shape_xyz[0] * shape_xyz[1] * shape_xyz[2]
    multiplier = torch.as_tensor(multiplier, dtype=images.dtype, device=device)
    for iteration in range(iterations):
        relaxation = relaxation_a / (1 + iteration / 20.)
        order = range(study.num_frames) if frame_order is None else frame_order()
        for frame in order:
            image = images[frame].to(device, copy=True)
            for subset in range(study.num_subsets):
                counts, scatter, mask = study.frame_data(frame, subset, device)
                prediction = study.forward(frame, subset, image) + scatter + epsilon
                gradient = study.backward(frame, subset, counts / prediction - mask)
                image.add_(relaxation * image * multiplier * gradient).clamp_(min=epsilon)
            images[frame].copy_(image.to(images.device))
        snapshot.copy_(images[:, :fine_size])
        for frame in range(study.num_frames):
            roi = snapshot[frame].to(device)
            adjacent = [snapshot[index].to(device) for index in neighbors(frame)]
            residual = nlm_residual(roi, adjacent, shape_xyz, beta, bandwidth, epsilon)
            updated = (roi - relaxation * roi * multiplier[:fine_size] * residual).clamp_(min=epsilon)
            images[frame, :fine_size].copy_(updated.to(images.device))
        if checkpoint is not None:
            checkpoint(iteration + 1, images)
    return images
