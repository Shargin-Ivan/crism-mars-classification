"""Full-axis ABI; normalized values in, normalized spectra/log probabilities out.

Segmentation is temporary tensor slicing, not creation of persisted subcubes.
The backbone never convolves across a forbidden edge or detector transition.
BatchNorm during training still shares batch statistics; independence tests use
eval mode. Callers must apply their minimum-support rejection policy.
Wavelengths enforce finite/positive metadata and splits at nonpositive steps or
gaps >7.5 nm; they are not learned wavelength conditioning. Inputs must already
be normalized using frozen training statistics. LWNet requires real spatial
patches; repeating a single spectrum is not supported as spatial training data.
Short LWNet segments with BatchNorm require a training batch of at least two.
"""
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .vendor.lwnet import LWNet_1
from .vendor.n2n4m import Noise2Noise1D


def _edges(wavelengths: Tensor, detectors: Tensor, allowed: Tensor, channels: int) -> Tensor:
    if wavelengths.dim() != 1 or detectors.dim() != 1 or allowed.dim() != 1:
        raise ValueError("Axis metadata must be one-dimensional")
    if wavelengths.numel() != channels or detectors.numel() != channels or allowed.numel() != channels - 1:
        raise ValueError("Axis metadata length mismatch")
    if not bool(torch.all(torch.isfinite(wavelengths) & (wavelengths > 0))):
        raise ValueError("Wavelengths must be finite positive nanometres")
    if not bool(torch.all((detectors == 0) | (detectors == 1))):
        raise ValueError("Detector encoding is L=0, S=1")
    if allowed.dtype != torch.bool:
        raise ValueError("allowed_edges must be boolean")
    step = wavelengths[1:] - wavelengths[:-1]
    return allowed & (detectors[1:] == detectors[:-1]) & (step > 0) & (step <= 7.5)


class N2N4MFullAxis(nn.Module):
    def __init__(self, kernel_size: int = 5, depth: int = 3, num_blocks: int = 4,
                 padded_length: int = 512):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0 or depth < 1 or num_blocks < 1:
            raise ValueError("Require odd kernel >=3 and positive depth/num_blocks")
        if padded_length < 2 ** num_blocks or padded_length % (2 ** num_blocks):
            raise ValueError("padded_length must be a positive multiple of 2**num_blocks")
        self.padded_length = padded_length
        self.backbone = Noise2Noise1D(kernel_size, depth, padded_length, num_blocks)

    def forward(self, values: Tensor, valid: Tensor, wavelengths_nm: Tensor,
                detectors: Tensor, allowed_edges: Tensor) -> Tensor:
        if values.dim() != 2 or valid.shape != values.shape or valid.dtype != torch.bool:
            raise ValueError("Expected values and boolean valid with shape B,C")
        channels = values.size(1)
        if channels < 1:
            raise ValueError("Empty spectral axis")
        edges = _edges(wavelengths_nm, detectors, allowed_edges, channels)
        mask = valid & torch.isfinite(values)
        clean = torch.where(mask, values, torch.zeros_like(values))
        output = torch.zeros_like(values)
        start = 0
        for stop in range(1, channels + 1):
            boundary = stop == channels
            if stop < channels:
                boundary = not bool(edges[stop - 1])
            if boundary:
                length = stop - start
                if length > self.padded_length:
                    raise ValueError("Connected segment exceeds padded_length")
                segment = F.pad(clean[:, start:stop].unsqueeze(1), (0, self.padded_length - length))
                prediction = self.backbone(segment).squeeze(1)[:, :length]
                output[:, start:stop] = prediction
                start = stop
        return torch.where(mask, output, torch.zeros_like(output))


class LWNetFullAxis(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError("Require at least two classes")
        self.num_classes = num_classes
        self.backbone = LWNet_1(num_classes=num_classes)

    def forward(self, values: Tensor, valid: Tensor, wavelengths_nm: Tensor,
                detectors: Tensor, allowed_edges: Tensor) -> Tensor:
        if values.dim() != 5 or values.size(1) != 1 or valid.shape != values.shape or valid.dtype != torch.bool:
            raise ValueError("Expected values and boolean valid with shape B,1,C,P,P")
        patch = values.size(3)
        if patch < 13 or patch % 2 == 0 or values.size(4) != patch:
            raise ValueError("Require square odd patches >=13")
        channels = values.size(2)
        if channels < 1:
            raise ValueError("Empty spectral axis")
        edges = _edges(wavelengths_nm, detectors, allowed_edges, channels)
        mask = valid & torch.isfinite(values)
        clean = torch.where(mask, values, torch.zeros_like(values))
        center = mask[:, 0, :, patch // 2, patch // 2]
        probabilities = values.new_zeros((values.size(0), self.num_classes))
        support = values.new_zeros((values.size(0), 1))
        start = 0
        for stop in range(1, channels + 1):
            boundary = stop == channels
            if stop < channels:
                boundary = not bool(edges[stop - 1])
            if boundary:
                weight = center[:, start:stop].sum(1, keepdim=True).to(values.dtype)
                if bool(torch.any(weight > 0)):
                    segment = clean[:, :, start:stop, :, :]
                    if stop - start < 17:
                        segment = F.pad(segment, (0, 0, 0, 0, 0, 17 - (stop - start)))
                    probabilities = probabilities + self.backbone(segment).exp() * weight
                    support = support + weight
                start = stop
        probabilities = probabilities / support.clamp_min(1.0)
        probabilities = torch.where(support > 0, probabilities,
                                    torch.full_like(probabilities, 1.0 / self.num_classes))
        return probabilities.clamp_min(1e-30).log()
