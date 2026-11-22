"""Factorized LWNet variant for spectrum pretraining and real spatial patches."""
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .full_axis import _edges


class SpectralEncoder(nn.Module):
    """Reduce each measured spectrum to wavelength-local features before spatial mixing."""

    def __init__(self):
        super().__init__()
        # Overlapping wavelength windows keep a band at the same physical location
        # across grids. Separate detector windows preserve overlapping L/S bands.
        # The full-axis bundle accepts wavelengths between 200 and 5000 nm.
        centers = torch.arange(200., 5001., 50.)
        self.register_buffer("centers_nm", centers)
        self.local = nn.Sequential(nn.Conv1d(4, 8, 1), nn.GELU())
        self.project = nn.Sequential(nn.Flatten(), nn.Linear(8 * 2 * len(centers), 32), nn.LayerNorm(32), nn.GELU())

    def forward(self, values: Tensor, valid: Tensor, wavelengths_nm: Tensor,
                detectors: Tensor, allowed_edges: Tensor) -> Tensor:
        if values.dim() != 2 or values.size(1) == 0 or valid.shape != values.shape or valid.dtype != torch.bool:
            raise ValueError("Expected nonempty spectra and boolean mask of shape B,C")
        edges = _edges(wavelengths_nm, detectors, allowed_edges, values.size(1))
        valid = valid & torch.isfinite(values)
        clean = torch.where(valid, values, 0.)
        if bool(torch.any((wavelengths_nm < 200.) | (wavelengths_nm > 5000.))):
            raise ValueError("Wavelength outside the 200-5000 nm model domain")
        wave = wavelengths_nm.to(values.dtype)
        basis = (1. - (wave[:, None] - self.centers_nm).abs() / 50.).clamp_min(0.)
        windows = torch.cat((basis * (detectors == 0)[:, None], basis * (detectors == 1)[:, None]), dim=1)
        support = valid.to(values.dtype) @ windows
        mean = (clean @ windows) / support.clamp_min(1.e-6)

        # Learn from rising/falling slopes separately, so narrow absorption
        # features survive averaging. A missing endpoint always breaks a pair.
        pairs = valid[:, 1:] & valid[:, :-1] & edges
        step = (wavelengths_nm[1:] - wavelengths_nm[:-1]).to(values.dtype)
        safe_step = torch.where(edges, step, 1.).clamp_min(1.e-3)
        slope = torch.where(pairs, (clean[:, 1:] - clean[:, :-1]) / safe_step, 0.)
        pair_windows = windows[1:]
        pair_support = (pairs.to(values.dtype) @ pair_windows).clamp_min(1.e-6)
        rising = (F.relu(slope) @ pair_windows) / pair_support
        falling = (F.relu(-slope) @ pair_windows) / pair_support
        occupied = support > 0
        features = torch.stack((mean, rising, falling, occupied.to(values.dtype)), dim=1)
        local = self.local(features) * occupied[:, None, :]
        embedding = self.project(local)
        return torch.where(valid.any(1, keepdim=True), embedding, 0.)


class LWNetSpectrum(nn.Module):
    """Supervised pretraining on independent spectra."""

    def __init__(self, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError("Require at least two classes")
        self.num_classes = num_classes
        self.encoder = SpectralEncoder()
        self.head = nn.Linear(32, num_classes)

    def forward(self, values: Tensor, valid: Tensor, wavelengths_nm: Tensor,
                detectors: Tensor, allowed_edges: Tensor) -> Tensor:
        embedding = self.encoder(values, valid, wavelengths_nm, detectors, allowed_edges)
        log_probs = F.log_softmax(self.head(embedding), dim=1)
        return torch.where((valid & torch.isfinite(values)).any(1, keepdim=True),
                           log_probs, -math.log(self.num_classes))


class LWNetSpatial(nn.Module):
    """Trainable spectral encoder followed by lightweight 2D patch convolutions."""

    def __init__(self, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError("Require at least two classes")
        self.num_classes = num_classes
        self.encoder = SpectralEncoder()
        self.spatial = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 32, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 32, 1, bias=False),
        )
        self.head = nn.Linear(32, num_classes)

    def load_spectral_state(self, state_dict, transfer_head=True):
        """Validate the whole source checkpoint before copying selected modules."""
        source = LWNetSpectrum(self.num_classes if transfer_head else state_dict["head.bias"].numel())
        source.load_state_dict(state_dict, strict=True)
        self.encoder.load_state_dict(source.encoder.state_dict(), strict=True)
        if transfer_head:
            self.head.load_state_dict(source.head.state_dict(), strict=True)

    def forward(self, values: Tensor, valid: Tensor, wavelengths_nm: Tensor,
                detectors: Tensor, allowed_edges: Tensor) -> Tensor:
        if values.dim() != 5 or values.size(1) != 1 or valid.shape != values.shape or valid.dtype != torch.bool:
            raise ValueError("Expected values and boolean valid with shape B,1,C,P,P")
        patch = values.size(3)
        if patch < 3 or patch % 2 == 0 or values.size(4) != patch:
            raise ValueError("Require square odd spatial patches >=3")
        batch, channels = values.size(0), values.size(2)
        spectra = values[:, 0].permute(0, 2, 3, 1).reshape(batch * patch * patch, channels)
        mask = valid[:, 0].permute(0, 2, 3, 1).reshape_as(spectra)
        encoded = self.encoder(spectra, mask, wavelengths_nm, detectors, allowed_edges)
        pixels = encoded.reshape(batch, patch, patch, 32).permute(0, 3, 1, 2)
        supported = (valid & torch.isfinite(values)).any(2)
        context_map = self.spatial(pixels) * supported
        context = context_map.sum((2, 3)) / supported.sum((2, 3)).clamp_min(1)
        center = patch // 2
        logits = self.head(pixels[:, :, center, center] + context_map[:, :, center, center] + context)
        return torch.where(supported[:, :, center, center], F.log_softmax(logits, dim=1),
                           -math.log(self.num_classes))
