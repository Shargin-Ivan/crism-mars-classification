"""Read-only, NumPy-only access to explicitly prepared training datasets.

Manifest references are relative to the manifest directory. No radiometric
conversion, interpolation, missing-value guessing or spatial synthesis occurs.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .io import read_json


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array_ref(base, ref, name):
    _require(isinstance(ref, dict), f"{name}: expected path/sha256 reference")
    relative = Path(ref.get("path", ""))
    _require(bool(str(relative)) and not relative.is_absolute(), f"{name}: expected relative path")
    path = (base / relative).resolve()
    _require(path.is_relative_to(base.resolve()), f"{name}: path escapes dataset directory")
    _require(path.suffix.lower() == ".npy", f"{name}: only .npy arrays are supported")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    _require(digest.hexdigest() == ref.get("sha256"), f"{name}: SHA256 mismatch")
    return path, np.load(path, mmap_mode="r", allow_pickle=False)


@dataclass
class PreparedDataset:
    path: Path
    model: str
    stage: str
    patch_size: int
    held_out_scene_id: str
    development_scene_ids: list[str]
    class_names: list[str]
    shared_lwnet: bool = False

    def __post_init__(self):
        try:
            self._initialize()
        except BaseException:
            self.close()
            raise

    def _initialize(self):
        self.path = Path(self.path).resolve()
        if self.path.is_dir():
            self.path = self.path / "manifest.json"
        self.model = self.model.lower()
        _require(self.model in {"n2", "n2n", "n2n4m", "lw", "lwnet"}, "unsupported model")
        self.is_lw = self.model in {"lw", "lwnet"}
        _require(type(self.shared_lwnet) is bool and (not self.shared_lwnet or self.is_lw),
                 "shared_lwnet requires an LWNet model")
        _require(self.stage in {"plebani350", "hymars_full"}, "unsupported stage")
        self.spectral_classifier = self.shared_lwnet and self.stage == "plebani350"
        _require(type(self.patch_size) is int and self.patch_size > 0 and self.patch_size % 2 == 1,
                 "patch_size must be a positive odd integer")
        self.descriptor = read_json(self.path)
        d = self.descriptor
        _require(d.get("schema_version") == 1, "expected schema_version 1")
        scene = d.get("scene_id")
        _require(isinstance(scene, str) and bool(scene.strip()), "scene_id is required")
        _require(scene != self.held_out_scene_id, "held-out scene cannot be used for training or validation")
        _require(d.get("role") in {"train", "validation"}, "role must be train or validation")
        if self.stage == "hymars_full":
            _require(scene in self.development_scene_ids, "scene is not in development_scene_ids")
        _require(d.get("metadata_verified") is True, "metadata_verified must be true")
        for name in ("units", "domain"):
            value = d.get(name)
            _require(isinstance(value, str) and value.strip().lower() not in
                     {"", "unknown", "unspecified", "none", "null", "n/a"}, f"verified {name} is required")
        provenance = d.get("provenance", {})
        ids = provenance.get("source_observation_ids")
        _require(isinstance(ids, list) and len(ids) > 0 and
                 all(isinstance(i, str) and i.strip() for i in ids), "source_observation_ids are required")
        _require(self.held_out_scene_id not in ids, "held-out source observation is forbidden")
        _require(provenance.get("spatial_context") in {"real", "none"}, "spatial_context must be real or none")
        _require(d.get("format") in {"cube", "spectra"}, "format must be cube or spectra")
        if self.spectral_classifier:
            _require(d["format"] == "spectra" and provenance["spatial_context"] == "none",
                     "LWNet spectral pretraining requires individual spectra without spatial context")
        elif self.is_lw:
            _require(d["format"] == "cube" and provenance["spatial_context"] == "real",
                     "LWNet requires a real spatial cube; spectra cannot be tiled into patches")
        paths = {}
        for name in ("values", "valid", "roi"):
            paths[name], array = _array_ref(self.path.parent, d.get(name), name)
            setattr(self, name, array)
        self.labels = None
        if d.get("labels") is not None:
            paths["labels"], self.labels = _array_ref(self.path.parent, d["labels"], "labels")
        _require(self.values.ndim == (3 if d["format"] == "cube" else 2), "values dimensions disagree with format")
        _require(self.values.dtype.kind in "fiu", "values must be real numeric data")
        _require(self.values.shape == self.valid.shape and self.valid.dtype == np.bool_, "valid must be boolean with values shape")
        _require(self.roi.shape == self.values.shape[:-1] and self.roi.dtype == np.bool_, "roi must be boolean with spatial/sample shape")
        _require(all(s > 0 for s in self.values.shape), "empty arrays are forbidden")
        if d.get("source_channel_indices") is not None:
            from .denoising_data import project_channels
            self.values = project_channels(self.values, d["source_channel_indices"])
            self.valid = project_channels(self.valid, d["source_channel_indices"])
        channels = self.values.shape[-1]
        axis = d.get("axis", {})
        wavelengths = np.asarray(axis.get("wavelengths_nm", []), dtype=np.float64)
        detectors = np.asarray(axis.get("detectors", []))
        segments = np.asarray(axis.get("segment_ids", []))
        _require(wavelengths.shape == detectors.shape == segments.shape == (channels,), "axis lengths must equal channel count")
        _require(np.all(np.isfinite(wavelengths)) and np.all(wavelengths > 0), "wavelengths must be finite positive nm")
        _require(np.all(np.isin(detectors, ["L", "S"])), "detectors must be L or S")
        _require(segments.dtype.kind in "iu" and np.all(segments >= 0), "segment_ids must be nonnegative integers")
        for segment in np.unique(segments):
            positions = np.flatnonzero(segments == segment)
            _require(np.all(np.diff(positions) == 1), "segment_ids must describe contiguous segments")
            _require(np.all(np.diff(wavelengths[positions]) > 0), "wavelengths must strictly increase within each segment")
            _require(np.all(detectors[positions] == detectors[positions[0]]), "segment cannot cross detectors")
            _require(np.all(np.diff(wavelengths[positions]) <= 7.5), "segment cannot cross a wavelength gap greater than 7.5 nm")
        delta = np.diff(wavelengths)
        allowed = (segments[1:] == segments[:-1]) & (detectors[1:] == detectors[:-1]) & (delta > 0) & (delta <= 7.5)
        self.axis_arrays = dict(wavelengths_nm=wavelengths, detectors=detectors,
                                segment_ids=segments, allowed_edges=allowed)
        membership = np.asarray(provenance.get("plebani_membership", []))
        _require(membership.shape == (channels,) and membership.dtype == np.bool_, "plebani_membership must be one boolean per channel")
        self.channel_mask = membership.copy() if self.stage == "plebani350" else np.ones(channels, dtype=bool)
        _require(np.any(self.channel_mask), "first stage has no Plebani wavelength coverage")
        if self.spectral_classifier:
            profile = read_json(Path(__file__).parent / "configs/plebani350_axis.json")
            _require(channels == 350 and np.all(membership) and np.all(detectors == "L")
                     and np.allclose(wavelengths, profile["wavelengths_nm"], rtol=0,
                                     atol=profile["match_tolerance_nm"]),
                     "LWNet spectral pretraining requires the ordered 350 Plebani L channels")
        if self.labels is not None:
            _require(self.labels.shape == self.roi.shape and self.labels.dtype.kind in "iu", "labels must be integer with roi shape")
            _require(len(self.class_names) > 0 and len(set(self.class_names)) == len(self.class_names), "class_names must be nonempty and unique")
            declared_classes = d.get("class_names")
            _require(isinstance(declared_classes, list) and declared_classes == self.class_names,
                     "manifest class_names must exactly match the configured canonical class order")
            _require(np.all((self.labels >= -1) & (self.labels < len(self.class_names))), "labels must be -1 or a canonical class index")
        _require(not self.is_lw or self.labels is not None, "LWNet requires labels")
        flat_values = self.values.reshape(-1, channels)
        flat_valid = self.valid.reshape(-1, channels)
        candidates = np.asarray(self.roi).ravel().copy()
        for start in range(0, len(candidates), 16384):
            end = start + 16384
            block_valid = flat_valid[start:end]
            _require(not np.any(block_valid & ~np.isfinite(flat_values[start:end])), "valid data contains NaN or infinity")
            candidates[start:end] &= np.any(block_valid & self.channel_mask, axis=1)
        if self.spectral_classifier:
            candidates &= self.labels >= 0
        elif self.is_lw:
            # Integral image checks every pixel in each patch without allocating
            # an H*W*P*P sliding-window array.
            h, w = self.roi.shape
            p, r = self.patch_size, self.patch_size // 2
            patch_roi = np.zeros((h, w), dtype=bool)
            if p <= min(h, w):
                sums = np.pad(np.asarray(self.roi, dtype=np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
                counts = sums[p:, p:] - sums[:-p, p:] - sums[p:, :-p] + sums[:-p, :-p]
                patch_roi[r:h-r if r else h, r:w-r if r else w] = counts == p * p
            candidates &= patch_roi.ravel() & (self.labels.ravel() >= 0)
        self.indices = np.flatnonzero(candidates)
        _require(len(self.indices) > 0, "dataset has no usable samples after masks, labels and patch ROI checks")
        d["resolved_paths"] = paths

    def __len__(self):
        return len(self.indices)

    def sample(self, index):
        flat_index = int(self.indices[index])
        if self.is_lw and not self.spectral_classifier:
            y, x = divmod(flat_index, self.values.shape[1])
            r = self.patch_size // 2
            region = (slice(y-r, y+r+1), slice(x-r, x+r+1))
            values = np.asarray(self.values[region], dtype=np.float32).transpose(2, 0, 1)[None].copy()
            valid = np.asarray(self.valid[region]).transpose(2, 0, 1)[None].copy()
            valid &= self.channel_mask[None, :, None, None]
        else:
            values = np.asarray(self.values.reshape(-1, self.values.shape[-1])[flat_index], dtype=np.float32).copy()
            valid = self.valid.reshape(-1, self.values.shape[-1])[flat_index].copy() & self.channel_mask
        # Invalid values never enter model arithmetic; valid zero stays valid.
        _require(np.all(np.isfinite(values[valid])), "valid values overflowed float32 conversion")
        values[~valid] = 0
        result = dict(values=values, valid=valid, **self.axis_arrays)
        if self.labels is not None:
            result["label"] = int(self.labels.ravel()[flat_index])
        if not self.is_lw:
            result["target"] = values.copy()
        return result

    __getitem__ = sample

    def close(self):
        """Release Windows file mappings; samples returned earlier are copies."""
        for name in ("values", "valid", "roi", "labels"):
            array = getattr(self, name, None)
            mmap = getattr(array, "_mmap", None)
            if mmap is not None and not mmap.closed:
                mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def open_dataset(path, model, stage, patch_size, held_out_scene_id, development_scene_ids, class_names,
                 *, shared_lwnet=False):
    return PreparedDataset(path, model, stage, patch_size, held_out_scene_id, development_scene_ids,
                           class_names, shared_lwnet=shared_lwnet)


def validate_dataset(path, model, stage, patch_size, held_out_scene_id, development_scene_ids, class_names,
                     *, shared_lwnet=False):
    with open_dataset(path, model, stage, patch_size, held_out_scene_id, development_scene_ids, class_names,
                      shared_lwnet=shared_lwnet) as dataset:
        return dataset.descriptor
