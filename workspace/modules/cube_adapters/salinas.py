"""Frozen AVIRIS Salinas LWNet transfer, retaining its native land-cover output."""
import importlib.util
from pathlib import Path

import numpy as np

from modules.pipeline_contracts import RejectionReason as Reason, contract_dict, require
from modules.resource_store import ResourceStore, read_json, verify_ref
from .common import (absolute_ref, check_classifier_input, compatible, inference_range,
                     output_directory, save_classification)

ARCHITECTURE = 'LWNet_3_Salinas_group4_v1'


def match_channels(source_nm, good, target_nm, tolerance):
    """Nearest measured band, in AVIRIS channel order; no extrapolation over range endpoints."""
    source_nm = np.asarray(source_nm)
    available = np.flatnonzero(good)
    compatible(available.size > 0, 'insufficient_channels', 'No valid source bands')
    source = source_nm[available]
    targets = np.asarray(target_nm)
    distances = np.abs(targets[:, None] - source)
    nearest = np.argmin(distances, axis=1)
    matched = ((distances[np.arange(len(targets)), nearest] <= tolerance)
               & (targets >= source.min()) & (targets <= source.max()))
    return np.where(matched, available[nearest], -1)


def _extent_and_centers(view, mapping, threshold):
    indices = mapping[mapping >= 0]
    eligible = np.zeros(view.cube.shape[:2], dtype=bool)
    empty = np.zeros_like(eligible)
    low, high = np.inf, -np.inf
    for start in range(0, len(eligible), 16):
        rows = slice(start, start + 16)
        valid = view.valid_samples[rows][:, :, indices]
        empty[rows] = ~view.valid_samples[rows].any(axis=2)
        eligible[rows] = valid.sum(axis=2) >= threshold
        values = view.cube[rows][:, :, indices]
        low = min(low, float(np.min(values, where=valid, initial=np.inf)))
        high = max(high, float(np.max(values, where=valid, initial=-np.inf)))
    compatible(eligible.any(), 'insufficient_channels', 'No pixel has enough supported Salinas channels')
    compatible(high > low, 'constant_scene', 'Global min-max scaling needs a nonzero scene range')
    return low, high, eligible, empty


def _padded_slab(view, mapping, low, high, start, stop, radius):
    """Project once per row slab with a patch halo, normalize and zero-pad missing values."""
    h, w = view.cube.shape[:2]
    a, b = max(0, start - radius), min(h, stop + radius)
    positions = np.flatnonzero(mapping >= 0)
    source = mapping[positions]
    valid = view.valid_samples[a:b][:, :, source]
    values = view.cube[a:b][:, :, source]
    normalized = np.zeros(values.shape, dtype=np.float32)
    np.subtract(values, low, where=valid, out=normalized)
    normalized /= (high - low)
    block = np.zeros((stop - start + 2 * radius, w + 2 * radius, len(mapping)), dtype=np.float32)
    offset = a - start + radius
    inner = block[offset:offset+b-a, radius:radius+w]
    inner[:, :, positions] = normalized
    return block


def _model(checkpoint, execution):
    import torch
    compatible(execution.device == 'cpu' or torch.cuda.is_available(),
               'device_unavailable', 'Requested CUDA device is unavailable')
    path = Path(__file__).resolve().parents[2] / 'LWNet/training/models/models_lw_3D.py'
    spec = importlib.util.spec_from_file_location('_salinas_upstream_lwnet', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    model = module.LWNet_3(num_classes=16)
    model.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()}, strict=True)
    return model.eval().requires_grad_(False).to(device=execution.device, memory_format=torch.channels_last_3d)


def classify_salinas(view, bundle_ref, resource_base, taxonomy, output_dir, *, branch_id, spec, execution):
    import torch
    store = ResourceStore(resource_base)
    bundle = store.model_bundle(bundle_ref, algorithm='lwnet')
    parent = verify_ref(bundle_ref, resource_base).parent
    parameters = read_json(verify_ref(bundle.preprocessing, parent))['parameters']
    require(bundle.architecture_version == ARCHITECTURE and bundle.taxonomy.taxonomy_id == 'salinas_aviris_land_cover'
            and bundle.output_class_ids == tuple(range(16)), 'Invalid Salinas architecture or output ordering')
    check_classifier_input(view, branch_id, 'lwnet', taxonomy, bundle.taxonomy)
    compatible(view.prepared.domain in ('hymars_photometric_atmospheric_corrected_if', 'crism_mtrdr_corrected_if')
               and view.prepared.units == 'dimensionless', 'domain_mismatch', 'Expected CRISM corrected I/F')
    require(parameters['patch_size'] == 27 and spec.patch_size in (19, 27),
            'Salinas supports upstream default 27 and project diagnostic patch size 19')
    target_waves = parameters['input_wavelengths_nm']
    require(len(target_waves) == 204, 'Expected 204 retained AVIRIS channels')
    mapping = match_channels(view.prepared.spectral_axis.wavelengths_nm, view.good_channels,
                             target_waves, parameters['max_distance_nm'])
    minimum = parameters['min_center_bands']
    compatible((mapping >= 0).sum() >= minimum, 'insufficient_channels', 'Too few overlapping Salinas channels')
    low, high, eligible, empty = _extent_and_centers(view, mapping, minimum)
    used = np.zeros(view.cube.shape[-1], dtype=bool)
    used[mapping[mapping >= 0]] = True
    labels = np.full(eligible.shape, -1, dtype=np.int32)
    reasons = np.where(empty, int(Reason.EMPTY_PIXEL), int(Reason.INSUFFICIENT_COVERAGE)).astype(np.uint8)
    radius = spec.patch_size // 2
    with output_directory(output_dir, 'classification') as path, torch.inference_mode():
        model = _model(verify_ref(bundle.weights, parent), execution)
        for start in inference_range(len(labels), 16, f'Salinas {branch_id}', 'rows'):
            stop = min(start + 16, len(labels))
            rows, cols = np.nonzero(eligible[start:stop])
            if not len(rows):
                continue
            block = _padded_slab(view, mapping, low, high, start, stop, radius)
            patches = np.lib.stride_tricks.sliding_window_view(block, (spec.patch_size, spec.patch_size), axis=(0, 1))
            for offset in range(0, len(rows), spec.batch_size):
                batch = slice(offset, offset + spec.batch_size)
                r, c = rows[batch], cols[batch]
                features = torch.from_numpy(np.ascontiguousarray(patches[r, c])).unsqueeze(1)
                scores = model(features.to(device=execution.device, memory_format=torch.channels_last_3d))
                scores = scores.cpu().numpy()
                require(scores.shape == (len(r), 16) and np.isfinite(scores).all(), 'Invalid Salinas predictions')
                labels[start+r, c] = scores.argmax(axis=1)
                reasons[start+r, c] = int(Reason.NONE)
        return save_classification(path, view, branch_id, absolute_ref(bundle_ref, resource_base), taxonomy,
                                   labels, reasons, used, {
                                       'transfer': 'Earth Salinas land-cover classes applied to Mars',
                                       'salinas_channel_to_source_index': mapping.tolist(),
                                       'matched_salinas_channels': int((mapping >= 0).sum()),
                                       'distinct_crism_channels': int(used.sum()),
                                       'min_center_bands': minimum, 'scene_min': low, 'scene_max': high,
                                       'patch_protocol': ('upstream_preprocess_default' if spec.patch_size == 27
                                                          else 'project_19px_transfer_diagnostic'),
                                       'spec': contract_dict(spec), 'execution': contract_dict(execution)})
