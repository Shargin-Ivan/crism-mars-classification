"""Export a pinned inference bundle without importing PyTorch or the parent project."""
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
from scipy.optimize import linear_sum_assignment

from .configuration import uses_spectral_transfer
from .io import digest, read_json, write_json


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _known(value):
    return isinstance(value, str) and value.strip().lower() not in {
        '', 'unknown', 'unspecified', 'none', 'null', 'n/a'}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _axis(document):
    axis = document.get('axis', {})
    _require(isinstance(axis, dict), 'axis must be an object')
    wave, detectors, segments = [axis.get(key) for key in
                                  ('wavelengths_nm', 'detectors', 'segment_ids')]
    _require(all(isinstance(v, list) for v in (wave, detectors, segments)), 'Incomplete axis')
    _require(len(wave) >= 2 and len(wave) == len(detectors) == len(segments), 'Axis length mismatch')
    _require(all(_number(w) and 200 <= w <= 5000 for w in wave), 'Invalid wavelengths in nm')
    _require(all(d in ('L', 'S') for d in detectors), 'Unknown detector')
    _require(all(type(s) is int and s >= 0 for s in segments), 'Invalid segment_ids')
    seen = set()
    for i, segment in enumerate(segments):
        if i == 0 or segment != segments[i - 1]:
            _require(segment not in seen, 'Segments must be contiguous')
            seen.add(segment)
        else:
            _require(detectors[i] == detectors[i - 1] and 1e-8 < wave[i] - wave[i - 1] <= 7.5,
                     'A segment crosses a detector boundary, gap or unordered wavelength')
    indices = axis.get('channel_indices', list(range(len(wave))))
    _require(isinstance(indices, list) and all(type(i) is int for i in indices) and
             indices == list(range(len(wave))), 'channel_indices must cover the complete axis in order')
    return dict(channel_indices=indices, wavelengths_nm=wave, detectors=detectors,
                segment_ids=segments, metadata_status='verified', unit='nm')


def _unseen_channel_indices(axis, support):
    """Describe stage-2 coverage on the target axis in its original order."""
    _require(isinstance(support, list) and bool(support), 'Missing training feature_support')
    for item in support:
        _require(isinstance(item, dict) and _number(item.get('wavelength_nm')) and
                 200 <= item['wavelength_nm'] <= 5000 and item.get('detector') in ('L', 'S'),
                 'Invalid training feature_support')
    wave = np.asarray(axis['wavelengths_nm'])
    detectors = np.asarray(axis['detectors'])
    seen = np.zeros(len(wave), dtype=bool)
    for detector in ('L', 'S'):
        available = sorted({item['wavelength_nm'] for item in support if item['detector'] == detector})
        indices = np.flatnonzero(detectors == detector)
        distance = np.abs(wave[indices, None] - available)
        eligible = distance <= 0.100000000001
        # An out-of-tolerance pair costs more than all eligible distances combined.
        # This maximizes the number of one-to-one matches, then prefers closer wavelengths.
        target, source = linear_sum_assignment(np.where(eligible, distance, len(wave) + 1.0))
        matched_targets = target[eligible[target, source]]
        seen[indices[matched_targets]] = True
    return np.flatnonzero(~seen).tolist()


def export_bundle(run_path, scene_manifest_path):
    """Return the new bundle.json path; never modify the run or target descriptor.

    Stage-2 feature coverage is recorded for interpretation of full-axis predictions.
    Model bytes are hash-pinned, not deserialized or executed by this exporter.
    """
    return _export_bundle(run_path, scene_manifest_path)


def _export_bundle(run_path, scene_manifest_path, *, scene_policy=None):
    run = Path(run_path).resolve()
    training_path = run / 'training.json'
    training_hash = digest(training_path)
    training = read_json(training_path)
    result_path = run / 'result.json'
    result_hash = digest(result_path)
    result = read_json(result_path)
    _require(isinstance(result, dict) and result.get('status') == 'completed',
             'Only completed training runs can be exported')
    exported = result.get('export')
    _require(isinstance(exported, dict) and exported.get('path') == 'model.ts',
             'Completed result must reference model.ts')
    target = read_json(scene_manifest_path)
    normalization = read_json(run / 'normalization.json')
    _require(training.get('stage') == 'hymars_full', 'Only hymars_full can be exported')
    model = training.get('model')
    _require(model in ('lwnet', 'n2n4m'), 'Unsupported model')
    binding = None
    if scene_policy is None:
        _require(target.get('metadata_verified') is True, 'Target metadata must be verified')
        _require(_known(training.get('domain')) and target.get('domain') == training['domain'],
                 'Target and training domain must be known and identical')
    else:
        binding = scene_policy(training, target, Path(scene_manifest_path))
    _require(_known(training.get('units')) and target.get('units') == training['units'],
             'Target and training units must be known and identical')
    # A held-out calibration sidecar carries the native axis without any image or labels.
    if 'spectral_axis' in target:
        _require('axis' not in target, 'Ambiguous target axis declarations')
        axis = _axis({'axis': target['spectral_axis']})
    else:
        axis = _axis(target)
    unseen_channels = _unseen_channel_indices(axis, training.get('feature_support'))
    mean, scale = normalization.get('mean'), normalization.get('scale')
    _require(_number(mean) and _number(scale) and scale > 0, 'Invalid scalar normalization')
    frozen = training.get('normalization')
    _require(isinstance(frozen, dict) and _number(frozen.get('mean')) and
             _number(frozen.get('scale')) and frozen['mean'] == mean and frozen['scale'] == scale,
             'Normalization differs from frozen training normalization')
    params = training.get('parameters', {})
    _require(isinstance(params, dict), 'Invalid training parameters')
    fraction = params.get('min_valid_fraction', 0.5)
    _require(_number(fraction) and 0 <= fraction <= 1, 'Invalid min_valid_fraction')
    preprocessing = dict(adapter_version=1, backend='torchscript_full_axis_v1', weights_format='torchscript',
                         architecture={}, units=training['units'], normalization='standardize_npz',
                         missing_policy='constant_normalized', fill_value=0.0, min_valid_bands=2,
                         stage2_unseen_channel_indices=unseen_channels,
                         min_valid_fraction=fraction,
                         output_kind='log_probabilities' if model == 'lwnet' else 'normalized_spectrum')
    if binding is not None:
        preprocessing['scene_binding'] = binding
    taxonomy, class_ids = None, []
    if model == 'lwnet':
        names = training.get('class_names')
        _require(isinstance(names, list) and len(names) >= 2 and
                 all(isinstance(n, str) and n.strip() for n in names) and len(set(names)) == len(names),
                 'Invalid class_names')
        patch = params.get('patch_size')
        _require(type(patch) is int and patch >= 13 and patch % 2 == 1, 'Invalid LWNet patch_size')
        preprocessing.update(patch_size=patch, border_policy='reject', min_patch_valid_fraction=0.5)
        version = hashlib.sha256(json.dumps(names, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()[:12]
        taxonomy = dict(taxonomy_id='finetuning_classes', version=version, class_names=names)
        class_ids = list(range(len(names)))
    model_hash = digest(run / 'model.ts')
    _require(model_hash == exported.get('sha256'), 'Completed model SHA-256 mismatch')
    _require(digest(training_path) == training_hash, 'Training manifest changed during export')
    _require(digest(result_path) == result_hash, 'Training result changed during export')
    identifier = uuid.uuid4().hex
    folder = run / 'bundles' / identifier
    folder.mkdir(parents=True, exist_ok=False)

    def reference(path, artifact_id, known_hash=None):
        return dict(artifact_id=artifact_id, path=Path(os.path.relpath(path, folder)).as_posix(),
                    sha256=known_hash or digest(path))

    scaler = folder / 'scaler.npz'
    np.savez(scaler, mean=np.full(len(axis['channel_indices']), mean, dtype=np.float64),
             scale=np.full(len(axis['channel_indices']), scale, dtype=np.float64))
    preprocess_path = folder / 'preprocessing.json'
    write_json(preprocess_path, dict(schema_version=1, scaler=reference(scaler, identifier + ':scaler'),
                                    parameters=preprocessing))
    architecture = 'LWNetFullAxis_v1' if model == 'lwnet' else 'N2N4MFullAxis_v1'
    if uses_spectral_transfer(training):
        architecture = 'LWNetSpatial_shared_spectral_v1'
    bundle = dict(bundle_id=identifier, algorithm=model, architecture_version=architecture,
                  weights=reference(run / 'model.ts', identifier + ':weights', model_hash),
                  preprocessing=reference(preprocess_path, identifier + ':preprocessing'),
                  training_manifest=reference(training_path, identifier + ':training', training_hash),
                  domain=target['domain'], input_layout='segmented', spectral_axis=axis,
                  taxonomy=taxonomy, output_class_ids=class_ids)
    path = folder / 'bundle.json'
    write_json(path, bundle)
    return path
