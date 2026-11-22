"""Bind a completed HyMars model to one of the experiment's three native scenes."""
import hashlib
import json
from pathlib import Path

from finetuning.configuration import observation_keys
from finetuning.export import _export_bundle
from finetuning.io import digest


HYMARS_DOMAIN = 'hymars_photometric_atmospheric_corrected_if'
MTRDR_DOMAIN = 'crism_mtrdr_corrected_if'
# SHA-256 of the verified configs/scenes manifests, excluding relocatable paths
# and optional evaluation labels. This pins the cube, axis, quality flags,
# detector boundaries, radiometry, and the independent calibration evidence.
SCENES = {
    'hymars_v2_HC': (HYMARS_DOMAIN, 'FRT0000580C',
                     '4ff3478ef56722b042f7b38db1a56398560fb7b3adf4968fdcf6264b6ae16361'),
    'FRT00019468_MTRDR': (MTRDR_DOMAIN, 'FRT00019468',
                          '63b64e73e120c4a031c03d78c313b68646777649e8d49394a5f29d5df71f4eaf'),
    'HRL000120CD_MTRDR': (MTRDR_DOMAIN, 'HRL000120CD',
                          '26019bd2edfa85130b81028e9c15931b66cfc1d4913a2746acdc74acad7694e9'),
}
# Ordered class names of configs/evaluation/hymars_plebani_taxonomy.json.
CLASS_NAMES_HASH = '4870953e5e8dea1ac0b3da50a828a4346a31a41d3f33b9115351723557237620'


def _hash_json(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _identity(ref):
    return {key: value for key, value in ref.items() if key != 'path'}


def _scene_policy(training, target, manifest_path):
    scene_id = target.get('scene_id')
    if scene_id not in SCENES:
        raise ValueError(f'Unknown experiment scene: {scene_id}')
    domain, observation, signature = SCENES[scene_id]
    if (training.get('domain') != HYMARS_DOMAIN or target.get('domain') != domain
            or target.get('units') != 'dimensionless'):
        raise ValueError('Export requires the documented dimensionless corrected I/F domains')
    science = {key: value for key, value in target.items() if key not in ('ground_truth', 'ground_truth_variable')}
    for key in ('cube', 'header'):
        if science[key] is not None:
            science[key] = _identity(science[key])
    science['metadata_sources'] = [_identity(ref) for ref in science['metadata_sources']]
    if _hash_json(science) != signature:
        raise ValueError(f'Source manifest differs from verified calibration for {scene_id}')
    # Preparation verifies the large cube itself. Only calibration artifacts
    # and the ENVI header need to be present/read at export time.
    for ref in [*target['metadata_sources'], target['header']]:
        if ref is not None and digest(manifest_path.parent / ref['path']) != ref['sha256']:
            raise ValueError(f'Calibration SHA-256 mismatch: {ref["artifact_id"]}')
    if training['model'] == 'lwnet' and _hash_json(training.get('class_names')) != CLASS_NAMES_HASH:
        raise ValueError('LWNet output order differs from the experiment taxonomy')
    for key in ('training_scene_ids', 'validation_scene_ids', 'source_observation_ids'):
        values = training.get(key)
        if not isinstance(values, list) or not values or not all(isinstance(s, str) and s for s in values):
            raise ValueError(f'Missing training lineage: {key}')
        if scene_id in values or observation_keys([observation]) & observation_keys(values):
            raise ValueError(f'Inference scene appears in training/validation lineage: {scene_id}')
    return dict(training_domain=training['domain'], inference_domain=domain, target_scene_id=scene_id,
                target_cube_sha256=target['cube']['sha256'])


def export_scene_bundle(run_path, scene_manifest_path) -> Path:
    """Export a scene-specific native-axis bundle for HC or exploratory MTRDR transfer."""
    return _export_bundle(run_path, scene_manifest_path, scene_policy=_scene_policy)
