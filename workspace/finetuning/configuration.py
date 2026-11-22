"""Fail-closed training preflight; it never imports Torch or fits statistics."""
from pathlib import Path
import re
import numpy as np

from .data import PreparedDataset
from .io import digest, inside, pinned, read_json

ROOT = Path(__file__).resolve().parent
FIELDS = {'schema_version', 'state', 'model', 'stage', 'class_names', 'held_out_scene_id',
          'held_out_observation_ids', 'development_scene_ids', 'datasets', 'initialization',
          'parameters'}
PARAMETERS = {'epochs', 'batch_size', 'learning_rate', 'patience', 'seed', 'device', 'threads',
              'max_samples_per_dataset', 'patch_size', 'noise_std', 'band_dropout',
              'min_valid_fraction', 'normalization', 'denoising_protocol', 'allow_domain_shift'}

SHARED_LWNET = 'shared_spectral_v1'
LEGACY_LWNET = 'legacy_patch_v1'


def uses_spectral_transfer(config):
    return config['model'] == 'lwnet' and config.get('lwnet_architecture') == SHARED_LWNET


def observation_keys(ids):
    """Plebani stores short hexadecimal numbers, HyMars full CRISM identifiers."""
    keys = set()
    for name in ids:
        token = name.strip().upper()
        match = re.fullmatch(r'(?:(?:FRT|FRS|HRL|HRS)([0-9A-F]{8})|([0-9A-F]{5,8}))', token)
        keys.add(('crism', int(match.group(1) or match.group(2), 16)) if match else ('literal', token))
    return keys


def load(path):
    config = read_json(path)
    if set(config) - {'lwnet_architecture'} != FIELDS or config['schema_version'] != 1:
        raise ValueError('Unknown or missing training configuration fields')
    if config['state'] not in ('draft', 'ready') or config['model'] not in ('n2n4m', 'lwnet'):
        raise ValueError('Invalid state/model')
    if config['stage'] not in ('plebani350', 'hymars_full'):
        raise ValueError('Invalid training stage')
    if 'lwnet_architecture' in config and (config['model'] != 'lwnet' or
                                           config['lwnet_architecture'] not in (SHARED_LWNET, LEGACY_LWNET)):
        raise ValueError('Unknown or inapplicable LWNet architecture')
    names = config['class_names']
    if not isinstance(names, list) or any(not isinstance(x, str) or not x.strip() for x in names) or len(set(names)) != len(names):
        raise ValueError('class_names must be unique nonempty names')
    for key in ('held_out_observation_ids', 'development_scene_ids', 'datasets'):
        values = config[key]
        if not isinstance(values, list) or any(not isinstance(x, str) or not x.strip() for x in values) or len(set(values)) != len(values):
            raise ValueError(f'Invalid {key}')
    if config['held_out_scene_id'] is not None and not isinstance(config['held_out_scene_id'], str):
        raise ValueError('Invalid held_out_scene_id')
    init = config['initialization']
    if not isinstance(init, dict) or set(init) != {'checkpoint', 'kind', 'allow_random', 'reset_classifier'}:
        raise ValueError('Invalid initialization fields')
    if init['kind'] not in ('backbone', 'training_checkpoint') or type(init['allow_random']) is not bool or type(init['reset_classifier']) is not bool:
        raise ValueError('Invalid initialization policy')
    if init['checkpoint'] is not None:
        if not isinstance(init['checkpoint'], dict) or set(init['checkpoint']) != {'path', 'sha256'}:
            raise ValueError('Initializer requires path and sha256')
    if uses_spectral_transfer(config) and init['reset_classifier']:
        raise ValueError('Shared LWNet uses a single canonical classifier across stages')
    if uses_spectral_transfer(config) and config['stage'] == 'plebani350' and init['checkpoint'] is not None:
        raise ValueError('Shared LWNet spectral pretraining starts from scratch on Plebani')
    p = config['parameters']
    if not isinstance(p, dict) or set(p) != PARAMETERS:
        raise ValueError('Unknown or missing training parameters')
    for key in ('epochs', 'batch_size', 'patience', 'threads', 'max_samples_per_dataset', 'patch_size'):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if p['batch_size'] < 2 and not uses_spectral_transfer(config):
        raise ValueError('BatchNorm models require batch_size>=2')
    if p['patch_size'] < 13 or p['patch_size'] % 2 == 0:
        raise ValueError('Require an odd patch>=13 for whole-cube export')
    if type(p['seed']) is not int or p['seed'] < 0:
        raise ValueError('Invalid seed')
    for key in ('learning_rate', 'noise_std', 'band_dropout', 'min_valid_fraction'):
        if type(p[key]) not in (int, float) or not np.isfinite(p[key]):
            raise ValueError(f'Invalid numeric parameter: {key}')
    if not (p['learning_rate'] > 0 and p['noise_std'] > 0 and 0 <= p['band_dropout'] < 1 and 0 < p['min_valid_fraction'] <= 1):
        raise ValueError('Invalid optimizer/noise/coverage parameters')
    if p['device'] not in ('cpu', 'cuda') or p['normalization'] != 'train_scalar_standardize':
        raise ValueError('Unsupported device/normalization')
    if p['denoising_protocol'] != 'synthetic_proxy':
        raise ValueError('Only explicitly labelled synthetic_proxy denoising is implemented')
    if type(p['allow_domain_shift']) is not bool:
        raise ValueError('allow_domain_shift must be explicit boolean')
    return config


def _check_membership(dataset):
    profile = read_json(ROOT / 'configs/plebani350_axis.json')
    wave = dataset.axis_arrays['wavelengths_nm']
    detector = dataset.axis_arrays['detectors']
    distance = np.abs(wave[:, None] - np.asarray(profile['wavelengths_nm'])[None, :])
    matches = (distance <= profile['match_tolerance_nm']) & (detector[:, None] == 'L')
    if np.any(matches.sum(0) > 1) or np.any(matches.sum(1) > 1):
        raise ValueError('Ambiguous Plebani band correspondence')
    actual = matches.any(1)
    declared = dataset.descriptor['provenance']['plebani_membership']
    if not np.array_equal(actual, declared):
        raise ValueError('Plebani membership differs from the pinned wavelength grid')


def _split_checks(datasets, config):
    train = [d for d in datasets if d.descriptor['role'] == 'train']
    val = [d for d in datasets if d.descriptor['role'] == 'validation']
    if not train or not val:
        raise ValueError('Explicit train and validation datasets are required')
    if not uses_spectral_transfer(config) and any(len(d) < 2 for d in train):
        raise ValueError('Each train dataset needs at least two usable samples')
    if config['stage'] == 'hymars_full':
        if {d.descriptor['scene_id'] for d in train} != set(config['development_scene_ids']):
            raise ValueError('Full-cube training must use exactly the two development scenes')
    if len({(d.descriptor['units'], d.descriptor['domain']) for d in datasets}) != 1:
        raise ValueError('Do not mix radiometric domains/units inside one training stage')
    forbidden = observation_keys(config['held_out_observation_ids'])
    for d in datasets:
        if forbidden & observation_keys(d.descriptor['provenance']['source_observation_ids']):
            raise ValueError('Held-out observation occurs in training/validation provenance')
    for left in train:
        for right in val:
            a, b = left.descriptor, right.descriptor
            same_values = a['values']['sha256'] == b['values']['sha256']
            same_source = bool(observation_keys(a['provenance']['source_observation_ids']) &
                               observation_keys(b['provenance']['source_observation_ids']))
            if uses_spectral_transfer(config) and config['stage'] == 'plebani350' and same_source:
                raise ValueError('Plebani spectral train/validation must use disjoint observations')
            if same_values or same_source or a['scene_id'] == b['scene_id']:
                if not same_values or left.roi.shape != right.roi.shape:
                    raise ValueError('Overlapping observations with unverified crop/geometry')
                if np.any(left.roi & right.roi):
                    raise ValueError('Train/validation spatial footprints overlap')


def preflight(config, root=ROOT):
    """Return diagnostics and opened read-only datasets. Caller closes datasets."""
    issues, datasets = [], []
    if config['state'] != 'ready':
        issues.append('configuration_is_draft')
    if not config['held_out_scene_id'] or not config['held_out_observation_ids']:
        issues.append('held_out_scene_and_observation_ids_required')
    if len(config['development_scene_ids']) != 2 or config['held_out_scene_id'] in config['development_scene_ids']:
        issues.append('exactly_two_development_scenes_disjoint_from_holdout_required')
    if config['model'] == 'lwnet' and len(config['class_names']) < 2:
        issues.append('approved_canonical_classes_required')
    init = config['initialization']
    if config['stage'] == 'hymars_full' and (init['kind'] != 'training_checkpoint' or init['checkpoint'] is None or init['allow_random']):
        issues.append('stage2_requires_stage1_training_checkpoint')
    if init['checkpoint'] is None:
        if not init['allow_random']:
            issues.append('frozen_initializer_required_or_explicit_from_scratch')
    else:
        try:
            pinned(root, init['checkpoint'])
        except (OSError, ValueError) as error:
            issues.append(str(error))
    if not config['datasets']:
        issues.append('prepared_train_and_validation_manifests_required')
    for relative in config['datasets']:
        try:
            path = inside(root, relative)
            read_json(path)  # Reject duplicate keys/nonfinite JSON before opening arrays.
            d = PreparedDataset(path, config['model'], config['stage'], config['parameters']['patch_size'],
                                config['held_out_scene_id'], config['development_scene_ids'], config['class_names'],
                                shared_lwnet=uses_spectral_transfer(config))
            datasets.append(d)
            _check_membership(d)
            if uses_spectral_transfer(config):
                wave = d.axis_arrays['wavelengths_nm']
                if np.any((wave < 200) | (wave > 5000)):
                    raise ValueError('Wavelength outside the 200-5000 nm model domain')
            fraction = config['parameters']['min_valid_fraction']
            count = d.values.shape[-1]
            required = int(np.ceil(fraction * (350 if config['stage'] == 'plebani350' else count)))
            mask = d.valid.reshape(-1, count)
            accepted = []
            for start in range(0, len(d.indices), 4096):
                indices = d.indices[start:start+4096]
                accepted.append(indices[(mask[indices] & d.channel_mask).sum(1) >= required])
            d.indices = np.concatenate(accepted) if accepted else np.empty(0, dtype=np.int64)
            if len(d) == 0:
                raise ValueError('No samples meet the declared full-profile coverage')
            segments = d.axis_arrays['segment_ids']
            if config['model'] == 'n2n4m' and max(np.unique(segments, return_counts=True)[1]) > 512:
                raise ValueError('Connected spectral segment exceeds this N2N4M padded_length=512')
        except (OSError, ValueError, TypeError, KeyError) as error:
            issues.append(f'{relative}: {error}')
    if datasets and not issues:
        try:
            _split_checks(datasets, config)
        except ValueError as error:
            issues.append(str(error))
    return {'ready': not issues, 'issues': issues,
            'datasets': [{'scene': d.descriptor['scene_id'], 'role': d.descriptor['role'],
                          'samples': len(d), 'channels': d.values.shape[-1]} for d in datasets]}, datasets
