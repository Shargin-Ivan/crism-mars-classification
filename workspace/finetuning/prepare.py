"""Explicit MAT/NPY conversion after metadata, mappings and spatial ROIs exist."""
from pathlib import Path
import numpy as np

from .export import _axis
from .io import digest, inside, read_json, write_json


def _read(root, ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256', 'format', 'variable'}:
        raise ValueError('Sources need path, sha256, format and variable')
    path = inside(root, ref['path'])
    if digest(path) != ref['sha256']:
        raise ValueError(f'Invalid source hash: {path}')
    if ref['format'] == 'npy' and ref['variable'] is None:
        data = np.load(path, allow_pickle=False)
    elif ref['format'] == 'mat' and isinstance(ref['variable'], str):
        from scipy.io import loadmat
        data = loadmat(path, variable_names=[ref['variable']])[ref['variable']]
    else:
        raise ValueError('Supported sources: NPY or MATLAB v5/v7 with explicit variable')
    if digest(path) != ref['sha256']:
        raise ValueError('Source changed during conversion')
    return data


def prepare(recipe_path, output_relative, root):
    recipe = read_json(recipe_path)
    if recipe.get('schema_version') != 1 or recipe.get('state') != 'ready' or recipe.get('metadata_verified') is not True:
        raise ValueError('Preparation requires a ready recipe with verified physical metadata')
    axis = _axis(recipe)
    for key in ('units', 'domain'):
        if not isinstance(recipe.get(key), str) or recipe[key].strip().lower() in ('', 'unknown', 'none'):
            raise ValueError(f'Confirmed {key} is required')
    observations = recipe.get('source_observation_ids')
    if not isinstance(observations, list) or not observations or not all(isinstance(i,str) and i.strip() for i in observations):
        raise ValueError('Original observation IDs are required')
    if recipe.get('format') not in ('cube', 'spectra'):
        raise ValueError('Prepared format must be cube or spectra')
    sources = recipe['sources']
    if set(sources) != {'values', 'labels', 'valid', 'train_roi', 'validation_roi'}:
        raise ValueError('Unknown/missing sources')
    if sources['train_roi'] is None or sources['validation_roi'] is None:
        raise ValueError('Explicit train/validation ROI masks are required')
    values = _read(root, sources['values'])
    if values.dtype.kind != 'f' or values.ndim != (3 if recipe['format']=='cube' else 2):
        raise ValueError('Expected floating HWC cube or NC spectra')
    if values.shape[-1] != len(axis['channel_indices']):
        raise ValueError('Spectral metadata length differs from the actual data')
    quality = recipe.get('quality_policy')
    if quality not in ('finite_only_declared', 'explicit_mask'):
        raise ValueError('An explicit quality policy is required')
    if (sources['valid'] is not None) != (quality == 'explicit_mask'):
        raise ValueError('Quality policy and mask binding disagree')
    valid = np.isfinite(values)
    invalid_values = recipe.get('invalid_values')
    if not isinstance(invalid_values, list) or not all(type(v) in (int,float) and np.isfinite(v) for v in invalid_values):
        raise ValueError('invalid_values must list confirmed finite sentinels')
    for value in invalid_values:
        valid &= values != value
    if sources['valid'] is not None:
        quality_mask = _read(root, sources['valid'])
        if quality_mask.dtype != bool or quality_mask.shape != values.shape:
            raise ValueError('Measurement quality mask must be boolean with the original shape')
        valid &= quality_mask
    rois = {role: _read(root, sources[role+'_roi']) for role in ('train', 'validation')}
    if any(roi.dtype != bool or roi.shape != values.shape[:-1] or not roi.any() for roi in rois.values()):
        raise ValueError('ROIs must be nonempty boolean arrays with unchanged spatial/sample geometry')
    if np.any(rois['train'] & rois['validation']):
        raise ValueError('Train and validation footprints overlap')
    labels = None
    if sources['labels'] is not None:
        source_labels = _read(root, sources['labels'])
        if recipe['format'] == 'spectra' and source_labels.shape == (values.shape[0], 1):
            source_labels = source_labels[:, 0]
        if source_labels.shape != values.shape[:-1] or source_labels.dtype.kind not in 'iuf' or not np.isfinite(source_labels).all() or not np.equal(source_labels, np.floor(source_labels)).all():
            raise ValueError('Labels must be finite integer IDs in the original geometry')
        mapping = recipe['source_to_class']
        names = recipe['class_names']
        if not isinstance(names, list) or len(names) < 2 or len(set(names)) != len(names):
            raise ValueError('An approved canonical taxonomy is required for labels')
        parsed = {int(k):v for k,v in mapping.items()}
        if len(parsed) != len(mapping) or recipe['unlabeled_id'] in parsed or any(type(v) is not int or not 0 <= v < len(names) for v in parsed.values()):
            raise ValueError('Invalid class mapping or unlabeled ID mapped to a class')
        present = set(np.unique(source_labels).tolist()) - {recipe['unlabeled_id']}
        if not present.issubset(parsed):
            raise ValueError('Every observed source class must have an explicit canonical mapping')
        labels = np.full(source_labels.shape, -1, dtype=np.int32)
        for source_id, target in parsed.items():
            labels[source_labels == source_id] = target
    plebani = read_json(Path(root)/'configs/plebani350_axis.json')
    wave = np.asarray(axis['wavelengths_nm'])
    matches = np.abs(wave[:,None] - np.asarray(plebani['wavelengths_nm'])[None,:]) <= plebani['match_tolerance_nm']
    matches &= np.asarray(axis['detectors'])[:,None] == 'L'
    if np.any(matches.sum(0)>1) or np.any(matches.sum(1)>1):
        raise ValueError('Ambiguous Plebani wavelength correspondence')
    output = inside(root, output_relative)
    output.mkdir(parents=True, exist_ok=False)
    refs = {}
    arrays = {'values':values, 'valid':valid, 'train_roi':rois['train'], 'validation_roi':rois['validation']}
    if labels is not None:
        arrays['labels'] = labels
    for key, array in arrays.items():
        path = output/(key+'.npy')
        np.save(path, array, allow_pickle=False)
        refs[key] = {'path':path.name, 'sha256':digest(path)}
    for role in ('train', 'validation'):
        document = {'schema_version':1, 'scene_id':recipe['scene_id'], 'role':role,
                    'format':recipe['format'], 'values':refs['values'], 'valid':refs['valid'],
                    'roi':refs[role+'_roi'], 'labels':refs.get('labels'), 'axis':recipe['axis'],
                    'class_names':recipe['class_names'] if labels is not None else [],
                    'units':recipe['units'], 'domain':recipe['domain'], 'metadata_verified':True,
                    'provenance':{'source_observation_ids':observations,
                                  'spatial_context':'real' if recipe['format']=='cube' else 'none',
                                  'plebani_membership':matches.any(1).tolist(),
                                  'sources':sources, 'quality_policy':quality}}
        write_json(output/(role+'.json'),document)
    write_json(output/'preparation.json', {'recipe':recipe, 'recipe_sha256':digest(recipe_path)})
    return output
