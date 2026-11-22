"""Prepare measured-spectrum denoising manifests from pinned spatial pairs and ROIs."""
from contextlib import ExitStack
import errno
from math import ceil
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from .export import _axis, _known
from .io import digest, inside, pinned, read_json, write_json


def _bind_array(source, destination, expected_hash):
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.ENOTSUP, errno.EPERM, errno.EACCES, errno.ENOSYS}:
            raise
        shutil.copyfile(source, destination)
        if digest(destination) != expected_hash:
            raise ValueError('Array changed while making a portable copy')


class _ChannelView:
    """Read-only spectral selection, materialized only for a requested row or block."""

    def __init__(self, array, indices):
        self._array = array
        self._indices = indices
        self.dtype = array.dtype
        self.shape = (*array.shape[:-1], len(indices))
        self.ndim = array.ndim
        self._mmap = getattr(array, '_mmap', None)

    def __getitem__(self, spatial):
        return np.take(self._array[spatial], self._indices, axis=-1)

    def __len__(self):
        return self.shape[0]

    def reshape(self, rows, channels):
        if rows != -1 or channels != self.shape[-1]:
            raise ValueError('Only flattening spatial dimensions is supported')
        if self._array.ndim == 3 and not self._array.flags.c_contiguous:
            return _FlatCube(self._array, self._indices)
        return _ChannelView(self._array.reshape(-1, self._array.shape[-1]), self._indices)


class _FlatCube:
    """Read C-order pixel IDs from a Fortran-order cube without reshaping the cube."""

    def __init__(self, cube, indices):
        self.cube = cube
        self.indices = indices
        self.shape = (cube.shape[0] * cube.shape[1], len(indices))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, rows):
        if isinstance(rows, slice):
            rows = np.arange(*rows.indices(len(self)))
        y, x = np.divmod(rows, self.cube.shape[1])
        return np.take(self.cube[y, x], self.indices, axis=-1)


def project_channels(array, indices):
    indices = np.asarray(indices)
    if (indices.ndim != 1 or not indices.size or indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices >= array.shape[-1])
            or np.unique(indices).size != indices.size):
        raise ValueError('Invalid or repeated source channel indices')
    return _ChannelView(array, indices)


def _observation(pair):
    provenance = pair['provenance']
    value = provenance.get('sources', {}).get('observation_id')
    if value is None:
        suffix = provenance['recipe']['observation_suffix']
        if not re.fullmatch(r'[0-9a-fA-F]{5}', suffix):
            raise ValueError('Expected a five-digit FRT suffix in spatial Plebani provenance')
        value = 'FRT000' + suffix
    if not isinstance(value, str) or not re.fullmatch(r'FRT[0-9A-F]{8}', value.upper()):
        raise ValueError('Expected a complete FRT observation ID')
    return value.upper()


def _channels(pair, stage, profile):
    source = pair.get('spectral_axis')
    channels = pair['shape_hwc'][-1]
    if source is None:
        if stage == 'plebani350':
            raise ValueError('A verified Plebani channel mapping is required')
        return None, None, None
    wave = np.array(source['wavelengths_nm'], dtype=float)
    detector = np.array(source['detectors'])
    if wave.shape != (channels,) or detector.shape != wave.shape:
        raise ValueError('Spectral metadata differs from the source cube shape')
    grid = np.asarray(profile['wavelengths_nm'])
    if grid.shape != (350,) or not np.all(np.diff(grid) > 0):
        raise ValueError('Expected the increasing 350-channel Plebani grid')
    match = ((np.abs(wave[:, None] - grid) <= profile['match_tolerance_nm'])
             & (detector[:, None] == 'L'))
    if np.any(match.sum(0) > 1) or np.any(match.sum(1) > 1):
        raise ValueError('Ambiguous wavelength mapping')
    if stage == 'plebani350':
        indices = np.where(match.T)[1]
        if len(indices) != len(grid) or source['plebani350_to_source_indices'] != indices.tolist():
            raise ValueError('Expected the verified ordered 350-band mapping')
    else:
        if not np.isfinite(wave).all() or not np.isin(detector, ['S', 'L']).all():
            return None, None, None
        if len(set(zip(wave, detector))) != channels:
            raise ValueError('Duplicate wavelength within a detector')
        indices = np.arange(channels)
    wave, detector = wave[indices], detector[indices]
    edges = (detector[1:] == detector[:-1]) & (np.diff(wave) > 0) & (np.diff(wave) <= 7.5)
    edges &= np.abs(np.diff(indices)) == 1
    if 'source_band_ids' in source:
        bands = np.asarray(source['source_band_ids'])
        if (bands.shape != (channels,) or bands.dtype.kind not in 'iu' or (bands < 0).any()
                or len(set(zip(source['detectors'], bands))) != channels):
            raise ValueError('Invalid original band IDs')
        edges &= np.abs(np.diff(bands[indices].astype(np.int64))) == 1
    if 'segment_ids' in source:
        source_segments = np.asarray(source['segment_ids'])
        if source_segments.shape != (channels,) or source_segments.dtype.kind not in 'iu':
            raise ValueError('Invalid declared spectral segments')
        edges &= np.diff(source_segments[indices]) == 0
    segments = np.r_[0, np.cumsum(~edges)].tolist()
    if (stage == 'hymars_full' and 'segment_ids' in source
            and np.array_equal(np.diff(source_segments) == 0, edges)):
        segments = source_segments.tolist()
    axis = _axis({'axis': {'wavelengths_nm': wave.tolist(), 'detectors': detector.tolist(),
                           'segment_ids': segments}})
    return indices.tolist(), axis, match.any(1)[indices].tolist()


def _scene(folder, root, pair_path, pair, split_path, split_item, metadata, profile, fraction):
    stage = split_item['stage']
    verified_binding = metadata.get('verified_metadata')
    if verified_binding is not None:
        metadata = read_json(pinned(root, verified_binding))
        if (metadata.get('metadata_verified') is not True
                or metadata.get('observation_id') != _observation(pair)
                or metadata.get('cube_sha256') != pair['cube']['sha256']
                or metadata.get('scene_id', pair['scene_id']) != pair['scene_id']
                or metadata.get('shape_hwc', pair['shape_hwc']) != pair['shape_hwc']
                or not _known(metadata.get('evidence'))):
            raise ValueError('Verified physical metadata must identify this exact cube and observation')
        pair = {**pair, 'spectral_axis': metadata['spectral_axis']}
    source_domain = pair['provenance'].get('recipe', {}).get('domain')
    if source_domain is not None and metadata.get('domain') != source_domain:
        raise ValueError('Declared radiometric domain differs from the source product')
    indices, axis, membership = _channels(pair, stage, profile)
    metadata_ready = (axis is not None and all(_known(metadata.get(k)) for k in ('units', 'domain'))
                      and (stage == 'plebani350' or verified_binding is not None))
    folder.mkdir()
    with ExitStack() as resources:
        def bind(name, base, ref):
            path = pinned(base, ref)
            local = folder / (name + '.npy')
            _bind_array(path, local, ref['sha256'])
            array = np.load(local, mmap_mode='r', allow_pickle=False)
            resources.callback(array._mmap.close)
            return {'path': local.name, 'sha256': ref['sha256']}, array

        values_ref, values = bind('values', pair_path.parent, pair['cube'])
        valid_ref, valid = bind('valid', pair_path.parent, pair['valid'])
        labels_ref, labels = bind('labels', pair_path.parent, pair['labels'])
        if (values.ndim != 3 or list(values.shape) != pair['shape_hwc'] or values.dtype.kind != 'f'
                or valid.shape != values.shape or valid.dtype != bool
                or labels.shape != values.shape[:2] or labels.dtype.kind not in 'iu'):
            raise ValueError('Invalid cube or measurement mask')
        rois, refs = {}, {}
        for role in ('train', 'validation'):
            key = role + '_roi'
            refs[role], rois[role] = bind(key, split_path.parent, split_item['masks'][key])
            if rois[role].shape != values.shape[:2] or rois[role].dtype != bool:
                raise ValueError('Invalid spatial ROI')
        if (rois['train'] & rois['validation']).any():
            raise ValueError('Train and validation ROIs overlap')
        expected = len(indices) if indices is not None else values.shape[-1]
        required = ceil(fraction * expected)
        counts = {'train': 0, 'validation': 0}
        for y in range(0, values.shape[0], 8):
            rows = slice(y, y + 8)
            mask = valid[rows]
            if (mask & ~np.isfinite(values[rows])).any():
                raise ValueError('Nonfinite source value is marked valid')
            if indices is not None:
                mask = np.take(mask, indices, axis=-1)
            usable = mask.sum(axis=-1) >= required
            for role, roi in rois.items():
                counts[role] += int((usable & roi[rows]).sum())
        if min(counts.values()) == 0:
            raise ValueError('No spectra meet coverage in one or both ROIs')
    for role in rois:
        doc = {'schema_version': 1, 'scene_id': pair['scene_id'], 'role': role, 'format': 'cube',
               'stage': stage, 'values': values_ref, 'valid': valid_ref, 'roi': refs[role],
               'labels': None, 'class_names': [], 'source_channel_indices': indices,
               'axis': axis, 'units': metadata.get('units'), 'domain': metadata.get('domain'),
               'metadata_verified': metadata_ready, 'provenance': {
                   'source_observation_ids': [_observation(pair)], 'spatial_context': 'real',
                   'plebani_membership': membership, 'pair_sha256': digest(pair_path),
                   'source_domain_evidence': metadata.get('evidence'),
                   'verified_metadata': verified_binding,
                   'wavelength_convention': metadata.get('wavelength_convention'),
                   'target': 'measured_spectrum', 'noise': 'added_online_on_valid_normalized_channels',
                   'sampling': 'any_roi_pixel_meeting_coverage_regardless_of_label'}}
        write_json(folder / (role + '.json'), doc)
        write_json(folder / ('lwnet_' + role + '.json'), {
            **doc, 'labels': labels_ref, 'class_names': pair['class_names'],
            'provenance': {
                key: value for key, value in doc['provenance'].items()
                if key not in {'target', 'noise', 'sampling'}}})
    return {'scene_id': pair['scene_id'], 'observation_id': _observation(pair), 'stage': stage,
            'physical_metadata_ready': metadata_ready, 'axis_known': axis is not None,
            'source_channels': values.shape[-1], 'selected_channels': expected,
            'min_valid_channels': required, 'usable_pixels': counts,
            'datasets': [f'{folder.name}/{role}.json' for role in rois],
            'lwnet_datasets': [f'{folder.name}/lwnet_{role}.json' for role in rois]}


def prepare_denoising(recipe_path, output_relative, root):
    """Create a portable directory of NPY hard links and training descriptors, atomically."""
    recipe = read_json(recipe_path)
    if recipe.get('schema_version') != 1:
        raise ValueError('Unsupported denoising recipe')
    fraction = recipe['min_valid_fraction']
    if type(fraction) not in (int, float) or not 0 < fraction <= 1:
        raise ValueError('Invalid minimum measurement coverage')
    if recipe.get('mode') == 'prepared_spectra':
        return _prepare_spectra(recipe_path, recipe, output_relative, root)
    split_path = pinned(root, recipe['split'])
    split = read_json(split_path)
    held_scene = split['recipe']['held_out_scene_id']
    if held_scene != recipe['held_out_scene_id'] or held_scene in recipe['scenes']:
        raise ValueError('Held-out scene cannot participate in preparation')
    stages = recipe.get('stages', ['hymars_full', 'plebani350'])
    if (not isinstance(stages, list) or 'hymars_full' not in stages
            or not set(stages) <= {'plebani350', 'hymars_full'}):
        raise ValueError('Invalid preparation stage selection')
    rows = [row for row in split['pairs'] if row['stage'] in stages]
    row_by_id = {r['scene_id']: r for r in rows}
    if len(row_by_id) != len(rows):
        raise ValueError('Repeated scene in split')
    pairs, observations, cubes = {}, set(), set()
    for entry in split['recipe']['pairs']:
        if entry['stage'] not in stages:
            continue
        if entry['role'] == 'held_out_test' and 'held_out_fingerprint' in recipe:
            fingerprint = recipe['held_out_fingerprint']
            row = row_by_id[held_scene]
            if (entry['pair']['sha256'] != fingerprint['pair_sha256']
                    or row['role'] != entry['role'] or row['stage'] != entry['stage']
                    or held_scene in pairs or fingerprint['cube_sha256'] in cubes
                    or not re.fullmatch('[0-9a-f]{64}', fingerprint['cube_sha256'])):
                raise ValueError('Held-out fingerprint disagrees with split')
            pairs[held_scene] = None
            cubes.add(fingerprint['cube_sha256'])
            observations.add(recipe['held_out_observation_id'])
            continue
        path = pinned(root, entry['pair'])
        pair = read_json(path)
        scene = pair['scene_id']
        row = row_by_id[scene]
        observation = _observation(pair)
        held = scene == held_scene
        if (scene in pairs or observation in observations or pair['cube']['sha256'] in cubes
                or not pair.get('pair_ready') or pair.get('spatial_context') != 'real'
                or (scene != Path(scene).name or scene in {'.', '..'})
                or row['stage'] != entry['stage'] or row['stage'] not in ('plebani350', 'hymars_full')
                or row['role'] != entry['role'] or row['role'] != ('held_out_test' if held else 'development')
                or pair['role'] not in (('held_out_test',) if held else ('development', 'development_candidate'))
                or held != (observation == recipe['held_out_observation_id'])):
            raise ValueError('Split, scene, source observation or held-out identity mismatch')
        pairs[scene] = (path, pair)
        observations.add(observation)
        cubes.add(pair['cube']['sha256'])
    if (set(pairs) != set(row_by_id) or held_scene not in pairs
            or set(recipe['scenes']) != set(pairs) - {held_scene}):
        raise ValueError('Recipe must cover exactly the development scenes')
    profile = read_json(pinned(root, recipe['plebani_grid']))
    output = inside(root, output_relative)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as staging:
        folder = Path(staging) / 'prepared'
        folder.mkdir()
        records = []
        for scene, metadata in recipe['scenes'].items():
            path, pair = pairs[scene]
            records.append(_scene(folder / scene, root, path, pair, split_path, row_by_id[scene],
                                  metadata, profile, fraction))
        write_json(folder / 'denoising.json', {
            'schema_version': 1, 'recipe': recipe, 'recipe_sha256': digest(recipe_path),
            'held_out_scene_id': held_scene, 'held_out_observation_id': recipe['held_out_observation_id'],
            'min_valid_fraction': fraction, 'scenes': records})
        folder.rename(output)
    return output


def _prepare_spectra(recipe_path, recipe, output_relative, root):
    """Reuse NC measured spectra without a class requirement or a spatial product."""
    from .configuration import observation_keys
    from .data import PreparedDataset

    profile = read_json(pinned(root, recipe['plebani_grid']))
    grid = np.asarray(profile['wavelengths_nm'])
    excluded = observation_keys([recipe['held_out_observation_id'], *recipe.get('excluded_observation_ids', [])])
    domain = recipe['target_domain']
    if domain not in ('ratioed_if', 'corrected_unratioed_if'):
        raise ValueError('Explicit ratioed_if or corrected_unratioed_if domain required')
    output = inside(root, output_relative)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records, seen = [], set()
    with TemporaryDirectory(dir=output.parent) as staging:
        dest = Path(staging) / 'prepared'
        dest.mkdir()
        for role in ('train', 'validation'):
            path = inside(root, recipe['inputs'][role])
            original_hash = digest(path)
            source = read_json(path)
            if (source.get('role') != role or source.get('format') != 'spectra'
                    or source.get('scene_id') == recipe['held_out_scene_id']
                    or source.get('metadata_verified') is not True
                    or source.get('domain') != domain or source.get('units') != 'dimensionless'):
                raise ValueError('Expected measured spectra in the declared domain and split')
            ids = observation_keys(source['provenance']['source_observation_ids'])
            if not ids or any(kind != 'crism' for kind, _ in ids) or ids & (seen | excluded):
                raise ValueError('Held-out or overlapping observation IDs in Plebani spectrum split')
            seen.update(ids)
            axis = _axis(source)
            wave = np.asarray(axis['wavelengths_nm'])
            if (grid.shape != (350,) or wave.shape != grid.shape
                    or not np.all(np.abs(wave - grid) <= profile['match_tolerance_nm'])
                    or set(axis['detectors']) != {'L'}):
                raise ValueError('Spectra must have the ordered Plebani-350 axis')
            folder = dest / role
            folder.mkdir()
            doc = {**source, 'labels': None, 'class_names': [], 'provenance': {
                **source['provenance'],
                'target': 'measured_spectrum', 'sampling': 'any_roi_pixel_meeting_coverage_regardless_of_label',
                'noise': 'added_online_on_valid_normalized_channels', 'source_manifest_sha256': original_hash}}
            names = ['values', 'valid', 'roi']
            if source.get('source_rows') is not None:
                names.append('source_rows')
            for name in names:
                ref = source[name]
                _bind_array(pinned(path.parent, ref), folder / (name + '.npy'), ref['sha256'])
                doc[name] = {'path': name + '.npy', 'sha256': ref['sha256']}
            manifest = folder / 'manifest.json'
            write_json(manifest, doc)
            required = ceil(recipe['min_valid_fraction'] * 350)
            with PreparedDataset(manifest, 'n2n4m', 'plebani350', 1, recipe['held_out_scene_id'], [], []) as data:
                if not data.channel_mask.all():
                    raise ValueError('Disagreement between source membership and Plebani-350 axis')
                count = sum(int(np.count_nonzero(data.valid[rows].sum(axis=1) >= required))
                            for rows in np.array_split(data.indices, max(1, ceil(len(data) / 4096))))
                if count < (2 if role == 'train' else 1):
                    raise ValueError('Insufficient spectra with the required measurement coverage')
            records.append({'role': role, 'dataset': f'{role}/manifest.json', 'usable_spectra': count,
                            'source_observation_ids': source['provenance']['source_observation_ids'],
                            'source_manifest_sha256': original_hash})
        write_json(dest / 'denoising.json', {'schema_version': 1, 'recipe': recipe,
                   'recipe_sha256': digest(recipe_path), 'physical_metadata_ready': True,
                   'target_domain': domain, 'min_valid_fraction': recipe['min_valid_fraction'],
                   'datasets': records})
        dest.rename(output)
    return output
