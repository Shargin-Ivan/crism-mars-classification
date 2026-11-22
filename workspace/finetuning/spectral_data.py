"""Prepare observation-disjoint supervised spectra from the published Plebani MAT."""
from contextlib import contextmanager
from pathlib import Path
import re
import struct
import tempfile
import zlib

import numpy as np

from .io import digest, inside, pinned, read_json, write_json


@contextmanager
def _spectra_matrix(mat, rows, scratch):
    """Inflate the pinned MAT-v5 first matrix to disk, avoiding a 1.66 GB RAM array."""
    shape = (rows, 350)
    size = rows * 350 * 8
    # Standard little-endian miMATRIX, real mxDOUBLE, dimensions, name, miDOUBLE.
    header = struct.pack('<10I2I8s2I', 14, size + 56, 6, 8, 6, 0, 5, 8,
                         *shape, 1, 7, b'pixspec', 9, size)
    with mat.open('rb') as source, tempfile.TemporaryFile(dir=scratch) as expanded:
        source.seek(126)
        if source.read(2) != b'IM':
            raise ValueError('Expected a little-endian MAT-v5 file')
        kind, remaining = struct.unpack('<II', source.read(8))
        if kind != 15:
            raise ValueError('Expected a compressed pixspec matrix as the first MAT variable')
        decoder = zlib.decompressobj()
        while remaining:
            data = source.read(min(1024 * 1024, remaining))
            if not data:
                raise ValueError('Incomplete compressed pixspec')
            remaining -= len(data)
            while data:
                expanded.write(decoder.decompress(data, 4 * 1024 * 1024))
                data = decoder.unconsumed_tail
                if expanded.tell() > len(header) + size:
                    raise ValueError('Unexpected expanded pixspec length')
        if not decoder.eof or decoder.unused_data or expanded.tell() != len(header) + size:
            raise ValueError('Invalid compressed pixspec payload')
        expanded.seek(0)
        if expanded.read(len(header)) != header:
            raise ValueError('Expected pixspec as a real N x 350 MATLAB double matrix')
        matrix = np.memmap(expanded, mode='r', offset=len(header), dtype='<f8',
                           shape=shape, order='F')
        try:
            yield matrix
        finally:
            matrix._mmap.close()


def _split_metadata(mat, recipe, taxonomy, hymars):
    from scipy.io import loadmat

    data = loadmat(mat, variable_names=['im_names', 'pixims', 'pixlabs'])
    suffixes = [s.strip().upper() for s in data['im_names'].ravel().tolist()]
    if (not suffixes or len(suffixes) != len(set(suffixes)) or
            any(not re.fullmatch('[0-9A-F]{5}', s) for s in suffixes)):
        raise ValueError('im_names must contain unique five-digit observation numbers')
    observations, raw = data['pixims'], data['pixlabs']
    if (observations.ndim != 2 or observations.shape[1] != 1 or
            observations.shape != raw.shape or observations.dtype.kind not in 'iu' or
            not np.all((observations >= 1) & (observations <= len(suffixes)))):
        raise ValueError('pixims and pixlabs must have aligned rows with valid observation indices')
    observations = observations.ravel()
    raw = raw.ravel()
    names = taxonomy['class_names']
    mapping = taxonomy['plebani_raw_to_dense']
    if (not names or len(names) != len(set(names)) or not all(isinstance(n, str) and n for n in names)
            or len(mapping) != len(names) or set(mapping.values()) != set(range(len(names)))):
        raise ValueError('Taxonomy must define unique names and a full dense mapping')
    labels = np.full(raw.shape, -1, dtype=np.int32)
    for source, target in mapping.items():
        labels[raw == int(source)] = target
    if (labels < 0).any():
        raise ValueError('pixlabs contains unknown or nonintegral source labels')

    forbidden_ids = [scene['observation_id'] for scene in hymars['scenes']]
    if (len(forbidden_ids) != 3 or not all(re.fullmatch('FRT000[0-9A-F]{5}', obs)
                                           for obs in forbidden_ids) or len(set(forbidden_ids)) != 3):
        raise ValueError('All three distinct HyMars observation identifiers are required')
    forbidden = {obs[-5:] for obs in forbidden_ids}
    validation = recipe['validation_observation_suffixes']
    if (not isinstance(validation, list) or not validation or len(set(validation)) != len(validation)
            or not set(validation) <= set(suffixes) - forbidden):
        raise ValueError('Validation observations must be present, unique and disjoint from HyMars')
    roles = {}
    for role, members in [('train', set(suffixes) - forbidden - set(validation)),
                          ('validation', set(validation))]:
        indices = np.flatnonzero(np.isin(observations, [i for i, s in enumerate(suffixes, 1)
                                                        if s in members]))
        if not indices.size:
            raise ValueError(f'Empty {role} split')
        ids = [suffixes[i-1] for i in np.unique(observations[indices])]
        roles[role] = indices, ids
    return labels, roles, forbidden_ids


def _save_split(folder, role, indices, observations, matrix, labels, axis, names,
                recipe, recipe_sha, excluded):
    folder.mkdir()
    refs = {}
    shape = (len(indices), matrix.shape[1])
    values = np.lib.format.open_memmap(folder / 'values.npy', 'w+', np.float32, shape)
    try:
        valid = np.lib.format.open_memmap(folder / 'valid.npy', 'w+', bool, shape)
        try:
            invalid_count = 0
            for start in range(0, len(indices), 2048):
                sl = slice(start, start + 2048)
                block = matrix[indices[sl]]
                mask = np.isfinite(block) & (block != 65535)
                if np.any(np.abs(block[mask]) > np.finfo(np.float32).max):
                    raise ValueError('A valid measurement exceeds the model float32 range')
                block[~mask] = 0
                values[sl] = block
                valid[sl] = mask
                invalid_count += int(mask.size - mask.sum())
        finally:
            valid._mmap.close()
    finally:
        values._mmap.close()
    selected_labels = labels[indices]
    np.save(folder / 'labels.npy', selected_labels)
    np.save(folder / 'roi.npy', np.ones(len(indices), dtype=bool))
    np.save(folder / 'source_rows.npy', indices.astype(np.int64, copy=False))
    for key in ('values', 'valid', 'labels', 'roi', 'source_rows'):
        path = folder / (key + '.npy')
        refs[key] = {'path': path.name, 'sha256': digest(path)}
    manifest = dict(schema_version=1, scene_id='plebani_spectra_' + role, role=role,
                    format='spectra', metadata_verified=True, units='dimensionless', domain='ratioed_if',
                    class_names=names, axis=axis, **refs)
    manifest['provenance'] = dict(source_observation_ids=observations, spatial_context='none',
                                  observation_id_convention='Five-digit CRISM observation number from im_names.',
                                  plebani_membership=[True] * 350, excluded_hymars_observation_ids=excluded,
                                  mat=recipe['mat'], axis_profile=recipe['axis'], taxonomy=recipe['taxonomy'],
                                  recipe_sha256=recipe_sha, source_rows_base=0,
                                  quality_policy='Finite and unequal to fill value 65535; valid zeros retained.',
                                  invalid_measurements=invalid_count,
                                  dense_label_counts=np.bincount(selected_labels, minlength=len(names)).tolist())
    write_json(folder / 'manifest.json', manifest)


def prepare_lwnet_spectra(recipe_path, output_relative, root):
    """Publish two complete portable manifests atomically to a new directory."""
    recipe = read_json(recipe_path)
    if recipe.get('schema_version') != 1 or recipe.get('layout') != 'plebani_ratioed_pixspec_mat5':
        raise ValueError('Expected a Plebani ratioed-pixel preparation recipe')
    destination = inside(root, output_relative)
    if destination.exists():
        raise FileExistsError(destination)
    mat = pinned(root, recipe['mat'])
    taxonomy = read_json(pinned(root, recipe['taxonomy']))
    profile = read_json(pinned(root, recipe['axis']))
    hymars = read_json(pinned(root, recipe['hymars_scenes']))
    waves = np.asarray(profile['wavelengths_nm'], dtype=np.float64)
    if (profile['detector'] != 'L' or waves.shape != (350,) or not np.isfinite(waves).all()
            or not np.all(waves > 0) or not np.all(np.diff(waves) > 0)):
        raise ValueError('Expected 350 ordered Plebani L wavelengths')
    segments = np.r_[0, np.cumsum(np.diff(waves) > 7.5)]
    axis = dict(wavelengths_nm=waves.tolist(), detectors=['L'] * 350, segment_ids=segments.tolist())
    labels, roles, excluded = _split_metadata(mat, recipe, taxonomy, hymars)
    recipe_sha = digest(recipe_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        pending = Path(scratch) / 'prepared'
        pending.mkdir()
        with _spectra_matrix(mat, len(labels), scratch) as matrix:
            for role, (indices, observations) in roles.items():
                _save_split(pending / role, role, indices, observations, matrix, labels, axis,
                            taxonomy['class_names'], recipe, recipe_sha, excluded)
        pending.rename(destination)
    return destination
