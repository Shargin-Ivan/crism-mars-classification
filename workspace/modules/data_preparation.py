"""Prepare one complete H,W,C cube without classifiers, training or partitions."""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re

import numpy as np

from modules.pipeline_contracts import (
    ArtifactRef, PreparedCube, SpectralAxis, contract_dict, integer, number,
    require, sequence, text,
)
from modules.pipeline_config import PreparationSpec, load_config
from modules.resource_store import (
    StorePaths, array_reference, decode_contract, read_json, resolved_path,
    sha256_file, verify_ref,
)
from modules.run_store import unique_directory, write_manifest


@dataclass(frozen=True)
class SceneManifest:
    schema_version: int
    scene_id: str
    format: str
    cube: ArtifactRef
    shape: tuple[int, int, int]
    dtype: str
    variable: str | None
    header: ArtifactRef | None
    spectral_axis: SpectralAxis
    good_channels: tuple[bool, ...] | None
    invalid_values: tuple[float, ...]
    units: str
    domain: str
    metadata_sources: tuple[ArtifactRef, ...]
    ground_truth: ArtifactRef | None
    ground_truth_variable: str | None

    def __post_init__(self):
        require(type(self.schema_version) is int and self.schema_version == 1, 'Unsupported scene schema')
        require(isinstance(self.scene_id, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', self.scene_id),
                'scene_id must be a safe directory identifier')
        require(self.format in ('npy', 'mat', 'envi'), 'Supported inputs: NPY, MATLAB v5/v7, ENVI')
        require(isinstance(self.cube, ArtifactRef), 'Expected pinned source cube')
        shape = sequence(self.shape, 'scene shape')
        require(len(shape) == 3, 'Scene axes must be H,W,C')
        for size in shape:
            integer(size, 'scene dimension', 1)
        object.__setattr__(self, 'shape', shape)
        require(isinstance(self.dtype, str), 'Source dtype must be explicit')
        dtype = np.dtype(self.dtype)
        require(dtype.kind == 'f' and dtype.itemsize in (4, 8), 'Source must contain float32/float64 measurements')
        object.__setattr__(self, 'dtype', dtype.str)
        require(isinstance(self.spectral_axis, SpectralAxis), 'Expected spectral metadata')
        require(self.spectral_axis.channel_indices == tuple(range(shape[2])), 'All source channels are required')
        if self.format == 'mat':
            text(self.variable, 'MAT variable')
        else:
            require(self.variable is None, 'Only MAT inputs have a variable name')
        if self.format == 'envi':
            require(isinstance(self.header, ArtifactRef), 'ENVI requires a pinned header and binary')
        else:
            require(self.header is None, 'Only ENVI inputs have an HDR reference')
        if self.good_channels is not None:
            quality = sequence(self.good_channels, 'good_channels')
            require(len(quality) == shape[2] and all(type(v) is bool for v in quality), 'Invalid channel quality mask')
            object.__setattr__(self, 'good_channels', quality)
        invalid = sequence(self.invalid_values, 'invalid_values')
        for value in invalid:
            number(value, 'sentinel', -np.finfo(np.float64).max, np.finfo(np.float64).max)
        object.__setattr__(self, 'invalid_values', tuple(float(v) for v in invalid))
        for name in ('units', 'domain'):
            text(getattr(self, name), name)
        metadata = sequence(self.metadata_sources, 'metadata_sources')
        require(all(isinstance(ref, ArtifactRef) for ref in metadata), 'Expected pinned metadata sources')
        object.__setattr__(self, 'metadata_sources', metadata)
        require(self.ground_truth is None or isinstance(self.ground_truth, ArtifactRef), 'Invalid GT reference')
        if self.ground_truth_variable is not None:
            text(self.ground_truth_variable, 'GT variable')
            require(self.ground_truth is not None, 'GT variable requires a GT reference')

    def references(self):
        return tuple(ref for ref in (self.cube, self.header, self.ground_truth, *self.metadata_sources)
                     if ref is not None)

    def absolute(self, base):
        def absolute(ref):
            return replace(ref, path=str(resolved_path(ref.path, base))) if ref else None
        return replace(self, cube=absolute(self.cube), header=absolute(self.header),
                       ground_truth=absolute(self.ground_truth),
                       metadata_sources=tuple(absolute(ref) for ref in self.metadata_sources))


def _envi_input(scene, base):
    from spectral.io import envi
    image = envi.open(str(resolved_path(scene.header.path, base)), str(resolved_path(scene.cube.path, base)))
    if scene.spectral_axis.wavelengths_nm is not None:
        unit = str(image.metadata.get('wavelength units', '')).strip().lower()
        factors = {'nm': 1, 'nanometers': 1, 'nanometres': 1,
                   'um': 1000, 'micrometers': 1000, 'micrometres': 1000}
        require(unit in factors, 'Unknown ENVI wavelength unit')
        grid = np.asarray(image.bands.centers, dtype=float) * factors[unit]
        require(grid.shape == (scene.shape[2],) and
                np.allclose(grid, scene.spectral_axis.wavelengths_nm, rtol=0, atol=0.01),
                'ENVI header and declared wavelength order differ')
    if 'data ignore value' in image.metadata:
        sentinel = float(image.metadata['data ignore value'])
        require(not np.isfinite(sentinel) or sentinel in scene.invalid_values,
                'Declare the ENVI data ignore value in invalid_values')
    quality = None
    if 'bbl' in image.metadata:
        bbl = np.asarray(image.metadata['bbl'], dtype=float)
        require(bbl.shape == (scene.shape[2],) and np.isin(bbl, [0, 1]).all(),
                'ENVI bbl must contain one 0/1 quality flag per source channel')
        quality = bbl.astype(bool)
    return image, quality


@contextmanager
def _source_array(scene, base):
    path = resolved_path(scene.cube.path, base)
    cube = None
    try:
        if scene.format == 'npy':
            cube = np.load(path, mmap_mode='r', allow_pickle=False)
        elif scene.format == 'mat':
            from scipy.io import loadmat
            document = loadmat(path, variable_names=[scene.variable])
            require(scene.variable in document, f'MAT variable not found: {scene.variable}')
            cube = document[scene.variable]
            cube.flags.writeable = False
        else:
            image, _ = _envi_input(scene, base)
            cube = image.open_memmap(interleave='bip', writable=False)
        require(isinstance(cube, np.ndarray) and cube.shape == scene.shape, 'Source shape mismatch')
        require(cube.dtype == np.dtype(scene.dtype), 'Source dtype mismatch')
        yield cube
    finally:
        if isinstance(cube, np.memmap):
            cube._mmap.close()


def prepare_scene(manifest_path, results_root, *, expected_sha256=None, rows_per_chunk=16):
    """Archive a full cube; source files are only opened for reading.

    NPY/ENVI are streamed. Compressed MATLAB v5/v7 requires loading the selected
    variable into RAM once; MATLAB v7.3/HDF5 is not silently guessed or converted.
    Unknown metadata can be archived but never reported as inference-ready.
    """
    integer(rows_per_chunk, 'rows_per_chunk', 1)
    PreparationSpec()  # Only the full-cube profile is supported here.
    manifest_path = Path(manifest_path).resolve()
    source_hash = sha256_file(manifest_path)
    if expected_sha256 is not None:
        require(source_hash == expected_sha256, 'Scene manifest hash mismatch')
    scene = decode_contract(SceneManifest, read_json(manifest_path))
    require(sha256_file(manifest_path) == source_hash, 'Scene manifest changed while reading')
    scene = scene.absolute(manifest_path.parent)
    results_root = Path(results_root).resolve()
    for ref in scene.references():
        path = verify_ref(ref, manifest_path.parent)
        require(not path.is_relative_to(results_root), 'Original source must not reside in the results store')
    envi_quality = None
    if scene.format == 'envi':
        _, envi_quality = _envi_input(scene, manifest_path.parent)
        if envi_quality is not None:
            if scene.good_channels is not None:
                envi_quality &= scene.good_channels
            scene = replace(scene, good_channels=tuple(bool(v) for v in envi_quality))
    run = unique_directory(results_root / 'prepared_cubes')
    root = run / scene.scene_id
    (root / 'original').mkdir(parents=True)
    document = {'schema_version': 2, 'kind': 'prepared_cube', 'status': 'preparing',
                'scene_id': scene.scene_id,
                'source_manifest': contract_dict(ArtifactRef(scene.scene_id, str(manifest_path), source_hash)),
                'preparation': contract_dict(PreparationSpec()),
                'quality_policy': 'envi_bbl_and_declared_channels_and_finite_samples' if envi_quality is not None else
                                  'provided_channels_and_finite_samples' if scene.good_channels is not None
                                  else 'finite_samples_only_quality_unverified'}
    write_manifest(root / 'manifest.json', document)
    try:
        write_manifest(root / 'source_manifest.json', contract_dict(scene))
        write_manifest(root / 'spectral_axis.json', contract_dict(scene.spectral_axis))
        quality = np.array(scene.good_channels if scene.good_channels is not None
                           else [True] * scene.shape[2], dtype=bool)
        np.save(root / 'original/good_channels.npy', quality, allow_pickle=False)
        cube_path, valid_path = root / 'original/cube.npy', root / 'original/valid_samples.npy'
        output = valid = None
        try:
            with _source_array(scene, manifest_path.parent) as cube:
                output = np.lib.format.open_memmap(cube_path.with_suffix('.npy.tmp'), mode='w+',
                                                   dtype=cube.dtype, shape=cube.shape)
                valid = np.lib.format.open_memmap(valid_path.with_suffix('.npy.tmp'), mode='w+',
                                                  dtype=bool, shape=cube.shape)
                for start in range(0, scene.shape[0], rows_per_chunk):
                    block = np.array(cube[start:start + rows_per_chunk], copy=True)
                    measured = np.isfinite(block)
                    for sentinel in scene.invalid_values:
                        measured &= block != sentinel
                    block[~measured] = np.nan
                    output[start:start + len(block)] = block
                    valid[start:start + len(block)] = measured & quality
        finally:
            for array in (output, valid):
                if array is not None:
                    array.flush()
                    array._mmap.close()
        cube_path.with_suffix('.npy.tmp').replace(cube_path)
        valid_path.with_suffix('.npy.tmp').replace(valid_path)
        # Detect source replacement during preparation, before publishing success.
        for ref in scene.references():
            verify_ref(ref, manifest_path.parent)
        source_ref = ArtifactRef(scene.scene_id, 'source_manifest.json', sha256_file(root / 'source_manifest.json'))
        prepared = PreparedCube(scene.scene_id, source_ref, scene.shape,
                                array_reference(cube_path, 'prepared_cube', root),
                                array_reference(valid_path, 'valid_samples', root),
                                array_reference(root / 'original/good_channels.npy', 'good_channels', root),
                                scene.spectral_axis, scene.units, scene.domain)
        metadata_ready = scene.spectral_axis.metadata_status == 'verified' and scene.units != 'unknown' and scene.domain != 'unknown'
        document.update(status='completed', prepared=contract_dict(prepared), metadata_ready=metadata_ready,
                        spectral_axis_ref=contract_dict(ArtifactRef('spectral_axis', 'spectral_axis.json',
                                                                  sha256_file(root / 'spectral_axis.json'))))
        write_manifest(root / 'manifest.json', document)
        return root
    except BaseException as error:
        document.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_manifest(root / 'manifest.json', document)
        raise


def load_prepared_cube(root, *, verify_sources=True):
    """Validate an existing immutable archive and return its metadata contract."""
    root = Path(root).resolve()
    document = read_json(root / 'manifest.json')
    require(isinstance(document, dict) and type(document.get('schema_version')) is int
            and document['schema_version'] == 2 and document.get('kind') == 'prepared_cube'
            and document.get('status') == 'completed', 'Not a completed whole-cube archive')
    prepared = decode_contract(PreparedCube, document['prepared'])
    snapshot_path = verify_ref(prepared.source, root, root=root)
    scene = decode_contract(SceneManifest, read_json(snapshot_path))
    decode_contract(PreparationSpec, document['preparation'])
    metadata_ready = scene.spectral_axis.metadata_status == 'verified' and scene.units != 'unknown' and scene.domain != 'unknown'
    require(type(document.get('metadata_ready')) is bool and document['metadata_ready'] == metadata_ready,
            'Archive readiness disagrees with spectral/radiometric metadata')
    require(prepared.scene_id == scene.scene_id and prepared.source_shape == scene.shape
            and document.get('scene_id') == scene.scene_id
            and prepared.spectral_axis == scene.spectral_axis
            and prepared.units == scene.units and prepared.domain == scene.domain, 'Archive/source metadata mismatch')
    spectral_path = verify_ref(decode_contract(ArtifactRef, document['spectral_axis_ref']), root, root=root)
    require(decode_contract(SpectralAxis, read_json(spectral_path)) == prepared.spectral_axis,
            'Spectral annotations disagree with the cube')
    if verify_sources:
        for ref in scene.references():
            verify_ref(ref, snapshot_path.parent)
    arrays = []
    try:
        for ref in (prepared.cube, prepared.valid_samples, prepared.good_channels):
            array = np.load(verify_ref(ref, root, root=root), mmap_mode='r', allow_pickle=False)
            arrays.append(array)
        prepared.validate_arrays(*arrays)
        expected_quality = scene.good_channels if scene.good_channels is not None else [True] * scene.shape[2]
        require(np.array_equal(arrays[2], expected_quality), 'Quality mask differs from the source declaration')
    finally:
        for array in arrays:
            if isinstance(array, np.memmap):
                array._mmap.close()
    return prepared


def prepare_from_config(config_path, *, rows_per_chunk=16):
    config = load_config(config_path)
    stores = StorePaths.from_config(config.stores, config_path)
    require(config.scene_manifest is not None, 'Bind scene_manifest before preparing a cube')
    scene_path = verify_ref(config.scene_manifest, Path(config_path).resolve().parent)
    return prepare_scene(scene_path, stores.results, expected_sha256=config.scene_manifest.sha256,
                         rows_per_chunk=rows_per_chunk)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--config', type=Path)
    inputs.add_argument('--scene', type=Path)
    parser.add_argument('--output', type=Path, help='Results root; required with --scene only')
    parser.add_argument('--rows-per-chunk', type=int, default=16)
    args = parser.parse_args(argv)
    if bool(args.scene) != bool(args.output):
        parser.error('--scene requires --output; --config uses its declared results store')
    try:
        root = (prepare_from_config(args.config, rows_per_chunk=args.rows_per_chunk) if args.config else
                prepare_scene(args.scene, args.output, rows_per_chunk=args.rows_per_chunk))
    except (OSError, ValueError, TypeError, NotImplementedError) as error:
        parser.error(str(error))
    print(json.dumps({'archive': str(root), 'metadata_ready': read_json(root / 'manifest.json')['metadata_ready']},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
