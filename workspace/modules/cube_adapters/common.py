"""Read-only whole-cube views and isolated adapter results (no torch imports)."""
from contextlib import contextmanager
from dataclasses import dataclass, replace
import os
from pathlib import Path
from time import monotonic

import numpy as np

from modules.data_preparation import load_prepared_cube
from modules.pipeline_contracts import (
    ArtifactRef, ArrayRef, BRANCHES, ClassificationResult, ContractError,
    DenoisingResult, PreparedCube, Taxonomy, contract_dict, require,
)
from modules.resource_store import array_reference, decode_contract, read_json, resolved_path, verify_ref
from modules.run_store import write_manifest


def inference_range(total, step, label, unit):
    """Range of chunk starts; optionally report completed work and loop-only ETA."""
    starts = range(0, total, step)
    if os.environ.get('CRISM_INFERENCE_PROGRESS') != '1':
        return starts

    def measured():
        started = monotonic()
        last_report = None
        for first in starts:
            yield first
            now = monotonic()
            finished = min(first + step, total)
            elapsed = now - started
            if elapsed <= 0 or (last_report is not None and now - last_report < 60):
                continue
            # A near-empty initial row can give a misleading instantaneous rate.
            if last_report is None and elapsed < 1 and finished < total:
                continue
            speed = finished / elapsed
            print(f'{label} {finished}/{total} {unit} elapsed={elapsed:.1f}s '
                  f'rate={speed:.2f} {unit}/s remaining={(total-finished)/speed:.1f}s', flush=True)
            last_report = now

    return measured()


class AdapterUnavailable(ContractError):
    """Known unsupported input/resource; the future runner should mark skipped."""
    def __init__(self, reason_code, message):
        self.reason_code = reason_code
        super().__init__(message)


def compatible(condition, reason_code, message):
    if not condition:
        raise AdapterUnavailable(reason_code, message)


def absolute_ref(ref, base):
    return replace(ref, path=str(resolved_path(ref.path, base)))


def close_array(array):
    if isinstance(array, np.memmap):
        array._mmap.close()


@contextmanager
def read_array(ref, base, *, root=None):
    path = verify_ref(ref, base, root=root)
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        ref.validate_array(array)
        yield array
    finally:
        close_array(array)


@dataclass(frozen=True)
class CubeView:
    prepared: PreparedCube  # absolute references to the original archive
    cube_ref: ArrayRef
    valid_ref: ArrayRef
    cube: np.ndarray
    valid_samples: np.ndarray
    good_channels: np.ndarray
    stage: str

    def __post_init__(self):
        require(self.stage in ('prepared', 'denoised'), 'Unknown cube processing stage')
        self.cube_ref.validate_array(self.cube)
        self.valid_ref.validate_array(self.valid_samples)
        self.prepared.validate_arrays(self.cube, self.valid_samples, self.good_channels)
        require(self.cube.flags.c_contiguous and self.valid_samples.flags.c_contiguous,
                'Prepared arrays must be C-contiguous for bounded-memory inference')
        if self.stage == 'prepared':
            require(self.cube_ref == self.prepared.cube and self.valid_ref == self.prepared.valid_samples,
                    'Original view must refer to its prepared artifacts')
        for array in (self.cube, self.valid_samples, self.good_channels):
            array.flags.writeable = False

    def require_ready(self):
        compatible(self.prepared.spectral_axis.metadata_status == 'verified',
                   'metadata_incomplete', 'Confirmed wavelengths/detectors/continuity are required')
        compatible(self.prepared.domain != 'unknown' and self.prepared.units != 'unknown',
                   'radiometry_unknown', 'Confirmed measurement domain and units are required')


@contextmanager
def open_prepared(root, *, verify_sources=True):
    root = Path(root).resolve()
    original = load_prepared_cube(root, verify_sources=verify_sources)
    prepared = replace(original, **{name: absolute_ref(getattr(original, name), root)
                                    for name in ('source', 'cube', 'valid_samples', 'good_channels')})
    with read_array(prepared.cube, root) as cube, read_array(prepared.valid_samples, root) as valid, \
            read_array(prepared.good_channels, root) as quality:
        yield CubeView(prepared, prepared.cube, prepared.valid_samples, cube, valid, quality, 'prepared')


@contextmanager
def open_denoised(original, root):
    require(original.stage == 'prepared', 'Denoised views must be linked to an original cube')
    root = Path(root).resolve()
    document = read_json(root / 'manifest.json')
    require(document.get('schema_version') == 1 and document.get('kind') == 'denoising'
            and document.get('status') == 'completed', 'Not a completed denoising result')
    result = decode_contract(DenoisingResult, document['result'])
    require(result.source == original.prepared and result.valid_samples == original.valid_ref,
            'Denoised result belongs to a different original cube or validity mask')
    verify_ref(result.bundle, root)
    with read_array(result.cube, root, root=root) as cube, \
            read_array(result.denoised_mask, root, root=root) as denoised_mask:
        result.validate_arrays(original.cube, original.valid_samples, cube, original.valid_samples,
                               denoised_mask, good_channels=original.good_channels)
        yield CubeView(original.prepared, absolute_ref(result.cube, root), original.valid_ref,
                       cube, original.valid_samples, original.good_channels, 'denoised')


def check_classifier_input(view, branch_id, classifier, taxonomy, resource_taxonomy):
    view.require_ready()
    require(branch_id in BRANCHES and BRANCHES[branch_id].classifier == classifier,
            'Branch ID does not match this classifier')
    require(BRANCHES[branch_id].input == view.stage, 'Branch ID does not match the actual cube processing stage')
    require(isinstance(taxonomy, Taxonomy), 'Expected canonical taxonomy')
    compatible(taxonomy == resource_taxonomy, 'taxonomy_mismatch', 'Resource and output taxonomies disagree')


def exact_axis(actual, expected):
    return (len(actual.channel_indices) == len(expected.channel_indices)
            and actual.detectors == expected.detectors
            and actual.allowed_edges == expected.allowed_edges
            and np.allclose(actual.wavelengths_nm, expected.wavelengths_nm, rtol=0, atol=1e-8))


@contextmanager
def output_directory(path, kind):
    """Exclusive directory: an adapter invocation never overwrites another result."""
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    document = {'schema_version': 1, 'kind': kind, 'status': 'running'}
    write_manifest(path / 'manifest.json', document)
    try:
        yield path
        require(read_json(path / 'manifest.json').get('status') == 'completed', 'Adapter did not finish its result')
    except BaseException as error:
        document.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_manifest(path / 'manifest.json', document)
        raise


def complete_result(path, kind, result, parameters):
    write_manifest(Path(path) / 'manifest.json', {'schema_version': 1, 'kind': kind, 'status': 'completed',
                   'result': contract_dict(result), 'parameters': parameters})


def save_classification(path, view, branch_id, resource, taxonomy, labels, reasons, used_channels, parameters):
    """Called inside output_directory; returns an immutable ClassificationResult."""
    path = Path(path)
    arrays = {'class_map': labels.astype(np.int32), 'prediction_valid_mask': labels >= 0,
              'rejection_reason': reasons.astype(np.uint8), 'used_channels': np.asarray(used_channels, dtype=bool)}
    for name, array in arrays.items():
        np.save(path / (name + '.npy'), array, allow_pickle=False)
    refs = {name: array_reference(path / (name + '.npy'), branch_id + '_' + name, path) for name in arrays}
    result = ClassificationResult(branch_id, view.cube_ref, resource, taxonomy, **refs)
    result.validate_arrays(arrays['class_map'], arrays['prediction_valid_mask'], arrays['rejection_reason'],
                           arrays['used_channels'], view.valid_samples)
    complete_result(path, 'classification', result, parameters)
    return result
