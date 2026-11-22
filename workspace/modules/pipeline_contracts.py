"""Contracts for the new whole-cube pipeline; no model or filesystem access.

Artifacts describe immutable files. Array validation is explicit and never
loads a file, partitions a cube, changes an array, or starts inference.
"""
from dataclasses import dataclass, fields, is_dataclass
from enum import IntEnum
from math import isfinite
from numbers import Real
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np


class ContractError(ValueError):
    """Invalid data, metadata or configuration at a block boundary."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), f'{name}: expected nonempty text')
    require(not any(c in value for c in ('<', '>', '\x00'))
            and value.strip().lower() not in {'todo', 'tbd', 'placeholder'},
            f'{name}: placeholder or invalid text')
    return value


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f'{name}: expected integer >= {minimum}')
    return value


def number(value, name, minimum, maximum):
    require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            and isfinite(value) and minimum <= value <= maximum,
            f'{name}: expected finite number in [{minimum}, {maximum}]')
    return value


def sequence(value, name):
    require(isinstance(value, (list, tuple)), f'{name}: expected list or tuple')
    return tuple(value)


def contract_dict(value):
    """Return fresh JSON-compatible metadata; never serialize cube contents."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: contract_dict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: contract_dict(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [contract_dict(v) for v in value]
    if isinstance(value, IntEnum):
        return int(value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ContractError(f'Unsupported metadata type: {type(value).__name__}')


@dataclass(frozen=True)
class BranchSpec:
    input: str
    classifier: str

    def __post_init__(self):
        require(self.input in ('prepared', 'denoised'), 'Unknown branch input')
        require(self.classifier in ('sam', 'lwnet'), 'Unknown classifier')


BRANCHES = MappingProxyType({
    'sam_original': BranchSpec('prepared', 'sam'),
    'sam_denoised': BranchSpec('denoised', 'sam'),
    'lwnet_original': BranchSpec('prepared', 'lwnet'),
    'lwnet_denoised': BranchSpec('denoised', 'lwnet'),
})


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    path: str
    sha256: str

    def __post_init__(self):
        text(self.artifact_id, 'artifact_id')
        text(self.path, 'path')
        require(isinstance(self.sha256, str) and re.fullmatch(r'[0-9a-f]{64}', self.sha256),
                'sha256: expected 64 lowercase hexadecimal characters')


@dataclass(frozen=True)
class ArrayRef(ArtifactRef):
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self):
        super().__post_init__()
        shape = sequence(self.shape, 'shape')
        require(bool(shape), 'shape must not be empty')
        for size in shape:
            integer(size, 'shape dimension', 1)
        require(isinstance(self.dtype, str), 'dtype must be text')
        try:
            dtype = np.dtype(self.dtype)
        except (TypeError, ValueError) as error:
            raise ContractError('Invalid array dtype') from error
        require(dtype.kind in 'buif', 'Only boolean, integer and real floating arrays are supported')
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'dtype', dtype.str)

    def validate_array(self, array):
        require(isinstance(array, np.ndarray), f'{self.artifact_id}: expected ndarray')
        require(array.shape == self.shape, f'{self.artifact_id}: array shape mismatch')
        require(array.dtype == np.dtype(self.dtype), f'{self.artifact_id}: array dtype mismatch')


def array_layout(ref, shape, kinds, name):
    require(isinstance(ref, ArrayRef), f'{name}: expected ArrayRef')
    require(ref.shape == shape and np.dtype(ref.dtype).kind in kinds,
            f'{name}: incompatible shape or dtype')


@dataclass(frozen=True)
class SpectralAxis:
    """Annotations on the original channel axis, not separate S/L subcubes.

    channel_indices refer to the supplied source file, not guessed PDS band IDs.
    Unknown physical metadata stay None. Segment boundaries prohibit crossing
    detector boundaries and missing bands without changing the channel order.
    """
    channel_indices: tuple[int, ...]
    wavelengths_nm: tuple[float, ...] | None
    detectors: tuple[str, ...] | None
    segment_ids: tuple[int, ...] | None
    metadata_status: str = 'verified'
    unit: str = 'nm'

    def __post_init__(self):
        indices = sequence(self.channel_indices, 'channel_indices')
        require(bool(indices), 'Empty spectral axis')
        for index in indices:
            integer(index, 'channel index')
        require(len(set(indices)) == len(indices), 'Duplicate source channel indices')
        object.__setattr__(self, 'channel_indices', indices)
        require(self.unit == 'nm', 'Wavelength units must be nm; no implicit conversion')
        require(self.metadata_status in ('verified', 'incomplete'), 'Unknown metadata status')
        for name in ('wavelengths_nm', 'detectors', 'segment_ids'):
            value = getattr(self, name)
            if value is not None:
                value = sequence(value, name)
                require(len(value) == len(indices), f'{name}: channel count mismatch')
                object.__setattr__(self, name, value)
            elif self.metadata_status == 'verified':
                raise ContractError(f'{name}: required for verified metadata')
        if self.wavelengths_nm is not None:
            for wl in self.wavelengths_nm:
                number(wl, 'wavelength_nm', 200, 5000)
            object.__setattr__(self, 'wavelengths_nm', tuple(float(wl) for wl in self.wavelengths_nm))
        if self.detectors is not None:
            require(all(d in ('S', 'L') for d in self.detectors), 'Unknown detector ID')
        if self.segment_ids is not None:
            seen = set()
            for i, segment in enumerate(self.segment_ids):
                integer(segment, 'segment_id')
                if i == 0 or segment != self.segment_ids[i - 1]:
                    require(segment not in seen, 'A segment must occupy contiguous indices')
                    seen.add(segment)
                else:
                    if self.detectors is not None:
                        require(self.detectors[i] == self.detectors[i - 1],
                                'A segment cannot cross the S/L boundary')
                    if self.wavelengths_nm is not None:
                        step = self.wavelengths_nm[i] - self.wavelengths_nm[i - 1]
                        require(1e-8 < step <= 7.5, 'Unordered or missing channels within a segment')

    def require_verified(self):
        require(self.metadata_status == 'verified', 'Spectral metadata are incomplete')

    @property
    def allowed_edges(self):
        self.require_verified()
        return tuple(a == b for a, b in zip(self.segment_ids, self.segment_ids[1:]))


@dataclass(frozen=True)
class Taxonomy:
    taxonomy_id: str
    version: str
    class_names: tuple[str, ...]

    def __post_init__(self):
        text(self.taxonomy_id, 'taxonomy_id')
        text(self.version, 'taxonomy version')
        names = sequence(self.class_names, 'class_names')
        require(bool(names), 'Taxonomy must contain classes')
        for name in names:
            text(name, 'class name')
        require(len(set(names)) == len(names), 'Duplicate class names')
        object.__setattr__(self, 'class_names', names)

    def validate_ids(self, values):
        for value in values:
            integer(value, 'canonical class ID')
            require(value < len(self.class_names), 'Class ID is outside the taxonomy')


def map_ground_truth(labels, source_to_class, taxonomy, *, unlabeled_id=0):
    """Map local IDs explicitly; unlabelled pixels become -1, not background."""
    require(isinstance(labels, np.ndarray) and labels.ndim == 2
            and labels.dtype.kind in 'iu', 'GT must be a two-dimensional integer array')
    require(isinstance(taxonomy, Taxonomy), 'Expected Taxonomy')
    require(isinstance(source_to_class, Mapping), 'Expected GT class mapping')
    integer(unlabeled_id, 'unlabeled_id')
    for source, target in source_to_class.items():
        integer(source, 'source GT ID')
        require(source != unlabeled_id, 'The unlabelled ID cannot map to a class')
        taxonomy.validate_ids((target,))
    unknown = set(np.unique(labels).tolist()) - {unlabeled_id} - set(source_to_class)
    require(not unknown, f'Unknown GT IDs: {sorted(unknown)}')
    mapped = np.full(labels.shape, -1, dtype=np.int32)
    for source, target in source_to_class.items():
        mapped[labels == source] = target
    return mapped, labels != unlabeled_id


@dataclass(frozen=True)
class PreparedCube:
    scene_id: str
    source: ArtifactRef
    source_shape: tuple[int, int, int]
    cube: ArrayRef
    valid_samples: ArrayRef
    good_channels: ArrayRef
    spectral_axis: SpectralAxis
    units: str
    domain: str

    def __post_init__(self):
        text(self.scene_id, 'scene_id')
        require(isinstance(self.source, ArtifactRef), 'Expected source manifest reference')
        shape = sequence(self.source_shape, 'source_shape')
        require(len(shape) == 3, 'Source must have H,W,C axes')
        for size in shape:
            integer(size, 'source_shape dimension', 1)
        object.__setattr__(self, 'source_shape', shape)
        array_layout(self.cube, shape, 'f', 'whole cube')
        array_layout(self.valid_samples, shape, 'b', 'valid_samples')
        array_layout(self.good_channels, (shape[2],), 'b', 'good_channels')
        require(isinstance(self.spectral_axis, SpectralAxis), 'Expected SpectralAxis')
        require(self.spectral_axis.channel_indices == tuple(range(shape[2])),
                'Whole-cube processing requires every source channel in original order')
        text(self.units, 'units')
        text(self.domain, 'radiometric domain')

    def validate_arrays(self, cube, valid_samples, good_channels):
        self.cube.validate_array(cube)
        self.valid_samples.validate_array(valid_samples)
        self.good_channels.validate_array(good_channels)
        # Bounded auxiliary memory even when checking a whole scene.
        for row, valid in zip(cube, valid_samples):
            require(not np.any(valid & ~np.isfinite(row)), 'Nonfinite measurement marked valid')
            require(not np.any(valid[:, ~good_channels]), 'Excluded channels cannot contain valid measurements')

    def require_ready(self):
        self.spectral_axis.require_verified()
        require(self.domain != 'unknown' and self.units != 'unknown', 'Unknown radiometric metadata')


@dataclass(frozen=True)
class ModelBundle:
    bundle_id: str
    algorithm: str
    architecture_version: str
    weights: ArtifactRef
    preprocessing: ArtifactRef
    training_manifest: ArtifactRef
    domain: str
    input_layout: str
    spectral_axis: SpectralAxis | None
    taxonomy: Taxonomy | None = None
    output_class_ids: tuple[int, ...] = ()

    def __post_init__(self):
        for name in ('bundle_id', 'architecture_version', 'domain'):
            text(getattr(self, name), name)
        require(self.domain != 'unknown', 'Model input domain must be known')
        require(self.algorithm in ('n2n4m', 'lwnet'), 'Unknown model algorithm')
        for name in ('weights', 'preprocessing', 'training_manifest'):
            require(isinstance(getattr(self, name), ArtifactRef), f'{name}: required artifact reference')
        require(self.input_layout in ('fixed_grid', 'segmented'), 'Unknown model input layout')
        if self.input_layout == 'fixed_grid':
            require(isinstance(self.spectral_axis, SpectralAxis), 'Fixed grid requires its spectral axis')
        if self.spectral_axis is not None:
            require(isinstance(self.spectral_axis, SpectralAxis), 'Expected model SpectralAxis')
            self.spectral_axis.require_verified()
        ids = sequence(self.output_class_ids, 'output_class_ids')
        object.__setattr__(self, 'output_class_ids', ids)
        if self.algorithm == 'lwnet':
            require(isinstance(self.taxonomy, Taxonomy) and bool(ids), 'LWNet requires its class mapping')
            self.taxonomy.validate_ids(ids)
            require(len(set(ids)) == len(ids), 'LWNet output class mapping must be one-to-one')
        else:
            require(self.taxonomy is None and not ids, 'N2N4M predicts spectra, not class IDs')


@dataclass(frozen=True)
class ReferenceLibrary:
    library_id: str
    spectra: ArrayRef
    valid_samples: ArrayRef
    spectral_axis: SpectralAxis
    taxonomy: Taxonomy
    class_ids: tuple[int, ...]
    units: str
    domain: str
    aggregation: str = 'minimum_angle'

    def __post_init__(self):
        text(self.library_id, 'library_id')
        text(self.units, 'reference units')
        text(self.domain, 'reference domain')
        require(self.units != 'unknown' and self.domain != 'unknown', 'Unknown reference radiometry')
        require(isinstance(self.spectra, ArrayRef) and len(self.spectra.shape) == 2,
                'Reference spectra must have N,C axes')
        array_layout(self.spectra, self.spectra.shape, 'f', 'reference spectra')
        array_layout(self.valid_samples, self.spectra.shape, 'b', 'reference mask')
        require(isinstance(self.spectral_axis, SpectralAxis), 'Expected reference SpectralAxis')
        self.spectral_axis.require_verified()
        require(len(self.spectral_axis.channel_indices) == self.spectra.shape[1], 'Reference grid mismatch')
        require(isinstance(self.taxonomy, Taxonomy), 'Expected reference Taxonomy')
        ids = sequence(self.class_ids, 'reference class_ids')
        require(len(ids) == self.spectra.shape[0], 'Each reference needs a class mapping')
        self.taxonomy.validate_ids(ids)
        object.__setattr__(self, 'class_ids', ids)
        require(self.aggregation == 'minimum_angle', 'Unsupported reference aggregation')


@dataclass(frozen=True)
class DenoisingResult:
    source: PreparedCube
    bundle: ArtifactRef
    cube: ArrayRef
    valid_samples: ArrayRef
    denoised_mask: ArrayRef

    def __post_init__(self):
        require(isinstance(self.source, PreparedCube), 'Expected original PreparedCube')
        require(isinstance(self.bundle, ArtifactRef), 'Expected frozen denoiser bundle reference')
        shape = self.source.cube.shape
        array_layout(self.cube, shape, 'f', 'denoised cube')
        require(self.cube.dtype == self.source.cube.dtype, 'Denoising must preserve the prepared dtype')
        array_layout(self.valid_samples, shape, 'b', 'denoised validity')
        array_layout(self.denoised_mask, shape, 'b', 'denoised_mask')

    def validate_arrays(self, original, original_valid, output, output_valid, denoised_mask, *, good_channels):
        self.source.validate_arrays(original, original_valid, good_channels)
        for ref, array in ((self.cube, output), (self.valid_samples, output_valid),
                           (self.denoised_mask, denoised_mask)):
            ref.validate_array(array)
        require(denoised_mask.any(), 'No denoised support; report a skipped denoising stage instead')
        for before, valid, after, after_valid, changed in zip(
                original, original_valid, output, output_valid, denoised_mask):
            require(np.array_equal(valid, after_valid), 'Denoising must preserve measurement validity')
            require(not np.any(changed & ~valid), 'Missing samples cannot be declared denoised')
            require(not np.any(after_valid & ~np.isfinite(after)), 'Nonfinite denoised measurement')
            require(np.array_equal(before[~changed], after[~changed], equal_nan=True),
                    'Unprocessed samples must remain unchanged')


class RejectionReason(IntEnum):
    NONE = 0
    EMPTY_PIXEL = 1
    INSUFFICIENT_COVERAGE = 2
    PATCH_BORDER = 3
    THRESHOLD_REJECTION = 4
    MODEL_ABSTENTION = 5


@dataclass(frozen=True)
class ClassificationResult:
    branch_id: str
    input_cube: ArrayRef
    resource: ArtifactRef
    taxonomy: Taxonomy
    class_map: ArrayRef
    prediction_valid_mask: ArrayRef
    rejection_reason: ArrayRef
    used_channels: ArrayRef

    def __post_init__(self):
        require(self.branch_id in BRANCHES, 'Unknown classification branch')
        require(isinstance(self.input_cube, ArrayRef) and len(self.input_cube.shape) == 3,
                'Expected H,W,C input cube')
        require(isinstance(self.resource, ArtifactRef), 'Expected classifier resource reference')
        require(isinstance(self.taxonomy, Taxonomy), 'Expected output Taxonomy')
        shape = self.input_cube.shape[:2]
        array_layout(self.class_map, shape, 'i', 'class map with -1 for abstentions')
        array_layout(self.prediction_valid_mask, shape, 'b', 'prediction mask')
        array_layout(self.rejection_reason, shape, 'iu', 'rejection reasons')
        array_layout(self.used_channels, (self.input_cube.shape[2],), 'b', 'used_channels')

    def validate_arrays(self, class_map, predicted, reasons, used_channels, input_valid):
        for ref, array in ((self.class_map, class_map), (self.prediction_valid_mask, predicted),
                           (self.rejection_reason, reasons), (self.used_channels, used_channels)):
            ref.validate_array(array)
        require(isinstance(input_valid, np.ndarray) and input_valid.dtype == bool
                and input_valid.shape == self.input_cube.shape, 'Invalid input measurement mask')
        for labels, accepted, why, measured in zip(class_map, predicted, reasons, input_valid):
            require(np.all((labels >= -1) & (labels < len(self.taxonomy.class_names))), 'Unknown output class')
            require(np.array_equal(accepted, labels >= 0), 'Map and prediction mask disagree')
            available = (measured & used_channels).any(axis=1)
            require(not np.any(accepted & ~available), 'Empty or unsupported pixels cannot receive classes')
            require(np.isin(why, [int(reason) for reason in RejectionReason]).all(), 'Unknown rejection reason')
            require(np.array_equal(why == 0, accepted), 'Rejection reasons and prediction mask disagree')


@dataclass(frozen=True)
class BranchOutcome:
    branch_id: str
    status: str
    result: ArtifactRef | None = None
    reason_code: str | None = None

    def __post_init__(self):
        require(self.branch_id in BRANCHES, 'Unknown branch outcome')
        require(self.status in ('pending', 'running', 'completed', 'skipped', 'failed'), 'Unknown branch status')
        if self.status == 'completed':
            require(isinstance(self.result, ArtifactRef) and self.reason_code is None,
                    'Completed branch needs a result and no failure reason')
        else:
            require(self.result is None, 'Only completed branches have result artifacts')
            if self.status in ('skipped', 'failed'):
                text(self.reason_code, 'reason_code')
            else:
                require(self.reason_code is None, 'Pending/running branches have no terminal reason')


@dataclass(frozen=True)
class ComparisonResult:
    run_id: str
    analysis_id: str
    taxonomy: Taxonomy
    evaluation_manifest: ArtifactRef
    branches: tuple[BranchOutcome, ...]
    coverage: ArtifactRef
    agreement: ArtifactRef | None = None
    ground_truth: ArtifactRef | None = None
    metrics_gt: ArtifactRef | None = None
    not_evaluable_reason: str | None = None

    def __post_init__(self):
        text(self.run_id, 'run_id')
        text(self.analysis_id, 'analysis_id')
        require(isinstance(self.taxonomy, Taxonomy), 'Expected comparison Taxonomy')
        require(isinstance(self.evaluation_manifest, ArtifactRef), 'Expected evaluation protocol/ROI reference')
        branches = sequence(self.branches, 'branches')
        require(all(isinstance(b, BranchOutcome) for b in branches), 'Expected BranchOutcome entries')
        require(len(branches) == 4 and {b.branch_id for b in branches} == set(BRANCHES),
                'Comparison must retain all four branches')
        require(all(b.status in ('completed', 'skipped', 'failed') for b in branches),
                'Comparison requires terminal branch states')
        object.__setattr__(self, 'branches', branches)
        require(isinstance(self.coverage, ArtifactRef), 'Coverage report is required')
        for name in ('agreement', 'ground_truth', 'metrics_gt'):
            require(getattr(self, name) is None or isinstance(getattr(self, name), ArtifactRef),
                    f'{name}: expected artifact reference or None')
        if self.metrics_gt is not None:
            require(self.ground_truth is not None and self.not_evaluable_reason is None,
                    'GT metrics require ground truth')
        else:
            text(self.not_evaluable_reason, 'not_evaluable_reason')


@dataclass(frozen=True)
class VisualizationResult:
    run_id: str
    analysis_id: str
    artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self):
        text(self.run_id, 'run_id')
        text(self.analysis_id, 'analysis_id')
        artifacts = sequence(self.artifacts, 'visualization artifacts')
        require(bool(artifacts) and all(isinstance(a, ArtifactRef) for a in artifacts),
                'Visualization requires artifact references')
        require(len({a.artifact_id for a in artifacts}) == len(artifacts), 'Duplicate visualization artifact IDs')
        object.__setattr__(self, 'artifacts', artifacts)
