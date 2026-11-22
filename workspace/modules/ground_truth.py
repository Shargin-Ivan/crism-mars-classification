"""Explicit scene/label/taxonomy bindings for evaluation only; no model imports."""
from dataclasses import dataclass, replace

import numpy as np

from modules.pipeline_contracts import (
    ArtifactRef, ArrayRef, Taxonomy, integer, map_ground_truth, require, sequence, text,
)
from modules.resource_store import decode_contract, read_json, resolved_path, sha256_file, verify_ref


@dataclass(frozen=True)
class GroundTruthManifest:
    schema_version: int
    scene_id: str
    source_cube: ArtifactRef
    taxonomy: Taxonomy
    labels: ArtifactRef
    format: str
    variable: str | None
    unlabeled_id: int
    source_to_class: tuple[tuple[int, int], ...]
    roi: ArrayRef | None
    held_out_scene_id: str
    development_scene_ids: tuple[str, ...]

    def __post_init__(self):
        require(type(self.schema_version) is int and self.schema_version == 1, 'Unknown GT schema')
        text(self.scene_id, 'GT scene_id')
        require(isinstance(self.source_cube, ArtifactRef) and isinstance(self.labels, ArtifactRef), 'GT needs pinned sources')
        require(isinstance(self.taxonomy, Taxonomy), 'GT taxonomy is required')
        require(self.format in ('npy', 'mat'), 'GT supports NPY or MATLAB v5/v7')
        if self.format == 'mat':
            text(self.variable, 'GT MAT variable')
        else:
            require(self.variable is None, 'NPY GT has no variable name')
        integer(self.unlabeled_id, 'unlabeled_id')
        mapping = tuple(sequence(pair, 'GT mapping pair') for pair in sequence(self.source_to_class, 'GT mapping'))
        require(all(len(pair) == 2 for pair in mapping), 'GT mapping pairs need source and canonical ID')
        require(len({pair[0] for pair in mapping}) == len(mapping), 'Duplicate GT source ID')
        for source, target in mapping:
            integer(source, 'source GT ID')
            require(source != self.unlabeled_id, 'Unlabelled ID must not become a class')
            self.taxonomy.validate_ids((target,))
        object.__setattr__(self, 'source_to_class', mapping)
        require(self.roi is None or isinstance(self.roi, ArrayRef), 'ROI needs a pinned boolean NPY')
        text(self.held_out_scene_id, 'held_out_scene_id')
        development = sequence(self.development_scene_ids, 'development_scene_ids')
        for scene_id in development:
            text(scene_id, 'development scene ID')
        require(bool(development) and len(set(development)) == len(development)
                and self.held_out_scene_id not in development, 'Development/held-out scenes must be disjoint')
        require(self.scene_id in (*development, self.held_out_scene_id), 'GT scene is absent from the declared split')
        object.__setattr__(self, 'development_scene_ids', development)

    def absolute(self, base):
        return replace(self, **{key: replace(getattr(self, key), path=str(resolved_path(getattr(self, key).path, base)))
                                for key in ('source_cube', 'labels', 'roi') if getattr(self, key) is not None})


def load_ground_truth(ref, base, scene, taxonomy):
    """Return descriptor, canonical labels, spatial ROI and references actually read.

    Orientation is the source cube's H,W order: never transpose/crop/resize GT.
    MATLAB floating labels are accepted only when all values are finite integers.
    """
    path = verify_ref(ref, base)
    spec = decode_contract(GroundTruthManifest, read_json(path)).absolute(path.parent)
    require(sha256_file(path) == ref.sha256, 'GT descriptor changed while reading')
    require(spec.scene_id == scene.scene_id and spec.taxonomy == taxonomy, 'GT scene/taxonomy mismatch')
    require(spec.source_cube.sha256 == scene.cube.sha256
            and resolved_path(spec.source_cube.path, path.parent) == resolved_path(scene.cube.path, base),
            'GT belongs to another source cube')
    if scene.ground_truth is not None:
        require(spec.labels.sha256 == scene.ground_truth.sha256
                and resolved_path(spec.labels.path, path.parent) == resolved_path(scene.ground_truth.path, base)
                and spec.variable == scene.ground_truth_variable, 'GT disagrees with the scene annotation binding')
    labels_path = verify_ref(spec.labels, path.parent)
    if spec.format == 'mat':
        from scipy.io import loadmat
        from scipy.io.matlab import MatReadError
        try:
            document = loadmat(labels_path, variable_names=[spec.variable])
        except (MatReadError, NotImplementedError, EOFError) as error:
            raise ValueError(f'Cannot read GT MAT: {error}') from error
        require(spec.variable in document, 'GT MAT variable is missing')
        values = document[spec.variable]
    else:
        try:
            values = np.load(labels_path, allow_pickle=False)
        except EOFError as error:
            raise ValueError('Empty or truncated GT NPY') from error
    require(isinstance(values, np.ndarray) and values.shape == scene.shape[:2]
            and values.dtype.kind in 'iuf', 'GT must have the unchanged H,W geometry and numeric IDs')
    require(np.isfinite(values).all() and (values >= 0).all() and (values < 2 ** 63).all()
            and np.equal(values, np.floor(values)).all(), 'GT IDs must be finite nonnegative integers')
    labels, _ = map_ground_truth(values.astype(np.int64), dict(spec.source_to_class), taxonomy,
                                unlabeled_id=spec.unlabeled_id)
    require(sha256_file(labels_path) == spec.labels.sha256, 'GT labels changed while reading')
    roi = np.ones(scene.shape[:2], dtype=bool)
    refs = [replace(ref, path=str(path)), spec.labels]
    if spec.roi is not None:
        roi_path = verify_ref(spec.roi, path.parent)
        roi = np.load(roi_path, allow_pickle=False)
        spec.roi.validate_array(roi)
        require(roi.shape == scene.shape[:2] and roi.dtype == np.dtype(bool), 'GT ROI must be boolean H,W')
        require(sha256_file(roi_path) == spec.roi.sha256, 'GT ROI changed while reading')
        refs.append(spec.roi)
    return spec, labels, roi, refs
