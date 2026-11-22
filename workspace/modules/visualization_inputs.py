"""Pinned comparison snapshots and maps; no models or inference dependencies."""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np

from modules.pipeline_contracts import (
    ArtifactRef, ArrayRef, ClassificationResult, ComparisonResult, PreparedCube,
    RejectionReason, Taxonomy, contract_dict, require,
)
from modules.resource_store import decode_contract, read_json, resolved_path, sha256_file, verify_ref
from modules.run_store import read_run_manifest


def _reference(document):
    return decode_contract(ArrayRef if 'shape' in document else ArtifactRef, document)


class VisualizationInputs:
    def __init__(self, analysis, stack):
        self.path = Path(analysis).resolve()
        self.stack, self.consumed = stack, {}
        self.analysis_ref = ArtifactRef('analysis_manifest', str(self.path / 'manifest.json'),
                                         sha256_file(self.path / 'manifest.json'))
        document = read_json(self.analysis_ref.path)
        require(document.get('schema_version') == 1 and document.get('kind') == 'whole_cube_comparison'
                and document.get('status') == 'completed', 'Expected a completed comparison revision')
        self.result = decode_contract(ComparisonResult, document['result'])
        require(self.result.analysis_id == self.path.name == document['analysis_id'], 'Analysis identity mismatch')
        self.run_ref = decode_contract(ArtifactRef, document['source_run'])
        self.run = resolved_path(self.run_ref.path, self.path).parent
        require(self.path.parent == self.run / 'analysis' and document['run_id'] == self.result.run_id == self.run.name,
                'Comparison belongs to another run')
        self.consume(self.run_ref, self.path)
        run_doc = read_run_manifest(self.run)
        require(run_doc['schema_version'] == 2 and run_doc['branches'] == {
            b.branch_id: contract_dict(b) for b in self.result.branches}, 'Comparison and inference outcomes differ')
        self.run_registry = {resolved_path(item['ref']['path'], self.run): _reference(item['ref'])
                             for item in run_doc['artifacts']}
        self.local = {}
        for raw in document['artifacts']:
            ref = _reference(raw)
            require(ref.artifact_id not in self.local, 'Duplicate analysis artifact ID')
            require(not any(ref.path == item.path for item in self.local.values()), 'Duplicate analysis artifact path')
            self.local[ref.artifact_id] = ref
        self.external = {}
        for raw in document['inputs']:
            ref = _reference(raw)
            path = resolved_path(ref.path, self.path)
            require(path not in self.external or self.external[path] == ref, 'Conflicting snapshot input references')
            self.external[path] = ref
        require(self.local_json(self.local['comparison']) == document['result'], 'Comparison contract differs from manifest')
        self.evaluation = self.local_json(self.result.evaluation_manifest)
        require(self.evaluation['taxonomy'] == contract_dict(self.result.taxonomy), 'Evaluation taxonomy mismatch')
        self.valid = self.local_array(_reference(self.evaluation['standard_roi']))
        require(self.valid.dtype.kind == 'b' and self.valid.ndim == 2, 'Standard ROI must be boolean H,W')
        self.coverage = self.local_json(self.result.coverage)
        self.agreement = self.local_json(self.result.agreement)
        self.gt_info = self.evaluation['ground_truth']
        self.metrics_gt = self.local_json(self.result.metrics_gt) if self.result.metrics_gt else None
        self.gt, self.gt_mask = None, None
        k = len(self.result.taxonomy.class_names)
        if self.result.ground_truth:
            self.local_json(self.result.ground_truth)
            self.gt = self.local_array(self.local['gt_labels'])
            self.gt_mask = self.local_array(_reference(self.gt_info['evaluation_mask']))
            require(self.gt.shape == self.valid.shape and self.gt.dtype.kind in 'iu'
                    and ((self.gt >= -1) & (self.gt < k)).all(), 'Invalid saved canonical GT')
            require(self.gt_mask.shape == self.valid.shape and self.gt_mask.dtype.kind == 'b'
                    and not (self.gt_mask & (~self.valid | (self.gt < 0))).any(), 'Invalid fixed GT evaluation area')
        self.outcomes = {b.branch_id: {'status': b.status, 'reason_code': b.reason_code} for b in self.result.branches}
        completed = {key for key, branch in self.outcomes.items() if branch['status'] == 'completed'}
        declared = self.evaluation.get('branch_taxonomies')
        if declared is None:
            self.taxonomies = {key: self.result.taxonomy for key in completed}
        else:
            require(isinstance(declared, dict) and set(declared) == completed, 'Incomplete branch taxonomy snapshot')
            self.taxonomies = {key: decode_contract(Taxonomy, value) for key, value in declared.items()}
        self.maps, self.reasons, self.classifications = {}, {}, {}
        require(set(self.coverage) == set(self.outcomes), 'Coverage must retain all four branches')
        for branch in self.result.branches:
            key = branch.branch_id
            require(self.coverage[key]['status'] == branch.status, 'Coverage/branch status mismatch')
            if branch.status != 'completed':
                continue
            result_path = self.external_path(branch.result, self.run)
            classification = read_json(result_path)
            require(classification.get('kind') == 'classification' and classification.get('status') == 'completed',
                    'Incomplete classification snapshot')
            result = decode_contract(ClassificationResult, classification['result'])
            require(result.branch_id == key and result.taxonomy == self.taxonomies[key], 'Map taxonomy/branch mismatch')
            values = [self.external_array(getattr(result, name), result_path.parent) for name in (
                'class_map', 'prediction_valid_mask', 'rejection_reason')]
            labels, predicted, reasons = values
            k = len(result.taxonomy.class_names)
            require(labels.shape == self.valid.shape and labels.dtype.kind in 'iu'
                    and ((labels >= -1) & (labels < k)).all(), 'Invalid class map')
            require(predicted.shape == labels.shape and np.array_equal(predicted, labels >= 0)
                    and not (predicted & ~self.valid).any(), 'Prediction mask or geometry mismatch')
            require(reasons.shape == labels.shape and np.isin(reasons, [int(r) for r in RejectionReason]).all()
                    and np.array_equal(reasons == 0, predicted), 'Invalid rejection reasons')
            self.maps[key], self.reasons[key], self.classifications[key] = labels, reasons, result
            self.recorded_cube(result.input_cube)  # Identity only; cube values are optional.

    def consume(self, ref, base, *, root=None):
        path = verify_ref(ref, base, root=root)
        self.consumed[str(path)] = replace(ref, path=str(path))
        return path

    def local_path(self, ref):
        require(ref is not None and self.local.get(ref.artifact_id) == ref, 'Unregistered comparison artifact')
        return self.consume(ref, self.path, root=self.path)

    def local_json(self, ref):
        return read_json(self.local_path(ref))

    def array(self, path, ref):
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        require(isinstance(array, np.memmap), 'Expected an NPY array')
        self.stack.callback(array._mmap.close)
        ref.validate_array(array)
        return array

    def local_array(self, ref):
        return self.array(self.local_path(ref), ref)

    def external_path(self, ref, base):
        path = resolved_path(ref.path, base)
        recorded = self.external.get(path)
        require(recorded is not None and recorded.sha256 == ref.sha256, 'Map is not pinned by the selected analysis')
        if isinstance(ref, ArrayRef):
            require(isinstance(recorded, ArrayRef) and recorded.shape == ref.shape and recorded.dtype == ref.dtype,
                    'Array metadata differs from the selected analysis')
        return self.consume(ref, base)

    def external_array(self, ref, base):
        return self.array(self.external_path(ref, base), ref)

    def recorded_cube(self, ref):
        path = resolved_path(ref.path, self.run)
        recorded = self.run_registry.get(path)
        require(isinstance(recorded, ArrayRef) and recorded.sha256 == ref.sha256
                and recorded.shape == ref.shape and recorded.dtype == ref.dtype, 'Cube identity differs from inference registry')
        return path

    def dominant_spectrum(self, key):
        """Stream the actual branch input. Called only when spectra were requested."""
        result = self.classifications[key]
        labels = self.maps[key]
        ids, counts = np.unique(labels[labels >= 0], return_counts=True)
        require(len(ids) > 0, 'No classified pixels for a mean spectrum')
        dominant = int(ids[np.argmax(counts)])
        prepared_ref = next(ref for ref in self.external.values() if ref.artifact_id == 'prepared_manifest')
        prepared_path = self.external_path(prepared_ref, self.path)
        prepared = decode_contract(PreparedCube, read_json(prepared_path)['prepared'])
        cube_ref = result.input_cube
        path = self.recorded_cube(cube_ref)
        # Close the optional full-cube mappings after each branch, not at the end of rendering.
        before = dict(self.consumed)
        try:
            self.consume(cube_ref, self.run)
            mask_path = self.external_path(prepared.valid_samples, prepared_path.parent)
            with ExitStack() as mappings:
                cube = np.load(path, mmap_mode='r', allow_pickle=False)
                mappings.callback(cube._mmap.close)
                cube_ref.validate_array(cube)
                mask = np.load(mask_path, mmap_mode='r', allow_pickle=False)
                mappings.callback(mask._mmap.close)
                prepared.valid_samples.validate_array(mask)
                require(cube.shape == prepared.source_shape and labels.shape == cube.shape[:2], 'Spectrum/map geometry mismatch')
                totals, support = np.zeros(cube.shape[2]), np.zeros(cube.shape[2], dtype=np.int64)
                for row in range(0, cube.shape[0], 16):
                    block = cube[row:row + 16]
                    selected = mask[row:row + 16] & (labels[row:row + 16] == dominant)[..., None] & np.isfinite(block)
                    totals += np.where(selected, block, 0).sum(axis=(0, 1), dtype=np.float64)
                    support += selected.sum(axis=(0, 1))
            means = np.divide(totals, support, out=np.full(totals.shape, np.nan), where=support > 0)
            axis = prepared.spectral_axis
            verified = axis.metadata_status == 'verified'
            return {'status': 'completed', 'class_id': dominant, 'class_name': result.taxonomy.class_names[dominant],
                    'taxonomy': contract_dict(result.taxonomy), 'source_cube': contract_dict(cube_ref),
                    'units': prepared.units, 'domain': prepared.domain,
                    'x': list(axis.wavelengths_nm if verified else axis.channel_indices),
                    'x_label': 'Длина волны, нм' if verified else 'Индекс канала (метаданные не подтверждены)',
                    'allowed_edges': list(axis.allowed_edges) if verified else [False] * (cube.shape[2] - 1),
                    'mean': [float(v) if np.isfinite(v) else None for v in means], 'support': support.tolist()}
        except Exception:
            self.consumed = before  # Optional spectra errors must not discard the valid maps/report.
            raise

    def verify_unchanged(self):
        for ref in (self.analysis_ref, *self.consumed.values()):
            verify_ref(ref, self.path)


@contextmanager
def open_visualization_inputs(analysis):
    with ExitStack() as stack:
        yield VisualizationInputs(analysis, stack)
