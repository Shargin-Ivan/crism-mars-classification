"""Read only the artifacts consumed by analysis; never load cube values or weights."""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np

from modules.data_preparation import SceneManifest
from modules.pipeline_config import config_from_dict
from modules.pipeline_contracts import (
    ArtifactRef, ArrayRef, BRANCHES, BranchOutcome, ClassificationResult,
    DenoisingResult, ModelBundle, PreparedCube, Taxonomy, require,
)
from modules.resource_store import decode_contract, read_json, resolved_path, sha256_file, verify_ref
from modules.run_store import read_run_manifest


class AnalysisInputs:
    def __init__(self, run, stack):
        self.run = Path(run).resolve()
        self.stack = stack
        self.inputs = {}
        self.manifest_ref = ArtifactRef('inference_manifest', str(self.run / 'manifest.json'),
                                        sha256_file(self.run / 'manifest.json'))
        self.manifest = read_run_manifest(self.run)
        require(self.manifest['schema_version'] == 2 and self.manifest['status'] in ('completed', 'partial'),
                'Analysis requires a completed or explicitly selected partial v2 run')
        self.registry = {entry['ref']['artifact_id']: decode_contract(
            ArrayRef if 'shape' in entry['ref'] else ArtifactRef, entry['ref']) for entry in self.manifest['artifacts']}
        self.outcomes = tuple(decode_contract(BranchOutcome, self.manifest['branches'][key]) for key in BRANCHES)
        self.config = config_from_dict(self.json(self.registry['configuration'], self.run))
        if self.config.configuration_state == 'draft':
            execution = self.json(self.registry['execution'], self.run)
            require(execution.get('requested_branches') == ['sam_original'] and all(
                outcome.status == 'completed' if outcome.branch_id == 'sam_original'
                else outcome.status == 'skipped' and outcome.reason_code == 'not_requested'
                for outcome in self.outcomes), 'Draft analysis requires an explicit completed SAM baseline')
            self.config.require_sam_baseline()
        else:
            self.config.require_ready()
        self.taxonomy = decode_contract(Taxonomy, self.json(self.config.taxonomy, self.run))
        archive_ref = self.registry['prepared_manifest']
        archive = resolved_path(archive_ref.path, self.run).parent
        archive_doc = self.json(archive_ref, self.run)
        require(archive_doc.get('status') == 'completed' and archive_doc.get('kind') == 'prepared_cube',
                'Incomplete prepared archive')
        original = decode_contract(PreparedCube, archive_doc['prepared'])
        self.prepared = replace(original, **{name: replace(getattr(original, name), path=str(
            resolved_path(getattr(original, name).path, archive)))
            for name in ('source', 'cube', 'valid_samples', 'good_channels')})
        self.scene = decode_contract(SceneManifest, self.json(self.prepared.source, archive)).absolute(archive)
        require(self.scene.scene_id == self.prepared.scene_id and self.scene.shape == self.prepared.source_shape
                and self.scene.spectral_axis == self.prepared.spectral_axis, 'Prepared/source geometry mismatch')
        self.measured = self.array(self.prepared.valid_samples, archive)
        quality = self.array(self.prepared.good_channels, archive)
        for row in self.measured:
            require(not np.any(row & ~quality), 'Invalid prepared quality mask')
        self.valid_pixels = self.measured.any(axis=2)
        self.maps, self.reasons, self.taxonomies = {}, {}, {}
        expected_denoised = None
        for outcome in self.outcomes:
            if outcome.status != 'completed':
                continue
            document = self.json(outcome.result, self.run)
            require(document.get('kind') == 'classification' and document.get('status') == 'completed',
                    'Branch is not a completed classification')
            result = decode_contract(ClassificationResult, document['result'])
            key = outcome.branch_id
            resource = self.config.resources.sam_library if key.startswith('sam') else self.config.resources.lwnet_bundle
            expected_taxonomy = self.taxonomy
            if result.taxonomy != expected_taxonomy and key.startswith('lwnet'):
                bundle = decode_contract(ModelBundle, self.json(resource, self.run))
                require(bundle.architecture_version == 'LWNet_3_Salinas_group4_v1', 'Unexpected native-class transfer')
                expected_taxonomy = bundle.taxonomy
            require(result.branch_id == key and result.taxonomy == expected_taxonomy and result.resource == resource,
                    'Branch identity/taxonomy/resource mismatch')
            self.taxonomies[key] = result.taxonomy
            if BRANCHES[key].input == 'prepared':
                expected = self.prepared.cube
            else:
                if expected_denoised is None:
                    ref = self.registry['n2n4m_result']
                    denoising = self.json(ref, self.run)
                    require(denoising.get('kind') == 'denoising' and denoising.get('status') == 'completed',
                            'Denoising dependency is incomplete')
                    clean = decode_contract(DenoisingResult, denoising['result'])
                    require(clean.source == self.prepared and clean.bundle == self.config.resources.n2n4m_bundle
                            and clean.valid_samples == self.prepared.valid_samples, 'Denoising belongs to another source')
                    expected_denoised = replace(clean.cube, path=str(resolved_path(
                        clean.cube.path, resolved_path(ref.path, self.run).parent)))
                expected = expected_denoised
            require(result.input_cube == expected, 'Classification uses another cube version')
            self.registered(expected, self.run)  # Compare recorded identity; do not read the spectral cube.
            base = resolved_path(outcome.result.path, self.run).parent
            arrays = [self.array(getattr(result, name), base) for name in (
                'class_map', 'prediction_valid_mask', 'rejection_reason', 'used_channels')]
            result.validate_arrays(*arrays, self.measured)
            require(not np.any(arrays[3] & ~quality), 'Classification claims excluded spectral channels')
            self.maps[key], self.reasons[key] = arrays[0], arrays[2]

    def registered(self, ref, base):
        path = resolved_path(ref.path, base)
        matches = [value for value in self.registry.values() if resolved_path(value.path, self.run) == path]
        require(len(matches) == 1 and matches[0].sha256 == ref.sha256, 'Unregistered or mismatched analysis input')
        if isinstance(ref, ArrayRef):
            require(isinstance(matches[0], ArrayRef) and matches[0].shape == ref.shape
                    and matches[0].dtype == ref.dtype, 'Array metadata differs from the artifact registry')
        return replace(matches[0], path=str(path))

    def verified(self, ref, base):
        ref = self.registered(ref, base)
        path = verify_ref(ref, self.run)
        self.inputs[ref.artifact_id] = ref
        return path

    def json(self, ref, base):
        path = self.verified(ref, base)
        document = read_json(path)
        require(sha256_file(path) == ref.sha256, 'Analysis input changed while reading')
        return document

    def array(self, ref, base):
        path = self.verified(ref, base)
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        self.stack.callback(array._mmap.close)
        ref.validate_array(array)
        return array

    def verify_unchanged(self):
        for ref in (self.manifest_ref, *self.inputs.values()):
            verify_ref(ref, self.run)


@contextmanager
def open_analysis_inputs(run):
    with ExitStack() as stack:
        yield AnalysisInputs(run, stack)
