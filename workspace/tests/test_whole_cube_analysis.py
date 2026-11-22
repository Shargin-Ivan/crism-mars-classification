"""Synthetic inference outputs and GT only; no project data or trained weights."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

import test_pipeline_runner as runner_fixtures
from analysis.whole_cube import analyze_run
from modules.ground_truth import GroundTruthManifest, load_ground_truth
from modules.pipeline_contracts import ArtifactRef, ComparisonResult, ContractError, contract_dict
from modules.resource_store import decode_contract, read_json, sha256_file, verify_ref
from modules.run_store import write_manifest


class WholeCubeAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.fixture = runner_fixtures.PipelineRunnerTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def gt(self, mode='companion', *, labels=None, mat=False, roi=None):
        if labels is None:
            labels = np.array([[7, 7, 9], [7, 9, 9], [9, 7, 0]], dtype=np.int16)
        name = 'gt.mat' if mat else 'gt.npy'
        if mat:
            from scipy.io import savemat
            savemat(self.root / name, {'labels': labels.astype(float)})
        else:
            np.save(self.root / name, labels)
        roi_ref = None
        if roi is not None:
            np.save(self.root / 'roi.npy', roi)
            roi_ref = self.fixture.array('roi.npy', 'roi')
        spec = GroundTruthManifest(1, 'synthetic', self.fixture.ref('scene.npy', 'source'), self.fixture.taxonomy,
            self.fixture.ref(name, 'labels'), 'mat' if mat else 'npy', 'labels' if mat else None,
            0, ((7, 0), (9, 1)), roi_ref, 'synthetic', ('development1', 'development2'))
        descriptor = self.root / ('scene.ground_truth.json' if mode == 'companion' else 'evaluation.json')
        write_manifest(descriptor, contract_dict(spec))
        scene = read_json(self.root / 'scene.json')
        scene['ground_truth'] = contract_dict(spec.labels)
        scene['ground_truth_variable'] = spec.variable
        write_manifest(self.root / 'scene.json', scene)
        config = read_json(self.fixture.config)
        config['scene_manifest'] = contract_dict(self.fixture.ref('scene.json', 'scene'))
        if mode == 'config':
            config['evaluation']['ground_truth'] = contract_dict(self.fixture.ref(descriptor.name, 'gt_descriptor'))
        write_manifest(self.fixture.config, config)
        return descriptor, spec

    def check_report(self, report):
        document = read_json(report / 'manifest.json')
        self.assertEqual(document['status'], 'completed')
        for raw in document['artifacts']:
            verify_ref(ArtifactRef(raw['artifact_id'], raw['path'], raw['sha256']), report)
        result = decode_contract(ComparisonResult, read_json(report / 'comparison.json'))
        self.assertEqual(len(result.branches), 4)
        return result

    def test_no_gt_standard_comparison_four_rows_and_immutable_reanalysis(self):
        run = self.fixture.run_pipeline()
        before = {p.relative_to(run): sha256_file(p) for p in run.rglob('*') if p.is_file()}
        first = analyze_run(run)
        result = self.check_report(first)
        self.assertIsNone(result.metrics_gt)
        self.assertEqual(result.not_evaluable_reason, 'ground_truth_missing')
        agreement = read_json(first / 'agreement.json')
        self.assertEqual(len(agreement['pairs']), 6)
        self.assertEqual(agreement['sam_original_baseline']['lwnet_original']['agreement_on_joint'], 0)
        self.assertEqual(len(read_json(first / 'coverage.json')), 4)
        self.assertFalse((first / 'metrics_gt.json').exists())
        first_bytes = {p.name: p.read_bytes() for p in first.iterdir() if p.is_file()}
        second = analyze_run(run)
        self.assertNotEqual(first, second)
        self.assertEqual(first_bytes, {p.name: p.read_bytes() for p in first.iterdir() if p.is_file()})
        self.assertEqual(before, {p: sha256_file(run / p) for p in before})

    def test_gt_companion_mapping_fixed_roi_and_separate_sam_agreement(self):
        self.gt()
        report = analyze_run(self.fixture.run_pipeline())
        result = self.check_report(report)
        self.assertIsNotNone(result.metrics_gt)
        metrics = read_json(report / 'metrics_gt.json')
        info = metrics['ground_truth']
        self.assertEqual(info['origin'], 'scene_companion')
        self.assertEqual(info['role'], 'held_out')
        self.assertEqual(info['split_verification'], 'lineage_unverified')
        self.assertEqual(info['labeled_empty_pixels'], 1)
        for key, score in metrics['branches'].items():
            self.assertEqual(score['n_evaluated'], 7)
            self.assertAlmostEqual(score['oa'], (3 if key.startswith('sam') else 4) / 7)
            self.assertEqual(np.load(report / (key + '_confusion_gt.npy')).shape, (2, 3))
        self.assertEqual(np.load(report / 'gt_labels.npy')[0, 1], 0)  # Local 7, canonical 0.
        self.assertEqual(np.load(report / 'gt_labels.npy')[2, 2], -1)
        self.assertEqual(read_json(report / 'agreement.json')['sam_original_baseline']['lwnet_original']['agreement_on_joint'], 0)

    def test_configured_mat_gt_custom_roi_and_all_unlabeled(self):
        roi = np.ones((3, 3), dtype=bool)
        roi[0, 1] = False
        self.gt('config', mat=True, roi=roi)
        report = analyze_run(self.fixture.run_pipeline())
        self.assertEqual(read_json(report / 'evaluation.json')['ground_truth']['evaluated_pixels'], 6)
        self.assertEqual(read_json(report / 'evaluation.json')['ground_truth']['origin'], 'run_configuration')
        # Fresh descriptor/source for a fresh run, preserving old analysis/inference artifacts.
        self.gt('config', labels=np.zeros((3, 3), dtype=np.int16), mat=True)
        empty_report = analyze_run(self.fixture.run_pipeline())
        self.assertEqual(self.check_report(empty_report).not_evaluable_reason, 'no_labeled_pixels_in_fixed_roi')
        self.assertFalse((empty_report / 'metrics_gt.json').exists())

    def test_raw_labels_without_mapping_keep_standard_comparison(self):
        path, _ = self.gt('unbound')
        run = self.fixture.run_pipeline()
        report = analyze_run(run)
        self.assertEqual(self.check_report(report).not_evaluable_reason, 'labels_present_mapping_missing')
        explicit = analyze_run(run, ground_truth_manifest=path)
        self.assertIsNotNone(self.check_report(explicit).metrics_gt)

    def test_invalid_gt_does_not_erase_standard_comparison(self):
        path, _ = self.gt()
        run = self.fixture.run_pipeline()
        original = read_json(path)
        changes = [{'scene_id': 'development1'}, {'source_to_class': [[7, 0]]},
                   {'source_to_class': [[7, 0], [9, 8]]}, {'held_out_scene_id': 'development1'}]
        for change in changes:
            with self.subTest(change=change):
                write_manifest(path, {**original, **change})
                report = analyze_run(run)
                self.assertEqual(self.check_report(report).not_evaluable_reason, 'ground_truth_invalid')
                self.assertTrue((report / 'agreement.json').exists())
                self.assertFalse((report / 'metrics_gt.json').exists())

    def test_sam_absent_gt_still_evaluates_lwnet_and_partial_cli_is_explicit(self):
        self.gt()
        with patch('modules.pipeline_runner.classify_sam', side_effect=RuntimeError('synthetic SAM failure')):
            run = self.fixture.run_pipeline()
        report = analyze_run(run)
        metrics = read_json(report / 'metrics_gt.json')['branches']
        self.assertEqual(metrics['sam_original']['status'], 'not_evaluable')
        self.assertEqual(metrics['lwnet_original']['n_evaluated'], 7)
        self.assertEqual(len(read_json(report / 'coverage.json')), 4)
        command = [sys.executable, '-B', str(self.fixture.project / 'run_analysis.py'), '--output-root',
                   str(self.root / 'results'), '--run', str(run)]
        refused = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=60)
        self.assertEqual(refused.returncode, 2)
        accepted = subprocess.run(command + ['--allow-partial'], cwd=self.root, capture_output=True, text=True, timeout=60)
        self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)

    def test_analysis_never_reads_weights_or_cube_values_or_imports_torch(self):
        run = self.fixture.run_pipeline()
        # These test-only files are unnecessary for analysis. Their absence must not trigger inference.
        for path in (self.root / 'lwnet/weights.pt', self.root / 'n2n4m/weights.pt'):
            path.unlink()
        code = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
original = np.load
def guarded(path, *args, **kwargs):
    assert Path(path).name not in ('cube.npy', 'weights.pt'), path
    return original(path, *args, **kwargs)
np.load = guarded
import run_analysis
from analysis.whole_cube import analyze_run
analyze_run(sys.argv[2])
assert 'torch' not in sys.modules
assert 'modules.pipeline_runner' not in sys.modules
assert 'modules.cube_splitter' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-B', '-c', code, str(self.fixture.project), str(run)],
                                cwd=self.root, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_corrupt_map_rejected_before_new_analysis(self):
        run = self.fixture.run_pipeline()
        (run / 'maps/sam_original/class_map.npy').write_bytes(b'corrupted fixture')
        with self.assertRaises(ContractError):
            analyze_run(run)
        self.assertFalse(any((run / 'analysis').iterdir()))

    def test_optional_lineage_missing_and_unsupported_mat_keep_standard_analysis(self):
        self.gt('config', mat=True)
        run = self.fixture.run_pipeline()
        (self.root / 'lwnet/training.json').unlink()
        report = analyze_run(run)
        self.assertIsNotNone(self.check_report(report).metrics_gt)
        info = read_json(report / 'evaluation.json')['ground_truth']
        self.assertEqual(info['neural_training_lineage']['lwnet'], 'unverified')
        from scipy.io.matlab import MatReadError
        for error in (NotImplementedError('synthetic unsupported MAT v7.3'), MatReadError('synthetic damaged MAT')):
            with self.subTest(error=error), patch('scipy.io.loadmat', side_effect=error):
                unsupported = analyze_run(run)
            self.assertEqual(self.check_report(unsupported).not_evaluable_reason, 'ground_truth_invalid')
            self.assertTrue((unsupported / 'agreement.json').exists())

    def test_gt_loader_rejects_shape_fractional_ids_and_wrong_cube(self):
        descriptor, spec = self.gt('unbound')
        from modules.data_preparation import SceneManifest
        scene = decode_contract(SceneManifest, read_json(self.root / 'scene.json')).absolute(self.root)
        for labels in (np.ones((2, 3), dtype=int), np.full((3, 3), 7.5), np.full((3, 3), np.nan)):
            np.save(self.root / 'gt.npy', labels)
            modified = replace(spec, labels=self.fixture.ref('gt.npy', 'labels'))
            # Keep scene's source binding consistent to exercise label-array validation.
            bound_scene = replace(scene, ground_truth=replace(modified.labels, path=str(self.root / 'gt.npy')))
            write_manifest(descriptor, contract_dict(modified))
            with self.assertRaises(ContractError):
                load_ground_truth(self.fixture.ref(descriptor.name, 'gt'), self.root, bound_scene, self.fixture.taxonomy)
        modified = replace(spec, source_cube=replace(spec.source_cube, sha256='0' * 64))
        write_manifest(descriptor, contract_dict(modified))
        with self.assertRaises(ContractError):
            load_ground_truth(self.fixture.ref(descriptor.name, 'gt'), self.root, scene, self.fixture.taxonomy)

    def test_abstention_stays_in_fixed_gt_roi_and_holdout_overlap_is_reported(self):
        self.gt()
        folder = self.root / 'lwnet'
        preprocessing = read_json(folder / 'preprocess.json')
        preprocessing['parameters']['min_valid_fraction'] = 1.0
        write_manifest(folder / 'preprocess.json', preprocessing)
        write_manifest(folder / 'training.json', {'training_scene_ids': ['synthetic'], 'validation_scene_ids': []})
        bundle = read_json(folder / 'bundle.json')
        bundle['preprocessing']['sha256'] = sha256_file(folder / 'preprocess.json')
        bundle['training_manifest']['sha256'] = sha256_file(folder / 'training.json')
        write_manifest(folder / 'bundle.json', bundle)
        config = read_json(self.fixture.config)
        config['resources']['lwnet_bundle'] = contract_dict(self.fixture.ref('lwnet/bundle.json', 'lwnet_bundle'))
        write_manifest(self.fixture.config, config)
        report = analyze_run(self.fixture.run_pipeline())
        metrics = read_json(report / 'metrics_gt.json')
        self.assertEqual(metrics['ground_truth']['split_verification'], 'overlap_detected_diagnostic_only')
        for key, value in metrics['branches'].items():
            self.assertEqual(value['n_evaluated'], 7)
            if key.startswith('lwnet'):
                self.assertAlmostEqual(value['oa'], 3 / 7)
                self.assertAlmostEqual(value['coverage'], 6 / 7)
                self.assertEqual(value['confusion'][1][-1], 1)


if __name__ == '__main__':
    unittest.main()
