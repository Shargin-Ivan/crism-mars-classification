"""Whole-cube orchestration with synthetic files and a deterministic CPU backend."""
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from modules.cube_adapters.common import AdapterUnavailable
from modules.data_preparation import SceneManifest
from modules.pipeline_contracts import (
    ArtifactRef, ContractError, ModelBundle, ReferenceLibrary, SpectralAxis, Taxonomy, contract_dict,
)
from modules.pipeline_run_store import validate_manifest
from modules.resource_store import array_reference, read_json, sha256_file
from modules.run_store import write_manifest


class PipelineRunnerTests(unittest.TestCase):
    def setUp(self):
        self.project = Path(__file__).resolve().parents[1]
        temporary = (self.project / 'output/adapter_tests_tmp').resolve()
        self.assertTrue(temporary.is_relative_to(self.project))
        temporary.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.axis = SpectralAxis((0, 1, 2, 3), (1000, 1006, 1000, 1006),
                                 ('S', 'S', 'L', 'L'), (0, 0, 1, 1))
        self.taxonomy = Taxonomy('synthetic', '1', ('A', 'B'))
        self.values = np.ones((3, 3, 4), dtype=np.float32)
        self.values[0, 0] = np.nan
        self.values[1, 1, 1] = np.nan
        np.save(self.root / 'scene.npy', self.values)
        self.source_hash = sha256_file(self.root / 'scene.npy')
        scene = SceneManifest(1, 'synthetic', 'npy', self.ref('scene.npy', 'source'),
                              self.values.shape, self.values.dtype.str, None, None, self.axis,
                              (True,) * 4, (), 'I/F', 'corrected_if', (), None, None)
        write_manifest(self.root / 'scene.json', contract_dict(scene))
        write_manifest(self.root / 'taxonomy.json', contract_dict(self.taxonomy))
        np.save(self.root / 'references.npy', np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]))
        np.save(self.root / 'reference_valid.npy', np.ones((2, 4), dtype=bool))
        library = ReferenceLibrary('synthetic_library', self.array('references.npy', 'spectra'),
                                   self.array('reference_valid.npy', 'reference_valid'),
                                   self.axis, self.taxonomy, (0, 1), 'I/F', 'corrected_if')
        write_manifest(self.root / 'library.json', contract_dict(library))
        config = read_json(self.project / 'configs/pipelines/four_branches.json')
        config.update(configuration_state='ready', scene_manifest=contract_dict(self.ref('scene.json', 'scene')),
                      taxonomy=contract_dict(self.ref('taxonomy.json', 'taxonomy')))
        config['resources'] = {'n2n4m_bundle': contract_dict(self.bundle('n2n4m')),
                               'lwnet_bundle': contract_dict(self.bundle('lwnet')),
                               'sam_library': contract_dict(self.ref('library.json', 'library'))}
        config['stores'] = dict(inputs_root='inputs', resources_root='resources', results_root='results')
        config['lwnet'].update(patch_size=3, batch_size=2)
        config['n2n4m']['batch_size'] = 2
        self.config = self.root / 'pipeline.json'
        write_manifest(self.config, config)

    def ref(self, relative, name):
        return ArtifactRef(name, relative, sha256_file(self.root / relative))

    def array(self, relative, name):
        return array_reference(self.root / relative, name, self.root)

    def bundle(self, algorithm):
        folder = self.root / algorithm
        folder.mkdir()
        (folder / 'weights.pt').write_bytes(b'SYNTHETIC: backend is mocked, not trained weights')
        parameters = dict(adapter_version=1, backend='torchscript_full_axis_v1', weights_format='torchscript',
                          architecture={}, units='I/F', normalization='identity', missing_policy='constant_normalized',
                          fill_value=0, min_valid_bands=2, min_valid_fraction=.5,
                          output_kind='normalized_spectrum' if algorithm == 'n2n4m' else 'logits')
        if algorithm == 'lwnet':
            parameters.update(patch_size=3, border_policy='reflect', min_patch_valid_fraction=0)
        write_manifest(folder / 'preprocess.json', dict(schema_version=1, scaler=None, parameters=parameters))
        write_manifest(folder / 'training.json', dict(synthetic_fixture=True, trained=False))
        refs = [ArtifactRef(algorithm + '_' + name, name, sha256_file(folder / name))
                for name in ('weights.pt', 'preprocess.json', 'training.json')]
        bundle = ModelBundle(algorithm, algorithm, 'full_axis_v1', *refs, 'corrected_if', 'fixed_grid', self.axis,
                             self.taxonomy if algorithm == 'lwnet' else None,
                             (1, 0) if algorithm == 'lwnet' else ())
        write_manifest(folder / 'bundle.json', contract_dict(bundle))
        return self.ref(algorithm + '/bundle.json', algorithm + '_bundle')

    @staticmethod
    def backend(plan, view, execution):
        if plan.bundle.algorithm == 'n2n4m':
            return lambda values, valid: (values + .05).astype(np.float32)
        return lambda values, valid: np.tile([2., -1.], (len(values), 1))

    def run_pipeline(self):
        from modules.pipeline_runner import run_pipeline
        with patch('modules.cube_adapters.neural.load_backend', side_effect=self.backend):
            return run_pipeline(self.config)

    def manifests(self, run):
        document = read_json(run / 'manifest.json')
        validate_manifest(document, run, verify_artifacts=True)
        return document, read_json(run / 'execution.json')

    def test_complete_graph_single_shared_stages_and_immutable_repeat(self):
        from modules import pipeline_runner as runner
        with ExitStack() as stack:
            calls = {name: stack.enter_context(patch.object(runner, name, wraps=getattr(runner, name)))
                     for name in ('prepare_scene', 'denoise_n2n4m', 'classify_sam', 'classify_lwnet')}
            run = self.run_pipeline()
        self.assertEqual([calls[name].call_count for name in calls], [1, 1, 2, 2])
        document, execution = self.manifests(run)
        self.assertEqual(document['status'], 'completed')
        self.assertEqual({b['status'] for b in document['branches'].values()}, {'completed'})
        for stage in ('preparation', 'n2n4m'):
            self.assertEqual(execution['stages'][stage]['status'], 'completed')
            self.assertTrue(execution['stages'][stage]['result_id'])
        self.assertFalse(execution['branch_errors'])
        inputs = {}
        for branch, outcome in document['branches'].items():
            self.assertEqual(outcome['result']['path'], f'maps/{branch}/manifest.json')
            result = read_json(run / outcome['result']['path'])['result']
            inputs[branch] = result['input_cube']
        self.assertEqual(inputs['sam_original'], inputs['lwnet_original'])
        self.assertEqual(inputs['sam_denoised'], inputs['lwnet_denoised'])
        self.assertNotEqual(inputs['sam_original']['artifact_id'], inputs['sam_denoised']['artifact_id'])
        denoising = read_json(run / 'processed/n2n4m/manifest.json')['result']
        denoised = np.load(run / 'processed/n2n4m' / denoising['cube']['path'])
        np.testing.assert_array_equal(np.isnan(denoised), np.isnan(self.values))
        np.testing.assert_array_equal(np.load(self.root / 'scene.npy'), self.values)
        self.assertEqual(sha256_file(self.root / 'scene.npy'), self.source_hash)
        self.assertFalse(any((run / 'analysis').iterdir()))
        before = {p.relative_to(run): sha256_file(p) for p in run.rglob('*') if p.is_file()}
        again = self.run_pipeline()
        self.assertNotEqual(run, again)
        self.assertEqual(before, {p.relative_to(run): sha256_file(p) for p in run.rglob('*') if p.is_file()})
        self.assertEqual(self.manifests(again)[0]['status'], 'completed')

    def test_denoiser_failure_blocks_only_dependants(self):
        with patch('modules.pipeline_runner.denoise_n2n4m', side_effect=RuntimeError('synthetic N2 failure')):
            run = self.run_pipeline()
        document, execution = self.manifests(run)
        self.assertEqual(document['status'], 'partial')
        self.assertEqual(execution['stages']['n2n4m']['status'], 'failed')
        for branch, outcome in document['branches'].items():
            self.assertEqual(outcome['status'], 'skipped' if branch.endswith('denoised') else 'completed')

    def test_classifier_failure_is_independent(self):
        from modules import pipeline_runner as runner
        original = runner.classify_sam
        count = 0
        def fail_first(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                raise RuntimeError('synthetic single branch failure')
            return original(*args, **kwargs)
        with patch.object(runner, 'classify_sam', side_effect=fail_first):
            run = self.run_pipeline()
        document, execution = self.manifests(run)
        statuses = [b['status'] for b in document['branches'].values()]
        self.assertEqual(statuses.count('completed'), 3)
        self.assertEqual(statuses.count('failed'), 1)
        self.assertEqual(document['status'], 'partial')
        self.assertEqual(len(execution['branch_errors']), 1)

    def test_unavailable_classifier_keeps_reason(self):
        with patch('modules.pipeline_runner.classify_sam',
                   side_effect=AdapterUnavailable('synthetic_unsupported', 'unsupported synthetic fixture')):
            run = self.run_pipeline()
        document, _ = self.manifests(run)
        self.assertEqual(document['status'], 'partial')
        for branch, outcome in document['branches'].items():
            if branch.startswith('sam'):
                self.assertEqual(outcome['status'], 'skipped')
                self.assertEqual(outcome['reason_code'], 'synthetic_unsupported')
            else:
                self.assertEqual(outcome['status'], 'completed')

    def test_missing_lwnet_weights_does_not_block_sam_or_denoising(self):
        from modules import pipeline_runner as runner
        (self.root / 'lwnet/weights.pt').unlink()
        with patch.object(runner, 'denoise_n2n4m', wraps=runner.denoise_n2n4m) as denoise:
            run = self.run_pipeline()
        self.assertEqual(denoise.call_count, 1)
        document, execution = self.manifests(run)
        self.assertEqual(document['status'], 'partial')
        self.assertEqual(execution['stages']['n2n4m']['status'], 'completed')
        for branch, outcome in document['branches'].items():
            self.assertEqual(outcome['status'], 'failed' if branch.startswith('lwnet') else 'completed')

    def test_preparation_failure_skips_all_branches(self):
        with patch('modules.pipeline_runner.prepare_scene', side_effect=RuntimeError('synthetic preparation failure')):
            run = self.run_pipeline()
        document, execution = self.manifests(run)
        self.assertEqual(document['status'], 'failed')
        self.assertEqual({b['status'] for b in document['branches'].values()}, {'skipped'})
        self.assertEqual(execution['stages']['preparation']['status'], 'failed')

    def test_draft_rejected_before_output(self):
        config = read_json(self.config)
        config['configuration_state'] = 'draft'
        write_manifest(self.config, config)
        with self.assertRaises(ContractError):
            self.run_pipeline()
        self.assertFalse((self.root / 'results').exists())

    def test_explicit_sam_baseline_on_draft_is_analyzable_without_neural_resources(self):
        from analysis.whole_cube import analyze_run
        from main import main
        config = read_json(self.config)
        config.update(configuration_state='draft')
        config['resources'].update(lwnet_bundle=None, n2n4m_bundle=None)
        write_manifest(self.config, config)
        with patch('modules.pipeline_runner._inference_seed') as seed:
            self.assertEqual(main(['--config', str(self.config), '--sam-only']), 0)
        seed.assert_not_called()
        run, = (self.root / 'results/runs').iterdir()
        document, execution = self.manifests(run)
        self.assertEqual(execution['requested_branches'], ['sam_original'])
        self.assertEqual(execution['stages']['n2n4m']['reason_code'], 'not_requested')
        self.assertEqual(document['status'], 'partial')
        for key, outcome in document['branches'].items():
            if key == 'sam_original':
                self.assertEqual(outcome['status'], 'completed')
            else:
                self.assertEqual((outcome['status'], outcome['reason_code']), ('skipped', 'not_requested'))
        coverage = read_json(analyze_run(run) / 'coverage.json')['sam_original']
        self.assertEqual(coverage['predicted_pixels'], 8)

    def test_unbound_sam_baseline_rejected_before_output(self):
        from modules.pipeline_runner import run_pipeline
        config = read_json(self.config)
        config.update(configuration_state='draft')
        config['resources']['sam_library'] = None
        write_manifest(self.config, config)
        with self.assertRaisesRegex(ContractError, 'resources.sam_library'):
            run_pipeline(self.config, sam_only=True)
        self.assertFalse((self.root / 'results').exists())

    def test_keyboard_interrupt_bubbles_and_finalizes_manifest(self):
        with patch('modules.pipeline_runner.denoise_n2n4m', side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                self.run_pipeline()
        runs = list((self.root / 'results/runs').iterdir())
        self.assertEqual(len(runs), 1)
        document, _ = self.manifests(runs[0])
        self.assertEqual(document['status'], 'failed')
        self.assertIn('KeyboardInterrupt', document['error'])
        self.assertTrue(all(b['status'] in ('completed', 'failed', 'skipped') for b in document['branches'].values()))

    def test_fresh_import_does_not_load_models_legacy_or_create_directories(self):
        cwd = self.root / 'import_only'
        cwd.mkdir()
        code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
before = set(Path.cwd().iterdir())
import main
import modules.pipeline_runner
assert set(Path.cwd().iterdir()) == before
for name in ('torch', 'legacy_main', 'modules.lwnet_module', 'modules.n2n4m_module', 'modules.splitting'):
    assert name not in sys.modules, name
assert not any('splitter' in name for name in sys.modules)
"""
        result = subprocess.run([sys.executable, '-B', '-c', code, str(self.project)], cwd=cwd,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_incomplete_metadata_skips_models_without_neural_initialization(self):
        scene = read_json(self.root / 'scene.json')
        scene['spectral_axis'] = contract_dict(SpectralAxis((0, 1, 2, 3), None, None, None, 'incomplete'))
        write_manifest(self.root / 'scene.json', scene)
        config = read_json(self.config)
        config['scene_manifest'] = contract_dict(self.ref('scene.json', 'scene'))
        write_manifest(self.config, config)
        with patch('modules.pipeline_runner._inference_seed', side_effect=AssertionError('must not initialize')) as seed:
            run = self.run_pipeline()
        seed.assert_not_called()
        document, execution = self.manifests(run)
        self.assertEqual(document['status'], 'failed')
        self.assertEqual(execution['stages']['preparation']['status'], 'completed')
        self.assertEqual(execution['stages']['n2n4m']['reason_code'], 'metadata_incomplete')
        for key in ('sam_original', 'lwnet_original'):
            self.assertEqual(document['branches'][key]['reason_code'], 'metadata_incomplete')

    def test_seed_is_repeatable_restores_rng_and_does_not_initialize_cuda(self):
        import random
        import torch
        from modules.pipeline_config import ExecutionSpec
        from modules.pipeline_runner import _inference_seed
        before_python, before_numpy, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        draws = []
        with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CPU must not probe CUDA')):
            for _ in range(2):
                with _inference_seed(ExecutionSpec(seed=37)):
                    draws.append((random.random(), np.random.random(), torch.rand(1).item()))
            with self.assertRaises(RuntimeError):
                with _inference_seed(ExecutionSpec(seed=7)):
                    torch.rand(1)
                    raise RuntimeError('synthetic abort')
        self.assertEqual(draws[0], draws[1])
        self.assertEqual(before_python, random.getstate())
        after_numpy = np.random.get_state()
        self.assertEqual(before_numpy[0], after_numpy[0])
        np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
        self.assertEqual(before_numpy[2:], after_numpy[2:])
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))

    def test_cli_real_synthetic_exports_and_exit_statuses(self):
        import torch
        # New tiny test exports, not the project's trained models. Exercise CLI without mocks.
        metadata = (torch.tensor(self.axis.wavelengths_nm), torch.tensor([1, 1, 0, 0]),
                    torch.tensor([True, False, True]))
        config = read_json(self.config)
        for algorithm in ('n2n4m', 'lwnet'):
            values = torch.ones((2, 4) if algorithm == 'n2n4m' else (2, 1, 4, 3, 3))
            fn = (lambda x, mask, wave, detector, edge: x + .05) if algorithm == 'n2n4m' else (
                lambda x, mask, wave, detector, edge: torch.stack((x.mean((1, 2, 3, 4)), -x.mean((1, 2, 3, 4))), 1))
            exported = torch.jit.trace(fn, (values, torch.ones_like(values, dtype=torch.bool), *metadata))
            folder = self.root / algorithm
            exported.save(str(folder / 'weights.pt'))
            bundle = read_json(folder / 'bundle.json')
            bundle['weights']['sha256'] = sha256_file(folder / 'weights.pt')
            write_manifest(folder / 'bundle.json', bundle)
            config['resources'][algorithm + '_bundle'] = contract_dict(self.ref(algorithm + '/bundle.json', algorithm + '_bundle'))
        write_manifest(self.config, config)
        command = [sys.executable, '-B', str(self.project / 'main.py'), '--config', str(self.config)]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        import json
        report = json.loads(result.stdout)
        self.assertEqual(report['status'], 'completed')
        self.manifests(Path(report['run']))
        (self.root / 'lwnet/weights.pt').unlink()
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'partial')
        config['configuration_state'] = 'draft'
        write_manifest(self.config, config)
        before = set((self.root / 'results/runs').iterdir())
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(before, set((self.root / 'results/runs').iterdir()))


if __name__ == '__main__':
    unittest.main()
