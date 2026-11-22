"""Saved synthetic comparisons only; figures never invoke inference or training."""
from pathlib import Path
from contextlib import nullcontext
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import test_whole_cube_analysis as fixtures
from analysis.visualize_comparison import (
    _class_differences, _html, _native_class_maps, _native_palettes, _spectra_figure,
    render_analysis, sam_difference, gt_difference,
)
from analysis.whole_cube import analyze_run
from modules.pipeline_contracts import ArtifactRef, BRANCHES, ContractError, Taxonomy, VisualizationResult
from modules.resource_store import decode_contract, read_json, sha256_file, verify_ref
from modules.visualization_inputs import open_visualization_inputs


class WholeCubeVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.case = fixtures.WholeCubeAnalysisTests(methodName='runTest')
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.fixture = self.case.fixture

    def report(self, gt=False, absent_sam=False):
        if gt:
            self.case.gt()
        with patch('modules.pipeline_runner.classify_sam', side_effect=RuntimeError('synthetic absent SAM')) if absent_sam else nullcontext():
            run = self.fixture.run_pipeline()
        return run, analyze_run(run)

    def check_visual(self, path, analysis):
        document = read_json(path / 'manifest.json')
        self.assertEqual(document['status'], 'completed')
        self.assertEqual(document['analysis_id'], analysis.name)
        result = decode_contract(VisualizationResult, read_json(path / 'visualization.json'))
        self.assertEqual(result.analysis_id, analysis.name)
        for ref in document['artifacts']:
            verify_ref(ArtifactRef(**ref), path)
        self.assertEqual(document['source_analysis']['sha256'], sha256_file(analysis / 'manifest.json'))
        for item in document['figures']:
            self.assertTrue((path / item['png']).read_bytes().startswith(b'\x89PNG'))
            self.assertIn('<svg', (path / item['svg']).read_text(encoding='utf-8'))
        return document

    def test_difference_categories_keep_abstentions_and_roi_distinct(self):
        left = np.array([[0, 0, 0, -1, -1, -1]])
        right = np.array([[0, 1, -1, 1, -1, -1]])
        valid = np.array([[True, True, True, True, True, False]])
        np.testing.assert_array_equal(sam_difference(left, right, valid), [[1, 2, 3, 4, 5, 0]])
        np.testing.assert_array_equal(gt_difference(np.zeros_like(left), right, valid), [[1, 2, 3, 2, 3, 0]])

    def test_overlapping_native_ids_have_separate_colors_labels_and_comparisons(self):
        minerals = Taxonomy('minerals', '1', ('Gypsum', 'mineral B'))
        land = Taxonomy('land', '1', ('Celery', 'crop B', 'crop C'))
        labels = np.array([[0, 1, -1]])
        valid = np.ones(labels.shape, dtype=bool)
        view = SimpleNamespace(
            result=SimpleNamespace(taxonomy=minerals), gt=labels, gt_mask=valid, valid=valid,
            maps={key: labels.copy() for key in BRANCHES},
            taxonomies={key: minerals if key.startswith('sam') else land for key in BRANCHES},
            outcomes={key: {'status': 'completed'} for key in BRANCHES})
        for reference in ('sam_original', 'gt'):
            self.assertEqual(set(_class_differences(view, reference)), {'sam_original', 'sam_denoised'})
        palettes = _native_palettes(view)
        self.assertNotEqual(palettes[minerals]['classes'][0], palettes[land]['classes'][0])
        figure = _native_class_maps(view, palettes)
        self.assertNotEqual(figure.axes[0].images[0].get_array()[0, 0],
                            figure.axes[2].images[0].get_array()[0, 0])
        legends = [text.get_text() for text in figure.legends[0].get_texts()]
        self.assertIn('minerals · 0: Гипс', legends)
        self.assertIn('land · 0: Сельдерей', legends)
        self.assertIn('land · 1: crop B', legends)
        spectra = {key: dict(status='completed', x=[1000], mean=[1.], allowed_edges=[],
                             class_id=0, x_label='nm', units='I/F') for key in BRANCHES}
        figure = _spectra_figure(spectra, view, palettes)
        self.assertIn('Сельдерей', figure.axes[2].lines[0].get_label())

    def test_full_gt_figures_and_correct_spectra_sources(self):
        run, analysis = self.report(gt=True)
        before = {str(p): sha256_file(p) for p in run.rglob('*') if p.is_file()}
        visual = render_analysis(analysis, spectra=True)
        document = self.check_visual(visual, analysis)
        self.assertEqual(set(item['section'] for item in document['figures']), {'overview', 'sam', 'gt', 'spectra'})
        self.assertEqual(len([f for f in document['figures'] if 'confusion_gt' in f['png']]), 4)
        spectra = read_json(visual / 'spectra.json')
        for key, values in spectra.items():
            self.assertEqual(values['status'], 'completed')
            expected = 1.05 if key.endswith('denoised') else 1.0
            np.testing.assert_allclose(values['mean'], expected, atol=1e-6)
            self.assertEqual(values['allowed_edges'], [True, False, True])
        self.assertEqual(spectra['sam_denoised']['source_cube'], spectra['lwnet_denoised']['source_cube'])
        self.assertEqual(before, {p: sha256_file(p) for p in before})

    def test_no_gt_partial_sam_absence_still_visualizes_lwnet(self):
        run, analysis = self.report(absent_sam=True)
        visual = render_analysis(analysis)
        document = self.check_visual(visual, analysis)
        self.assertFalse(any(f['section'] == 'gt' for f in document['figures']))
        text = (visual / 'report.html').read_text(encoding='utf-8')
        self.assertIn('ground_truth_missing', text)
        self.assertIn('adapter_failed', text)
        self.assertFalse((visual / 'spectra.json').exists())
        self.assertEqual(read_json(run / 'manifest.json')['status'], 'partial')

    def test_html_legend_keeps_scientific_name_and_escapes_unknown_names(self):
        _, analysis = self.report()
        space = Taxonomy('test', '1', ('Gypsum', 'X & Y'))
        with open_visualization_inputs(analysis) as view:
            _html(view, self.fixture.root, [], {space: {'classes': ['#abcdef', '#eeeeee']}},
                  dict.fromkeys(('overview', 'sam', 'gt', 'spectra'), ''))
        report = (self.fixture.root / 'report.html').read_text(encoding='utf-8')
        self.assertIn('title="Gypsum"', report)
        self.assertIn('0: Гипс</li>', report)
        self.assertIn('1: X &amp; Y</li>', report)

    def test_rerender_is_separate_cli_uses_snapshot_without_models_or_cube_values(self):
        _, analysis = self.report()
        original = (analysis / 'manifest.json').read_bytes()
        code = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
original = np.load
def guarded(path, *a, **kw):
    assert Path(path).name not in ('cube.npy', 'weights.pt'), path
    return original(path, *a, **kw)
np.load = guarded
from analysis.visualize_comparison import main
assert main(['--analysis', sys.argv[2]]) == 0
assert 'torch' not in sys.modules
assert 'modules.pipeline_runner' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-B', '-c', code, str(self.fixture.project), str(analysis)],
                                cwd=self.fixture.root, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        first = next((analysis / 'visualizations').iterdir())
        before = {p.name: sha256_file(p) for p in first.iterdir() if p.is_file()}
        second = render_analysis(analysis)
        self.assertNotEqual(first, second)
        self.assertEqual(before, {p.name: sha256_file(p) for p in first.iterdir() if p.is_file()})
        self.assertEqual(original, (analysis / 'manifest.json').read_bytes())

    def test_corrupted_snapshot_or_map_is_rejected_before_rendering(self):
        run, analysis = self.report()
        original = (analysis / 'coverage.json').read_bytes()
        (analysis / 'coverage.json').write_bytes(b'{}')
        with self.assertRaises(ContractError):
            render_analysis(analysis)
        self.assertFalse((analysis / 'visualizations').exists())
        (analysis / 'coverage.json').write_bytes(original)
        (run / 'maps/lwnet_original/class_map.npy').write_bytes(b'bad map')
        with self.assertRaises(ContractError):
            render_analysis(analysis)

    def test_missing_optional_cube_does_not_block_other_spectra(self):
        run, analysis = self.report()
        (run / 'processed/n2n4m/cube.npy').unlink()
        with open_visualization_inputs(analysis) as view:
            original = view.dominant_spectrum('sam_original')
            self.assertEqual(original['status'], 'completed')
            with self.assertRaises(ContractError):
                view.dominant_spectrum('sam_denoised')
            view.verify_unchanged()

    def test_render_failure_keeps_completed_analysis_and_marks_visualization_failed(self):
        _, analysis = self.report()
        before = (analysis / 'manifest.json').read_bytes()
        with patch('matplotlib.figure.Figure.savefig', side_effect=RuntimeError('synthetic image write failure')):
            with self.assertRaises(RuntimeError):
                render_analysis(analysis)
        revisions = list((analysis / 'visualizations').iterdir())
        self.assertEqual(len(revisions), 1)
        self.assertEqual(read_json(revisions[0] / 'manifest.json')['status'], 'failed')
        self.assertEqual(before, (analysis / 'manifest.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
