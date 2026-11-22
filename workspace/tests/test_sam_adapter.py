"""CPU-only synthetic archives: no local datasets or pretrained resources."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from modules.cube_adapters.common import AdapterUnavailable, CubeView
from modules.cube_adapters.sam import _align_references, classify_sam
from modules.pipeline_config import SamSpec
from modules.pipeline_contracts import (
    ArtifactRef, ContractError, PreparedCube, ReferenceLibrary, SpectralAxis, Taxonomy, contract_dict,
)
from modules.resource_store import array_reference, read_json, sha256_file
from modules.run_store import write_manifest


def axis(wavelengths, detectors=None, segments=None):
    n = len(wavelengths)
    return SpectralAxis(tuple(range(n)), tuple(wavelengths), tuple(detectors or ['S'] * n),
                        tuple(segments or [0] * n))


class SamAdapterTests(unittest.TestCase):
    def setUp(self):
        project_root = Path(__file__).resolve().parents[1]
        temporary_root = (project_root / 'output/adapter_tests_tmp').resolve()
        self.assertTrue(temporary_root.is_relative_to(project_root))
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.old_tempdir = tempfile.tempdir
        tempfile.tempdir = str(temporary_root)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.taxonomy = Taxonomy('canonical', '1', ('first', 'second', 'third'))

    def tearDown(self):
        self.temp.cleanup()
        tempfile.tempdir = self.old_tempdir

    def array(self, name, values):
        path = self.root / (name + '.npy')
        np.save(path, values)
        return replace(array_reference(path, name, self.root), path=str(path))

    def view(self, values, grid, good=None):
        cube = np.asarray(values, dtype=np.float32)
        good = np.ones(cube.shape[-1], bool) if good is None else np.asarray(good, bool)
        valid = np.isfinite(cube) & good
        cube_ref = self.array('cube', cube)
        valid_ref = self.array('valid', valid)
        good_ref = self.array('good', good)
        source_path = self.root / 'source.json'
        write_manifest(source_path, {'synthetic': True})
        source = ArtifactRef('source', str(source_path), sha256_file(source_path))
        prepared = PreparedCube('synthetic', source, cube.shape, cube_ref, valid_ref, good_ref,
                                grid, 'I/F', 'corrected_if')
        return CubeView(prepared, cube_ref, valid_ref, cube, valid, good, 'prepared')

    def library(self, values, grid, ids, mask=None, **kwargs):
        spectra = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(spectra) if mask is None else np.asarray(mask, bool)
        library = ReferenceLibrary('synthetic', self.array('spectra', spectra), self.array('ref_valid', valid),
                                   grid, self.taxonomy, tuple(ids), kwargs.get('units', 'I/F'),
                                   kwargs.get('domain', 'corrected_if'))
        path = self.root / 'library.json'
        write_manifest(path, contract_dict(library))
        return ArtifactRef('library', str(path), sha256_file(path))

    def run_sam(self, view, library, name='result', **kwargs):
        path = self.root / name
        result = classify_sam(view, library, self.root, self.taxonomy, path, **kwargs)
        return result, np.load(path / result.class_map.path), np.load(path / result.rejection_reason.path)

    def test_class_ids_multiple_references_reasons_and_immutable_input(self):
        grid = axis([1000, 1005, 1010, 1015])
        view = self.view([[[1, 2, 3, 4], [4, 3, 2, 1], [np.nan] * 4,
                           [1, np.nan, np.nan, np.nan], [0] * 4, [1, -1, 1, -1],
                           [1, np.nan, 3, 4]]], grid)
        before = view.cube.copy()
        library = self.library([[4, 3, 2, 1], [2, 4, 6, 8], [1, 2, 3, 4]], grid, [0, 2, 2])
        result, labels, reasons = self.run_sam(view, library, batch_size=2)
        np.testing.assert_array_equal(labels, [[2, 0, -1, -1, -1, -1, 2]])
        np.testing.assert_array_equal(reasons, [[0, 0, 1, 2, 2, 4, 0]])
        np.testing.assert_array_equal(view.cube, before)
        self.assertFalse(view.cube.flags.writeable)
        self.assertEqual(result.input_cube.shape, view.cube.shape)
        permuted = self.library([[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 6, 8]], grid, [2, 0, 2])
        _, reordered, _ = self.run_sam(view, permuted, name='permuted')
        np.testing.assert_array_equal(labels, reordered)

    def test_alignment_preserves_gaps_masks_detectors_and_full_grid(self):
        source = axis([1000, 1005, 1010, 1020, 1025, 1000, 1005],
                      ['S'] * 5 + ['L'] * 2, [0, 0, 0, 1, 1, 2, 2])
        target = axis([997, 1000, 1002.5, 1007.5, 1015, 1022.5, 1027,
                       1002.5, 1007.5], ['S'] * 7 + ['L'] * 2, list(range(9)))
        spectra = np.array([[1, 3, np.nan, 7, 9, 10, 20], [2, 4, 6, 8, 10, 20, 40.]])
        mask = np.isfinite(spectra)
        mask[1, 1] = False
        aligned = _align_references(spectra, mask, source, target)
        np.testing.assert_allclose(aligned[0], [np.nan, 1, 2, np.nan, np.nan, 8, np.nan, 15, np.nan], equal_nan=True)
        self.assertTrue(np.isnan(aligned[1, 2:5]).all())
        # Equal wavelengths in S and L are valid, but never mix detectors.
        self.assertEqual(aligned[1, 7], 30)
        view = self.view([[[1] * 9]], target, good=[True] * 7 + [False, True])
        library = self.library(spectra, source, [2, 0], mask=mask)
        result, labels, reasons = self.run_sam(view, library)
        np.testing.assert_array_equal(np.load(self.root / 'result' / result.used_channels.path),
                                      [False, True, True, False, False, True, False, False, False])
        self.assertEqual(reasons[0, 0], 2)  # 3 supported < ceil(full C * 0.5) = 5
        self.assertEqual(labels[0, 0], -1)
        parameters = read_json(self.root / 'result/manifest.json')['parameters']
        self.assertAlmostEqual(parameters['supported_channel_fraction'], 3 / 9)
        self.assertEqual(parameters['sam'], contract_dict(SamSpec()))

    def test_incompatible_inputs_fail_before_output(self):
        grid = axis([1000, 1005])
        view = self.view([[[1, 2]]], grid)
        for field, value, code in [('units', 'radiance', 'units_mismatch'),
                                   ('domain', 'raw', 'domain_mismatch')]:
            library = self.library([[1, 2]], grid, [1], **{field: value})
            with self.assertRaises(AdapterUnavailable) as error:
                self.run_sam(view, library)
            self.assertEqual(error.exception.reason_code, code)
        library = self.library([[1, 2]], grid, [1])
        with self.assertRaises(AdapterUnavailable) as error:
            classify_sam(view, library, self.root, replace(self.taxonomy, version='2'), self.root / 'result')
        self.assertEqual(error.exception.reason_code, 'taxonomy_mismatch')
        with self.assertRaises(ContractError):
            self.run_sam(view, library, branch_id='sam_denoised')
        with self.assertRaises(ContractError):
            self.run_sam(view, library, spec=SamSpec(min_valid_bands=1))
        self.assertFalse((self.root / 'result').exists())

    def test_duplicate_wavelengths_and_overlapping_segments_are_rejected(self):
        target = axis([1001, 1004])
        for source in (axis([1000, 1000], segments=[0, 1]),
                       axis([1000, 1005, 1000.5, 1005.5], segments=[0, 0, 1, 1])):
            spectra = np.ones((1, len(source.channel_indices)))
            with self.assertRaises(AdapterUnavailable) as error:
                _align_references(spectra, np.ones_like(spectra, bool), source, target)
            self.assertEqual(error.exception.reason_code, 'ambiguous_reference_axis')


if __name__ == '__main__':
    unittest.main()
