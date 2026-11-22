from dataclasses import FrozenInstanceError, replace
import json
import unittest

import numpy as np

from modules.pipeline_contracts import (
    ArtifactRef, ArrayRef, BRANCHES, BranchOutcome, ClassificationResult,
    ComparisonResult, ContractError, DenoisingResult, ModelBundle, PreparedCube,
    ReferenceLibrary, SpectralAxis, Taxonomy, VisualizationResult,
    contract_dict, map_ground_truth,
)


def artifact(name):
    return ArtifactRef(name, name + '.json', 'a' * 64)


def array_ref(name, shape, dtype):
    return ArrayRef(name, name + '.npy', 'b' * 64, shape, dtype)


def axis(channels=4):
    return SpectralAxis(tuple(range(channels)), tuple(1000 + 6 * i for i in range(channels)),
                        ('L',) * channels, (0,) * channels)


def prepared(shape=(2, 3, 4)):
    return PreparedCube('scene', artifact('source'), shape, array_ref('cube', shape, 'float32'),
                        array_ref('valid', shape, 'bool'), array_ref('quality', (shape[2],), 'bool'),
                        axis(shape[2]), 'I/F', 'corrected_if')


TAXONOMY = Taxonomy('minerals', '1', ('class_A', 'class_B'))


class InputContracts(unittest.TestCase):
    def test_whole_cube_varies_in_channel_count_without_any_partition(self):
        for channels in (3, 17, 440):
            with self.subTest(channels=channels):
                cube = prepared((2, 3, channels))
                values = np.ones(cube.cube.shape, dtype=np.float32)
                mask = np.ones(values.shape, dtype=bool)
                values[0, 0, 0] = np.nan
                mask[0, 0, 0] = False
                values.flags.writeable = mask.flags.writeable = False
                cube.validate_arrays(values, mask, np.ones(channels, dtype=bool))
                cube.require_ready()
                self.assertTrue(np.isnan(values[0, 0, 0]))
                self.assertEqual(cube.cube.shape, cube.source_shape)
                json.dumps(contract_dict(cube), allow_nan=False)

    def test_crop_channel_subset_reordering_and_invalid_measurements_fail(self):
        cube = prepared()
        for altered in (
                dict(cube=array_ref('crop', (1, 3, 4), 'float32')),
                dict(cube=array_ref('subset', (2, 3, 3), 'float32')),
                dict(spectral_axis=replace(axis(), channel_indices=(3, 2, 1, 0))),
                dict(valid_samples=array_ref('badmask', (2, 3), 'bool')),
                dict(valid_samples=array_ref('badmask', (2, 3, 4), 'uint8'))):
            with self.subTest(altered=altered), self.assertRaises(ContractError):
                replace(cube, **altered)
        values = np.ones(cube.cube.shape, dtype=np.float32)
        values[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ContractError, 'Nonfinite'):
            cube.validate_arrays(values, np.ones(values.shape, dtype=bool), np.ones(4, dtype=bool))
        with self.assertRaisesRegex(ContractError, 'dtype'):
            cube.validate_arrays(values.astype(np.float64), np.isfinite(values), np.ones(4, dtype=bool))

    def test_excluded_channels_stay_in_cube_but_cannot_be_valid(self):
        cube = prepared()
        values = np.ones(cube.cube.shape, dtype=np.float32)
        mask = np.ones(values.shape, dtype=bool)
        quality = np.array([True, False, True, True])
        with self.assertRaises(ContractError):
            cube.validate_arrays(values, mask, quality)
        mask[..., 1] = False
        cube.validate_arrays(values, mask, quality)
        self.assertEqual(values.shape[-1], 4)
        np.testing.assert_array_equal(values, 1)

    def test_detector_overlap_and_missing_channels_are_metadata_boundaries(self):
        grid = SpectralAxis((0, 1, 2, 3), (1000, 1006, 1000, 1020),
                            ('S', 'S', 'L', 'L'), (0, 0, 1, 2))
        self.assertEqual(grid.allowed_edges, (True, False, False))
        for altered in (dict(segment_ids=(0, 0, 0, 1)), dict(segment_ids=(0, 0, 1, 1)),
                        dict(wavelengths_nm=(1000, 999, 1000, 1020)),
                        dict(wavelengths_nm=(1, 1.006, 1, 1.02)), dict(unit='um'),
                        dict(detectors=('S', 'S', 'L', 'unknown')),
                        dict(channel_indices=(0, 0, 2, 3)), dict(segment_ids=(0, 1, 0, 2))):
            with self.subTest(altered=altered), self.assertRaises(ContractError):
                replace(grid, **altered)

    def test_missing_metadata_stays_incomplete_and_cannot_claim_readiness(self):
        incomplete = SpectralAxis((0, 1, 2, 3), None, None, None, 'incomplete')
        cube = replace(prepared(), spectral_axis=incomplete)
        self.assertIsNone(contract_dict(cube)['spectral_axis']['wavelengths_nm'])
        with self.assertRaises(ContractError):
            cube.require_ready()
        with self.assertRaises(ContractError):
            _ = incomplete.allowed_edges
        with self.assertRaises(ContractError):
            replace(incomplete, metadata_status='verified')

    def test_references_and_metadata_are_immutable_and_json_serializable(self):
        mutable = [0, 1, 2, 3]
        grid = replace(axis(), channel_indices=mutable)
        mutable.reverse()
        self.assertEqual(grid.channel_indices, (0, 1, 2, 3))
        with self.assertRaises(FrozenInstanceError):
            grid.unit = 'um'
        for changes in (dict(sha256='not-a-hash'), dict(path='<weights>')):
            with self.assertRaises(ContractError):
                replace(artifact('weights'), **changes)
        for dtype in ('object', 'complex64', 'invalid'):
            with self.assertRaises(ContractError):
                array_ref('bad', (2, 3), dtype)


class TaxonomyContracts(unittest.TestCase):
    def test_gt_zero_is_unlabelled_and_local_ids_do_not_become_model_ids(self):
        labels = np.array([[0, 7, 3]], dtype=np.uint8)
        converted, labeled = map_ground_truth(labels, {7: 0, 3: 1}, TAXONOMY)
        np.testing.assert_array_equal(converted, [[-1, 0, 1]])
        np.testing.assert_array_equal(labeled, [[False, True, True]])
        np.testing.assert_array_equal(labels, [[0, 7, 3]])
        for mapping in ({0: 0, 7: 0, 3: 1}, {7: 0}, {7: 2, 3: 1}):
            with self.assertRaises(ContractError):
                map_ground_truth(labels, mapping, TAXONOMY)

    def test_model_class_mapping_and_reference_mapping_have_distinct_sizes(self):
        model = ModelBundle('model', 'lwnet', 'v1', artifact('weights'), artifact('preprocess'),
                            artifact('training'), 'corrected_if', 'fixed_grid', axis(), TAXONOMY, (1, 0))
        self.assertEqual(model.output_class_ids, (1, 0))
        for changed in (dict(weights=None), dict(output_class_ids=(1, 2)),
                        dict(output_class_ids=(1, 1)), dict(taxonomy=None)):
            with self.subTest(changed=changed), self.assertRaises(ContractError):
                replace(model, **changed)
        library = ReferenceLibrary('mica', array_ref('spectra', (3, 4), 'float32'),
                                   array_ref('valid_refs', (3, 4), 'bool'), axis(), TAXONOMY,
                                   (1, 0, 1), 'I/F', 'corrected_if')
        self.assertEqual(len(library.class_ids), 3)  # two classes, three references
        with self.assertRaises(ContractError):
            replace(library, class_ids=(0, 1))
        with self.assertRaises(ContractError):
            replace(model, algorithm='n2n4m')
        denoiser = replace(model, algorithm='n2n4m', taxonomy=None, output_class_ids=(),
                           input_layout='segmented', spectral_axis=None)
        self.assertEqual(denoiser.input_layout, 'segmented')


class OutputContracts(unittest.TestCase):
    def test_denoising_preserves_shape_missingness_and_unprocessed_values(self):
        cube = prepared()
        result = DenoisingResult(cube, artifact('denoiser'), array_ref('denoised', cube.cube.shape, 'float32'),
                                 cube.valid_samples, array_ref('changed', cube.cube.shape, 'bool'))
        before = np.ones(cube.cube.shape, dtype=np.float32)
        before[0, 0, :] = np.nan
        valid = np.isfinite(before)
        changed = valid.copy()
        changed[..., 0] = False
        after = np.where(changed, before * 2, before)
        quality = np.ones(4, dtype=bool)
        result.validate_arrays(before, valid, after, valid, changed, good_channels=quality)
        bad = after.copy()
        bad[1, 0, 0] = 100  # unsupported first channel must be copied exactly
        with self.assertRaises(ContractError):
            result.validate_arrays(before, valid, bad, valid, changed, good_channels=quality)
        with self.assertRaises(ContractError):
            result.validate_arrays(before, valid, after, np.ones_like(valid), changed, good_channels=quality)
        with self.assertRaises(ContractError):
            result.validate_arrays(before, valid, after, valid, np.ones_like(changed), good_channels=quality)
        with self.assertRaises(ContractError):
            result.validate_arrays(before, valid, before, valid, np.zeros_like(changed), good_channels=quality)
        with self.assertRaises(ContractError):
            replace(result, cube=array_ref('wrong', (2, 3, 3), 'float32'))

    def test_classification_rejects_empty_pixels_unknown_ids_and_mask_disagreement(self):
        cube = prepared()
        result = ClassificationResult('lwnet_original', cube.cube, artifact('classifier'), TAXONOMY,
                                      array_ref('classes', (2, 3), 'int32'),
                                      array_ref('predicted', (2, 3), 'bool'),
                                      array_ref('reasons', (2, 3), 'uint8'),
                                      array_ref('channels', (4,), 'bool'))
        labels = np.zeros((2, 3), dtype=np.int32)
        labels[0, 0] = -1
        predicted = labels >= 0
        reasons = np.where(predicted, 0, 1).astype(np.uint8)
        measured = np.ones(cube.cube.shape, dtype=bool)
        measured[0, 0] = False
        channels = np.ones(4, dtype=bool)
        result.validate_arrays(labels, predicted, reasons, channels, measured)
        for target in (0, 2, -2):
            bad = labels.copy()
            bad[0, 0] = target
            with self.assertRaises(ContractError):
                result.validate_arrays(bad, bad >= 0, np.where(bad >= 0, 0, 1).astype(np.uint8), channels, measured)
        with self.assertRaises(ContractError):
            result.validate_arrays(labels, ~predicted, reasons, channels, measured)
        with self.assertRaises(ContractError):
            result.validate_arrays(labels, predicted, np.zeros_like(reasons), channels, measured)
        with self.assertRaises(ContractError):
            result.validate_arrays(labels, predicted, reasons, np.zeros_like(channels), measured)

    def test_reports_keep_four_states_and_cannot_invent_gt_metrics(self):
        branches = tuple(BranchOutcome(key, 'skipped', reason_code='no_resource') for key in BRANCHES)
        report = ComparisonResult('run', 'analysis', TAXONOMY, artifact('evaluation'), branches,
                                  artifact('coverage'), not_evaluable_reason='ground_truth_missing')
        self.assertEqual(len(contract_dict(report)['branches']), 4)
        for changes in (dict(branches=branches[:-1]), dict(metrics_gt=artifact('metrics')),
                        dict(not_evaluable_reason=None)):
            with self.assertRaises(ContractError):
                replace(report, **changes)
        with self.assertRaises(ContractError):
            BranchOutcome('sam_original', 'completed')
        with self.assertRaises(ContractError):
            BranchOutcome('sam_original', 'skipped')
        visual = VisualizationResult('run', 'analysis', (artifact('map'), artifact('chart')))
        self.assertEqual(len(contract_dict(visual)['artifacts']), 2)
        with self.assertRaises(ContractError):
            replace(visual, artifacts=(artifact('map'), artifact('map')))


if __name__ == '__main__':
    unittest.main()
