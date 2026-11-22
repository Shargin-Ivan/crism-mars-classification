"""Exercise the HC inference axis using calibration metadata and synthetic pixels."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import test_sam_adapter as sam_fixtures
from finetuning.export import export_bundle
from finetuning.io import digest, read_json, write_json
from modules.comparison_metrics import gt_scores
from modules.cube_adapters.common import open_denoised
from modules.cube_adapters.neural import classify_lwnet, denoise_n2n4m
from modules.cube_adapters.sam import classify_sam
from modules.pipeline_config import LWNetSpec
from modules.pipeline_contracts import ArtifactRef, SpectralAxis, Taxonomy
from modules.resource_store import decode_contract


class HCFullAxisInferenceTests(unittest.TestCase):
    def test_measured_hc_only_channels_reach_all_four_branches(self):
        fixture = sam_fixtures.SamAdapterTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        root = fixture.root
        research = Path(__file__).resolve().parents[1] / 'finetuning/configs/research'

        def features(scene):
            axis = read_json(research / f'hymars_{scene}_crism_v1.json')['spectral_axis']
            return list(zip(axis['wavelengths_nm'], axis['detectors']))

        support = set(features('nf')) | set(features('up'))
        hc = features('hc')
        unseen = np.array([band not in support for band in hc])
        self.assertEqual((len(hc), unseen.sum()), (440, 39))
        target = read_json(research / 'hymars_hc_crism_v1.json')
        # Use the radiometry of the synthetic SAM fixture with the native HC grid.
        target.update(units='I/F', domain='corrected_if')
        target_path = root / 'target.json'
        write_json(target_path, target)

        def exported(model):
            run = root / model
            run.mkdir()
            (run / 'model.ts').write_bytes(b'deterministic mock backend')
            normalization = dict(mean=0., scale=1.)
            write_json(run / 'normalization.json', normalization)
            write_json(run / 'result.json', dict(status='completed', export=dict(
                path='model.ts', sha256=digest(run / 'model.ts'))))
            write_json(run / 'training.json', dict(model=model, stage='hymars_full',
                       units=target['units'], domain=target['domain'], normalization=normalization,
                       parameters={'patch_size': 13}, class_names=['original', 'denoised'],
                       feature_support=[dict(wavelength_nm=w, detector=d) for w, d in sorted(support)]))
            path = export_bundle(run, target_path)
            return ArtifactRef(model, str(path), digest(path))

        n2n, lwnet = exported('n2n4m'), exported('lwnet')
        fixture.taxonomy = decode_contract(Taxonomy, read_json(lwnet.path)['taxonomy'])
        axis = SpectralAxis(tuple(range(440)), tuple(w for w, _ in hc), tuple(d for _, d in hc),
                            tuple(target['spectral_axis']['segment_ids']))
        spectrum = np.ones(440, dtype=np.float32)
        spectrum[unseen] = np.linspace(2., 4., 39)
        transformed = spectrum.copy()
        transformed[unseen] *= 2
        cube = np.tile(spectrum, (13, 13, 1))
        cube[0, 0] = np.nan
        cube[0, 1, np.flatnonzero(unseen)[0]] = np.nan
        original = fixture.view(cube, axis)
        library = fixture.library([spectrum, transformed], axis, [0, 1])
        valid = np.isfinite(cube)
        wanted = cube.copy()
        wanted[..., unseen] *= 2

        def denoiser(values, masks):
            selected = valid.any(axis=-1)
            np.testing.assert_array_equal(masks, valid[selected])
            np.testing.assert_array_equal(values, np.where(valid, cube, 0)[selected])
            result = values.copy()
            result[:, unseen] *= 2
            return result

        def check_classifications(view, suffix, expected_cube, label):
            def classifier(values, masks):
                # The 13x13 fixture has one eligible patch, including the partially missing pixel.
                np.testing.assert_array_equal(masks[0, 0], valid.transpose(2, 0, 1))
                np.testing.assert_array_equal(values[0, 0], np.where(valid, expected_cube, 0).transpose(2, 0, 1))
                return np.log([[.9, .1] if values[0, 0, unseen, 6, 6].mean() < 4 else [.1, .9]])

            sam_dir = root / f'sam_{suffix}'
            lw_dir = root / f'lwnet_{suffix}'
            sam = classify_sam(view, library, root, fixture.taxonomy, sam_dir, branch_id=sam_dir.name)
            with patch('modules.cube_adapters.neural.load_backend', return_value=classifier):
                lw = classify_lwnet(view, lwnet, root, fixture.taxonomy, lw_dir,
                                    branch_id=lw_dir.name, spec=LWNetSpec(13))
            self.assertEqual(read_json(lw_dir / 'manifest.json')['parameters']['model']
                             ['stage2_unseen_channel_indices'], np.flatnonzero(unseen).tolist())
            for result, folder, accepted in ((sam, sam_dir, 168), (lw, lw_dir, 1)):
                labels = np.load(folder / result.class_map.path)
                gt = np.full(labels.shape, label)
                roi = valid.any(axis=-1)
                scores = gt_scores(gt, labels, roi, 2)
                self.assertEqual(labels[6, 6], label)
                self.assertEqual(labels[0, 0], -1)
                self.assertEqual(scores['confusion'][label][label], accepted)
                self.assertEqual(scores['confusion'][label][-1], roi.sum() - accepted)

        check_classifications(original, 'original', cube, 0)
        with patch('modules.cube_adapters.neural.load_backend', return_value=denoiser):
            denoise_n2n4m(original, n2n, root, root / 'processed')
        self.assertEqual(read_json(root / 'processed/manifest.json')['parameters']['model']
                         ['stage2_unseen_channel_indices'], np.flatnonzero(unseen).tolist())
        with open_denoised(original, root / 'processed') as after:
            np.testing.assert_array_equal(after.valid_samples, valid)
            np.testing.assert_array_equal(after.cube, wanted)
            check_classifications(after, 'denoised', wanted, 1)
