"""Native pre-finetuning coverage and radiometry on a small sparse cube."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from modules.cube_adapters.backends import NeuralPlan
from modules.cube_adapters.common import open_prepared
from modules.cube_adapters.pretrained_n2n4m import denoise_native_grid
from modules.data_preparation import SceneManifest, prepare_scene
from modules.pipeline_config import ExecutionSpec, N2N4MSpec
from modules.pipeline_contracts import ArtifactRef, ModelBundle, SpectralAxis, contract_dict
from modules.resource_store import decode_contract, read_json, sha256_file
from modules.run_store import write_manifest


class NativePlebaniTests(TestCase):
    def test_original_positions_mask_and_physical_output(self):
        resources = Path(__file__).resolve().parents[1] / 'configs/models/n2n4m_pretrained'
        bundle = decode_contract(ModelBundle, read_json(resources / 'bundle.json'))
        parameters = read_json(resources / 'preprocessing.json')['parameters']
        native = np.array(parameters['native_wavelengths_nm'])
        # Duplicate wavelength on S, one missing native wavelength, off-grid
        # L wavelength, one globally bad channel and partially measured pixels.
        keep = np.delete(np.arange(182), 8)
        waves = np.r_[native[0], native[keep], 4100.]
        detectors = ('S',) + ('L',) * (len(waves) - 1)
        segments = np.cumsum(np.r_[False, (np.diff(waves) > 7.5) | (np.diff(waves) <= 0)])
        axis = SpectralAxis(tuple(range(len(waves))), tuple(waves), detectors, tuple(map(int, segments)))
        quality = [True] * len(waves)
        quality[2] = False
        values = np.full((1, 3, len(waves)), .25, dtype=np.float32)
        values[0, 0, 3:5] = [np.nan, 1.1]
        values[0, 1, 10:30] = np.nan
        values[0, 2] = np.nan
        temporary_root = resources.parents[2] / 'output' / 'adapter_tests_tmp'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temporary_root) as directory:
            work = Path(directory)
            source = work / 'source.npy'
            np.save(source, values)
            ref = ArtifactRef('test_data', str(source), sha256_file(source))
            scene = SceneManifest(1, 'fixture', 'npy', ref, values.shape, values.dtype.str, None, None,
                                  axis, tuple(quality), (), 'dimensionless', parameters['source_domains'][0],
                                  (), None, None)
            manifest = work / 'scene.json'
            write_manifest(manifest, contract_dict(scene))
            archive = prepare_scene(manifest, work / 'prepared')
            bundle_path = resources / 'bundle.json'
            plan = NeuralPlan(bundle, ArtifactRef('original', str(bundle_path), sha256_file(bundle_path)),
                              bundle_path, parameters, np.full(350, .2), np.full(350, .02))
            seen = []

            def predict(features, mask):
                seen.append(features)
                return np.full_like(features, .125)

            with open_prepared(archive) as view, patch(
                    'modules.cube_adapters.pretrained_n2n4m.load_backend', return_value=predict):
                denoise_native_grid(view, plan, work / 'out', N2N4MSpec(2), ExecutionSpec())
                output = np.load(work / 'out/cube.npy')
                changed = np.load(work / 'out/denoised_mask.npy')
                np.testing.assert_array_equal(output[~changed], view.cube[~changed])
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0].shape, (1, 350))
            self.assertEqual(np.count_nonzero(seen[0]), 178)
            np.testing.assert_array_equal(seen[0][0, [1, 2, 3, 8, 349]], 0)
            np.testing.assert_allclose(seen[0][0, [0, 4, 9]], 2.5)
            np.testing.assert_array_equal(output[changed], .125)
            self.assertFalse(changed[0, 1:].any())
            self.assertFalse(changed[0, 0, [0, 2, 3, 4, -1]].any())
            self.assertEqual(changed.sum(), 178)
            recorded = read_json(work / 'out/manifest.json')['parameters']
            self.assertEqual(recorded['internally_imputed_inputs'], 172)
            self.assertEqual(recorded['measured_native_count_histogram'][178], 1)
