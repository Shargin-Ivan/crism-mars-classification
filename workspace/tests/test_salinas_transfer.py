from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from modules.cube_adapters.common import open_prepared
from modules.cube_adapters.neural import classify_lwnet
from modules.cube_adapters.salinas import match_channels
from modules.data_preparation import SceneManifest, prepare_scene
from modules.pipeline_config import ExecutionSpec, LWNetSpec
from modules.pipeline_contracts import ArtifactRef, ModelBundle, RejectionReason, SpectralAxis, contract_dict
from modules.resource_store import decode_contract, read_json, sha256_file
from modules.run_store import write_manifest

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / 'configs/models/lwnet_salinas'


class SalinasTransferTests(TestCase):
    def test_nearest_measured_band_no_extrapolation_or_wide_gap(self):
        indices = match_channels([450, 455, 460, 480], [True, False, True, True],
                                 [448, 450, 455.1, 470, 482], 5)
        np.testing.assert_array_equal(indices, [-1, 0, 2, -1, -1])

    def test_native_terrestrial_outputs_on_original_and_denoised(self):
        # Test the public API and unmodified published weights on a small sparse scene.
        import torch
        profile = read_json(ROOT / 'configs/scenes/hymars_hc.json')
        axis = decode_contract(SpectralAxis, profile['spectral_axis'])
        values = np.full((2, 2, len(axis.channel_indices)), np.nan, dtype=np.float32)
        values[0, 0] = np.linspace(.05, .3, values.shape[2])
        values[1, 0, :10] = .15
        root = ROOT / 'output/adapter_tests_tmp'
        root.mkdir(exist_ok=True, parents=True)
        with TemporaryDirectory(dir=root) as tmp:
            work = Path(tmp)
            data = work / 'data.npy'
            np.save(data, values)
            ref = ArtifactRef('test_data', str(data), sha256_file(data))
            scene = SceneManifest(1, 'fixture', 'npy', ref, values.shape, values.dtype.str, None, None,
                                  axis, tuple([True] * values.shape[2]), (), 'dimensionless',
                                  profile['domain'], (), None, None)
            manifest = work / 'scene.json'
            write_manifest(manifest, contract_dict(scene))
            archive = prepare_scene(manifest, work / 'prepared')
            bundle = decode_contract(ModelBundle, read_json(RESOURCES / 'bundle.json'))
            resource = ArtifactRef('original', 'bundle.json', sha256_file(RESOURCES / 'bundle.json'))
            device = ExecutionSpec('cuda', True) if torch.cuda.is_available() else ExecutionSpec()
            with open_prepared(archive) as view:
                for stage, patch in [('prepared', 19), ('denoised', 27)]:
                    branch = 'lwnet_original' if stage == 'prepared' else 'lwnet_denoised'
                    result = classify_lwnet(replace(view, stage=stage), resource, RESOURCES, bundle.taxonomy,
                                            work / branch, spec=LWNetSpec(patch, 2), execution=device, branch_id=branch)
                    labels = np.load(work / branch / 'class_map.npy')
                    why = np.load(work / branch / 'rejection_reason.npy')
                    self.assertEqual(result.taxonomy.taxonomy_id, 'salinas_aviris_land_cover')
                    self.assertTrue(0 <= labels[0, 0] < 16)
                    self.assertEqual(why[1, 0], RejectionReason.INSUFFICIENT_COVERAGE)
                    self.assertEqual(why[0, 1], RejectionReason.EMPTY_PIXEL)
                    self.assertEqual(np.sum(labels >= 0), 1)
