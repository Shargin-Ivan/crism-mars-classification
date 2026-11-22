"""Small offline checks for data delivery, frozen profiles and campaign comparison."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import data_parts
import frozen_reference
import run_experiment as experiment
from infer import profile
from modules.pipeline_contracts import ArtifactRef
from modules.resource_store import ResourceStore, decode_contract


class ReleaseTests(unittest.TestCase):
    def test_data_parts_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, parts = root / "workspace", root / "parts"
            work.mkdir()
            data = b"science"
            names = ["cube.img", "another/cube.img"]
            rows = [{"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name in names]
            (work / "input_inventory.json").write_text(json.dumps({"files": rows}))
            (root / "large_inputs.json").write_text(json.dumps(names))
            for name in names:
                (work / name).parent.mkdir(exist_ok=True)
                (work / name).write_bytes(data)
            with patch.multiple(data_parts, ROOT=root, WORK=work, PART_BYTES=4):
                data_parts.transfer("pack", parts)
                self.assertEqual(len(list(parts.iterdir())), 2)
                for name in names:
                    (work / name).unlink()
                data_parts.transfer("restore", parts)
                for name in names:
                    self.assertEqual((work / name).read_bytes(), data)
                    (work / name).unlink()
                next(parts.glob("*.000")).write_bytes(b"xxxx")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    data_parts.transfer("restore", parts)
                self.assertFalse((work / names[0]).exists())

    def test_nine_frozen_profiles(self):
        store = ResourceStore(experiment.HERE)
        for version in ("standard", "25", "50"):
            for scene in experiment.SCENES:
                cfg = profile(version, scene)
                for model in experiment.MODELS:
                    reference = decode_contract(ArtifactRef, cfg["resources"][model + "_bundle"])
                    bundle = store.model_bundle(reference, algorithm=model)
                    self.assertEqual(bundle.algorithm, model)

    def test_frozen_reference(self):
        steps = frozen_reference.verified_steps(experiment.HERE / "reference25")
        self.assertEqual(set(steps), {f"{phase}/{scene}" for phase in ("before", "after") for scene in experiment.SCENES})


if __name__ == "__main__":
    unittest.main()
