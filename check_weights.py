"""Verify frozen weights, then run a small CPU forward pass for each model."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import run_experiment as experiment
from infer import profile
from modules.cube_adapters.backends import load_backend
from modules.cube_adapters.pretrained_n2n4m import load_pretrained_plan
from modules.cube_adapters.salinas import _model
from modules.pipeline_config import ExecutionSpec
from modules.pipeline_contracts import ArtifactRef, ModelBundle, SpectralAxis
from modules.resource_store import ResourceStore, decode_contract


def standard(axis):
    cfg = profile("standard", "hymars_hc")
    view = SimpleNamespace(prepared=SimpleNamespace(units="dimensionless",
        domain="hymars_photometric_atmospheric_corrected_if", spectral_axis=axis))
    spec = ExecutionSpec()
    for name in experiment.MODELS:
        ref = decode_contract(ArtifactRef, cfg["resources"][name + "_bundle"])
        store = ResourceStore(experiment.WORK)
        bundle, path = store.load_contract(ref, ModelBundle)
        store.model_bundle(ref, algorithm=name)
        if name == "lwnet":
            model = _model(path.parent / bundle.weights.path, spec)
            result = model(torch.zeros(1, 1, 204, 19, 19))
            if result.shape != (1, 16) or not result.isfinite().all():
                raise ValueError("Invalid Salinas prediction")
        else:
            plan = load_pretrained_plan(view, store, ref, bundle, path)
            predict = load_backend(plan, view, spec, native_channels=350)
            result = predict(np.zeros((1, 350), dtype=np.float32), None)
            if result.shape != (1, 350) or not np.isfinite(result).all():
                raise ValueError("Invalid N2N4M prediction")
        print(f"standard/{name}: CPU forward OK")


def frozen(root, scene_axis):
    training = experiment.read(root / "training.json")
    result = experiment.read(root / "result.json")
    for key in ("checkpoint", "last_checkpoint", "export"):
        if key in result and experiment.digest(root / result[key]["path"]) != result[key]["sha256"]:
            raise ValueError(f"{root}: {key} checksum mismatch")
    model = torch.jit.load(str(root / "model.ts"), map_location="cpu").eval()
    for key, epoch in (("checkpoint", result["best_epoch"]), ("last_checkpoint", result["epochs"])):
        if key not in result:
            continue
        checkpoint = torch.load(root / result[key]["path"], map_location="cpu", weights_only=True)
        if checkpoint["epoch"] != epoch:
            raise ValueError(f"{root}: {key} epoch mismatch")
        if key == "checkpoint":
            for name, value in model.state_dict().items():
                if not torch.equal(checkpoint["model_state"][name], value):
                    raise ValueError(f"{root}: TorchScript differs from best at {name}")
    if training["stage"] == "plebani350":
        wave = experiment.read(experiment.WORK / "finetuning/configs/plebani350_axis.json")["wavelengths_nm"]
        waves = torch.tensor(wave)
        detectors = torch.zeros(len(waves), dtype=torch.long)
        edges = (waves.diff() > 0) & (waves.diff() <= 7.5)
    else:
        waves = torch.tensor(scene_axis.wavelengths_nm)
        detectors = torch.tensor([int(d == "S") for d in scene_axis.detectors])
        edges = torch.tensor(scene_axis.allowed_edges)
    channels = len(waves)
    spatial = training["model"] == "lwnet" and training["stage"] == "hymars_full"
    patch = training["parameters"]["patch_size"] if spatial else None
    shape = (1, 1, channels, patch, patch) if spatial else (1, channels)
    features = torch.zeros(shape)
    prediction = model(features, torch.ones_like(features, dtype=torch.bool), waves, detectors, edges)
    width = channels if training["model"] == "n2n4m" else len(training["class_names"])
    if prediction.shape != (1, width) or not prediction.isfinite().all():
        raise ValueError(f"{root}: invalid CPU prediction")
    print(f"{root.relative_to(experiment.HERE)}: epoch {result['best_epoch']}, CPU forward OK")


if __name__ == "__main__":
    torch.set_num_threads(2)
    configuration = experiment.base_config("hymars_hc")
    scene = experiment.read(Path(configuration["scene_manifest"]["path"]))
    axis = decode_contract(SpectralAxis, scene["spectral_axis"])
    with torch.inference_mode():
        standard(axis)
        for version in ("25", "50"):
            for name in experiment.MODELS:
                for stage in experiment.STAGES:
                    frozen(experiment.HERE / "weights" / version / name / stage, axis)
