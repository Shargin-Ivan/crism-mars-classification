"""Run the four branches with frozen standard, 25+25 or 50+50 weights."""
import argparse
from pathlib import Path
import time

import run_experiment as experiment
from modules.pipeline_config import config_from_dict
from modules.pipeline_runner import run_pipeline
from modules.run_store import read_run_manifest
from modules.resource_store import ResourceStore


def profile(version, scene):
    config = experiment.base_config(scene)
    if version == "standard":
        return config
    scene_id = experiment.read(Path(config["scene_manifest"]["path"]))["scene_id"]
    for model in experiment.MODELS:
        folder = experiment.HERE / "weights" / version / model / "hymars_full" / "bundles"
        matches = []
        for path in sorted(folder.glob("*/bundle.json")):
            preprocessing = experiment.read(path.parent / "preprocessing.json")
            if preprocessing["parameters"]["scene_binding"]["target_scene_id"] == scene_id:
                matches.append(path)
        if len(matches) != 1:
            raise ValueError(f"Expected one {model} bundle for {scene_id}, found {len(matches)}")
        path = matches[0]
        config["resources"][model + "_bundle"] = {
            "artifact_id": experiment.read(path)["bundle_id"],
            "path": str(path),
            "sha256": experiment.digest(path),
        }
    config["n2n4m"]["batch_size"] = 128
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, choices=("standard", "25", "50"))
    parser.add_argument("--scene", required=True, choices=experiment.SCENES)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--lwnet-batch", type=int)
    parser.add_argument("--n2n4m-batch", type=int)
    parser.add_argument("--check", action="store_true", help="Verify model references without classifying the cube")
    args = parser.parse_args()
    config = profile(args.weights, args.scene)
    config["execution"].update(device=args.device, gpu_allowed=args.device != "cpu")
    for name, batch in (("n2n4m", args.n2n4m_batch), ("lwnet", args.lwnet_batch)):
        if batch is not None:
            if batch < 1:
                parser.error("Batch size must be positive")
            config[name]["batch_size"] = batch
    root = experiment.HERE / "outputs" / args.weights / args.scene
    config["stores"]["results_root"] = str(root)
    path = root / "profiles" / f"{time.time_ns()}.json"
    ready = config_from_dict(config, require_ready=True)
    store = ResourceStore(path.parent)
    for model in experiment.MODELS:
        store.model_bundle(getattr(ready.resources, model + "_bundle"), algorithm=model)
    if args.check:
        print(f"{args.weights} / {args.scene}: model bindings verified")
        return
    experiment.save(path, config)
    run = run_pipeline(path)
    manifest = read_run_manifest(run, verify_artifacts=True)
    if manifest["status"] != "completed" or any(
        manifest["branches"][name]["status"] != "completed" for name in experiment.BRANCHES
    ):
        raise RuntimeError(f"Inspect branch statuses in {run / 'execution.json'}")
    from analysis.whole_cube import analyze_run
    from analysis.visualize_comparison import render_analysis
    report = render_analysis(analyze_run(run)) / "report.html"
    print(report)


if __name__ == "__main__":
    main()
