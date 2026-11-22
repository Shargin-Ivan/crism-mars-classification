"""Собрать переносимые результаты из проверенных выгрузок двух экспериментов."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workspace"))
from modules.class_names_ru import to_ru

BRANCHES = {
    "sam_original": "SAM",
    "sam_denoised": "N2N4M → SAM",
    "lwnet_original": "LWNet",
    "lwnet_denoised": "N2N4M → LWNet",
}
SCENES = {
    "hymars_hc": "Кратер Холден",
    "frt00019468_mtrdr": "FRT00019468",
    "hrl000120cd_mtrdr": "HRL000120CD",
}
PHASES = (
    ("main", "before", "baseline", "Исходные веса"),
    ("main", "after", "epochs25", "25 + 25 эпох"),
    ("epochs50", "after", "epochs50", "50 + 50 эпох"),
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def palette(taxonomy_id, length):
    # Фиксированные цвета по номеру класса, в том числе для разных сцен.
    names = ("tab20", "tab20b", "tab20c") if taxonomy_id == "finetuning_classes" else ("tab20c",)
    colors = [to_hex(rgb) for name in names for rgb in matplotlib.colormaps[name].colors]
    if length > len(colors):
        raise ValueError(f"Недостаточно цветов для {taxonomy_id}")
    return colors[:length]


def render(path, title, arrays, taxonomies, branch_taxonomy):
    fig = plt.figure(figsize=(16.4, 10.8), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.24])
    fig.suptitle(title, fontsize=19, weight="bold")
    legend_items = {}
    for i, (branch, name) in enumerate(BRANCHES.items()):
        ax = fig.add_subplot(grid[i // 2, i % 2])
        tid = branch_taxonomy[branch]
        classes = taxonomies[tid]
        values = arrays[branch + "_class_map"]
        valid = arrays[branch + "_prediction_valid_mask"]
        ids = np.unique(values[valid])
        if np.any((ids < 0) | (ids >= len(classes))):
            raise ValueError(f"Неверный номер класса в {branch}")
        legend_items.setdefault(tid, set()).update(ids.tolist())
        cmap = ListedColormap([c["color"] for c in classes])
        cmap.set_bad("#212121")
        ax.imshow(np.ma.masked_where(~valid, values), cmap=cmap, vmin=-0.5,
                  vmax=len(classes)-0.5, interpolation="nearest")
        ax.set_title(name + (" · Salinas" if tid == "salinas_aviris_land_cover" else ""), fontsize=13)
        ax.set_axis_off()
    legend = fig.add_subplot(grid[:, 2])
    legend.axis("off")
    handles = []
    for tid, ids in legend_items.items():
        heading = "Минеральные классы CRISM" if tid == "finetuning_classes" else "Земные классы Salinas"
        handles.append(Patch(facecolor="none", edgecolor="none", label=heading))
        for idx in sorted(ids):
            item = taxonomies[tid][idx]
            text = fill(f"{idx}  {item['ru']}", width=36)
            handles.append(Patch(facecolor=item["color"], label=text))
    handles.append(Patch(facecolor="#212121", label="Нет предсказания"))
    ncols = 2 if len(handles) > 29 else 1
    legend.legend(handles=handles, loc="center", ncols=ncols, frameon=False,
                  fontsize=8 if ncols == 2 else 10, handlelength=1.0, columnspacing=.9)
    fig.savefig(path, dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def copy_class_table(source, dest):
    with source.open(encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        fields = [*rows.fieldnames, "class_name_ru"]
        with dest.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "class_name_ru": to_ru(row["class_name"])})


def render_ground_truth(path, arrays, classes):
    labels, region = arrays["gt_labels"], arrays["gt_evaluation_mask"]
    cmap = ListedColormap([entry["color"] for entry in classes])
    cmap.set_bad("#e5e5e5")
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    ax.imshow(np.ma.masked_where(~region, labels), cmap=cmap, vmin=-.5,
              vmax=len(classes)-.5, interpolation="nearest")
    handles = []
    for value, count in zip(*np.unique(labels[region], return_counts=True)):
        item = classes[int(value)]
        handles.append(Patch(facecolor=item["color"], label=f"{item['ru']} ({count})"))
    handles.append(Patch(facecolor="#e5e5e5", label="Вне области оценки"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, .5), frameon=False)
    ax.set_title("Кратер Холден · разметка HyMars")
    ax.set_axis_off()
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def package_scene(root, step, out, title):
    out.mkdir(parents=True, exist_ok=True)
    run, analysis = root / step["run"], root / step["analysis"]
    manifest = read(run / "manifest.json")
    arrays, taxonomies, branch_taxonomy, map_hashes = {}, {}, {}, {}
    for branch in BRANCHES:
        record = manifest["branches"][branch]
        if record["status"] != "completed":
            raise ValueError(f"Незавершённая ветвь {branch}")
        descriptor = run / record["result"]["path"]
        detail = read(descriptor)["result"]
        tax = detail["taxonomy"]
        tid = tax["taxonomy_id"]
        colors = palette(tid, len(tax["class_names"]))
        classes = [dict(id=i, source=name, ru=to_ru(name), color=colors[i])
                   for i, name in enumerate(tax["class_names"])]
        if tid in taxonomies and taxonomies[tid] != classes:
            raise ValueError("Расхождение таксономий")
        taxonomies[tid] = classes
        branch_taxonomy[branch] = tid
        for field in ("class_map", "prediction_valid_mask", "rejection_reason", "used_channels"):
            ref = detail[field]
            src = descriptor.parent / ref["path"]
            with src.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != ref["sha256"]:
                    raise ValueError(f"Расходится SHA-256: {src}")
            arrays[branch + "_" + field] = np.load(src, allow_pickle=False)
        map_hashes[branch] = detail["class_map"]["sha256"]
    arrays["valid_pixels"] = np.load(analysis / "valid_pixels.npy", allow_pickle=False)
    if (analysis / "gt_labels.npy").is_file():
        for name in ("gt_labels", "gt_evaluation_mask"):
            arrays[name] = np.load(analysis / (name + ".npy"), allow_pickle=False)
        copy_class_table(analysis / "per_class_gt.csv", out / "per_class_gt.csv")
        if out.parent.name == "baseline":
            render_ground_truth(out / "ground_truth.png", arrays, taxonomies["finetuning_classes"])
    np.savez_compressed(out / "maps.npz", **arrays)
    save_json(out / "classes.json", dict(run_id=manifest["run_id"], branch_taxonomy=branch_taxonomy,
                                         taxonomies=taxonomies, original_npy_sha256=map_hashes))
    shutil.copyfile(analysis / "summary.csv", out / "metrics.csv")
    render(out / "maps.png", title, arrays, taxonomies, branch_taxonomy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "full_results")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    times = []
    for campaign in ("main", "epochs50"):
        state = read(args.archive / campaign / "campaign.json")
        if state["status"] != "completed":
            raise ValueError(campaign)
        times += [dict(campaign=campaign, step=step, seconds=rec["seconds"])
                  for step, rec in state["steps"].items()]
    with (args.out / "timing.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["campaign", "step", "seconds"])
        writer.writeheader()
        writer.writerows(times)
    shutil.copyfile(args.archive / "epochs50/comparison.csv", args.out / "comparison.csv")
    for campaign, phase, folder, label in PHASES:
        root = args.archive / campaign
        steps = read(root / "campaign.json")["steps"]
        for scene, title in SCENES.items():
            package_scene(root, steps[f"{phase}/{scene}"], args.out / folder / scene,
                          f"{title} · {label}")
            print(folder, scene, flush=True)


if __name__ == "__main__":
    main()
