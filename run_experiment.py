"""Run the sealed H100 experiment. All persistent paths are inside this folder."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from package_workspace import digest, read, verify
from gpu_profile import PROFILE, inference_batches


HERE = Path(__file__).resolve().parent
WORK = HERE / 'workspace'
sys.path.insert(0, str(WORK))
CAMPAIGN = WORK / 'finetuning/storage' / ('h100_campaign' if PROFILE == 'h100' else 'h100_campaign_portable') / 'main'
SCENES = ('hymars_hc', 'frt00019468_mtrdr', 'hrl000120cd_mtrdr')
MODELS = ('lwnet', 'n2n4m')
STAGES = ('plebani350', 'hymars_full')
BRANCHES = {'sam_original', 'sam_denoised', 'lwnet_original', 'lwnet_denoised'}
STEPS = ([f'before/{s}' for s in SCENES]
         + [f'train/{model}/{stage}' for stage in STAGES for model in MODELS]
         + [f'after/{s}' for s in SCENES])


def save(path, value):
    from modules.run_store import write_manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(path, value)


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def base_config(scene):
    from modules.pipeline_config import load_config
    from modules.pipeline_runner import _resolved_config
    path = WORK / 'configs/pipelines' / f'pretrained_gpu_{scene}.json'
    config = _resolved_config(load_config(path, require_ready=True), path).to_dict()
    if (set(config['branches']) != BRANCHES or config['execution']['device'] != 'cuda:0'
            or not config['execution']['gpu_allowed'] or config['sam']['threshold_radians'] != 0.15
            or config['resources']['sam_library']['artifact_id'] != 'mica_crism_corrected_if_v1'
            or bool(config['evaluation']['ground_truth']) != (scene == 'hymars_hc')):
        raise ValueError(f'Inconsistent cohort/profile: {scene}')
    return config


def preflight():
    from training_h100 import PENDING, preflight_stage
    inventory = verify(WORK)
    print(f"Verified {len(inventory['files'])} files, {inventory['total_bytes']/1e9:.2f} GB", flush=True)
    shapes = {}
    for scene in SCENES:
        config = base_config(scene)
        refs = [config['scene_manifest'], config['taxonomy'], *config['resources'].values()]
        if config['evaluation']['ground_truth']:
            refs.append(config['evaluation']['ground_truth'])
        for ref in refs:
            path = Path(ref['path'])
            if not path.is_relative_to(WORK) or digest(path) != ref['sha256']:
                raise ValueError(f"Invalid binding: {ref['artifact_id']}")
        shapes[scene] = read(Path(config['scene_manifest']['path']))['shape']
    training = {}
    for stage in STAGES:
        for model in MODELS:
            info = preflight_stage(WORK, CAMPAIGN, model, stage)
            pending = stage == STAGES[1] and info['issues'] == [PENDING]
            if not info['data_ready'] or not (info['ready'] or pending):
                raise ValueError(f"{model}/{stage}: {info['issues']}")
            training[f'{model}/{stage}'] = info
            print(f"{model}/{stage}: data OK, batch={info['batch_size']}, epochs={info['epochs']}, "
                  f"batches/epoch={sum(d['batches_per_epoch'] for d in info['datasets'])}", flush=True)
    return dict(files=len(inventory['files']), input_bytes=inventory['total_bytes'],
                shapes=shapes, training=training)


def hardware():
    import torch
    if sys.platform != 'linux' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected Linux with one visible CUDA device. Check nvidia-smi and CUDA_VISIBLE_DEVICES.')
    gpu = torch.cuda.get_device_properties(0)
    if PROFILE == 'h100' and ('H100' not in gpu.name or gpu.total_memory < 80 * 2**30):
        raise RuntimeError(f'Expected a full H100 NVL on GPU 0; found {gpu.name}, {gpu.total_memory/2**30:.1f} GiB')
    if torch.ones(1, device='cuda').item() != 1:
        raise RuntimeError('CUDA kernel failed')
    return dict(profile=PROFILE, gpu=gpu.name, vram_bytes=gpu.total_memory, torch=torch.__version__, cuda=torch.version.cuda)


def space_required(shapes, pending):
    # Source precision + boolean validity twice per scene (prepared/denoised), four
    # label maps + masks, analytics, GT and PNGs. Additional room for checkpoints and temp files.
    required = 1_000_000_000
    for step in pending:
        parts = step.split('/')
        if parts[0] == 'train':
            required += 800_000_000
        else:
            h, w, bands = shapes[parts[1]]
            item_bytes = 8 if parts[1] == 'hymars_hc' else 4
            required += h * w * (2 * (item_bytes + 1) * bands + 85) + 100_000_000
    return required


def identity():
    names = ['run_experiment.py', 'training_h100.py', 'export.py', 'package_workspace.py',
             'start.sh', 'requirements.txt', 'workspace/input_inventory.json', 'gpu_profile.py']
    return hashlib.sha256(''.join(digest(HERE / name) for name in names).encode()).hexdigest()


def load_state(state_path, release):
    if not state_path.exists():
        return dict(release=release, started=stamp(), steps={})
    state = read(state_path)
    if state['release'] != release or list(state['steps']) != STEPS[:len(state['steps'])]:
        raise ValueError('Saved experiment uses another release or an invalid stage order')
    for step, record in state['steps'].items():
        for file, sha in record['pins'].items():
            path = (CAMPAIGN / file).resolve()
            if not path.is_relative_to(CAMPAIGN) or digest(path) != sha:
                raise ValueError(f'Saved result changed: {file}')
        if not step.startswith('train/'):
            from modules.run_store import read_run_manifest
            run = CAMPAIGN / record['run']
            read_run_manifest(run, verify_artifacts=True)
            for folder in [CAMPAIGN / record['analysis'], (CAMPAIGN / record['report']).parent]:
                for ref in read(folder / 'manifest.json')['artifacts']:
                    asset = (folder / ref['path']).resolve()
                    if not asset.is_relative_to(folder) or digest(asset) != ref['sha256']:
                        raise ValueError(f'Saved report artifact changed: {asset}')
    return state


def relative(path):
    return path.relative_to(CAMPAIGN).as_posix()


def pins(files):
    return {relative(f): digest(f) for f in files}


def training(model, stage):
    from training_h100 import train_stage
    path = train_stage(WORK, CAMPAIGN, model, stage)
    result = read(path / 'result.json')
    files = [path / n for n in ('training.json', 'result.json', 'config.json', 'model.ts', 'normalization.json')]
    files += [path / result['checkpoint']['path'], CAMPAIGN / model / stage / 'selected.json']
    return dict(run=relative(path), epochs=result['epochs'], best_epoch=result['best_epoch'], pins=pins(files))


def inference(phase, scene, completed):
    from analysis.visualize_comparison import render_analysis
    from analysis.whole_cube import analyze_run
    from export import export_scene_bundle
    from modules.pipeline_runner import run_pipeline
    from modules.run_store import read_run_manifest
    config = base_config(scene)
    root = CAMPAIGN / 'inference' / phase / scene
    config['stores']['results_root'] = str(root)
    if phase == 'after':
        for model in MODELS:
            source = CAMPAIGN / completed[f'train/{model}/hymars_full']['run']
            bundle = export_scene_bundle(source, Path(config['scene_manifest']['path']))
            config['resources'][model + '_bundle'] = dict(
                artifact_id=read(bundle)['bundle_id'], path=str(bundle), sha256=digest(bundle))
        config['n2n4m']['batch_size'] = 128
    inference_batches(config)
    config_path = root / 'profiles' / f'{time.time_ns()}.json'
    save(config_path, config)
    run = run_pipeline(config_path)
    manifest = read_run_manifest(run, verify_artifacts=True)
    if (manifest['status'] != 'completed' or set(manifest['branches']) != BRANCHES
            or any(b['status'] != 'completed' for b in manifest['branches'].values())):
        raise RuntimeError(f"Incomplete branches; see {run / 'execution.json'}: {manifest['branches']}")
    analysis = analyze_run(run)
    gt = read(analysis / 'evaluation.json')['ground_truth']
    if scene == 'hymars_hc':
        if (gt['status'] != 'available' or gt['role'] != 'held_out' or gt['evaluated_pixels'] != 20090
                or not (analysis / 'metrics_gt.json').is_file()):
            raise ValueError(f'Held-out evaluation failed: {gt}')
    elif gt['reason'] != 'ground_truth_missing' or (analysis / 'metrics_gt.json').exists():
        raise ValueError(f'Unexpected ground truth: {gt}')
    visuals = render_analysis(analysis)
    sam = manifest['branches']['sam_original']['result']
    sam_path = run / sam['path']
    map_hash = read(sam_path)['result']['class_map']['sha256']
    if phase == 'after' and map_hash != completed['before/' + scene]['sam_map_sha256']:
        raise ValueError(f'Original MICA/SAM map changed for {scene}')
    files = [run / 'manifest.json', analysis / 'manifest.json', analysis / 'summary.csv',
             analysis / 'evaluation.json', analysis / 'valid_pixels.npy', visuals / 'manifest.json',
             visuals / 'report.html', config_path]
    if scene == 'hymars_hc':
        files += [analysis / 'metrics_gt.json', analysis / 'gt_evaluation_mask.npy', analysis / 'per_class_gt.csv']
    return dict(run=relative(run), analysis=relative(analysis), report=relative(visuals / 'report.html'),
                sam_map_sha256=map_hash, pins=pins(files))


def compare(steps):
    rows = []
    for scene in SCENES:
        before = CAMPAIGN / steps['before/' + scene]['analysis']
        after = CAMPAIGN / steps['after/' + scene]['analysis']
        for name in ['valid_pixels.npy'] + (['gt_evaluation_mask.npy'] if scene == 'hymars_hc' else []):
            if digest(before / name) != digest(after / name):
                raise ValueError(f'Evaluation mask changed: {scene}/{name}')
        for phase, folder in [('before', before), ('after', after)]:
            tax = read(folder / 'evaluation.json')['branch_taxonomies']
            with (folder / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
                summary = list(csv.DictReader(stream))
                if len(summary) != 4 or {row['branch_id'] for row in summary} != BRANCHES:
                    raise ValueError(f'Incomplete comparison for {phase}/{scene}')
                for row in summary:
                    rows.append(dict(scene=scene, phase=phase, taxonomy=tax[row['branch_id']]['taxonomy_id'], **row))
    path = CAMPAIGN / 'comparison.csv'
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def execute():
    # start.sh holds a separate lock through environment setup and execution.
    import fcntl
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    with (CAMPAIGN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        machine = hardware()
        state_path = CAMPAIGN / 'campaign.json'
        state = load_state(state_path, identity())
        report = preflight()
        save(CAMPAIGN / 'preflight.json', dict(hardware=machine, **report))
        pending = [s for s in STEPS if s not in state['steps']]
        for index, step in enumerate(pending):
            remaining = space_required(report['shapes'], pending[index:])
            available = shutil.disk_usage(CAMPAIGN).free
            if available < remaining:
                raise OSError(f'Need {remaining/1e9:.1f} GB free for remaining stages; available {available/1e9:.1f} GB')
            state.update(status='running', current=step, error=None)
            save(state_path, state)
            start = time.monotonic()
            print(f'[{stamp()}] {step}', flush=True)
            try:
                kind, *args = step.split('/')
                record = training(*args) if kind == 'train' else inference(kind, args[0], state['steps'])
            except BaseException as error:
                state.update(status='interrupted', error=f'{type(error).__name__}: {error}')
                save(state_path, state)
                raise
            record.update(seconds=round(time.monotonic() - start, 2), finished=stamp())
            state['steps'][step] = record
            save(state_path, state)
            print(f"Completed {step} in {record['seconds']/60:.1f} min", flush=True)
        compare(state['steps'])
        state.update(status='completed', current=None, finished=state.get('finished') or stamp())
        save(state_path, state)
        print(f"Results: {CAMPAIGN / 'comparison.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run', 'status'))
    args = parser.parse_args()
    if args.action == 'check':
        report = preflight()
        print(f"Output space budget: {space_required(report['shapes'], STEPS)/1e9:.1f} GB")
    elif args.action == 'status':
        path = CAMPAIGN / 'campaign.json'
        state = read(path) if path.exists() else dict(status='not_started', steps={})
        print(json.dumps({k: v for k, v in state.items() if k != 'steps'}, indent=2))
        for step, record in state['steps'].items():
            print(f"{step}: {record['seconds']/60:.1f} min")
    else:
        execute()


if __name__ == '__main__':
    main()
