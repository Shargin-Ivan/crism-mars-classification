"""Two-stage, single-GPU training for one portable H100 campaign."""
from contextlib import contextmanager
import gc
import math
import os
from pathlib import Path
from uuid import uuid4

from gpu_profile import PROFILE
from finetuning.configuration import _split_checks, load, observation_keys, preflight
from finetuning.io import digest, pinned, read_json, write_json


STAGES = ('plebani350', 'hymars_full')
EPOCHS_PER_STAGE = 25
BATCHES = {'n2n4m': (128, 64), 'lwnet': (128, 32)}
if PROFILE == 'portable':
    BATCHES = {'n2n4m': (16, 16), 'lwnet': (32, 4)}
DEVELOPMENT = {'hymars_v2_NF': 'FRT00003E12', 'hymars_v2_UP': 'FRT0000527D'}
HOLDOUT = ['FRT0000580C', 'FRT00019468', 'HRL000120CD']
PENDING = 'stage1_best_checkpoint_pending'
CHECKPOINT_ISSUES = {'stage2_requires_stage1_training_checkpoint', 'frozen_initializer_required_or_explicit_from_scratch'}


def _paths(workspace_root, experiment_run_dir):
    root = (Path(workspace_root) / 'finetuning').resolve()
    campaign = Path(experiment_run_dir).resolve()
    if not campaign.is_relative_to(root / 'storage') or campaign == root / 'storage':
        raise ValueError('Campaign directory must be below workspace/finetuning/storage')
    return root, campaign


def _recipe(root, model, stage, batch_size):
    if model not in BATCHES or stage not in STAGES:
        raise ValueError('Unknown experiment model or stage')
    cfg = load(root / 'configs' / f'{model}_{stage}.json')
    if (cfg['model'] != model or cfg['stage'] != stage or cfg['held_out_scene_id'] != 'hymars_v2_HC'
            or cfg['held_out_observation_ids'] != [HOLDOUT[0]] or set(cfg['development_scene_ids']) != set(DEVELOPMENT)):
        raise ValueError('Training recipe differs from the experiment split')
    cfg['held_out_observation_ids'] = list(HOLDOUT)
    default_batch = BATCHES[model][STAGES.index(stage)]
    if batch_size is None:
        batch_size = default_batch
    if type(batch_size) is not int or not 2 <= batch_size <= default_batch:
        raise ValueError(f'Expected batch size 2 through {default_batch}')
    cfg['parameters'].update(epochs=EPOCHS_PER_STAGE, patience=EPOCHS_PER_STAGE,
                             batch_size=batch_size, device='cuda', threads=8)
    cfg['state'] = 'ready'
    return cfg


def _selected(root, campaign, model, stage, selection=None):
    """Accept a published, verified result below this model and campaign only."""
    parent = campaign / model / stage
    if selection is None:
        selection = read_json(parent / 'selected.json')
    result_path = pinned(parent, selection['result'])
    result = read_json(result_path)
    run_dir = result_path.parent
    lineage = read_json(pinned(run_dir, selection['lineage']))
    config_path = run_dir / 'config.json'
    config = load(config_path)
    if (result_path.name != 'result.json' or result['status'] != 'completed' or result['epochs'] != EPOCHS_PER_STAGE
            or not 1 <= result['best_epoch'] <= result['epochs']
            or result['checkpoint']['path'] != f"best_epoch_{result['best_epoch']:04d}.pt"
            or lineage['epoch'] != result['best_epoch'] or lineage['config_sha256'] != digest(config_path)
            or lineage['model'] != model or lineage['stage'] != stage or lineage['parameters'] != config['parameters']
            or lineage['initializer'] != config['initialization'] or lineage['class_names'] != config['class_names']
            or config['model'] != model or config['stage'] != stage or config['parameters']['epochs'] != EPOCHS_PER_STAGE
            or lineage['held_out_scene_id'] != 'hymars_v2_HC'
            or observation_keys(lineage['held_out_observation_ids']) != observation_keys(HOLDOUT)
            or observation_keys(lineage['source_observation_ids']) & observation_keys(HOLDOUT)
            or [ref['path'] for ref in lineage['inputs']] != config['datasets']):
        raise ValueError(f'Invalid campaign result for {model}/{stage}')
    for ref in lineage['inputs']:
        pinned(root, ref)
    if not lineage['code_sha256']:
        raise ValueError('Missing training source hashes')
    for filename, checksum in lineage['code_sha256'].items():
        pinned(root, {'path': filename, 'sha256': checksum})
    checkpoint = pinned(run_dir, result['checkpoint'])
    pinned(run_dir, result['export'])
    return run_dir, {'path': checkpoint.relative_to(root).as_posix(), 'sha256': result['checkpoint']['sha256']}


def _check_sources(datasets, cfg):
    sources = [observation_keys(d.descriptor['provenance']['source_observation_ids']) for d in datasets]
    if cfg['stage'] == STAGES[0]:
        forbidden = observation_keys([*HOLDOUT, *DEVELOPMENT.values()])
        if (len(datasets) != 2 or {d.descriptor['role'] for d in datasets} != {'train', 'validation'}
                or any(d.descriptor['scene_id'] != 'plebani_spectra_' + d.descriptor['role'] for d in datasets)
                or sources[0] & sources[1] or any(ids & forbidden for ids in sources)):
            raise ValueError('Expected disjoint Plebani observations excluding all three inference scenes and NF/UP')
    else:
        expected = {(scene, role) for scene in DEVELOPMENT for role in ('train', 'validation')}
        if len(datasets) != 4 or {(d.descriptor['scene_id'], d.descriptor['role']) for d in datasets} != expected:
            raise ValueError('Expected exactly NF and UP train and validation splits')
        for d, ids in zip(datasets, sources):
            if ids != observation_keys([DEVELOPMENT[d.descriptor['scene_id']]]):
                raise ValueError('Unexpected observation in NF/UP data')


@contextmanager
def _checked(root, campaign, model, stage, batch_size):
    cfg = _recipe(root, model, stage, batch_size)
    waiting = False
    previous = None
    if stage == STAGES[1]:
        if (campaign / model / STAGES[0] / 'selected.json').exists():
            previous_run, ref = _selected(root, campaign, model, STAGES[0])
            previous = read_json(previous_run / 'training.json')
            cfg['initialization']['checkpoint'] = ref
        else:
            waiting = True
    report, datasets = preflight(cfg, root)
    try:
        if waiting:
            report['issues'] = [issue for issue in report['issues'] if issue not in CHECKPOINT_ISSUES]
        if not report['issues']:
            try:
                if waiting:
                    _split_checks(datasets, cfg)
                _check_sources(datasets, cfg)
                if previous is not None:
                    from finetuning.training import check_transfer
                    check_transfer(previous, datasets, cfg, root)
            except ValueError as error:
                report['issues'].append(str(error))
        report['data_ready'] = not report['issues']
        if waiting:
            report['issues'].append(PENDING)
        report['ready'] = not report['issues']
        p = cfg['parameters']
        for info in report['datasets']:
            n = min(info['samples'], p['max_samples_per_dataset'])
            batches = math.ceil(n / p['batch_size'])
            if model == 'n2n4m' and info['role'] == 'train' and n % p['batch_size'] == 1:
                batches -= 1
            info['batches_per_epoch'] = batches
        report.update(batch_size=p['batch_size'], epochs=p['epochs'])
        yield cfg, datasets, report
    finally:
        for d in datasets:
            d.close()


def preflight_stage(workspace_root, experiment_run_dir, model, stage, *, batch_size=None):
    """Validate all source data on CPU, also when the stage-one result is still pending."""
    root, campaign = _paths(workspace_root, experiment_run_dir)
    with _checked(root, campaign, model, stage, batch_size) as (_, _, report):
        return report


def train_stage(workspace_root, experiment_run_dir, model, stage, *, batch_size=None):
    """Return the verified best run directory; failed attempts can be retried at a smaller batch."""
    from finetuning.training import run

    root, campaign = _paths(workspace_root, experiment_run_dir)
    if model not in BATCHES or stage not in STAGES:
        raise ValueError('Unknown experiment model or stage')
    parent = campaign / model / stage
    with _checked(root, campaign, model, stage, batch_size) as (cfg, datasets, report):
        if not report['ready']:
            raise ValueError(report['issues'])
        if (parent / 'selected.json').exists():
            path, _ = _selected(root, campaign, model, stage)
            actual = load(path / 'config.json')
            # A prior successful smaller batch remains reusable on a normal restart.
            if batch_size is None:
                cfg['parameters']['batch_size'] = actual['parameters']['batch_size']
            if cfg != actual:
                raise ValueError('Existing stage uses different settings; use a new campaign')
            return path
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('This experiment requires CUDA on GPU 0')
        torch.cuda.set_device(0)
        while True:
            attempt = parent / uuid4().hex
            attempt.mkdir(parents=True, exist_ok=False)
            config_path = attempt / 'config.json'
            write_json(config_path, cfg)
            try:
                path = run(load(config_path), config_path, datasets, root, run_parent=attempt, report_timing=True)
                break
            except torch.cuda.OutOfMemoryError as error:
                size = cfg['parameters']['batch_size']
                if 'cuda' not in str(error).lower() or size <= 2:
                    raise
                next_size = max(2, size // 2)
                write_json(attempt / 'retry.json', {'reason': str(error), 'next_batch_size': next_size})
            # Outside the exception handler, the failed autograd graph can be released.
            gc.collect()
            torch.cuda.empty_cache()
            cfg['parameters']['batch_size'] = next_size
            print(f'CUDA memory exhausted. Restarting {model}/{stage} at batch {next_size}.', flush=True)
        # Immutable exclusive publication. A failed job has no selected.json.
        selection = {'result': {'path': (path / 'result.json').relative_to(parent).as_posix(),
                                'sha256': digest(path / 'result.json')},
                     'lineage': {'path': 'training.json', 'sha256': digest(path / 'training.json')}}
        selected, _ = _selected(root, campaign, model, stage, selection)
        publication = attempt / 'selection.json'
        write_json(publication, selection)
        if os.name == 'nt':
            publication.rename(parent / 'selected.json')  # Windows refuses to replace an existing file.
        else:
            os.link(publication, parent / 'selected.json')
        return selected
