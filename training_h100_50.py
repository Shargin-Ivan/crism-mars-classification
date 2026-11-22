"""50-epoch training loop with validation-best and last model/Adam checkpoints."""
from datetime import datetime, timezone
import shutil
from pathlib import Path
import platform
from time import monotonic
import uuid

import numpy as np

from finetuning import training as core
from finetuning.configuration import LEGACY_LWNET, uses_spectral_transfer
from finetuning.io import digest, pinned, write_json
from finetuning.training import _epoch, check_transfer, fit_normalization, initialize


CHECKPOINT_STATE = {'model_state', 'optimizer_state', 'torch_rng_state', 'cuda_rng_state'}


def run(config, config_path, datasets, root, model_factory=None, *, run_parent=None, report_timing=False):
    import torch
    from finetuning.models import LWNetFullAxis, LWNetSpatial, LWNetSpectrum, N2N4MFullAxis
    p = config['parameters']
    if config['state'] != 'ready':
        raise ValueError('Training configuration is not ready')
    if p['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; choose a compatible wheel/device explicitly')
    torch.set_num_threads(p['threads'])
    torch.manual_seed(p['seed'])
    if p['device'] == 'cuda':
        torch.cuda.manual_seed_all(p['seed'])
    if model_factory is not None:
        model = model_factory()
    elif config['model'] == 'n2n4m':
        model = N2N4MFullAxis()
    else:
        classifier = LWNetFullAxis
        if uses_spectral_transfer(config):
            classifier = LWNetSpectrum if config['stage'] == 'plebani350' else LWNetSpatial
        model = classifier(len(config['class_names']))
    previous = initialize(model, config, root)
    check_transfer(previous, datasets, config, root)
    normalization, feature_support = fit_normalization(datasets)
    model.to(p['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'])
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:10]
    if run_parent is None:
        run_parent = Path(root) / 'storage/runs' / config['model'] / config['stage']
    destination = Path(run_parent) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / 'config.json', config)
    write_json(destination / 'normalization.json', normalization)
    training_scenes = sorted({d.descriptor['scene_id'] for d in datasets if d.descriptor['role'] == 'train'})
    validation_scenes = sorted({d.descriptor['scene_id'] for d in datasets if d.descriptor['role'] == 'validation'})
    observations = sorted({s for d in datasets for s in d.descriptor['provenance']['source_observation_ids']})
    lineage = {
        'model': config['model'], 'stage': config['stage'], 'class_names': config['class_names'],
        'units': datasets[0].descriptor['units'], 'domain': datasets[0].descriptor['domain'],
        'parameters': p, 'held_out_scene_id': config['held_out_scene_id'],
        'held_out_observation_ids': config['held_out_observation_ids'],
        'training_scene_ids': sorted(set(training_scenes + (previous['training_scene_ids'] if previous else []))),
        'validation_scene_ids': sorted(set(validation_scenes + (previous['validation_scene_ids'] if previous else []))),
        'current_training_scene_ids': training_scenes, 'current_validation_scene_ids': validation_scenes,
        'source_observation_ids': sorted(set(observations + (previous['source_observation_ids'] if previous else []))),
        'feature_support': [], 'normalization': normalization,
        'normalization_feature_support': feature_support,
        'normalization_policy': 'refit_train_only_each_stage',
        'previous_domain': previous.get('domain') if previous else None,
        'previous_normalization': previous.get('normalization') if previous else None,
        'pretraining_lineage_status': (previous.get('pretraining_lineage_status', 'unverified') if previous else
             ('unverified_external_initializer' if config['initialization']['checkpoint'] else 'from_scratch')),
        'initializer': config['initialization'], 'config_sha256': digest(config_path),
        'inputs': [{'path': d.path.relative_to(Path(root).resolve()).as_posix(), 'sha256': digest(d.path)} for d in datasets],
        'code_sha256': {p.relative_to(Path(core.__file__).parent).as_posix():digest(p)
                        for p in sorted(Path(core.__file__).parent.rglob('*.py'))
                        if 'storage' not in p.relative_to(Path(core.__file__).parent).parts
                        and '.venv' not in p.relative_to(Path(core.__file__).parent).parts},
        'versions': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__},
    }
    lineage['extension_sha256'] = digest(Path(__file__))
    if config['model'] == 'lwnet':
        lineage['lwnet_architecture'] = config.get('lwnet_architecture', LEGACY_LWNET)
    if uses_spectral_transfer(config):
        lineage['transferred_modules'] = ['encoder', 'head'] if previous else []
        lineage['trained_modules'] = ['encoder', 'head', 'spatial'] if previous else ['encoder', 'head']
    history, best, stale, checkpoint = [], float('inf'), 0, None
    seen_features = set()
    started = monotonic()
    try:
        for epoch in range(p['epochs']):
            epoch_started = monotonic()
            train = _epoch(model, datasets, config, normalization, optimizer, epoch)
            val = _epoch(model, datasets, config, normalization, None, epoch)
            seen_features.update((f['wavelength_nm'],f['detector']) for f in train['feature_support'])
            lineage['feature_support'] = [{'wavelength_nm':w,'detector':d} for w,d in sorted(seen_features)]
            row = {'epoch': epoch + 1, 'train': train, 'validation': val}
            if report_timing:
                elapsed = monotonic() - started
                row['seconds'] = monotonic() - epoch_started
                row['remaining_epoch_seconds_estimate'] = elapsed / (epoch + 1) * (p['epochs'] - epoch - 1)
            history.append(row)
            write_json(destination / f'epoch_{epoch+1:04d}.json', row)
            print(f"epoch={epoch+1} train_loss={train['loss']:.6g} validation_loss={val['loss']:.6g}", flush=True)
            if report_timing:
                print(f"epoch_seconds={row['seconds']:.1f} "
                      f"remaining_stage_hours={row['remaining_epoch_seconds_estimate']/3600:.2f}", flush=True)
            last = destination / 'last.pt'
            temporary = destination / 'last.tmp'
            torch.save({**lineage, 'epoch': epoch + 1, 'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'torch_rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state() if p['device'] == 'cuda' else None},
                       temporary)
            temporary.replace(last)
            if val['loss'] < best:
                best, stale = val['loss'], 0
                prior_best = checkpoint
                checkpoint = destination / f'best_epoch_{epoch+1:04d}.pt'
                shutil.copyfile(last, checkpoint)
                if prior_best is not None:
                    prior_best.unlink()
            else:
                stale += 1
            if stale >= p['patience']:
                break
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        checked = set()
        for dataset in datasets:
            for name in ('values','valid','roi','labels'):
                ref = dataset.descriptor.get(name)
                if ref is not None:
                    key = (dataset.path.parent,ref['path'],ref['sha256'])
                    if key not in checked:
                        pinned(dataset.path.parent,ref)
                        checked.add(key)
        for ref in lineage['inputs']:
            pinned(root,ref)
        model.cpu().load_state_dict(saved['model_state'], strict=True)
        write_json(destination / 'training.json', {k:v for k,v in saved.items() if k not in CHECKPOINT_STATE})
        torch.jit.script(model.eval()).save(str(destination / 'model.ts'))
        write_json(destination / 'result.json', {'status': 'completed', 'best_epoch': saved['epoch'],
                   'best_validation_loss': best, 'checkpoint': {'path': checkpoint.name, 'sha256': digest(checkpoint)},
                   'last_checkpoint': {'path': last.name, 'sha256': digest(last)},
                   'export': {'path': 'model.ts', 'sha256': digest(destination / 'model.ts')},
                   'metric_scope': 'development_validation_only', 'epochs': len(history)})
    except BaseException as error:
        write_json(destination / 'failure.json', {'status': 'failed', 'error': str(error)})
        raise
    return destination
