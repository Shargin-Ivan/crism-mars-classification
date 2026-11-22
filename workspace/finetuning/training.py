"""Explicit training entry point; never called by imports, checks or packaging."""
from datetime import datetime, timezone
from pathlib import Path
import platform
from time import monotonic
import uuid

import numpy as np

from .configuration import LEGACY_LWNET, observation_keys, uses_spectral_transfer
from .io import digest, pinned, read_json, write_json


def fit_normalization(datasets):
    """One scalar affine normalization, fitted only to measured train ROI values.

    A scalar supports different C without inventing statistics for missing bands.
    It does not make different radiometric domains scientifically equivalent.
    """
    count, mean, m2 = 0, 0.0, 0.0
    support = set()
    for dataset in datasets:
        if dataset.descriptor['role'] != 'train':
            continue
        channels = dataset.values.shape[-1]
        values = dataset.values.reshape(-1, channels)
        valid = dataset.valid.reshape(-1, channels)
        roi = dataset.roi.ravel()
        bands = np.zeros(channels, dtype=bool)
        for start in range(0, len(values), 4096):
            stop = start + 4096
            mask = valid[start:stop] & roi[start:stop, None] & dataset.channel_mask
            bands |= mask.any(0)
            block = np.asarray(values[start:stop][mask], dtype=np.float64)
            if not block.size:
                continue
            n = block.size
            block_mean = float(block.mean())
            delta = block_mean - mean
            total = count + n
            m2 += float(np.square(block - block_mean).sum()) + delta * delta * count * n / total
            mean += delta * n / total
            count = total
        for wave, detector in zip(dataset.axis_arrays['wavelengths_nm'][bands], dataset.axis_arrays['detectors'][bands]):
            support.add((float(wave), str(detector)))
    scale = float(np.sqrt(m2 / count)) if count else 0
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Train-only normalization has empty, constant or nonfinite support')
    return {'mean': mean, 'scale': scale, 'sample_count': count, 'method': 'train_scalar_standardize'}, [
        {'wavelength_nm': wave, 'detector': detector} for wave, detector in sorted(support)]


def initialize(model, config, root):
    import torch
    init = config['initialization']
    if init['checkpoint'] is None:
        if not init['allow_random'] or config['stage'] != 'plebani350':
            raise ValueError('Random initialization was not explicitly enabled for stage 1')
        return None
    path = pinned(root, init['checkpoint'])
    state = torch.load(path, map_location='cpu', weights_only=True)
    if init['kind'] == 'training_checkpoint':
        result = read_json(path.parent / 'result.json')
        if (result.get('status') != 'completed' or result.get('checkpoint', {}).get('path') != path.name
                or result['checkpoint'].get('sha256') != init['checkpoint']['sha256']):
            raise ValueError('Stage 1 must be completed and its selected checkpoint pinned')
        if (not isinstance(state, dict) or state.get('model') != config['model']
                or state.get('stage') != 'plebani350' or config['stage'] != 'hymars_full'
                or state.get('class_names') != config['class_names']):
            raise ValueError('Stage 2 needs a matching stage-1 checkpoint and canonical class order')
        forbidden = observation_keys(config['held_out_observation_ids'])
        if (state.get('held_out_scene_id') != config['held_out_scene_id'] or
                observation_keys(state.get('held_out_observation_ids', [])) != forbidden):
            raise ValueError('The held-out split changed after stage 1')
        if forbidden & observation_keys(state.get('source_observation_ids', [])):
            raise ValueError('Stage-1 checkpoint was exposed to the held-out observation')
        if config['model'] == 'lwnet' and (state.get('lwnet_architecture', LEGACY_LWNET)
                                           != config.get('lwnet_architecture', LEGACY_LWNET)):
            raise ValueError('Stage-1 LWNet architecture differs from the requested stage-2 architecture')
        if uses_spectral_transfer(config):
            # The spatial branch is new. The encoder and classifier retain their class order.
            model.load_spectral_state(state['model_state'])
        else:
            model.load_state_dict(state['model_state'], strict=True)
    else:
        if config['stage'] != 'plebani350':
            raise ValueError('Backbone initialization is only for stage 1')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if not isinstance(state, dict) or not state:
            raise ValueError('Expected a nonempty tensor state_dict')
        cleaned = {key.removeprefix('module.'): value for key, value in state.items()}
        if len(cleaned) != len(state) or not all(isinstance(v, torch.Tensor) for v in cleaned.values()):
            raise ValueError('Invalid initializer state_dict')
        if config['model'] == 'lwnet' and init['reset_classifier']:
            cleaned = {k: v for k, v in cleaned.items() if k not in ('fc.weight', 'fc.bias')}
            result = model.backbone.load_state_dict(cleaned, strict=False)
            if set(result.missing_keys) != {'fc.weight', 'fc.bias'} or result.unexpected_keys:
                raise ValueError('Unexpected backbone mismatch beyond the explicitly reset head')
        else:
            model.backbone.load_state_dict(cleaned, strict=True)
    pinned(root, init['checkpoint'])
    return state if init['kind'] == 'training_checkpoint' else None


def check_transfer(previous, datasets, config, root):
    if previous is None:
        return
    domain = datasets[0].descriptor['domain']
    units = datasets[0].descriptor['units']
    if units != previous.get('units'):
        raise ValueError('Initializer and stage-2 units differ; perform a documented conversion first')
    if domain != previous.get('domain') and not config['parameters']['allow_domain_shift']:
        raise ValueError('Domain shift requires explicit allow_domain_shift=true and an adaptation protocol')
    if not previous.get('inputs'):
        raise ValueError('Stage-1 dataset lineage is missing')
    # Keep the old footprint declarations available alongside the checkpoint.
    # SHA checks prevent changing a former train ROI into a validation ROI later.
    for ref in previous['inputs']:
        path = pinned(root, ref)
        old = read_json(path)
        for current in datasets:
            new = current.descriptor
            if old['role'] == new['role']:
                continue
            overlap = (old['values']['sha256'] == new['values']['sha256'] or
                       old['scene_id'] == new['scene_id'] or
                       bool(observation_keys(old['provenance']['source_observation_ids']) &
                            observation_keys(new['provenance']['source_observation_ids'])))
            if not overlap:
                continue
            if old['values']['sha256'] != new['values']['sha256']:
                raise ValueError('Cross-stage observations overlap but pixel geometry is unverified')
            roi_path = pinned(path.parent, old['roi'])
            old_roi = np.load(roi_path, allow_pickle=False)
            if old_roi.shape != current.roi.shape or np.any(old_roi & current.roi):
                raise ValueError('Cross-stage train/validation footprints overlap')


def _epoch(model, datasets, config, normalization, optimizer, epoch):
    import torch
    import torch.nn.functional as F
    training = optimizer is not None
    model.train(training)
    p = config['parameters']
    rng = np.random.default_rng(p['seed'] + (epoch if training else 100000))
    noise_rng = torch.Generator().manual_seed(p['seed'] + (epoch if training else 100000))
    total_loss, support, correct = 0.0, 0, 0
    seen_features = set()
    for dataset in datasets:
        if dataset.descriptor['role'] != ('train' if training else 'validation'):
            continue
        order = rng.permutation(len(dataset))[:p['max_samples_per_dataset']]
        axis = dataset.axis_arrays
        wave = torch.tensor(axis['wavelengths_nm'], dtype=torch.float32, device=p['device'])
        detectors = torch.tensor((axis['detectors'] == 'S').astype(np.int64), device=p['device'])
        edges = torch.tensor(axis['allowed_edges'], dtype=torch.bool, device=p['device'])
        for start in range(0, len(order), p['batch_size']):
            batch = [dataset.sample(int(i)) for i in order[start:start + p['batch_size']]]
            if training and len(batch) < 2 and not uses_spectral_transfer(config):
                continue
            valid = torch.from_numpy(np.stack([s['valid'] for s in batch]))
            is_patch = valid.ndim == 5
            center = p['patch_size'] // 2
            raw = np.stack([s['values'] for s in batch]).astype(np.float64)
            normalized = np.where(valid.numpy(), (raw - normalization['mean']) / normalization['scale'], 0)
            values = torch.from_numpy(normalized.astype(np.float32))
            if not torch.isfinite(values).all():
                raise ValueError('Normalization produced nonfinite float32 values')
            if training and p['band_dropout']:
                channels = raw.shape[2] if is_patch else raw.shape[1]
                keep = torch.rand((len(batch), channels), generator=noise_rng) >= p['band_dropout']
                if is_patch:
                    keep = keep[:, None, :, None, None]
                proposed = valid & keep
                # Keep original support if augmentation would empty the centre.
                center_mask = proposed[:, 0, :, center, center] if is_patch else proposed
                selected = center_mask.any(1)
                valid[selected] = proposed[selected]
            values = torch.where(valid, values, torch.zeros_like(values))
            used = (valid[:, 0, :, center, center] if is_patch else valid).any(0)
            for wavelength, detector in zip(axis['wavelengths_nm'][used.numpy()], axis['detectors'][used.numpy()]):
                seen_features.add((float(wavelength), str(detector)))
            if config['model'] == 'n2n4m':
                target = values.to(p['device'])
                inputs = values + torch.randn(values.shape, generator=noise_rng) * p['noise_std'] * valid
            else:
                target = torch.tensor([s['label'] for s in batch], dtype=torch.long, device=p['device'])
                inputs = values
            valid = valid.to(p['device'])
            with torch.set_grad_enabled(training):
                prediction = model(inputs.to(p['device']), valid, wave, detectors, edges)
                if config['model'] == 'n2n4m':
                    loss = torch.square(prediction[valid] - target[valid]).mean()
                    weight = int(valid.sum())
                else:
                    loss = F.nll_loss(prediction, target)
                    weight = len(batch)
                    correct += int((prediction.argmax(1) == target).sum())
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training/validation loss')
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            total_loss += float(loss.detach()) * weight
            support += weight
    if not support:
        raise ValueError('No batches remain; increase train ROI/sample cap or reduce batch size')
    return {'loss': total_loss / support, 'support': support,
            'accuracy': correct / support if config['model'] == 'lwnet' else None,
            'feature_support': [{'wavelength_nm':w, 'detector':d} for w,d in sorted(seen_features)]}


def run(config, config_path, datasets, root, model_factory=None, *, run_parent=None, report_timing=False):
    import torch
    from .models import LWNetFullAxis, LWNetSpatial, LWNetSpectrum, N2N4MFullAxis
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
        'code_sha256': {p.relative_to(Path(__file__).parent).as_posix():digest(p)
                        for p in sorted(Path(__file__).parent.rglob('*.py'))
                        if 'storage' not in p.relative_to(Path(__file__).parent).parts
                        and '.venv' not in p.relative_to(Path(__file__).parent).parts},
        'versions': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__},
    }
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
            if val['loss'] < best:
                best, stale = val['loss'], 0
                checkpoint = destination / f'best_epoch_{epoch+1:04d}.pt'
                torch.save({**lineage, 'model_state': model.state_dict(), 'epoch': epoch+1}, checkpoint)
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
        write_json(destination / 'training.json', {k:v for k,v in saved.items() if k != 'model_state'})
        torch.jit.script(model.eval()).save(str(destination / 'model.ts'))
        write_json(destination / 'result.json', {'status': 'completed', 'best_epoch': saved['epoch'],
                   'best_validation_loss': best, 'checkpoint': {'path': checkpoint.name, 'sha256': digest(checkpoint)},
                   'export': {'path': 'model.ts', 'sha256': digest(destination / 'model.ts')},
                   'metric_scope': 'development_validation_only', 'epochs': len(history)})
    except BaseException as error:
        write_json(destination / 'failure.json', {'status': 'failed', 'error': str(error)})
        raise
    return destination
