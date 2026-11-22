"""N2N4M and LWNet adapters over complete H,W,C arrays, with bounded batches."""
from pathlib import Path

import numpy as np

from modules.pipeline_config import ExecutionSpec, LWNetSpec, N2N4MSpec
from modules.pipeline_contracts import DenoisingResult, RejectionReason as Reason, contract_dict, require
from modules.resource_store import array_reference
from .backends import load_backend, load_plan
from .common import (check_classifier_input, close_array, compatible, complete_result,
                     inference_range, output_directory, save_classification)


def _required(parameters, channels):
    return max(parameters['min_valid_bands'], int(np.ceil(channels * parameters['min_valid_fraction'])))


def denoise_n2n4m(view, bundle_ref, resource_base, output_dir, *,
                  spec=N2N4MSpec(), execution=ExecutionSpec()):
    require(view.stage == 'prepared', 'N2N4M requires the original prepared cube, not an already denoised input')
    require(isinstance(spec, N2N4MSpec) and isinstance(execution, ExecutionSpec), 'Invalid inference configuration')
    compatible(execution.num_workers == 0, 'unsupported_workers', 'This streaming adapter uses num_workers=0')
    plan = load_plan(view, bundle_ref, resource_base, 'n2n4m')
    if plan.bundle.architecture_version == 'N2N4M_Plebani350_original_v1':
        from .pretrained_n2n4m import denoise_native_grid
        return denoise_native_grid(view, plan, output_dir, spec, execution)
    channels = view.cube.shape[2]
    original = view.cube.reshape(-1, channels)
    measured = view.valid_samples.reshape(-1, channels)
    eligible = measured.sum(axis=1) >= _required(plan.parameters, channels)
    compatible(eligible.any(), 'no_denoised_support', 'No spectrum satisfies the model coverage policy')
    with output_directory(output_dir, 'denoising') as path:
        output = denoised = None
        try:
            output = np.lib.format.open_memmap(path / 'cube.npy', mode='w+', shape=view.cube.shape, dtype=view.cube.dtype)
            denoised = np.lib.format.open_memmap(path / 'denoised_mask.npy', mode='w+', shape=view.cube.shape, dtype=bool)
            output_flat, denoised_flat = output.reshape(-1, channels), denoised.reshape(-1, channels)
            predict = None
            for start in inference_range(len(original), spec.batch_size, 'N2N4M full-axis', 'pixels'):
                stop = min(start + spec.batch_size, len(original))
                current = np.array(original[start:stop], copy=True)
                selected = eligible[start:stop]
                mask = measured[start:stop][selected]
                if selected.any():
                    features = plan.normalize(current[selected], mask)
                    if predict is None:
                        predict = load_backend(plan, view, execution)
                    prediction = np.asarray(predict(features, mask))
                    require(prediction.shape == features.shape and prediction.dtype.kind == 'f',
                            'N2N4M must return B,C floating spectra without dropping channels')
                    with np.errstate(over='ignore', invalid='ignore'):
                        restored = (prediction * plan.scale + plan.mean).astype(view.cube.dtype)
                    require(np.isfinite(restored[mask]).all(), 'Nonfinite N2N4M predictions on measured channels')
                    current[selected] = np.where(mask, restored, current[selected])
                output_flat[start:stop] = current
                denoised_flat[start:stop] = measured[start:stop] & selected[:, None]
            output.flush()
            denoised.flush()
            result = DenoisingResult(view.prepared, plan.resource_ref,
                                    array_reference(path / 'cube.npy', 'n2n4m_cube', path), view.valid_ref,
                                    array_reference(path / 'denoised_mask.npy', 'denoised_mask', path))
            result.validate_arrays(view.cube, view.valid_samples, output, view.valid_samples, denoised,
                                   good_channels=view.good_channels)
            parameters = {'model': plan.parameters, 'execution': contract_dict(execution),
                          'batch_size': spec.batch_size, 'processed_spectra': int(eligible.sum()),
                          'processed_samples': int(denoised.sum()), 'channels': channels}
            # Close writable mappings before publishing a completed result.
            close_array(output)
            close_array(denoised)
            output = denoised = None
            complete_result(path, 'denoising', result, parameters)
            return result
        finally:
            close_array(output)
            close_array(denoised)


def _check_scores(scores, count, classes, kind):
    scores = np.asarray(scores)
    require(scores.shape == (count, classes) and scores.dtype.kind == 'f' and np.isfinite(scores).all(),
            'LWNet returned invalid shape/type or nonfinite class scores')
    if kind == 'probabilities':
        require((scores >= 0).all() and (scores <= 1).all()
                and np.allclose(scores.sum(axis=1), 1, rtol=0, atol=1e-5), 'Invalid LWNet probabilities')
    elif kind == 'log_probabilities':
        require((scores <= 1e-6).all() and np.allclose(np.exp(scores).sum(axis=1), 1, rtol=0, atol=1e-5),
                'Invalid LWNet log probabilities')
    return scores


def classify_lwnet(view, bundle_ref, resource_base, taxonomy, output_dir, *,
                   branch_id='lwnet_original', spec=LWNetSpec(), execution=ExecutionSpec()):
    require(isinstance(spec, LWNetSpec) and isinstance(execution, ExecutionSpec), 'Invalid inference configuration')
    compatible(execution.num_workers == 0, 'unsupported_workers', 'This streaming adapter uses num_workers=0')
    if bundle_ref is not None:
        from modules.pipeline_contracts import ModelBundle
        from modules.resource_store import ResourceStore
        from .salinas import ARCHITECTURE, classify_salinas
        bundle, _ = ResourceStore(resource_base).load_contract(bundle_ref, ModelBundle)
        if bundle.architecture_version == ARCHITECTURE:
            return classify_salinas(view, bundle_ref, resource_base, taxonomy, output_dir,
                                    branch_id=branch_id, spec=spec, execution=execution)
    plan = load_plan(view, bundle_ref, resource_base, 'lwnet', taxonomy=taxonomy, spec=spec)
    check_classifier_input(view, branch_id, 'lwnet', taxonomy, plan.bundle.taxonomy)
    h, w, channels = view.cube.shape
    half = spec.patch_size // 2
    labels = np.full((h, w), -1, dtype=np.int32)
    reasons = np.full((h, w), int(Reason.INSUFFICIENT_COVERAGE), dtype=np.uint8)
    center_counts = view.valid_samples.sum(axis=2)
    reasons[center_counts == 0] = int(Reason.EMPTY_PIXEL)
    required = _required(plan.parameters, channels)
    # Only coordinate vectors are padded. No padded cube or subcube is materialized.
    row_ids = np.pad(np.arange(h), half, mode='reflect')
    column_ids = np.pad(np.arange(w), half, mode='reflect')
    local_to_canonical = np.array(plan.bundle.output_class_ids, dtype=np.int32)
    with output_directory(output_dir, 'classification') as path:
        features, masks, coordinates = [], [], []
        predict = None

        def flush():
            nonlocal predict
            if not coordinates:
                return
            if predict is None:
                predict = load_backend(plan, view, execution)
            scores = _check_scores(predict(np.stack(features), np.stack(masks)),
                                   len(coordinates), len(local_to_canonical), plan.parameters['output_kind'])
            classes = local_to_canonical[np.argmax(scores, axis=1)]
            for (row, column), label in zip(coordinates, classes):
                labels[row, column] = label
                reasons[row, column] = int(Reason.NONE)
            features.clear()
            masks.clear()
            coordinates.clear()

        for row in inference_range(h, 1, f'LWNet {branch_id}', 'rows'):
            for column in range(w):
                if center_counts[row, column] < required:
                    continue
                if plan.parameters['border_policy'] == 'reject' and (
                        row < half or row >= h - half or column < half or column >= w - half):
                    reasons[row, column] = int(Reason.PATCH_BORDER)
                    continue
                rows = row_ids[row:row + spec.patch_size]
                columns = column_ids[column:column + spec.patch_size]
                patch_mask = view.valid_samples[rows[:, None], columns[None, :], :]
                if patch_mask.mean() < plan.parameters['min_patch_valid_fraction']:
                    continue
                patch = view.cube[rows[:, None], columns[None, :], :]
                normalized = plan.normalize(patch, patch_mask)
                features.append(np.ascontiguousarray(normalized.transpose(2, 0, 1)[None]))
                masks.append(np.ascontiguousarray(patch_mask.transpose(2, 0, 1)[None]))
                coordinates.append((row, column))
                if len(coordinates) == spec.batch_size:
                    flush()
        flush()
        return save_classification(path, view, branch_id, plan.resource_ref, taxonomy, labels, reasons,
                                   view.good_channels, {'model': plan.parameters, 'spec': contract_dict(spec),
                                                        'execution': contract_dict(execution), 'channels': channels})
