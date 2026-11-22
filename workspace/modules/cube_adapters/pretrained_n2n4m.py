"""Frozen upstream Plebani-350 inference on the measured native L-detector grid."""
from io import BytesIO
import hashlib

import numpy as np

from modules.pipeline_contracts import ArtifactRef, DenoisingResult, contract_dict, require
from modules.resource_store import array_reference, decode_contract, read_json, verify_ref
from .backends import NeuralPlan, load_backend
from .common import absolute_ref, close_array, compatible, complete_result, inference_range, output_directory


def load_pretrained_plan(view, store, bundle_ref, bundle, path):
    # The original artifacts are immutable. In particular, never unpickle a
    # replacement joblib artifact just because its hash matches a caller's JSON.
    require(bundle.domain == 'crism_trr3_if' and bundle.input_layout == 'segmented'
            and bundle.spectral_axis is None
            and bundle.weights.sha256 == '8cac162c44891e372834d076b2ff7e0c10536f5e780632895ecae8251bd56bc9'
            and bundle.preprocessing.sha256 == '32800f4574de40c4e94d714c1860658d1466a864a03da2579f77d531d3a159ac',
            'Expected the original N2N4M checkpoint and native-grid preprocessing')
    store.model_bundle(bundle_ref, algorithm='n2n4m')
    preprocess_path = verify_ref(bundle.preprocessing, path.parent)
    document = read_json(preprocess_path)
    parameters = document['parameters']
    compatible(view.prepared.units == parameters['units'], 'units_mismatch', 'Expected dimensionless I/F')
    compatible(view.prepared.domain in parameters['source_domains'], 'domain_mismatch',
               'Input is outside the documented corrected-I/F transfer domains')
    import joblib
    scaler_ref = decode_contract(ArtifactRef, document['scaler'])
    scaler_bytes = verify_ref(scaler_ref, preprocess_path.parent).read_bytes()
    require(hashlib.sha256(scaler_bytes).hexdigest() ==
            '3882e60f38016b8d37fb89f859ed160308f40b57dd7cd8cf4d489396b70ea45c',
            'Expected the distributed RobustScaler')
    scaler = joblib.load(BytesIO(scaler_bytes))
    center, scale = scaler.center_, scaler.scale_
    require(center.shape == scale.shape == (350,) and np.isfinite(center).all()
            and np.isfinite(scale).all() and (scale > 0).all(), 'Invalid robust scaler')
    parameters = dict(parameters, weights_format='state_dict',
                      architecture=dict(kernel_size=5, depth=3, num_blocks=4))
    return NeuralPlan(bundle, absolute_ref(bundle_ref, store.base), path, parameters, center, scale)


def _common_indices(axis, good, native, tolerance):
    """Match only measured L band centres. No interpolation or extrapolation."""
    waves = np.asarray(axis.wavelengths_nm)
    matches = np.abs(waves[:, None] - native) <= tolerance
    matches &= (np.asarray(axis.detectors) == 'L')[:, None] & good[:, None]
    require(matches.sum(axis=0).max() <= 1 and matches.sum(axis=1).max() <= 1,
            'Ambiguous native wavelength matches')
    return np.nonzero(matches)


def _predict_native(values, usable, model_indices, plan, predict):
    # The fixed CNN was trained on this ordered 350-vector, which includes
    # intentional gaps in physical wavelength. Keep all original positions.
    features = np.zeros((len(values), 350), dtype=np.float32)
    measured = np.where(usable, values, plan.mean[model_indices])
    features[:, model_indices] = (measured - plan.mean[model_indices]) / plan.scale[model_indices]
    require(np.isfinite(features).all(), 'Nonfinite standardized I/F')
    prediction = predict(features, None)
    require(prediction.shape == features.shape and np.isfinite(prediction).all(),
            'Invalid pretrained N2N4M prediction')
    # Upstream train.py scales X only; the model predicts unscaled I/F targets.
    return prediction[:, model_indices]


def denoise_native_grid(view, plan, output_dir, spec, execution):
    p = plan.parameters
    native = np.asarray(p['native_wavelengths_nm'])
    common, model_indices = _common_indices(view.prepared.spectral_axis, view.good_channels,
                                            native, p['match_atol_nm'])
    compatible(len(common) >= p['min_valid_bands'], 'no_denoised_support',
               'Insufficient common measured native L bands')
    shape = view.cube.shape
    source = view.cube.reshape(-1, shape[-1])
    measured = view.valid_samples.reshape(source.shape)
    with output_directory(output_dir, 'denoising') as path:
        output = denoised = None
        try:
            output = np.lib.format.open_memmap(path / 'cube.npy', mode='w+', shape=shape, dtype=source.dtype)
            denoised = np.lib.format.open_memmap(path / 'denoised_mask.npy', mode='w+', shape=shape, dtype=bool)
            out, mask = output.reshape(source.shape), denoised.reshape(source.shape)
            predict = None
            spectra = samples = differences = 0
            coverage_histogram = np.zeros(len(native) + 1, dtype=np.int64)
            for start in inference_range(len(source), spec.batch_size, 'N2N4M pretrained', 'pixels'):
                block = slice(start, start + spec.batch_size)
                values = source[block]
                common_values = values[:, common]
                usable = measured[block][:, common] & (common_values <= p['maximum_input_if'])
                counts = usable.sum(axis=1)
                coverage_histogram += np.bincount(counts, minlength=len(coverage_histogram))
                rows = np.flatnonzero(counts >= p['min_valid_bands'])
                restored = values.copy()
                processed = np.zeros(values.shape, dtype=bool)
                if len(rows):
                    if predict is None:
                        predict = load_backend(plan, view, execution, native_channels=350)
                    prediction = _predict_native(common_values[rows], usable[rows], model_indices, plan, predict)
                    idx = np.ix_(rows, common)
                    restored[idx] = np.where(usable[rows], prediction, common_values[rows])
                    processed[idx] = usable[rows]
                out[block] = restored
                mask[block] = processed
                spectra += int(processed.any(axis=1).sum())
                samples += int(processed.sum())
                differences += int((processed & (restored != values)).sum())
            compatible(samples > 0, 'no_denoised_support', 'No spectra have sufficient measured native coverage')
            output.flush()
            denoised.flush()
            result = DenoisingResult(view.prepared, plan.resource_ref,
                                    array_reference(path / 'cube.npy', 'n2n4m_cube', path), view.valid_ref,
                                    array_reference(path / 'denoised_mask.npy', 'denoised_mask', path))
            result.validate_arrays(view.cube, view.valid_samples, output, view.valid_samples, denoised,
                                   good_channels=view.good_channels)
            parameters = dict(model=p, training_domain=plan.bundle.domain, input_domain=view.prepared.domain,
                              execution=contract_dict(execution), batch_size=spec.batch_size, channels=shape[-1],
                              common_channels=len(common), common_scene_indices=common.tolist(),
                              common_native_indices=model_indices.tolist(),
                              measured_native_count_histogram=coverage_histogram.tolist(),
                              internally_imputed_inputs=spectra * len(native) - samples,
                              processed_spectra=spectra, processed_samples=samples,
                              numerically_changed_samples=differences)
            close_array(output)
            close_array(denoised)
            output = denoised = None
            complete_result(path, 'denoising', result, parameters)
            return result
        finally:
            close_array(output)
            close_array(denoised)
