"""Explicit frozen neural resources; torch is imported only for actual inference."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np

from modules.pipeline_contracts import ArtifactRef, ModelBundle, integer, number, require
from modules.resource_store import ResourceStore, decode_contract, read_json, verify_ref
from .common import absolute_ref, compatible, exact_axis


@dataclass(frozen=True)
class NeuralPlan:
    bundle: ModelBundle
    resource_ref: ArtifactRef
    bundle_path: Path
    parameters: dict
    mean: np.ndarray
    scale: np.ndarray

    def normalize(self, values, mask):
        # Values have channels LAST; no statistics are fitted on the inference scene.
        with np.errstate(over='ignore', invalid='ignore'):
            transformed = np.where(mask, (values - self.mean) / self.scale, self.parameters['fill_value'])
            transformed = transformed.astype(np.float32)
        require(np.isfinite(transformed).all(), 'Preprocessing produced nonfinite model inputs')
        return transformed


def load_plan(view, bundle_ref, resource_base, algorithm, *, taxonomy=None, spec=None):
    view.require_ready()
    compatible(bundle_ref is not None, 'missing_bundle', 'A frozen trained bundle is required')
    store = ResourceStore(resource_base)
    bundle, path = store.load_contract(bundle_ref, ModelBundle)
    compatible(bundle.algorithm == algorithm, 'algorithm_mismatch', 'Wrong model algorithm')
    if algorithm == 'n2n4m' and bundle.architecture_version == 'N2N4M_Plebani350_original_v1':
        from .pretrained_n2n4m import load_pretrained_plan
        return load_pretrained_plan(view, store, bundle_ref, bundle, path)
    compatible(bundle.domain == view.prepared.domain, 'domain_mismatch', 'Model input domain differs from the cube')
    if taxonomy is not None:
        compatible(bundle.taxonomy == taxonomy, 'taxonomy_mismatch', 'Model and target taxonomies differ')
    if bundle.spectral_axis is not None:
        compatible(exact_axis(view.prepared.spectral_axis, bundle.spectral_axis),
                   'spectral_grid_mismatch', 'Model grid must match the entire cube, including order and boundaries')
    preprocess_path = verify_ref(bundle.preprocessing, path.parent)
    document = read_json(preprocess_path)
    require(isinstance(document, dict) and set(document) == {'schema_version', 'scaler', 'parameters'}
            and type(document['schema_version']) is int and document['schema_version'] == 1,
            'Invalid preprocessing manifest')
    p = document['parameters']
    keys = {'adapter_version', 'backend', 'weights_format', 'architecture', 'units', 'normalization',
            'missing_policy', 'fill_value', 'min_valid_bands', 'min_valid_fraction', 'output_kind'}
    if algorithm == 'lwnet':
        keys |= {'patch_size', 'border_policy', 'min_patch_valid_fraction'}
    optional = {'stage2_unseen_channel_indices', 'scene_binding'}
    require(isinstance(p, dict) and keys <= set(p) <= keys | optional,
            'Unsupported or incomplete adapter preprocessing parameters')
    require(type(p['adapter_version']) is int and p['adapter_version'] == 1, 'Unknown adapter version')
    require(p['backend'] in ('state_dict', 'torchscript_full_axis_v1'), 'Unknown neural backend')
    require(isinstance(p['architecture'], dict), 'architecture must be an object')
    compatible(p['units'] == view.prepared.units, 'units_mismatch', 'Model units differ from the cube')
    if 'scene_binding' in p:
        binding = p['scene_binding']
        require(isinstance(binding, dict) and set(binding) == {
            'training_domain', 'inference_domain', 'target_scene_id', 'target_cube_sha256'},
            'Invalid scene binding')
        source = read_json(verify_ref(view.prepared.source, path.parent))
        training = read_json(verify_ref(bundle.training_manifest, path.parent))
        compatible(binding['inference_domain'] == bundle.domain and
                   binding['training_domain'] == training.get('domain') and
                   training.get('units') == p['units'], 'domain_mismatch',
                   'Scene binding disagrees with pinned training or inference radiometry')
        compatible(binding['target_scene_id'] == view.prepared.scene_id == source.get('scene_id') and
                   binding['target_cube_sha256'] == source['cube']['sha256'], 'scene_mismatch',
                   'Model was exported for a different source scene')
    require(p['normalization'] in ('identity', 'standardize_npz'), 'Unsupported normalization')
    require(p['missing_policy'] == 'constant_normalized', 'Unsupported missing-sample policy')
    number(p['fill_value'], 'fill_value', -np.finfo(np.float32).max, np.finfo(np.float32).max)
    integer(p['min_valid_bands'], 'min_valid_bands', 1)
    number(p['min_valid_fraction'], 'min_valid_fraction', 0, 1)
    channels = view.cube.shape[2]
    if 'stage2_unseen_channel_indices' in p:
        indices = p['stage2_unseen_channel_indices']
        require(bundle.spectral_axis is not None and isinstance(indices, list)
                and all(type(i) is int and 0 <= i < channels for i in indices)
                and indices == sorted(set(indices)), 'Invalid stage-2 unseen channel indices')
    compatible(p['min_valid_bands'] <= channels, 'insufficient_channels', 'Model requires more input bands')
    if algorithm == 'lwnet':
        integer(p['patch_size'], 'patch_size', 1)
        require(p['patch_size'] % 2 == 1 and p['border_policy'] in ('reflect', 'reject'), 'Invalid patch/border policy')
        number(p['min_patch_valid_fraction'], 'min_patch_valid_fraction', 0, 1)
        require(p['output_kind'] in ('logits', 'log_probabilities', 'probabilities'), 'Unknown classifier output kind')
        compatible(spec is not None and spec.patch_size == p['patch_size'],
                   'patch_size_mismatch', 'Inference patch size differs from the training bundle')
    else:
        require(p['output_kind'] == 'normalized_spectrum', 'N2N4M must predict normalized spectra')

    if p['backend'] == 'state_dict':
        require(p['weights_format'] in ('state_dict', 'wrapped_state_dict'), 'Unknown checkpoint format')
        compatible(bundle.input_layout == 'fixed_grid', 'unsupported_layout', 'Native models require an exact fixed grid')
        compatible(all(view.prepared.spectral_axis.allowed_edges), 'unsupported_spectral_boundaries',
                   'Native CNNs cannot respect S/L boundaries or structural gaps; use an axis-aware export')
        if algorithm == 'n2n4m':
            require(bundle.architecture_version == 'Noise2Noise1D'
                    and set(p['architecture']) == {'kernel_size', 'depth', 'num_blocks'}, 'Unknown N2N4M architecture')
            for name in ('kernel_size', 'depth', 'num_blocks'):
                integer(p['architecture'][name], name, 1)
            require(p['architecture']['kernel_size'] >= 3 and p['architecture']['kernel_size'] % 2 == 1,
                    'Noise2Noise1D requires an odd kernel >=3')
            compatible(p['architecture']['num_blocks'] <= channels.bit_length() - 1,
                       'insufficient_channels', 'Input is too short for N2N4M pooling')
        else:
            require(bundle.architecture_version in ('LWNet_1', 'LWNet_2', 'LWNet_3', 'LWNet_4')
                    and not p['architecture'], 'Unknown native LWNet architecture')
            require(p['output_kind'] == 'log_probabilities', 'Native LWNet returns log probabilities')
            compatible(channels >= 17 and p['patch_size'] >= 13, 'input_too_small',
                       'Native LWNet needs C>=17 and odd patch size>=13')
    else:
        require(p['weights_format'] == 'torchscript' and not p['architecture'], 'Invalid full-axis export parameters')

    # Verify the immutable files only after checking the known input incompatibilities.
    store.model_bundle(bundle_ref, algorithm=algorithm)
    mean, scale = np.zeros(channels, dtype=np.float64), np.ones(channels, dtype=np.float64)
    if p['normalization'] == 'identity':
        require(document['scaler'] is None, 'Identity preprocessing must not silently ignore a scaler')
    else:
        require(document['scaler'] is not None, 'standardize_npz requires a pinned scaler')
        scaler = verify_ref(decode_contract(ArtifactRef, document['scaler']), preprocess_path.parent)
        with np.load(scaler, allow_pickle=False) as data:
            require(set(data.files) == {'mean', 'scale'}, 'Scaler must contain only mean and scale')
            mean, scale = np.array(data['mean'], dtype=np.float64), np.array(data['scale'], dtype=np.float64)
        require(mean.shape == scale.shape == (channels,) and np.isfinite(mean).all()
                and np.isfinite(scale).all() and (scale > 0).all(), 'Invalid scaler shape or coefficients')
    mean.flags.writeable = scale.flags.writeable = False
    return NeuralPlan(bundle, absolute_ref(bundle_ref, resource_base), path, p, mean, scale)


def _native_architecture(plan, channels):
    """Import architecture definitions only, without importing legacy inference wrappers."""
    import torch
    root = Path(__file__).resolve().parents[2]
    if plan.bundle.algorithm == 'n2n4m':
        path = root / 'n2n4m-main/n2n4m/model.py'
    else:
        path = root / 'LWNet/training/models/models_lw_3D.py'
    spec = importlib.util.spec_from_file_location('_whole_cube_' + plan.bundle.algorithm, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.device('cpu'):
        if plan.bundle.algorithm == 'n2n4m':
            model = module.Noise2Noise1D(num_input_features=channels, **plan.parameters['architecture'])
        else:
            model = getattr(module, plan.bundle.architecture_version)(num_classes=len(plan.bundle.output_class_ids))
    return model.float()


def load_backend(plan, view, execution, *, native_channels=None):
    import torch
    device = execution.device
    require(device == 'cpu' or execution.gpu_allowed, 'GPU is not authorized by the execution configuration')
    if device != 'cpu':
        compatible(torch.cuda.is_available(), 'device_unavailable', 'Requested CUDA device is unavailable')
    weights = verify_ref(plan.bundle.weights, plan.bundle_path.parent)
    scripted = plan.parameters['backend'] == 'torchscript_full_axis_v1'
    if scripted:
        model = torch.jit.load(str(weights), map_location='cpu')
    else:
        state = torch.load(weights, map_location='cpu', weights_only=True)
        if plan.parameters['weights_format'] == 'wrapped_state_dict':
            require(isinstance(state, dict) and 'state_dict' in state, 'Missing checkpoint state_dict')
            state = state['state_dict']
        require(isinstance(state, dict) and bool(state), 'Checkpoint must contain a nonempty state dict')
        cleaned = {}
        for key, value in state.items():
            require(isinstance(key, str) and isinstance(value, torch.Tensor), 'Invalid state_dict entry')
            key = key.removeprefix('module.')
            require(key not in cleaned, 'Checkpoint key collision after removing module. prefix')
            cleaned[key] = value
        model = _native_architecture(plan, view.cube.shape[2] if native_channels is None else native_channels)
        model.load_state_dict(cleaned, strict=True)
    verify_ref(plan.bundle.weights, plan.bundle_path.parent)
    model = model.to(device).eval()
    axis = view.prepared.spectral_axis
    wavelengths = torch.tensor(axis.wavelengths_nm, dtype=torch.float32, device=device)
    detectors = torch.tensor([1 if d == 'S' else 0 for d in axis.detectors], dtype=torch.int64, device=device)
    edges = torch.tensor(axis.allowed_edges, dtype=torch.bool, device=device)

    def predict(values, valid):
        with torch.inference_mode():
            features = torch.from_numpy(np.ascontiguousarray(values)).to(device)
            if scripted:
                mask = torch.from_numpy(np.ascontiguousarray(valid)).to(device)
                output = model(features, mask, wavelengths, detectors, edges)
            elif plan.bundle.algorithm == 'n2n4m':
                output = model(features.unsqueeze(1)).squeeze(1)
            else:
                output = model(features)
            require(isinstance(output, torch.Tensor) and output.is_floating_point(), 'Model must return a floating tensor')
            return output.detach().cpu().numpy()
    return predict
