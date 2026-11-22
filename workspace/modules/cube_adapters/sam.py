"""Whole-cube SAM with reference-only, detector-aware spectral alignment."""
import numpy as np

from modules.cube_adapters.common import (
    absolute_ref, check_classifier_input, compatible, inference_range, output_directory,
    read_array, save_classification,
)
from modules.pipeline_config import ExecutionSpec, SamSpec
from modules.pipeline_contracts import ClassificationResult, RejectionReason, contract_dict, require
from modules.resource_store import ResourceStore, resolved_path
from modules.sam_core import _angles


def _align_references(spectra, valid, source_axis, target_axis):
    """Keep every target position; unsupported reference samples remain NaN.

    Interpolation uses adjacent *original* reference channels, never jumps a
    masked measurement, segment boundary, detector boundary or missing band.
    """
    wavelengths = np.asarray(source_axis.wavelengths_nm)
    detectors = np.asarray(source_axis.detectors)
    for detector in ('S', 'L'):
        ordered = np.sort(wavelengths[detectors == detector])
        compatible(not np.any(np.diff(ordered) <= 1e-8), 'ambiguous_reference_axis',
                   'Duplicate reference wavelengths within a detector')
    clean = np.where(valid & np.isfinite(spectra), spectra, np.nan)
    aligned = np.full((len(spectra), len(target_axis.channel_indices)), np.nan, dtype=np.float64)
    for target, (wavelength, detector) in enumerate(zip(target_axis.wavelengths_nm, target_axis.detectors)):
        exact = np.flatnonzero((detectors == detector) & np.isclose(wavelengths, wavelength, rtol=0, atol=1e-8))
        compatible(len(exact) <= 1, 'ambiguous_reference_axis',
                   'Multiple reference wavelengths match a target channel')
        if len(exact):
            aligned[:, target] = clean[:, exact[0]]
            continue
        candidates = [i for i, allowed in enumerate(source_axis.allowed_edges)
                      if allowed and detectors[i] == detector
                      and wavelengths[i] < wavelength < wavelengths[i + 1]
                      and wavelengths[i + 1] - wavelengths[i] <= 7.5]
        compatible(len(candidates) <= 1, 'ambiguous_reference_axis',
                   'Overlapping reference segments give ambiguous interpolation')
        if candidates:
            i = candidates[0]
            fraction = (wavelength - wavelengths[i]) / (wavelengths[i + 1] - wavelengths[i])
            aligned[:, target] = clean[:, i] * (1 - fraction) + clean[:, i + 1] * fraction
    return aligned


def classify_sam(view, library_ref, resource_base, taxonomy, output_dir, *,
                 branch_id='sam_original', spec=SamSpec(), batch_size=4096,
                 execution=ExecutionSpec()) -> ClassificationResult:
    """Classify using common finite support measured against the full C axis."""
    require(isinstance(spec, SamSpec), 'Expected SAM specification')
    require(spec.min_valid_bands >= 2, 'SAM requires at least two common valid bands')
    require(type(batch_size) is int and batch_size > 0, 'batch_size must be a positive integer')
    view.require_ready()
    compatible(library_ref is not None, 'missing_library', 'A pinned reference library is required')
    library = ResourceStore(resource_base).reference_library(library_ref)
    check_classifier_input(view, branch_id, 'sam', taxonomy, library.taxonomy)
    compatible(view.prepared.units == library.units, 'units_mismatch', 'SAM reference units disagree with cube')
    corrected_if = {'crism_mtrdr_corrected_if', 'crism_ter_corrected_if', 'hymars_photometric_atmospheric_corrected_if'}
    compatible(view.prepared.domain == library.domain or
               (library.domain == 'mica_ter_corrected_numerator_if' and view.prepared.domain in corrected_if),
               'domain_mismatch', 'SAM reference domain disagrees with cube')
    base = resolved_path(library_ref.path, resource_base).parent
    with read_array(library.spectra, base) as spectra, read_array(library.valid_samples, base) as valid:
        references = _align_references(spectra, valid, library.spectral_axis, view.prepared.spectral_axis)
    used_channels = np.isfinite(references).any(axis=0) & view.good_channels
    references[:, ~view.good_channels] = np.nan
    channels = view.cube.shape[2]
    pixels = view.cube.reshape(-1, channels)
    masks = view.valid_samples.reshape(pixels.shape)
    labels = np.full(len(pixels), -1, dtype=np.int32)
    reasons = np.full(len(pixels), RejectionReason.INSUFFICIENT_COVERAGE, dtype=np.uint8)
    class_ids = np.asarray(library.class_ids, dtype=np.int32)
    device_name = 'CPU'
    if execution.device == 'cpu':
        def angles_for_batch(values, valid):
            return _angles(values, references, valid, spec.min_valid_bands, spec.min_valid_fraction)
    else:
        import torch
        compatible(execution.gpu_allowed and torch.cuda.is_available(), 'device_unavailable',
                   'Requested SAM CUDA device unavailable')
        angles_for_batch = _cuda_angles(references, spec, execution.device)
        device_name = torch.cuda.get_device_name(execution.device)
    for start in inference_range(len(pixels), batch_size, f'SAM {branch_id}', 'pixels'):
        stop = min(start + batch_size, len(pixels))
        angles = angles_for_batch(pixels[start:stop], masks[start:stop])
        best = np.argmin(angles, axis=1)
        minimum = angles[np.arange(len(best)), best]
        accepted = minimum <= spec.threshold_radians
        labels[start:stop] = np.where(accepted, class_ids[best], -1)
        reasons[start:stop] = np.where(
            ~masks[start:stop].any(axis=1), RejectionReason.EMPTY_PIXEL,
            np.where(accepted, RejectionReason.NONE,
                     np.where(np.isfinite(minimum), RejectionReason.THRESHOLD_REJECTION,
                              RejectionReason.INSUFFICIENT_COVERAGE)))
    parameters = {'sam': contract_dict(spec), 'batch_size': batch_size, 'device': execution.device,
                  'device_name': device_name, 'reference_domain': library.domain, 'cube_domain': view.prepared.domain,
                  'supported_channel_fraction': float(used_channels.mean()),
                  'reference_supported_channel_fractions': np.isfinite(references).mean(axis=1).tolist(),
                  'alignment': 'same_detector_adjacent_segment_linear_no_extrapolation'}
    with output_directory(output_dir, 'classification') as path:
        return save_classification(path, view, branch_id, absolute_ref(library_ref, resource_base), taxonomy,
                                   labels.reshape(view.cube.shape[:2]), reasons.reshape(view.cube.shape[:2]),
                                   used_channels, parameters)


def _cuda_angles(references, spec, device):
    """Same pairwise common-support float64 calculation as sam_core, in GPU batches."""
    import torch
    refs = torch.tensor(references, device=device, dtype=torch.float64)
    finite = torch.isfinite(refs)
    refs = torch.where(finite, refs, 0.)
    tiny = torch.finfo(refs.dtype).tiny
    refs /= refs.abs().amax(dim=1, keepdim=True).clamp_min(tiny)
    rv = finite.to(refs.dtype).T.contiguous()
    rt = refs.T.contiguous()
    squared = rt.square()
    minimum = max(spec.min_valid_bands, int(np.ceil(refs.shape[1] * spec.min_valid_fraction)))

    def angles(values, valid):
        p = torch.tensor(values, device=device, dtype=torch.float64)
        mask = torch.tensor(valid, device=device, dtype=torch.bool)
        p = torch.where(mask, p, 0.)
        p /= p.abs().amax(dim=1, keepdim=True).clamp_min(tiny)
        pv = mask.to(p.dtype)
        denominator = (p.square() @ rv).sqrt() * (pv @ squared).sqrt()
        usable = (pv @ rv >= minimum) & (denominator > 0)
        cosine = ((p @ rt) / denominator).clamp(-1, 1)
        return torch.where(usable, cosine.acos(), torch.inf).cpu().numpy()

    return angles
