"""SAM on common finite measurements; missing samples never act as zeros."""
import numpy as np
from modules.validity import valid_samples


def _angles(pixels, references, valid, min_valid_bands, min_valid_fraction):
    channels = pixels.shape[1]
    if not isinstance(min_valid_bands, (int, np.integer)) or min_valid_bands < 2:
        raise ValueError('min_valid_bands должен быть целым числом >= 2')
    if not 0 <= min_valid_fraction <= 1:
        raise ValueError('min_valid_fraction должен быть в [0, 1]')
    required = max(min_valid_bands, int(np.ceil(channels * min_valid_fraction)))
    ref_valid = np.isfinite(references)
    p = np.where(valid, pixels, 0.0).astype(np.float64)
    r = np.where(ref_valid, references, 0.0).astype(np.float64)
    p /= np.maximum(np.max(np.abs(p), axis=1, keepdims=True), np.finfo(float).tiny)
    r /= np.maximum(np.max(np.abs(r), axis=1, keepdims=True), np.finfo(float).tiny)
    pv, rv = valid.astype(float), ref_valid.astype(float)
    counts = pv @ rv.T
    denominator = np.sqrt((p * p) @ rv.T) * np.sqrt(pv @ (r * r).T)
    usable = (counts >= required) & (denominator > 0)
    cosine = np.zeros_like(denominator)
    np.divide(p @ r.T, denominator, out=cosine, where=usable)
    return np.where(usable, np.arccos(np.clip(cosine, -1, 1)), np.inf)


def spectral_angle(pixel, reference, min_valid_bands=2, min_valid_fraction=0.5):
    """Radians; infinity for insufficient common support or zero norms."""
    pixel, reference = np.asarray(pixel), np.asarray(reference)
    if pixel.ndim != 1 or pixel.shape != reference.shape or not pixel.size:
        raise ValueError('Спектры должны быть непустыми векторами одинаковой длины')
    return float(_angles(pixel[None], reference[None], np.isfinite(pixel)[None],
                         min_valid_bands, min_valid_fraction)[0, 0])


def sam_classify_pixel(pixel, references, threshold=0.1, **kwargs):
    pixel = np.asarray(pixel)
    if pixel.ndim != 1:
        raise ValueError('Ожидается одномерный спектр')
    return int(sam_classify_cube(pixel[None, None], references, threshold,
                                verbose=False, **kwargs)[0, 0])


def sam_classify_cube(cube, references, threshold=0.1, verbose=True,
                      valid_mask=None, min_valid_bands=2, min_valid_fraction=0.5,
                      batch_size=4096):
    """Classify on pairwise common support, in bounded-memory batches.

    Default coverage: at least two samples and half the supplied spectral grid.
    These safeguards are configurable, not calibrated mineral-detection criteria.
    """
    cube, references = np.asarray(cube), np.asarray(references)
    valid = valid_samples(cube, valid_mask)
    if (references.ndim != 2 or references.shape[0] == 0
            or references.shape[1] != cube.shape[2]):
        raise ValueError('Эталоны должны иметь форму (N > 0, C)')
    if not np.isfinite(threshold) or not 0 <= threshold <= np.pi:
        raise ValueError('Порог SAM должен быть в [0, pi]')
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('batch_size должен быть положительным целым')
    flat = cube.reshape(-1, cube.shape[2])
    valid = valid.reshape(flat.shape)
    result = np.full(len(flat), -1, dtype=int)
    for start in range(0, len(flat), batch_size):
        stop = start + batch_size
        angles = _angles(flat[start:stop], references, valid[start:stop],
                         min_valid_bands, min_valid_fraction)
        best = np.argmin(angles, axis=1)
        result[start:stop] = np.where(angles[np.arange(len(best)), best] <= threshold, best, -1)
    if verbose:
        print(f'[SAM] Классифицировано: {np.count_nonzero(result >= 0)} / {len(result)}')
    return result.reshape(cube.shape[:2])
