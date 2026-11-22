"""Validity of measurements, separate from numerical imputation."""
import numpy as np


def valid_samples(cube, valid_mask=None):
    cube = np.asarray(cube)
    if cube.ndim != 3 or any(size == 0 for size in cube.shape):
        raise ValueError('Ожидается непустой куб (H, W, C)')
    valid = np.isfinite(cube)
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.dtype != np.bool_:
            raise ValueError('Маска валидности должна быть булевой')
        if mask.shape == cube.shape[:2]:
            mask = mask[..., None]
        elif mask.shape != cube.shape:
            raise ValueError('Форма маски должна быть (H, W) или (H, W, C)')
        valid &= mask
    return valid


def restore_missing(result, original_valid):
    result = np.asarray(result)
    if result.shape != original_valid.shape:
        raise ValueError('Форма результата не соответствует исходной маске')
    return np.where(original_valid & np.isfinite(result), result, np.nan)
