"""
metrics.py — Прежние метрики v1 и пространственные характеристики карт.

Строгие GT-метрики и описательное согласие v2 реализованы отдельно
в comparison_metrics.py; whole_cube.py не использует SAM как GT.

Поскольку ground truth отсутствует, эталоном служит карта SAM
на исходных данных (baseline). Метрики измеряют согласованность
с этим baseline + пространственные характеристики карт.

Интерпретация Cohen's kappa:
  > 0.8 — отличная согласованность
  0.6–0.8 — хорошая
  0.4–0.6 — умеренная
  < 0.4 — слабая
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score
from scipy import ndimage
from scipy.ndimage import generic_filter


def compute_agreement(map_a, map_b):
    """Доля пикселей с одинаковым классом (только валидные в обоих)."""
    valid = (map_a >= 0) & (map_b >= 0)
    if valid.sum() == 0:
        return 0.0
    return float(np.mean(map_a[valid] == map_b[valid]))


def compute_kappa(map_a, map_b):
    """Cohen's kappa между двумя картами."""
    valid = (map_a >= 0) & (map_b >= 0)
    if valid.sum() == 0:
        return 0.0
    return float(cohen_kappa_score(map_a[valid], map_b[valid]))


def compute_macro_f1(map_a, map_b):
    """
    macro-F1: F1, усреднённая по классам с одинаковым весом.

    map_a — эталон (SAM baseline)
    map_b — предсказание (другой метод)

    macro-усреднение честно при дисбалансе классов: редкие
    минералы получают тот же вес, что и частые.
    """
    valid = (map_a >= 0) & (map_b >= 0)
    if valid.sum() == 0:
        return 0.0
    return float(f1_score(map_a[valid], map_b[valid], average='macro',
                           zero_division=0))


def compute_class_distribution(class_map):
    """Распределение пикселей по классам."""
    total = class_map.size
    unique, counts = np.unique(class_map[class_map >= 0], return_counts=True)
    return {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}


def compute_spatial_compactness(class_map):
    """Средний размер связной области класса."""
    sizes = []
    for cls in np.unique(class_map[class_map >= 0]):
        mask = (class_map == cls)
        labeled, n = ndimage.label(mask)
        if n > 0:
            component_sizes = ndimage.sum(mask, labeled, range(1, n + 1))
            sizes.extend(component_sizes)
    return float(np.mean(sizes)) if sizes else 0.0


def _majority_diff(values):
    """1, если центральный пиксель отличается от большинства соседей."""
    center = values[len(values) // 2]
    others = np.delete(values, len(values) // 2)
    others = others[others >= 0]
    if len(others) == 0 or center < 0:
        return 0
    return 1 if center != np.bincount(others.astype(int)).argmax() else 0


def compute_salt_pepper(class_map):
    """Доля «одиночных» пикселей (класс отличается от соседей)."""
    is_outlier = generic_filter(
        class_map, _majority_diff, size=3, mode='constant', cval=-1
    )
    return float(np.mean(is_outlier))


def summarize_all(class_map):
    """Сводка метрик одной карты."""
    total = class_map.size
    n_classified = int(np.sum(class_map >= 0))
    dist = compute_class_distribution(class_map)

    if dist:
        dominant_cls = max(dist, key=dist.get)
        dominant_pct = 100 * dist[dominant_cls] / total
    else:
        dominant_cls, dominant_pct = -1, 0.0

    return {
        'classified_pct': 100 * n_classified / total,
        'unclassified_pct': 100 * (total - n_classified) / total,
        'n_unique_classes': len(dist),
        'dominant_class': dominant_cls,
        'dominant_pct': dominant_pct,
        'compactness_px': compute_spatial_compactness(class_map),
        'salt_pepper_pct': 100 * compute_salt_pepper(class_map),
    }
