"""Pure report figures: no files, model loading, pyplot or global style changes."""
import colorsys
import math
import textwrap

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.text import Text

from .class_names_ru import to_ru, to_ru_list
from .pipeline_contracts import BRANCHES


BRANCH_ORDER = tuple(BRANCHES)
BRANCH_LABELS = {
    'sam_original': 'SAM · исходный',
    'sam_denoised': 'N2N4M → SAM',
    'lwnet_original': 'LWNet · исходный',
    'lwnet_denoised': 'N2N4M → LWNet',
}


def palette(class_names):
    """Assign stable colors by canonical ID, independent of observed classes."""
    colors = [to_hex(colorsys.hsv_to_rgb((i * .618033988749895) % 1,
                                        .65, .82)) for i in range(len(class_names))]
    return {'classes': colors, 'missing': '#e5e7eb', 'abstention': '#4b5563'}


def _wrap(value, width=25):
    return '\n'.join(textwrap.wrap(str(value), width=width))


def _finish(fig):
    for artist in fig.findobj(match=Text):
        artist.set_fontfamily('DejaVu Sans')
    return fig


def _panels(title, height=8):
    fig = Figure(figsize=(13, height))
    axes = fig.subplots(2, 2).flat
    axes = list(axes)
    fig.suptitle(title, fontsize=15)
    for branch, ax in zip(BRANCH_ORDER, axes):
        ax.set_title(BRANCH_LABELS[branch])
    fig.subplots_adjust(top=.9, bottom=.08, hspace=.4, wspace=.3)
    return fig, axes


def _unavailable(ax, outcome):
    outcome = outcome or {}
    status = outcome.get('status', 'missing')
    translated = {'skipped': 'Пропущено', 'failed': 'Ошибка',
                  'missing': 'Нет данных', 'unavailable': 'Недоступно',
                  'completed': 'Завершено', 'ok': 'Готово'}
    message = f"Статус: {translated.get(status, status)}"
    reason = outcome.get('reason_code')
    if reason:
        message += '\nПричина: ' + _wrap(reason, 40)
    ax.text(.5, .5, message, ha='center', va='center', transform=ax.transAxes)
    ax.set_axis_off()


def _legend(fig, categories):
    cols = min(3, max(1, len(categories)))
    rows = math.ceil(len(categories) / cols)
    line_count = max((len(_wrap(label).splitlines()) for label, _ in categories), default=1)
    height = rows * (.23 * line_count + .12) + .3
    fig.set_figheight(fig.get_figheight() + height)
    fig.subplots_adjust(bottom=height / fig.get_figheight())
    fig.legend(handles=[Patch(facecolor=color, label=_wrap(label))
                        for label, color in categories], loc='lower center',
               ncol=cols, frameon=False, fontsize=9)


def _image(ax, values, colors):
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(colors) + 1) - .5, len(colors))
    ax.imshow(values, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])


def sam_baseline_map(prediction, valid, class_names, palette, *, gt=None):
    """Show the requested full-frame baseline and optional held-out labels at the same scale."""
    k = len(class_names)
    colors = [*palette['classes'], palette['abstention'], palette['missing']]
    panels = [('SAM · исходный куб', np.where(~valid, k + 1, np.where(prediction < 0, k, prediction)))]
    if gt is not None:
        panels.append(('Эталонная разметка', np.where(gt < 0, k + 1, gt)))
    fig = Figure(figsize=(7 * len(panels), 5.4))
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    shown = set()
    for ax, (title, values) in zip(axes, panels):
        _image(ax, values, colors)
        ax.set_title(title)
        shown.update(np.unique(values))
    labels = [*to_ru_list(class_names), 'Отказ SAM', 'Нет данных или метки']
    _legend(fig, [(labels[i], colors[i]) for i in sorted(shown)])
    fig.subplots_adjust(left=.02, right=.98, wspace=.06)
    return _finish(fig)


def class_maps(maps, valid_pixels, class_names, outcomes, palette):
    """Show class IDs, valid abstentions and invalid pixels as distinct colors."""
    count = len(class_names)
    colors = list(palette['classes']) + [palette['abstention'], palette['missing']]
    if len(palette['classes']) != count:
        raise ValueError('Palette must contain every canonical class')
    valid = np.asarray(valid_pixels)
    if valid.ndim != 2 or valid.dtype != np.bool_:
        raise ValueError('valid_pixels must be a two-dimensional boolean mask')
    fig, axes = _panels('Карты классов')
    for branch, ax in zip(BRANCH_ORDER, axes):
        if maps.get(branch) is None:
            _unavailable(ax, outcomes.get(branch))
            continue
        values = np.asarray(maps[branch])
        if (values.shape != valid.shape or values.dtype.kind not in 'iu'
                or np.any((values < -1) | (values >= count))):
            raise ValueError('Invalid class map')
        encoded = np.where(~valid, count + 1, np.where(values == -1, count, values))
        _image(ax, encoded, colors)
    categories = [(f'{i}: {to_ru(name)}', colors[i]) for i, name in enumerate(class_names)]
    categories += [('Отказ (−1)', palette['abstention']), ('Нет данных', palette['missing'])]
    _legend(fig, categories)
    return _finish(fig)


def category_maps(values, title, categories, outcomes):
    """Show discrete category IDs with one shared legend across all branches."""
    fig, axes = _panels(title)
    for branch, ax in zip(BRANCH_ORDER, axes):
        if values.get(branch) is None:
            _unavailable(ax, outcomes.get(branch))
            continue
        data = np.asarray(values[branch])
        if (data.ndim != 2 or data.dtype.kind not in 'iu'
                or np.any((data < 0) | (data >= len(categories)))):
            raise ValueError('Invalid category map')
        _image(ax, data, [color for _, color in categories])
    _legend(fig, categories)
    return _finish(fig)


def metrics_bars(rows, title, series, outcomes):
    """Compare metrics without replacing undefined values by zero."""
    fig = Figure(figsize=(13, 6))
    ax = fig.subplots()
    ax.set_title(title)
    width = .8 / max(1, len(series))
    colors = palette(range(len(series)))['classes']
    for index, (field, label) in enumerate(series):
        for group, branch in enumerate(BRANCH_ORDER):
            x = group + (index - (len(series) - 1) / 2) * width
            value = (rows.get(branch) or {}).get(field)
            if value is None:
                ax.text(x, .025, 'н/д', ha='center', va='bottom', rotation=90, fontsize=9)
            else:
                ax.bar(x, value, width=width, color=colors[index])
    ax.set_xticks(range(4), [BRANCH_LABELS[key] for key in BRANCH_ORDER])
    lower = -1 if any('kappa' in field.lower() for field, _ in series) else 0
    ax.set_ylim(lower, 1)
    ax.set_ylabel('Значение метрики')
    ax.axhline(0, linewidth=.6, color='#6b7280')
    for index, branch in enumerate(BRANCH_ORDER):
        if rows.get(branch) is None:
            outcome = outcomes.get(branch) or {}
            reason = outcome.get('reason_code') or outcome.get('status', 'нет данных')
            ax.text(index, .12, _wrap(reason, 18), ha='center',
                    transform=ax.get_xaxis_transform(), fontsize=8)
    fig.subplots_adjust(bottom=.17, top=.88)
    _legend(fig, [(label, colors[i]) for i, (_, label) in enumerate(series)])
    return _finish(fig)


def distributions(coverage, class_names, palette, outcomes):
    """Include every canonical class, including classes with zero predictions."""
    count = len(class_names)
    fig, axes = _panels('Распределение предсказанных классов', max(8, count * .65))
    labels = [f'{i}: {_wrap(to_ru(name))}' for i, name in enumerate(class_names)]
    for branch, ax in zip(BRANCH_ORDER, axes):
        if coverage.get(branch) is None:
            _unavailable(ax, outcomes.get(branch))
            continue
        counts = np.asarray(coverage[branch]['class_counts'])
        if counts.shape != (count,) or np.any(counts < 0):
            raise ValueError('class_counts must contain all canonical classes')
        ax.barh(range(count), counts, color=palette['classes'])
        ax.set_yticks(range(count), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Число пикселей')
    fig.tight_layout(rect=(0, 0, 1, .95))
    return _finish(fig)


def confusion_matrix(matrix, class_names, title):
    """Render raw counts; the final prediction column is always abstention."""
    count = len(class_names)
    data = np.asarray(matrix)
    if data.shape != (count, count + 1) or np.any(data < 0):
        raise ValueError('Expected a nonnegative K × (K + 1) confusion matrix')
    fig = Figure(figsize=(min(24, max(9, count * .65)), min(22, max(7, count * .6))))
    ax = fig.subplots()
    img = ax.imshow(data, interpolation='nearest', cmap='Blues', vmin=0)
    labels = [f'{i}: {_wrap(to_ru(name), 20)}' for i, name in enumerate(class_names)]
    ax.set_xticks(range(count + 1), labels + ['Отказ (−1)'], rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(count), labels, fontsize=8)
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Эталонный класс (GT)')
    ax.set_title(title)
    if count <= 15:
        maximum = data.max(initial=0)
        for (row, column), value in np.ndenumerate(data):
            ax.text(column, row, str(int(value)), ha='center', va='center',
                    color='white' if value > maximum / 2 else 'black', fontsize=8)
    fig.colorbar(img, ax=ax, label='Число пикселей')
    fig.tight_layout()
    return _finish(fig)
