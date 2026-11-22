"""Figures and an offline HTML report for one immutable comparison revision."""
import argparse
from html import escape
from pathlib import Path

import numpy as np

from modules.class_names_ru import to_ru
from modules.pipeline_contracts import ArtifactRef, VisualizationResult, contract_dict
from modules.resource_store import sha256_file
from modules.run_store import unique_directory, write_manifest
from modules.visualization_inputs import open_visualization_inputs


SAM_CATEGORIES = [('Вне исходной области', '#e5e7eb'), ('Совпадают', '#15803d'),
                  ('Разные классы', '#dc2626'), ('Предсказал только SAM', '#f59e0b'),
                  ('Предсказал только метод', '#2563eb'), ('Оба отказались', '#4b5563')]
GT_CATEGORIES = [('Вне GT-области', '#e5e7eb'), ('Верный класс', '#15803d'),
                 ('Ошибка класса', '#dc2626'), ('Отказ предсказания', '#4b5563')]
REASON_CATEGORIES = [('Есть предсказание', '#15803d'), ('Пустой пиксель', '#e5e7eb'),
                     ('Недостаточно каналов / контекста', '#f59e0b'), ('Край патча', '#7c3aed'),
                     ('Порог SAM', '#dc2626'), ('Отказ модели', '#4b5563')]


def sam_difference(left, right, valid):
    values = np.zeros(valid.shape, dtype=np.uint8)
    a, b = left >= 0, right >= 0
    values[valid & a & b & (left == right)] = 1
    values[valid & a & b & (left != right)] = 2
    values[valid & a & ~b] = 3
    values[valid & ~a & b] = 4
    values[valid & ~a & ~b] = 5
    return values


def gt_difference(labels, prediction, roi):
    values = np.zeros(roi.shape, dtype=np.uint8)
    values[roi & (prediction >= 0) & (prediction == labels)] = 1
    values[roi & (prediction >= 0) & (prediction != labels)] = 2
    values[roi & (prediction < 0)] = 3
    return values


def _ref(path, name, base):
    return ArtifactRef(name, path.relative_to(base).as_posix(), sha256_file(path))


def _native_palettes(view):
    from modules.comparison_plots import BRANCH_ORDER, palette
    spaces = dict.fromkeys([view.result.taxonomy] + [view.taxonomies[key] for key in BRANCH_ORDER if key in view.taxonomies])
    colors = palette(range(sum(len(space.class_names) for space in spaces)))
    offset = 0
    for space in spaces:
        size = len(space.class_names)
        spaces[space] = {**colors, 'classes': colors['classes'][offset:offset + size]}
        offset += size
    return spaces


def _native_class_maps(view, palettes):
    from modules import comparison_plots as plots
    categories = [('Нет данных', '#e5e7eb'), ('Отказ', '#4b5563')]
    lookup = {}
    for space, colors in palettes.items():
        observed = {int(i) for key, labels in view.maps.items() if view.taxonomies[key] == space
                    for i in np.unique(labels) if i >= 0}
        lookup[space] = np.zeros(len(space.class_names), dtype=np.int32)
        for i in sorted(observed):
            lookup[space][i] = len(categories)
            categories.append((f'{space.taxonomy_id} · {i}: {to_ru(space.class_names[i])}', colors['classes'][i]))
    display = {}
    for key, labels in view.maps.items():
        class_colors = lookup[view.taxonomies[key]][labels]
        display[key] = np.where(~view.valid, 0, np.where(labels < 0, 1, class_colors))
    fig = plots.category_maps(display, 'Карты в исходных системах классов', categories, view.outcomes)
    for key, axis in zip(plots.BRANCH_ORDER, fig.axes):
        if key in view.taxonomies:
            axis.set_title(f'{plots.BRANCH_LABELS[key]} · {view.taxonomies[key].taxonomy_id}')
    return fig


def _native_distributions(view, palettes):
    from matplotlib.figure import Figure
    from modules.comparison_plots import BRANCH_ORDER, BRANCH_LABELS
    fig = Figure(figsize=(15, max(8, max(len(s.class_names) for s in palettes) * .55)))
    for key, ax in zip(BRANCH_ORDER, fig.subplots(2, 2).flat):
        if key not in view.maps:
            ax.set_title(BRANCH_LABELS[key])
            ax.text(.5, .5, view.outcomes[key]['reason_code'] or view.outcomes[key]['status'],
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue
        space = view.taxonomies[key]
        labels = [f'{i}: {to_ru(name)}' for i, name in enumerate(space.class_names)]
        ax.barh(range(len(labels)), view.coverage[key]['class_counts'], color=palettes[space]['classes'])
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Число пикселей')
        ax.set_title(f'{BRANCH_LABELS[key]} · {space.taxonomy_id}')
    fig.tight_layout()
    return fig


def _class_differences(view, reference):
    if reference == 'gt':
        return {key: gt_difference(view.gt, labels, view.gt_mask) for key, labels in view.maps.items()
                if view.taxonomies[key] == view.result.taxonomy}
    if reference not in view.maps:
        return {}
    return {key: sam_difference(view.maps[reference], labels, view.valid) for key, labels in view.maps.items()
            if view.taxonomies[key] == view.taxonomies[reference]}


def _difference_outcomes(view, differences, reference_available):
    outcomes = dict(view.outcomes)
    for key in view.maps.keys() - differences.keys():
        outcomes[key] = {'status': 'unavailable', 'reason_code': 'Разные системы классов' if reference_available
                         else 'Исходный SAM недоступен'}
    return outcomes


def _gt_overview(view, colors):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.figure import Figure
    fig = Figure(figsize=(12, 5))
    axes = fig.subplots(1, 2)
    k = len(colors['classes'])
    cmap = ListedColormap(colors['classes'] + [colors['missing']])
    axes[0].imshow(np.where(view.gt >= 0, view.gt, k), cmap=cmap,
                   norm=BoundaryNorm(np.arange(k + 2) - .5, k + 1), interpolation='nearest')
    axes[0].set_title('Каноническая GT (вся разметка)')
    axes[1].imshow(view.gt_mask, cmap=ListedColormap(['#e5e7eb', '#0f766e']), vmin=0, vmax=1,
                   interpolation='nearest')
    axes[1].set_title(f'Фиксированная GT-область: {int(view.gt_mask.sum())} пикселей')
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f'GT и область оценки · {view.result.taxonomy.taxonomy_id}')
    fig.tight_layout()
    return fig


def _pairwise(view):
    from matplotlib.figure import Figure
    from modules.comparison_plots import BRANCH_ORDER, BRANCH_LABELS
    arrays = [np.full((4, 4), np.nan) for _ in range(2)]
    for i, left in enumerate(BRANCH_ORDER):
        for j, right in enumerate(BRANCH_ORDER):
            if i == j:
                n = int((view.maps[left] >= 0).sum()) if left in view.maps else 0
                scores = [1 if n else None, n / view.valid.sum() if view.valid.any() and left in view.maps else None]
            else:
                key = '__'.join((left, right) if i < j else (right, left))
                pair = view.agreement['pairs'][key]
                scores = [pair.get('agreement_on_joint'), pair.get('coverage_joint')]
            for array, score in zip(arrays, scores):
                if score is not None:
                    array[i, j] = score
    fig = Figure(figsize=(13, 6))
    axes = fig.subplots(1, 2)
    for ax, array, title in zip(axes, arrays, ('Согласие на совместных предсказаниях', 'Покрытие совместными предсказаниями')):
        image = ax.imshow(np.ma.masked_invalid(array), cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        ax.set_facecolor('#e5e7eb')
        ax.set_xticks(range(4), [BRANCH_LABELS[b] for b in BRANCH_ORDER], rotation=35, ha='right', fontsize=9)
        ax.set_yticks(range(4), [BRANCH_LABELS[b] for b in BRANCH_ORDER], fontsize=9)
        ax.set_title(title, fontsize=11)
        for (row, column), value in np.ndenumerate(array):
            ax.text(column, row, f'{value:.2f}' if np.isfinite(value) else 'н/д', ha='center', va='center',
                    color='white' if value > .6 else 'black')
        fig.colorbar(image, ax=ax, fraction=.045)
    fig.suptitle('Попарное согласие методов — не точность по GT')
    fig.tight_layout(rect=(0, 0, 1, .94))
    return fig


def _spectra_figure(spectra, view, palettes):
    from matplotlib.figure import Figure
    from modules.comparison_plots import BRANCH_ORDER, BRANCH_LABELS
    fig = Figure(figsize=(13, 8))
    for ax, key in zip(fig.subplots(2, 2).flat, BRANCH_ORDER):
        data = spectra[key]
        ax.set_title(BRANCH_LABELS[key])
        if data['status'] != 'completed':
            ax.text(.5, .5, 'Спектр недоступен\n' + data['reason'], ha='center', va='center',
                    transform=ax.transAxes, wrap=True, fontsize=9)
            ax.set_axis_off()
            continue
        xs, ys = [], []
        for i, (x, y) in enumerate(zip(data['x'], data['mean'])):
            if i and not data['allowed_edges'][i - 1]:
                xs.append(np.nan)
                ys.append(np.nan)
            xs.append(x)
            ys.append(np.nan if y is None else y)
        class_id = data['class_id']
        space = view.taxonomies[key]
        ax.plot(xs, ys, '.-', markersize=2, color=palettes[space]['classes'][class_id],
                label=f'{space.taxonomy_id} · {class_id}: {to_ru(space.class_names[class_id])}')
        ax.set_xlabel(data['x_label'])
        ax.set_ylabel(data['units'])
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('Средний спектр доминирующего предсказанного класса; фактический вход каждой ветви')
    fig.tight_layout(rect=(0, 0, 1, .95))
    return fig


def _html(view, root, figures, palettes, sections):
    from modules.comparison_plots import BRANCH_ORDER, BRANCH_LABELS
    e = lambda value: escape(str(value), quote=True)

    def number(value):
        return 'н/д' if value is None else f'{value:.3f}'

    rows = []
    for key in BRANCH_ORDER:
        outcome, cov = view.outcomes[key], view.coverage[key]
        sam = view.agreement['sam_original_baseline'][key]
        gt = view.metrics_gt['branches'][key] if view.metrics_gt else {}
        cells = [BRANCH_LABELS[key], view.taxonomies[key].taxonomy_id if key in view.taxonomies else '—',
                 outcome['status'], outcome['reason_code'] or '—',
                 number(cov.get('coverage_valid')), number(sam.get('agreement_on_joint')),
                 number(sam.get('coverage_joint')), number(gt.get('oa')), number(gt.get('macro_f1')),
                 number(gt.get('coverage'))]
        if len(palettes) > 1:
            cells.extend(number(record.get(metric)) for record in (sam, gt)
                         for metric in ('adjusted_rand', 'normalized_mutual_info'))
        rows.append('<tr>' + ''.join('<td>' + e(value) + '</td>' for value in cells) + '</tr>')
    legends = []
    for space, colors in palettes.items():
        items = ''.join(f'<li title="{e(name)}"><span style="background:{color}"></span>{i}: {e(to_ru(name))}</li>'
                        for i, (name, color) in enumerate(zip(space.class_names, colors['classes'])))
        legends.append(f'<h3>{e(space.taxonomy_id)} · {e(space.version)}</h3><ul>{items}</ul>')
    chunks = []
    titles = {'overview': 'Карты, покрытие и отказы', 'sam': 'Сравнение с SAM',
              'gt': 'Сравнение с GT', 'spectra': 'Спектры входных данных'}
    for section, title in titles.items():
        chunks.append(f'<section><h2>{title}</h2><p>{e(sections[section])}</p>')
        for figure in figures:
            if figure['section'] == section:
                chunks.append(f'<figure><img src="{e(figure["png"])}" alt="{e(figure["title"])}">'
                              f'<figcaption>{e(figure["title"])} · <a href="{e(figure["svg"])}">SVG</a></figcaption></figure>')
        chunks.append('</section>')
    html = '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    html += '<title>Сравнение четырёх ветвей</title><style>body{font:16px system-ui,sans-serif;background:#f3f5f7;color:#17212b;max-width:1400px;margin:32px auto;padding:0 24px}h1,h2{line-height:1.2}section,header{background:white;padding:24px;margin:24px 0;border:1px solid #dbe1e7;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #dbe1e7;text-align:left}figure{margin:24px 0}img{width:100%;height:auto}figcaption{color:#536170}ul{list-style:none;padding:0;columns:3}li{break-inside:avoid;padding:5px}li span{display:inline-block;width:16px;height:16px;margin-right:8px}a{color:#0754a6}.scroll{overflow:auto}code{overflow-wrap:anywhere} @media(max-width:700px){ul{columns:1}body{padding:0 10px}section,header{padding:12px}}</style>'
    html += f'<header><h1>Сравнение четырёх ветвей</h1><p>Запуск: <code>{e(view.result.run_id)}</code><br>Анализ: <code>{e(view.result.analysis_id)}</code></p>'
    html += '<p>Доли — от 0 до 1. Согласие с SAM описывает совпадение методов. GT-оценка относится к фиксированной размеченной области и учитывает отказы.</p>'
    html += f'<p>GT: {e(view.gt_info.get("status"))}; роль: {e(view.gt_info.get("role", "не задана"))}; проверка split: {e(view.gt_info.get("split_verification", "не применимо"))}.</p>'
    if len(palettes) > 1:
        html += '<p>Ветви используют разные системы классов. ARI и NMI сравнивают группировку совместно размеченных пикселей независимо от номеров классов. Значение ARI может быть отрицательным. Для оценки OA и F1 требуется общая система классов.</p>'
    headers = ['Ветвь', 'Система классов', 'Статус', 'Причина', 'Покрытие входа', 'Согласие с SAM', 'Общее покрытие SAM', 'GT OA', 'GT macro-F1', 'Покрытие GT']
    if len(palettes) > 1:
        headers.extend(['SAM ARI', 'SAM NMI', 'GT ARI', 'GT NMI'])
    html += '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + e(v) + '</th>' for v in headers) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></header>'
    html += '<section><h2>Легенда классов</h2>' + ''.join(legends) + '<p>Светло-серый обозначает область без данных, тёмно-серый обозначает отказ.</p></section>'
    html += ''.join(chunks) + '</html>'
    (root / 'report.html').write_text(html, encoding='utf-8')


def render_analysis(analysis_path, *, spectra=False):
    """Render only the selected analysis; every call creates a new visualization revision."""
    from modules import comparison_plots as plots
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with open_visualization_inputs(analysis_path) as view:
        root = unique_directory(view.path / 'visualizations')
        (root / 'figures').mkdir()
        document = {'schema_version': 1, 'kind': 'whole_cube_visualization', 'status': 'running',
                    'visualization_id': root.name, 'run_id': view.result.run_id, 'analysis_id': view.result.analysis_id,
                    'source_analysis': contract_dict(view.analysis_ref), 'source_run': contract_dict(view.run_ref),
                    'parameters': {'spectra': bool(spectra), 'formats': ['png', 'svg']}}
        write_manifest(root / 'manifest.json', document)
        refs, figures = [], []
        names = view.result.taxonomy.class_names
        palettes = _native_palettes(view)
        colors = palettes[view.result.taxonomy]
        mixed = len(palettes) > 1
        sections = {'overview': 'Общая каноническая палитра; исходные пустые пиксели отделены от отказов.',
                    'sam': 'Описательное согласие на совместных предсказаниях вместе с их покрытием. Оба отказа не считаются совпадением.',
                    'gt': 'GT недоступна: ' + str(view.gt_info.get('reason')),
                    'spectra': 'Не запрошены: значения кубов не читались.'}
        if mixed:
            sections['overview'] = 'Каждая карта и частоты предсказаний подписаны в исходной системе классов ветви.'
            sections['sam'] = 'Совпадения классов вычислены для общей системы классов. Для разных систем в таблице приведены ARI, NMI и совместное покрытие.'

        def save(name, section, title, figure):
            try:
                from matplotlib.text import Text
                for text in figure.findobj(match=Text):
                    text.set_fontfamily('DejaVu Sans')
                FigureCanvasAgg(figure)
                for extension in ('png', 'svg'):
                    path = root / 'figures' / (name + '.' + extension)
                    figure.savefig(path, dpi=130, bbox_inches='tight', format=extension)
                    refs.append(_ref(path, name + '_' + extension, root))
                figures.append({'section': section, 'title': title, 'png': 'figures/' + name + '.png',
                                'svg': 'figures/' + name + '.svg'})
            finally:
                figure.clear()

        try:
            palette_doc = {'taxonomy': contract_dict(view.result.taxonomy), **colors}
            if mixed:
                palette_doc['branches'] = {key: {'taxonomy': contract_dict(space), **palettes[space]}
                                           for key, space in view.taxonomies.items()}
            write_manifest(root / 'palette.json', palette_doc)
            refs.append(_ref(root / 'palette.json', 'palette', root))
            if mixed:
                save('class_maps', 'overview', 'Карты четырёх ветвей в исходных классах', _native_class_maps(view, palettes))
            elif 'sam_original' in view.maps and all(
                    item['reason_code'] == 'not_requested' for key, item in view.outcomes.items() if key != 'sam_original'):
                save('class_maps', 'overview', 'Базовая карта SAM', plots.sam_baseline_map(
                    view.maps['sam_original'], view.valid, names, colors, gt=view.gt))
            else:
                save('class_maps', 'overview', 'Карты четырёх ветвей', plots.class_maps(view.maps, view.valid, names, view.outcomes, colors))
            save('coverage', 'overview', 'Покрытие: весь кадр и исходные непустые пиксели', plots.metrics_bars(
                view.coverage, 'Покрытие предсказаниями', [('coverage_all', 'Весь кадр'), ('coverage_valid', 'Непустой вход')], view.outcomes))
            save('rejections', 'overview', 'Причины отсутствия предсказания', plots.category_maps(
                view.reasons, 'Предсказания и причины отказов', REASON_CATEGORIES, view.outcomes))
            save('class_distribution', 'overview', 'Распределение по классам каждой ветви',
                 _native_distributions(view, palettes) if mixed else plots.distributions(
                {key: view.coverage[key] for key in view.maps}, names, colors, view.outcomes))
            save('sam_metrics', 'sam', 'Согласие с исходным SAM и совместное покрытие', plots.metrics_bars(
                view.agreement['sam_original_baseline'], 'Сравнение с SAM — не GT-точность',
                [('agreement_on_joint', 'Согласие'), ('macro_f1_on_joint', 'Macro-F1'),
                 ('kappa_on_joint', 'Kappa'), ('coverage_joint', 'Совместное покрытие')], view.outcomes))
            save('pairwise_agreement', 'sam', 'Попарное согласие и его область поддержки', _pairwise(view))
            differences = _class_differences(view, 'sam_original')
            sam_outcomes = _difference_outcomes(view, differences, 'sam_original' in view.maps)
            save('sam_differences', 'sam', 'Совпадение классов и односторонние/двусторонние отказы', plots.category_maps(
                differences, 'Карты сравнения с исходным SAM', SAM_CATEGORIES, sam_outcomes))
            if view.gt is not None:
                sections['gt'] = ('Зафиксированная разметка; роль: ' + view.gt_info.get('role', 'не задана') +
                    '; проверка split: ' + view.gt_info.get('split_verification', 'не подтверждена') + '. ' +
                    ('OA и F1 учитывают отказы на общей GT-области.' if view.metrics_gt else
                     'Метрики недоступны: ' + str(view.gt_info.get('reason'))))
                save('gt_reference', 'gt', 'Разметка и фиксированная область оценки', _gt_overview(view, colors))
            if view.metrics_gt is not None:
                scores = view.metrics_gt['branches']
                save('gt_metrics', 'gt', 'Качество по GT отдельно от согласия с SAM', plots.metrics_bars(scores, 'Метрики по GT',
                    [('oa', 'OA'), ('macro_f1', 'Macro-F1'), ('kappa', 'Kappa'), ('coverage', 'Покрытие GT')], view.outcomes))
                gt_maps = _class_differences(view, 'gt')
                save('gt_errors', 'gt', 'Ошибки и отказы только внутри фиксированной GT-области', plots.category_maps(
                    gt_maps, 'Сравнение с GT', GT_CATEGORIES, _difference_outcomes(view, gt_maps, True)))
                for key, scores_for_branch in scores.items():
                    if scores_for_branch['status'] == 'evaluated':
                        matrix = view.local_array(_reference(scores_for_branch['confusion_ref']))
                        save(key + '_confusion_gt', 'gt', plots.BRANCH_LABELS[key] + ': матрица ошибок по GT',
                             plots.confusion_matrix(matrix, names, plots.BRANCH_LABELS[key] + ' · GT'))
            spectrum_data = {}
            if spectra:
                for key in plots.BRANCH_ORDER:
                    try:
                        spectrum_data[key] = view.dominant_spectrum(key) if key in view.maps else {
                            'status': 'skipped', 'reason': 'Ветвь недоступна'}
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        spectrum_data[key] = {'status': 'skipped', 'reason': f'{type(error).__name__}: {error}'}
                write_manifest(root / 'spectra.json', spectrum_data)
                refs.append(_ref(root / 'spectra.json', 'spectra', root))
                sections['spectra'] = 'Средние спектры доминирующих классов; каждый берётся из входа своей ветви. Разрывы S/L и сегментов не соединяются.'
                save('spectra', 'spectra', sections['spectra'], _spectra_figure(spectrum_data, view, palettes))
            _html(view, root, figures, palettes, sections)
            refs.append(_ref(root / 'report.html', 'report', root))
            result = VisualizationResult(view.result.run_id, view.result.analysis_id, tuple(refs))
            write_manifest(root / 'visualization.json', contract_dict(result))
            refs.append(_ref(root / 'visualization.json', 'visualization_contract', root))
            view.verify_unchanged()
            document.update(status='completed', result=contract_dict(result), figures=figures, sections=sections,
                            inputs=[contract_dict(ref) for ref in view.consumed.values()],
                            artifacts=[contract_dict(ref) for ref in refs])
            write_manifest(root / 'manifest.json', document)
            return root
        except BaseException as error:
            document.update(status='failed', error=f'{type(error).__name__}: {error}',
                            artifacts=[contract_dict(ref) for ref in refs])
            write_manifest(root / 'manifest.json', document)
            raise


def _reference(raw):
    from modules.pipeline_contracts import ArrayRef
    from modules.resource_store import decode_contract
    return decode_contract(ArrayRef, raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True, help='Completed analysis revision; no analysis is repeated')
    parser.add_argument('--spectra', action='store_true', help='Read saved branch input cubes for mean spectra')
    args = parser.parse_args(argv)
    try:
        result = render_analysis(args.analysis, spectra=args.spectra)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f'Визуализация: {result}\nОтчёт: {result / "report.html"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
