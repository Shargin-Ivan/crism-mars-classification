"""
run_analysis.py — Анализ сохранённого запуска без повторного инференса.

v2: четыре ветви, отдельные GT-метрики/согласие, ревизии анализа и визуализации.

"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def main(argv=None):
    import argparse
    from modules.run_store import resolve_run, read_run_manifest
    parser = argparse.ArgumentParser(description="Анализ одного сохранённого запуска")
    parser.add_argument('--run', help='Путь к output/runs/<run_id>; по умолчанию последний завершённый')
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'output', help='Корень сохранённых запусков')
    parser.add_argument('--allow-partial', action='store_true', help='Разрешить явно выбранный partial-запуск')
    parser.add_argument('--ground-truth', type=Path, help='Декларация GT со сценой, таксономией, mapping и split (v2)')
    parser.add_argument('--no-visualization', action='store_true', help='Только анализ без рисунков')
    parser.add_argument('--spectra', action='store_true', help='Дополнительно читать входные кубы ветвей для графиков спектров v2')
    args = parser.parse_args(argv)
    if args.allow_partial and not args.run:
        parser.error('--allow-partial требует явного --run')
    if args.spectra and args.no_visualization:
        parser.error('--spectra несовместим с --no-visualization')
    try:
        run = resolve_run(args.output_root, args.run, allow_partial=args.allow_partial)
        if read_run_manifest(run)['schema_version'] == 2:
            from analysis.whole_cube import analyze_run
            report = analyze_run(run, ground_truth_manifest=args.ground_truth)
            print(f'Запуск: {run}\nАнализ v2: {report}\nТаблица: {report / "summary.csv"}')
            if not args.no_visualization:
                from analysis.visualize_comparison import render_analysis
                visual = render_analysis(report, spectra=args.spectra)
                print(f'Визуальный отчёт: {visual / "report.html"}')
            return 0
        parser.error('Требуется манифест запуска четырёх ветвей версии 2')
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
