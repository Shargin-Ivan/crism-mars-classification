"""Isolated processing runs, versioned selection and analysis revisions. No models."""
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def unique_directory(parent):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid4().hex[:8])
    path.mkdir()
    return path


def write_manifest(path, document):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


@contextmanager
def processing_run(root, config):
    path = unique_directory(Path(root) / 'runs')
    for name in ('maps', 'processed', 'analysis'):
        (path / name).mkdir()
    manifest = {'schema_version': 1, 'run_id': path.name, 'status': 'running', 'config': config}
    write_manifest(path / 'manifest.json', manifest)
    try:
        yield path
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    else:
        manifest['status'] = 'completed'
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        manifest['artifacts'] = [p.relative_to(path).as_posix() for p in sorted(path.rglob('*.npy'))]
        write_manifest(path / 'manifest.json', manifest)


def read_run_manifest(path, *, verify_artifacts=False):
    path = Path(path).resolve()
    from modules.resource_store import read_json
    manifest = read_json(path / 'manifest.json')
    if not isinstance(manifest, dict) or manifest.get('run_id') != path.name:
        raise ValueError('Run identity mismatch')
    version = manifest.get('schema_version')
    if type(version) is not int or version not in (1, 2):
        raise ValueError('Unsupported run manifest version')
    if version == 2:
        from modules.pipeline_run_store import validate_manifest
        validate_manifest(manifest, path, verify_artifacts=verify_artifacts)
    return manifest


def resolve_run(output, requested=None, *, schema_versions=(1, 2), allow_partial=False):
    runs = (Path(output) / 'runs').resolve()
    candidates = [Path(requested).resolve()] if requested else sorted(runs.glob('*'), reverse=True)
    for path in candidates:
        path = path.resolve()
        if path.parent != runs:
            if requested:
                raise ValueError('Выбранный запуск должен находиться непосредственно в output/runs проекта.')
            continue
        try:
            manifest = read_run_manifest(path)
        except (OSError, ValueError):
            continue
        accepted_statuses = ('completed', 'partial') if requested and allow_partial else ('completed',)
        if (manifest.get('status') in accepted_statuses
                and manifest.get('schema_version') in schema_versions):
            return path
    raise ValueError('Нет подходящего запуска поддерживаемой версии. Укажите папку output/runs/<run_id>; partial требует явного выбора и allow_partial=True. Старые output/*.npy автоматически не смешиваются.')


def analysis_directory(run):
    return unique_directory(Path(run) / 'analysis')
