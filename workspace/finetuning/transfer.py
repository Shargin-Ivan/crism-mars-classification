"""Copy explicit immutable inputs; package the workspace without virtualenvs."""
from pathlib import Path
import shutil
import zipfile

from .io import digest, inside, read_json, write_json


def import_local(root, project):
    root, project = Path(root).resolve(), Path(project).resolve()
    spec = read_json(root / 'configs/import_sources.json')
    records = []
    for item in spec['files']:
        source = inside(project, item['source'])
        target = inside(root, item['destination'])
        expected = item.get('sha256') or digest(source)
        if digest(source) != expected:
            raise ValueError(f'Source differs from import declaration: {source}')
        if target.exists():
            if digest(target) != expected:
                raise ValueError(f'Refusing to overwrite different input: {target}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        if digest(target) != expected or digest(source) != expected:
            raise ValueError(f'Copy verification failed: {source}')
        records.append({'path': target.relative_to(root).as_posix(), 'sha256': expected,
                        'bytes': target.stat().st_size, 'source': item['source']})
    report = {'schema_version': 1, 'files': records}
    manifest = root / 'storage/import_manifest.json'
    if manifest.exists():
        if read_json(manifest) != report:
            raise ValueError('Existing import manifest differs; use a new workspace')
    else:
        write_json(manifest, report)
    return report


def verify_transfer(root):
    root = Path(root).resolve()
    report = read_json(root / 'storage/import_manifest.json')
    for item in report['files']:
        path = inside(root, item['path'])
        if digest(path) != item['sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError(f'Transfer verification failed: {path}')
    return len(report['files'])


def pack(root, output, with_data=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError('Archive must be a new file outside finetuning/')
    files = []
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in ('.venv', '__pycache__', '.git') for part in rel.parts):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'External link cannot be packaged: {path}')
        if rel.parts[:2] == ('storage', 'runs'):
            continue
        if rel.parts[0] == 'storage' and not with_data and path.name != 'README.md':
            continue
        files.append(path)
    # MAT/checkpoints are already compressed; Zip64 supports large archives.
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in files:
            archive.write(path, 'finetuning/' + path.relative_to(root).as_posix())
    return len(files)
