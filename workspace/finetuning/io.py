"""Portable, pinned files. No model/data imports or filesystem writes on import."""
import hashlib
import json
from pathlib import Path


def digest(path):
    path = Path(path)
    before = path.stat()
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'File changed during verification: {path}')
    return value.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path):
    def bad(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_pairs,
                      parse_constant=bad)


def write_json(path, value):
    # New files only; a new experiment must not silently replace an old one.
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def inside(base, relative):
    base = Path(base).resolve()
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('A nonempty relative path is required')
    path = (base / relative).resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError(f'Path leaves its storage directory: {relative}')
    return path


def pinned(base, reference):
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
        raise ValueError('Expected path and sha256')
    path = inside(base, reference['path'])
    if digest(path) != reference['sha256']:
        raise ValueError(f'SHA-256 mismatch: {path}')
    return path
