"""Read-only, hash-pinned resources for the three pipeline stores.

No torch/pickle deserialization, model initialization or directory creation.
Nested references are relative to the manifest containing them.
"""
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from pathlib import Path
from types import UnionType
from typing import get_args, get_origin, get_type_hints, Union

import numpy as np

from modules.pipeline_contracts import (
    ArtifactRef, ArrayRef, ContractError, ModelBundle, ReferenceLibrary,
    require,
)


def sha256_file(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f'File changed while hashing: {path}')
    return digest.hexdigest()


def resolved_path(path, base):
    path = Path(path)
    return (path if path.is_absolute() else Path(base) / path).resolve()


def within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path != root and path.is_relative_to(root), f'Artifact is outside its directory: {path}')
    return path


def verify_ref(ref, base, *, root=None):
    require(isinstance(ref, ArtifactRef), 'Expected pinned artifact reference')
    path = resolved_path(ref.path, base)
    if root is not None:
        within(path, root)
    require(path.is_file(), f'Missing artifact: {path}')
    require(sha256_file(path) == ref.sha256, f'SHA-256 mismatch: {path}')
    return path


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ContractError(f'Nonfinite JSON constant: {value}')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)


def decode_contract(cls, document):
    """Strict decoding of the finite dataclass contract vocabulary, no eval."""
    require(is_dataclass(cls) and isinstance(document, dict), 'Expected contract JSON object')
    require(set(document) == {f.name for f in fields(cls)}, f'Unexpected/missing fields for {cls.__name__}')
    hints = get_type_hints(cls)

    def decode(hint, value):
        origin = get_origin(hint)
        if origin in (Union, UnionType):
            candidates = [t for t in get_args(hint) if t is not type(None)]
            if value is None and type(None) in get_args(hint):
                return None
            require(len(candidates) == 1, 'Only optional contract types are supported')
            return decode(candidates[0], value)
        if is_dataclass(hint):
            return decode_contract(hint, value)
        if origin is tuple:
            require(isinstance(value, list), 'Expected JSON array')
            args = get_args(hint)
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(decode(args[0], v) for v in value)
            require(len(value) == len(args), 'Tuple length mismatch')
            return tuple(decode(t, v) for t, v in zip(args, value))
        if hint in (str, int, bool):
            require(type(value) is hint, f'Expected {hint.__name__} in contract JSON')
        elif hint is float:
            require(type(value) in (int, float), 'Expected numeric value in contract JSON')
        elif hint is type(None):
            require(value is None, 'Expected null in contract JSON')
        return value  # Primitive constraints are enforced by each dataclass.

    return cls(**{name: decode(hints[name], value) for name, value in document.items()})


def array_reference(path, artifact_id, base):
    """Index a completed NPY without loading the whole array into memory."""
    path = within(path, base)
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        shape, dtype = array.shape, array.dtype.str
    finally:
        if isinstance(array, np.memmap):
            array._mmap.close()
    return ArrayRef(artifact_id, path.relative_to(Path(base).resolve()).as_posix(),
                    sha256_file(path), shape, dtype)


@dataclass(frozen=True)
class StorePaths:
    inputs: Path
    resources: Path
    results: Path

    @classmethod
    def from_config(cls, spec, config_path):
        base = Path(config_path).resolve().parent
        paths = [resolved_path(getattr(spec, name), base)
                 for name in ('inputs_root', 'resources_root', 'results_root')]
        for i, path in enumerate(paths):
            for other in paths[i + 1:]:
                require(not path.is_relative_to(other) and not other.is_relative_to(path),
                        'Input, resource and result stores must not overlap')
        return cls(*paths)


class ResourceStore:
    def __init__(self, base):
        self.base = Path(base).resolve()

    def load_contract(self, ref, cls):
        path = verify_ref(ref, self.base)
        document = read_json(path)
        require(sha256_file(path) == ref.sha256, 'Manifest changed while reading')
        return decode_contract(cls, document), path

    def model_bundle(self, ref, *, algorithm):
        require(ref is not None, 'Model bundle is not configured; random initialization is forbidden')
        bundle, path = self.load_contract(ref, ModelBundle)
        require(bundle.algorithm == algorithm, 'Model algorithm mismatch')
        for item in (bundle.weights, bundle.preprocessing, bundle.training_manifest):
            verify_ref(item, path.parent)
        preprocessing_path = verify_ref(bundle.preprocessing, path.parent)
        preprocessing = read_json(preprocessing_path)
        require(isinstance(preprocessing, dict)
                and set(preprocessing) == {'schema_version', 'scaler', 'parameters'}
                and type(preprocessing['schema_version']) is int and preprocessing['schema_version'] == 1
                and isinstance(preprocessing['parameters'], dict), 'Invalid preprocessing manifest')
        if preprocessing['scaler'] is not None:
            verify_ref(decode_contract(ArtifactRef, preprocessing['scaler']), preprocessing_path.parent)
        return bundle

    def reference_library(self, ref):
        library, path = self.load_contract(ref, ReferenceLibrary)
        arrays = []
        try:
            for item in (library.spectra, library.valid_samples):
                array = np.load(verify_ref(item, path.parent), mmap_mode='r', allow_pickle=False)
                arrays.append(array)
                item.validate_array(array)
            for spectrum, mask in zip(*arrays):
                require(not np.any(mask & ~np.isfinite(spectrum)), 'Nonfinite valid reference measurement')
        finally:
            for array in arrays:
                if isinstance(array, np.memmap):
                    array._mmap.close()
        return library

    def verify_bindings(self, config):
        """Independent results: a missing LWNet does not invalidate available SAM."""
        results = {}
        for name in ('sam_library', 'n2n4m_bundle', 'lwnet_bundle'):
            ref = getattr(config.resources, name)
            if ref is None:
                results[name] = {'status': 'missing', 'reason': 'not_configured'}
                continue
            try:
                if name == 'sam_library':
                    self.reference_library(ref)
                else:
                    self.model_bundle(ref, algorithm=name.removesuffix('_bundle'))
                results[name] = {'status': 'verified'}
            except (OSError, ValueError, TypeError) as error:
                results[name] = {'status': 'invalid', 'reason': str(error)}
        return results
