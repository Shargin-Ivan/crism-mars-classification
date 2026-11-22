"""Version 2 result manifests; legacy run writing remains unchanged."""
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modules.pipeline_contracts import ArtifactRef, ArrayRef, BRANCHES, BranchOutcome, contract_dict, require, text
from modules.pipeline_config import PipelineConfig
from modules.resource_store import array_reference, decode_contract, sha256_file, verify_ref, within
from modules.run_store import unique_directory, write_manifest


def validate_manifest(document, run_path, *, verify_artifacts=False):
    require(isinstance(document, dict) and document.get('schema_version') == 2, 'Expected v2 run manifest')
    require(document.get('run_id') == Path(run_path).name, 'Run identity mismatch')
    branches = document.get('branches')
    require(isinstance(branches, dict) and set(branches) == set(BRANCHES), 'Run must contain four branches')
    outcomes = {key: decode_contract(BranchOutcome, value) for key, value in branches.items()}
    require(all(key == value.branch_id for key, value in outcomes.items()), 'Branch identity mismatch')
    status = document.get('status')
    require(status in ('running', 'completed', 'partial', 'failed'), 'Unknown run status')
    completed = sum(value.status == 'completed' for value in outcomes.values())
    if status != 'running':
        require(all(value.status in ('completed', 'failed', 'skipped') for value in outcomes.values()),
                'Terminal run contains unfinished branches')
        expected = 'completed' if completed == 4 else 'partial' if completed else 'failed'
        require(status == expected or (status == 'failed' and bool(document.get('error'))), 'Run/branch statuses disagree')
    artifacts = document.get('artifacts')
    require(isinstance(artifacts, list), 'Missing artifact registry')
    registry = {}
    for entry in artifacts:
        require(isinstance(entry, dict) and set(entry) == {'kind', 'ref', 'bytes', 'dependencies', 'external'},
                'Invalid artifact entry')
        raw = entry['ref']
        require(isinstance(raw, dict), 'Invalid artifact reference')
        ref = decode_contract(ArrayRef if 'shape' in raw else ArtifactRef, raw)
        require(ref.artifact_id not in registry, 'Duplicate artifact ID')
        require(type(entry['external']) is bool and type(entry['bytes']) is int and entry['bytes'] >= 0,
                'Invalid artifact size/location')
        text(entry['kind'], 'artifact kind')
        require(isinstance(entry['dependencies'], list)
                and all(isinstance(dep, str) and dep in registry for dep in entry['dependencies']),
                'Unregistered artifact dependency')
        if not entry['external']:
            within(Path(run_path) / ref.path, run_path)
        if verify_artifacts:
            path = verify_ref(ref, run_path, root=None if entry['external'] else run_path)
            require(path.stat().st_size == entry['bytes'], 'Artifact size mismatch')
            if isinstance(ref, ArrayRef):
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                try:
                    ref.validate_array(array)
                finally:
                    if isinstance(array, np.memmap):
                        array._mmap.close()
        registry[ref.artifact_id] = entry
    for outcome in outcomes.values():
        if outcome.result is not None:
            entry = registry.get(outcome.result.artifact_id)
            require(entry is not None and not entry['external'], 'Branch result is not registered in this run')
            require(all(entry['ref'][key] == getattr(outcome.result, key)
                        for key in ('artifact_id', 'path', 'sha256')), 'Branch result/reference mismatch')
    return document


class WholeCubeRun:
    def __init__(self, root, config):
        require(isinstance(config, PipelineConfig), 'Expected validated PipelineConfig')
        self.path = unique_directory(Path(root).resolve() / 'runs')
        for name in ('maps', 'processed', 'branch_metadata', 'analysis'):
            (self.path / name).mkdir()
        self._closed = False
        self._branches = {key: BranchOutcome(key, 'pending') for key in BRANCHES}
        self._artifacts = []
        self._document = {'schema_version': 2, 'run_id': self.path.name, 'status': 'running',
                          'created_at': datetime.now(timezone.utc).isoformat()}
        write_manifest(self.path / 'config.resolved.json', config.to_dict())
        self.register_artifact('config.resolved.json', 'configuration', 'configuration')

    def _save(self):
        self._document.update(branches={key: contract_dict(value) for key, value in self._branches.items()},
                              artifacts=self._artifacts)
        validate_manifest(self._document, self.path)
        write_manifest(self.path / 'manifest.json', self._document)

    def _register(self, ref, kind, size, dependencies, external):
        require(not self._closed, 'Run has already finished')
        text(kind, 'artifact kind')
        require(isinstance(dependencies, (tuple, list)) and all(isinstance(dep, str) for dep in dependencies),
                'Dependencies must be a list/tuple of artifact IDs')
        known = {item['ref']['artifact_id'] for item in self._artifacts}
        require(ref.artifact_id not in known, 'Artifact ID is already registered')
        require(all(dep in known for dep in dependencies), 'Register dependencies before their consumers')
        require(not any(item['ref']['path'] == ref.path and item['external'] == external
                        for item in self._artifacts), 'Artifact path is already registered')
        self._artifacts.append({'kind': kind, 'ref': contract_dict(ref), 'bytes': size,
                                'dependencies': list(dependencies), 'external': external})
        self._save()
        return ref

    def register_artifact(self, path, artifact_id, kind, *, dependencies=()):
        """Call only after a file is fully written and closed."""
        path = within(self.path / path, self.path)
        require(path.is_file() and path.suffix != '.tmp', 'Only completed files can be registered')
        ref = (array_reference(path, artifact_id, self.path) if path.suffix == '.npy' else
               ArtifactRef(artifact_id, path.relative_to(self.path).as_posix(), sha256_file(path)))
        return self._register(ref, kind, path.stat().st_size, dependencies, False)

    def register_external(self, ref, base, kind, *, dependencies=()):
        path = verify_ref(ref, base)
        require(not path.is_relative_to(self.path), 'Internal artifact cannot be registered as external')
        absolute = replace(ref, path=str(path))
        return self._register(absolute, kind, path.stat().st_size, dependencies, True)

    def set_branch(self, branch_id, status, *, result=None, reason_code=None):
        require(not self._closed and branch_id in self._branches, 'Closed run or unknown branch')
        allowed = {'pending': ('running', 'skipped', 'failed'), 'running': ('completed', 'skipped', 'failed')}
        require(status in allowed.get(self._branches[branch_id].status, ()), 'Invalid branch state transition')
        if result is not None:
            require(isinstance(result, ArtifactRef), 'Expected result reference')
            result = ArtifactRef(result.artifact_id, result.path, result.sha256)
            registered = next((item for item in self._artifacts if item['ref']['artifact_id'] == result.artifact_id), None)
            require(registered is not None and not registered['external']
                    and all(registered['ref'][key] == getattr(result, key) for key in ('artifact_id', 'path', 'sha256')),
                    'Branch result must match a registered internal artifact')
        self._branches[branch_id] = BranchOutcome(branch_id, status, result, reason_code)
        self._save()

    def finish(self, error=None):
        require(not self._closed, 'Run has already finished')
        for key, branch in self._branches.items():
            if branch.status in ('pending', 'running'):
                self._branches[key] = BranchOutcome(key, 'failed', reason_code='interrupted' if error else 'not_executed')
        count = sum(branch.status == 'completed' for branch in self._branches.values())
        self._document['status'] = 'completed' if count == 4 else 'partial' if count else 'failed'
        if error is not None:
            self._document.update(status='failed', error=f'{type(error).__name__}: {error}')
        self._document['finished_at'] = datetime.now(timezone.utc).isoformat()
        self._document.update(branches={key: contract_dict(value) for key, value in self._branches.items()},
                              artifacts=self._artifacts)
        if error is None:
            try:
                validate_manifest(self._document, self.path, verify_artifacts=True)
            except BaseException as validation_error:
                self._document.update(status='failed', error=f'Artifact validation failed: {validation_error}')
                self._save()
                self._closed = True
                raise
        self._save()
        self._closed = True


@contextmanager
def processing_run_v2(root, config):
    run = WholeCubeRun(root, config)
    try:
        yield run
    except BaseException as error:
        run.finish(error)
        raise
    else:
        run.finish()
