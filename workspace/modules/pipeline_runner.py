"""Sequential whole-cube graph: one preparation, one denoising, four branches."""
import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import subprocess

import numpy as np

from modules.cube_adapters.common import (
    AdapterUnavailable, absolute_ref, open_denoised, open_prepared, read_array,
)
from modules.cube_adapters.neural import classify_lwnet, denoise_n2n4m
from modules.cube_adapters.sam import classify_sam
from modules.data_preparation import SceneManifest, prepare_scene
from modules.pipeline_config import load_config
from modules.pipeline_contracts import (
    ArtifactRef, BRANCHES, ClassificationResult, DenoisingResult, ModelBundle,
    ReferenceLibrary, Taxonomy, contract_dict, require,
)
from modules.pipeline_run_store import processing_run_v2
from modules.resource_store import (
    ResourceStore, StorePaths, decode_contract, read_json, sha256_file, verify_ref,
)
from modules.run_store import read_run_manifest, write_manifest


def _resolved_config(config, path):
    base = path.parent
    stores = StorePaths.from_config(config.stores, path)
    return replace(config, scene_manifest=absolute_ref(config.scene_manifest, base),
                   taxonomy=absolute_ref(config.taxonomy, base),
                   resources=replace(config.resources, **{
                       key: absolute_ref(getattr(config.resources, key), base)
                       for key in ('sam_library', 'n2n4m_bundle', 'lwnet_bundle')
                       if getattr(config.resources, key) is not None}),
                   evaluation=replace(config.evaluation, ground_truth=(
                       absolute_ref(config.evaluation.ground_truth, base)
                       if config.evaluation.ground_truth else None)),
                   stores=replace(config.stores, inputs_root=str(stores.inputs),
                                  resources_root=str(stores.resources), results_root=str(stores.results)))


def _provenance():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / 'modules').glob('*.py')) + sorted((root / 'modules/cube_adapters').glob('*.py'))
    paths += [root / p for p in ('main.py', 'LWNet/training/models/models_lw_3D.py',
                                 'n2n4m-main/n2n4m/model.py')]
    revisions = {}
    for label, folder in (('project', root),):
        try:
            revisions[label] = subprocess.run(
                ['git', '-C', str(folder), 'rev-parse', 'HEAD'], check=True,
                capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            revisions[label] = None
    revisions['LWNet'] = 'ca8361cd4a1c6418e3366fcc2ebc91fb72ac8c78'
    return {'python': platform.python_version(), 'numpy': np.__version__, 'git_revisions': revisions,
            'source_sha256': {p.relative_to(root).as_posix(): sha256_file(p) for p in paths if p.is_file()}}


@contextmanager
def _inference_seed(execution):
    """Seed only explicit neural calls; restore the caller's RNG state afterwards."""
    import torch
    device = execution.device
    devices = []
    if device != 'cpu':
        require(execution.gpu_allowed, 'CUDA requires explicit permission in the configuration')
        if not torch.cuda.is_available():
            raise AdapterUnavailable('device_unavailable', 'Requested CUDA device is unavailable')
        devices = [torch.device(device).index if ':' in device else torch.cuda.current_device()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(execution.seed)
            np.random.seed(execution.seed % (2 ** 32))
            torch.random.default_generator.manual_seed(execution.seed % (2 ** 64))
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(execution.seed % (2 ** 64))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class _Registry:
    """Namespace resource IDs and deduplicate shared external files by absolute path."""
    def __init__(self, run):
        self.run = run
        self.external_refs = {}

    def external(self, ref, base, role, *, dependencies=()):
        ref = absolute_ref(ref, base)
        previous = self.external_refs.get(ref.path)
        if previous is not None:
            require(previous.sha256 == ref.sha256, 'Shared resource changed between branches')
            return previous.artifact_id
        pinned = self.run.register_external(replace(ref, artifact_id=role), base, role, dependencies=dependencies)
        self.external_refs[ref.path] = pinned
        return pinned.artifact_id

    def resource(self, ref, algorithm):
        cls = ReferenceLibrary if algorithm == 'sam' else ModelBundle
        resource, path = ResourceStore(self.run.path).load_contract(ref, cls)
        role = 'sam_library' if algorithm == 'sam' else algorithm + '_bundle'
        items = (('spectra', resource.spectra), ('valid_samples', resource.valid_samples)) if algorithm == 'sam' else (
            ('weights', resource.weights), ('training_manifest', resource.training_manifest))
        deps = [self.external(item, path.parent, role + '.' + name) for name, item in items]
        if algorithm != 'sam':
            prep = verify_ref(resource.preprocessing, path.parent)
            scaler = read_json(prep)['scaler']
            scaler_deps = [self.external(decode_contract(ArtifactRef, scaler), prep.parent, role + '.scaler')] if scaler else []
            deps.append(self.external(resource.preprocessing, path.parent, role + '.preprocessing', dependencies=scaler_deps))
        return self.external(ref, path.parent, role, dependencies=deps)

    def prepared(self, view, archive, scene_id):
        scene = decode_contract(SceneManifest, read_json(view.prepared.source.path))
        source_deps = [self.external(ref, archive, 'source.' + str(i)) for i, ref in enumerate(scene.references())]
        snapshot = self.external(view.prepared.source, archive, 'source_manifest', dependencies=[scene_id, *source_deps])
        arrays = [self.external(ref, archive, ref.artifact_id, dependencies=[snapshot])
                  for ref in (view.cube_ref, view.valid_ref, view.prepared.good_channels)]
        document = read_json(archive / 'manifest.json')
        spectral = self.external(decode_contract(ArtifactRef, document['spectral_axis_ref']), archive, 'spectral_axis')
        ref = ArtifactRef('prepared_manifest', str(archive / 'manifest.json'), sha256_file(archive / 'manifest.json'))
        return self.external(ref, archive, 'prepared_manifest', dependencies=[snapshot, spectral, *arrays])

    def result(self, folder, result, kind, dependencies):
        document = read_json(folder / 'manifest.json')
        require(document.get('status') == 'completed' and document.get('kind') == kind
                and document.get('result') == contract_dict(result), 'Adapter manifest/result mismatch')
        names = ('cube', 'denoised_mask') if kind == 'denoising' else (
            'class_map', 'prediction_valid_mask', 'rejection_reason', 'used_channels')
        registered = []
        for name in names:
            ref = getattr(result, name)
            path = verify_ref(ref, folder, root=folder)
            saved = self.run.register_artifact(path, ref.artifact_id, name, dependencies=dependencies)
            require(saved.sha256 == ref.sha256, 'Result changed during registration')
            registered.append(saved.artifact_id)
        role = 'n2n4m_result' if kind == 'denoising' else result.branch_id + '_result'
        return self.run.register_artifact(folder / 'manifest.json', role, kind,
                                          dependencies=[*dependencies, *registered])


class _Journal:
    def __init__(self, run, requested_branches):
        self.run = run
        self.document = {'schema_version': 1, 'stages': {key: {'status': 'pending'}
                         for key in ('preparation', 'n2n4m')}, 'branch_errors': {},
                         'requested_branches': list(requested_branches),
                         'analysis': 'not_run', 'visualization': 'not_run'}
        self.save()

    def save(self):
        write_manifest(self.run.path / 'execution.json', self.document)

    def stage(self, key, status, **details):
        self.document['stages'][key].update(status=status, updated_at=datetime.now(timezone.utc).isoformat(), **details)
        self.save()

    def error(self, key, error):
        self.document['branch_errors'][key] = {'type': type(error).__name__, 'message': str(error)}
        self.save()


def _classify(run, registry, journal, config, view, taxonomy, key, input_dependency):
    run.set_branch(key, 'running')
    algorithm = BRANCHES[key].classifier
    resource = config.resources.sam_library if algorithm == 'sam' else config.resources.lwnet_bundle
    folder = run.path / 'maps' / key
    try:
        view.require_ready()
        if algorithm == 'sam':
            result = classify_sam(view, resource, run.path, taxonomy, folder, branch_id=key, spec=config.sam,
                                  execution=config.execution)
        else:
            bundle, _ = ResourceStore(run.path).load_contract(resource, ModelBundle)
            if bundle.architecture_version == 'LWNet_3_Salinas_group4_v1':
                taxonomy = bundle.taxonomy
            with _inference_seed(config.execution):
                result = classify_lwnet(view, resource, run.path, taxonomy, folder, branch_id=key,
                                        spec=config.lwnet, execution=config.execution)
        require(isinstance(result, ClassificationResult) and result.branch_id == key
                and result.input_cube == view.cube_ref and result.resource == resource
                and result.taxonomy == taxonomy, 'Classification result belongs to another branch/input/resource')
        with ExitStack() as arrays:
            values = [arrays.enter_context(read_array(getattr(result, name), folder, root=folder)) for name in (
                'class_map', 'prediction_valid_mask', 'rejection_reason', 'used_channels')]
            result.validate_arrays(*values, view.valid_samples)
        resource_id = registry.resource(resource, algorithm)
        ref = registry.result(folder, result, 'classification', [input_dependency, 'taxonomy', resource_id])
        run.set_branch(key, 'completed', result=ref)
    except AdapterUnavailable as error:
        journal.error(key, error)
        run.set_branch(key, 'skipped', reason_code=error.reason_code)
    except Exception as error:
        journal.error(key, error)
        run.set_branch(key, 'failed', reason_code='adapter_failed')


def _execute(run, config, journal):
    registry = _Registry(run)
    requested = journal.document['requested_branches']
    sam_only = requested == ['sam_original']
    for key in BRANCHES:
        if key not in requested:
            run.set_branch(key, 'skipped', reason_code='not_requested')
    if sam_only:
        journal.stage('n2n4m', 'skipped', reason_code='not_requested')
    journal.stage('preparation', 'running')
    with ExitStack() as contexts:
        try:
            taxonomy, _ = ResourceStore(run.path).load_contract(config.taxonomy, Taxonomy)
            registry.external(config.taxonomy, run.path, 'taxonomy')
            scene_id = registry.external(config.scene_manifest, run.path, 'scene_manifest')
            archive = prepare_scene(config.scene_manifest.path, config.stores.results_root,
                                    expected_sha256=config.scene_manifest.sha256)
            view = contexts.enter_context(open_prepared(archive))
            prepared_id = registry.prepared(view, archive, scene_id)
            journal.stage('preparation', 'completed', archive=str(archive), result_id=prepared_id)
        except Exception as error:
            journal.stage('preparation', 'failed', reason_code='preparation_failed',
                          error=f'{type(error).__name__}: {error}')
            if not sam_only:
                journal.stage('n2n4m', 'skipped', reason_code='dependency_preparation_failed')
            for key in requested:
                run.set_branch(key, 'skipped', reason_code='dependency_preparation_failed')
            return
        _classify(run, registry, journal, config, view, taxonomy, 'sam_original', prepared_id)
        if sam_only:
            return
        journal.stage('n2n4m', 'running')
        folder = run.path / 'processed' / 'n2n4m'
        try:
            view.require_ready()
            with _inference_seed(config.execution):
                result = denoise_n2n4m(view, config.resources.n2n4m_bundle, run.path, folder,
                                      spec=config.n2n4m, execution=config.execution)
            require(isinstance(result, DenoisingResult) and result.source == view.prepared
                    and result.bundle == config.resources.n2n4m_bundle, 'Denoising result belongs to another input/resource')
            denoised = contexts.enter_context(open_denoised(view, folder))
            resource_id = registry.resource(config.resources.n2n4m_bundle, 'n2n4m')
            ref = registry.result(folder, result, 'denoising', [prepared_id, resource_id])
            journal.stage('n2n4m', 'completed', result_id=ref.artifact_id)
        except Exception as error:
            unavailable = isinstance(error, AdapterUnavailable)
            journal.stage('n2n4m', 'skipped' if unavailable else 'failed',
                          reason_code=error.reason_code if unavailable else 'adapter_failed',
                          error=f'{type(error).__name__}: {error}')
            for key in ('sam_denoised', 'lwnet_denoised'):
                run.set_branch(key, 'skipped', reason_code='dependency_n2n4m_unavailable' if unavailable else 'dependency_n2n4m_failed')
            _classify(run, registry, journal, config, view, taxonomy, 'lwnet_original', prepared_id)
            return
        _classify(run, registry, journal, config, denoised, taxonomy, 'sam_denoised', ref.artifact_id)
        _classify(run, registry, journal, config, view, taxonomy, 'lwnet_original', prepared_id)
        _classify(run, registry, journal, config, denoised, taxonomy, 'lwnet_denoised', ref.artifact_id)


def run_pipeline(config_path, *, sam_only=False):
    """Execute the full graph or an explicit original-cube SAM baseline."""
    path = Path(config_path).resolve()
    config = load_config(path, require_ready=not sam_only)
    if sam_only:
        config.require_sam_baseline()
    config = _resolved_config(config, path)
    with processing_run_v2(config.stores.results_root, config) as run:
        write_manifest(run.path / 'provenance.json', _provenance())
        run.register_artifact('provenance.json', 'provenance', 'code_versions', dependencies=['configuration'])
        journal = _Journal(run, ('sam_original',) if sam_only else BRANCHES)
        try:
            _execute(run, config, journal)
        except BaseException:
            for key, stage in journal.document['stages'].items():
                if stage['status'] in ('pending', 'running'):
                    journal.stage(key, 'failed', reason_code='interrupted')
            raise
        finally:
            run.register_artifact('execution.json', 'execution', 'execution_status', dependencies=['configuration'])
    return run.path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--sam-only', action='store_true', help='Classify only the original full cube with SAM')
    args = parser.parse_args(argv)
    try:
        path = run_pipeline(args.config, sam_only=args.sam_only)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    document = read_run_manifest(path)
    print(json.dumps({'run': str(path), 'status': document['status'],
                      'branches': {key: value['status'] for key, value in document['branches'].items()}},
                     ensure_ascii=False, indent=2))
    if args.sam_only:
        return 0 if document['status'] == 'partial' and document['branches']['sam_original']['status'] == 'completed' else 1
    return 0 if document['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
