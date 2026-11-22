"""Strict, side-effect-free configuration of the four whole-cube branches.

This module validates declarations, not the contents of referenced files.
Resolving resources, checking their hashes and running blocks are later stages.
"""
import argparse
from dataclasses import dataclass, fields
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

from modules.pipeline_contracts import (
    ArtifactRef, BRANCHES, BranchSpec, ContractError, contract_dict,
    integer, number, require, text,
)


@dataclass(frozen=True)
class PreparationSpec:
    spectral_profile: str = 'full_cube'
    crop: None = None
    split_detectors: bool = False
    split_plebani: bool = False
    preserve_channel_order: bool = True

    def __post_init__(self):
        require(self.spectral_profile == 'full_cube', 'Only the whole-cube profile is enabled')
        require(self.crop is None, 'Spatial cropping is disabled in the new pipeline')
        require(self.split_detectors is False and self.split_plebani is False,
                'Subcube partitioning is disabled in the new pipeline')
        require(self.preserve_channel_order is True, 'All source channels must keep their original order')


@dataclass(frozen=True)
class ResourceSpec:
    n2n4m_bundle: ArtifactRef | None = None
    lwnet_bundle: ArtifactRef | None = None
    sam_library: ArtifactRef | None = None

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            require(value is None or isinstance(value, ArtifactRef), f'{field.name}: expected resource reference')


@dataclass(frozen=True)
class StorageSpec:
    inputs_root: str
    resources_root: str
    results_root: str

    def __post_init__(self):
        for field in fields(self):
            text(getattr(self, field.name), field.name)


@dataclass(frozen=True)
class ExecutionSpec:
    device: str = 'cpu'
    gpu_allowed: bool = False
    num_workers: int = 0
    seed: int = 0

    def __post_init__(self):
        require(isinstance(self.device, str) and
                (self.device == 'cpu' or re.fullmatch(r'cuda(?::[0-9]+)?', self.device)), 'Unknown device')
        require(type(self.gpu_allowed) is bool, 'gpu_allowed must be boolean')
        require(self.device == 'cpu' or self.gpu_allowed, 'CUDA requires explicit gpu_allowed=true')
        integer(self.num_workers, 'num_workers')
        integer(self.seed, 'seed')


@dataclass(frozen=True)
class SamSpec:
    threshold_radians: float = 0.15
    min_valid_bands: int = 2
    min_valid_fraction: float = 0.5

    def __post_init__(self):
        number(self.threshold_radians, 'threshold_radians', 0, 3.141592653589793)
        integer(self.min_valid_bands, 'min_valid_bands', 2)
        number(self.min_valid_fraction, 'min_valid_fraction', 0, 1)


@dataclass(frozen=True)
class N2N4MSpec:
    batch_size: int = 1000

    def __post_init__(self):
        integer(self.batch_size, 'N2N4M batch_size', 1)


@dataclass(frozen=True)
class LWNetSpec:
    patch_size: int = 19
    batch_size: int = 64

    def __post_init__(self):
        integer(self.patch_size, 'patch_size', 1)
        require(self.patch_size % 2 == 1, 'patch_size must be odd')
        integer(self.batch_size, 'LWNet batch_size', 1)


@dataclass(frozen=True)
class EvaluationSpec:
    ground_truth: ArtifactRef | None = None
    protocol: str = 'fixed_roi_v1'
    unlabeled_id: int = 0

    def __post_init__(self):
        require(self.ground_truth is None or isinstance(self.ground_truth, ArtifactRef), 'Invalid GT reference')
        require(self.protocol == 'fixed_roi_v1', 'Unknown evaluation protocol')
        integer(self.unlabeled_id, 'unlabeled_id')


@dataclass(frozen=True)
class PipelineConfig:
    schema_version: int
    configuration_state: str
    scene_manifest: ArtifactRef | None
    taxonomy: ArtifactRef | None
    preparation: PreparationSpec
    resources: ResourceSpec
    branches: Mapping[str, BranchSpec]
    execution: ExecutionSpec
    sam: SamSpec
    n2n4m: N2N4MSpec
    lwnet: LWNetSpec
    evaluation: EvaluationSpec
    stores: StorageSpec

    def __post_init__(self):
        require(type(self.schema_version) is int and self.schema_version == 2, 'Unsupported schema_version')
        require(self.configuration_state in ('draft', 'ready'), 'Unknown configuration_state')
        for name in ('scene_manifest', 'taxonomy'):
            value = getattr(self, name)
            require(value is None or isinstance(value, ArtifactRef), f'{name}: invalid reference')
        for name, cls in _SECTIONS.items():
            require(isinstance(getattr(self, name), cls), f'{name}: expected {cls.__name__}')
        require(isinstance(self.branches, Mapping) and set(self.branches) == set(BRANCHES),
                'Exactly the four declared branches are required')
        require(all(isinstance(self.branches[k], BranchSpec) and self.branches[k] == spec
                    for k, spec in BRANCHES.items()), 'Invalid branch routing')
        # Copy before freezing: later edits to the caller's dict cannot alter a run.
        object.__setattr__(self, 'branches', MappingProxyType({k: self.branches[k] for k in BRANCHES}))
        if self.configuration_state == 'ready':
            self.require_ready()

    @property
    def missing_bindings(self):
        missing = [name for name in ('scene_manifest', 'taxonomy') if getattr(self, name) is None]
        missing.extend('resources.' + f.name for f in fields(self.resources)
                       if getattr(self.resources, f.name) is None)
        return tuple(missing)

    def require_ready(self):
        require(not self.missing_bindings,
                'Unbound configuration: ' + ', '.join(self.missing_bindings))
        require(self.configuration_state == 'ready', 'Configuration is draft; not ready for execution')

    def require_sam_baseline(self):
        missing = [key for key in self.missing_bindings if key in (
            'scene_manifest', 'taxonomy', 'resources.sam_library')]
        require(not missing, 'Unbound SAM baseline: ' + ', '.join(missing))

    def to_dict(self):
        return contract_dict(self)


_SECTIONS = {
    'preparation': PreparationSpec, 'resources': ResourceSpec,
    'execution': ExecutionSpec, 'sam': SamSpec, 'n2n4m': N2N4MSpec,
    'lwnet': LWNetSpec, 'evaluation': EvaluationSpec, 'stores': StorageSpec,
}


def _object(document, cls, location):
    require(isinstance(document, dict), f'{location}: expected JSON object')
    require(all(isinstance(key, str) for key in document), f'{location}: keys must be strings')
    expected = {field.name for field in fields(cls)}
    actual = set(document)
    require(actual == expected,
            f'{location}: missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}')
    return dict(document)


def _reference(document, location):
    if document is None:
        return None
    return ArtifactRef(**_object(document, ArtifactRef, location))


def config_from_dict(document, *, require_ready=False):
    """Reject typos, partial branch graphs, unsafe defaults and fake bindings."""
    data = _object(document, PipelineConfig, 'configuration')
    for name in ('scene_manifest', 'taxonomy'):
        data[name] = _reference(data[name], name)
    for name, cls in _SECTIONS.items():
        section = _object(data[name], cls, name)
        if name == 'resources':
            section = {k: _reference(v, f'resources.{k}') for k, v in section.items()}
        elif name == 'evaluation':
            section['ground_truth'] = _reference(section['ground_truth'], 'evaluation.ground_truth')
        data[name] = cls(**section)
    require(isinstance(data['branches'], dict), 'branches: expected JSON object')
    data['branches'] = {
        k: BranchSpec(**_object(v, BranchSpec, f'branches.{k}')) for k, v in data['branches'].items()
    }
    config = PipelineConfig(**data)
    if require_ready:
        config.require_ready()
    return config


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ContractError(f'Nonfinite JSON constant: {value}')


def load_config(path, *, require_ready=False):
    """Read ONLY the given JSON; do not open cubes, weights, data or stores.

    Referenced relative paths are relative to this configuration's directory.
    A ready declaration still needs resource/hash/metadata checks in the runner.
    """
    try:
        document = json.loads(Path(path).read_text(encoding='utf-8'),
                              object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
    except json.JSONDecodeError as error:
        raise ContractError(f'Invalid JSON: {error.msg}') from error
    return config_from_dict(document, require_ready=require_ready)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Validate configuration without loading data or models')
    parser.add_argument('config', type=Path)
    parser.add_argument('--require-ready', action='store_true')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, require_ready=args.require_ready)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({'valid_structure': True, 'configuration_state': config.configuration_state,
                      'whole_cube': True, 'branches': list(config.branches),
                      'missing_bindings': config.missing_bindings}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
