"""Commands are inert by default; only `train` invokes an optimizer."""
import argparse
from pathlib import Path
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description='Portable Plebani -> HyMars fine-tuning workspace')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('check', 'train'):
        command = commands.add_parser(name)
        command.add_argument('--config', type=Path, required=True)
    imp = commands.add_parser('import-local')
    imp.add_argument('--project-root', type=Path, required=True)
    commands.add_parser('verify-transfer')
    verify_pairs_cmd = commands.add_parser('verify-pairs')
    verify_pairs_cmd.add_argument('--path', type=Path, required=True)
    commands.add_parser('environment')
    smoke_cmd = commands.add_parser('smoke-test')
    smoke_cmd.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    package = commands.add_parser('pack')
    package.add_argument('--output', type=Path, required=True)
    package.add_argument('--with-data', action='store_true')
    prepare_cmd = commands.add_parser('prepare')
    prepare_cmd.add_argument('--recipe', type=Path, required=True)
    prepare_cmd.add_argument('--output', required=True, help='New relative path inside finetuning/')
    for name in ('prepare-pairs', 'prepare-plebani-pair', 'split-pairs',
                 'prepare-lwnet-spectra', 'prepare-denoising'):
        pairs_cmd = commands.add_parser(name)
        pairs_cmd.add_argument('--recipe', type=Path, required=True)
        pairs_cmd.add_argument('--output', required=True, help='New relative path inside finetuning/')
    export = commands.add_parser('export')
    export.add_argument('--run', type=Path, required=True)
    export.add_argument('--target', type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    datasets = []
    try:
        if args.command in ('check', 'train'):
            from .configuration import load, preflight
            config = load(args.config)
            report, datasets = preflight(config, root)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if not report['ready']:
                return_code = 2
            elif args.command == 'train':
                from .training import run
                print(run(config, args.config, datasets, root))
                return_code = 0
            else:
                return_code = 0
            if return_code:
                raise SystemExit(return_code)
        elif args.command == 'import-local':
            from .transfer import import_local
            result = import_local(root, args.project_root)
            print(f"Verified copies: {len(result['files'])}")
        elif args.command == 'verify-transfer':
            from .transfer import verify_transfer
            print(f'Verified transferred files: {verify_transfer(root)}')
        elif args.command == 'verify-pairs':
            from .pairs import verify_pairs
            reports = verify_pairs(args.path)
            print(json.dumps(reports, ensure_ascii=False, indent=2))
        elif args.command == 'pack':
            from .transfer import pack
            print(f'Packaged files: {pack(root, args.output, args.with_data)}')
        elif args.command == 'prepare':
            from .prepare import prepare
            print(prepare(args.recipe, args.output, root))
        elif args.command == 'prepare-lwnet-spectra':
            from .spectral_data import prepare_lwnet_spectra
            print(prepare_lwnet_spectra(args.recipe, args.output, root))
        elif args.command == 'prepare-denoising':
            from .denoising_data import prepare_denoising
            print(prepare_denoising(args.recipe, args.output, root))
        elif args.command == 'prepare-pairs':
            from .pairs import prepare_pairs
            print(prepare_pairs(args.recipe, args.output, root))
        elif args.command == 'prepare-plebani-pair':
            from .pairs import prepare_plebani_pair
            print(prepare_plebani_pair(args.recipe, args.output, root))
        elif args.command == 'split-pairs':
            from .pair_split import prepare_pair_split
            print(prepare_pair_split(args.recipe, args.output, root))
        elif args.command == 'export':
            from .export import export_bundle
            print(export_bundle(args.run, args.target))
        elif args.command == 'environment':
            import platform
            import torch
            import numpy as np
            import scipy
            print(json.dumps({'python':platform.python_version(), 'torch':str(torch.__version__),
                              'numpy':np.__version__, 'scipy':scipy.__version__,
                              'cuda_available':torch.cuda.is_available(), 'cuda_build':torch.version.cuda}, indent=2))
        elif args.command == 'smoke-test':
            from .smoke import smoke
            smoke(args.device)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f'error: {error}\n')
    finally:
        for dataset in datasets:
            dataset.close()
