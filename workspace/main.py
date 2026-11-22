"""Explicit entry point; imports never load models, data or legacy processing."""


def main(argv=None):
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Run the four whole-cube branches')
    parser.add_argument('--config', type=Path, required=True, help='Ready whole-cube configuration')
    parser.add_argument('--sam-only', action='store_true', help='Run only the whole-cube SAM baseline')
    args = parser.parse_args(argv)
    from modules.pipeline_runner import main as run_main
    return run_main([str(args.config), *(['--sam-only'] if args.sam_only else [])])


if __name__ == '__main__':
    raise SystemExit(main())
