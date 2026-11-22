"""Run the independent H100 50+50 campaign against the completed 25+25 reference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import run_experiment as base


BASELINE = base.CAMPAIGN
CAMPAIGN = BASELINE.with_name('epochs50')
STEPS = [step for step in base.STEPS if not step.startswith('before/')]


def baseline_state():
    path = BASELINE / 'campaign.json'
    if not path.is_file():
        raise ValueError('Missing 25+25 reference campaign')
    state = base.load_state(path, base.identity())
    if state.get('status') != 'completed' or list(state['steps']) != base.STEPS:
        raise ValueError('The 25+25 reference campaign must finish before the 50+50 run')
    return state


def compare(reference, steps, destination):
    rows = []
    for scene in base.SCENES:
        phases = [('before', BASELINE, reference['before/' + scene]),
                  ('after25', BASELINE, reference['after/' + scene]),
                  ('after50', destination, steps['after/' + scene])]
        masks = ['valid_pixels.npy']
        if scene == 'hymars_hc':
            masks.append('gt_evaluation_mask.npy')
        reference_masks = None
        for phase, root, record in phases:
            folder = root / record['analysis']
            hashes = [base.digest(folder / name) for name in masks]
            if reference_masks is None:
                reference_masks = hashes
            if reference_masks != hashes or record['sam_map_sha256'] != phases[0][2]['sam_map_sha256']:
                raise ValueError(f'Evaluation mask or original SAM map changed for {scene}/{phase}')
            tax = base.read(folder / 'evaluation.json')['branch_taxonomies']
            with (folder / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
                summary = list(csv.DictReader(stream))
            if len(summary) != 4 or {row['branch_id'] for row in summary} != base.BRANCHES:
                raise ValueError(f'Incomplete comparison for {scene}/{phase}')
            for row in summary:
                rows.append(dict(scene=scene, phase=phase, taxonomy=tax[row['branch_id']]['taxonomy_id'], **row))
    path = destination / 'comparison.csv'
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def configure(reference):
    """Bind the existing orchestration to this dedicated command-line process only."""
    import training_h100
    import training_h100_50
    from finetuning import training

    release = hashlib.sha256(''.join([
        base.identity(), base.digest(Path(__file__)), base.digest(Path(training_h100_50.__file__)),
        base.digest(BASELINE / 'campaign.json') if reference else '',
    ]).encode()).hexdigest()
    original_inference = base.inference
    original_training = base.training

    def inference(phase, scene, steps):
        before = 'before/' + scene
        return original_inference(phase, scene, {**steps, before: reference[before]})

    def train(model, stage):
        record = original_training(model, stage)
        folder = CAMPAIGN / record['run']
        last = base.read(folder / 'result.json')['last_checkpoint']
        if last['path'] != 'last.pt' or base.digest(folder / 'last.pt') != last['sha256']:
            raise ValueError(f'Invalid last checkpoint for {model}/{stage}')
        record['pins'][base.relative(folder / 'last.pt')] = last['sha256']
        return record

    training.run = training_h100_50.run
    training_h100.EPOCHS_PER_STAGE = 50
    base.CAMPAIGN = CAMPAIGN
    base.STEPS = STEPS
    base.identity = lambda: release
    base.inference = inference
    base.training = train
    base.compare = lambda steps: compare(reference, steps, CAMPAIGN)


def main():
    global BASELINE, CAMPAIGN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run', 'status'))
    parser.add_argument('--frozen-reference', action='store_true',
                        help='Compare against the bundled original H100 reference without rerunning 25+25')
    args = parser.parse_args()
    action = args.action
    frozen = None
    if args.frozen_reference:
        from frozen_reference import verified_steps
        BASELINE = base.HERE / 'reference25'
        CAMPAIGN = CAMPAIGN.with_name('epochs50_frozen_reference')
        frozen = verified_steps(BASELINE)
    if action == 'status':
        path = CAMPAIGN / 'campaign.json'
        state = base.read(path) if path.exists() else dict(status='not_started', steps={})
        print(json.dumps({key: value for key, value in state.items() if key != 'steps'}, indent=2))
        for step, record in state['steps'].items():
            print(f"{step}: {record['seconds']/60:.1f} min")
    elif action == 'check':
        configure(frozen)
        report = base.preflight()
        print(f"Output space budget: {base.space_required(report['shapes'], STEPS)/1e9:.1f} GB")
    else:
        reference = frozen if frozen is not None else baseline_state()['steps']
        configure(reference)
        base.execute()


if __name__ == '__main__':
    main()
