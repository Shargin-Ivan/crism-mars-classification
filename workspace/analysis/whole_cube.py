"""Versioned four-branch comparison; GT accuracy and SAM agreement stay separate."""
import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from modules.analysis_inputs import open_analysis_inputs
from modules.comparison_metrics import coverage_summary, gt_scores, pair_agreement
from modules.ground_truth import load_ground_truth
from modules.pipeline_contracts import ArtifactRef, BRANCHES, ComparisonResult, ModelBundle, contract_dict, require
from modules.resource_store import array_reference, decode_contract, sha256_file, verify_ref
from modules.run_store import analysis_directory, write_manifest


def _ref(path, name, base=None):
    path = Path(path).resolve()
    return ArtifactRef(name, path.relative_to(base).as_posix() if base else str(path), sha256_file(path))


def _ground_truth(inputs, override):
    ref = None
    origin = 'none'
    try:
        if override is not None:
            ref, origin = _ref(override, 'gt_descriptor'), 'analysis_argument'
        elif inputs.config.evaluation.ground_truth is not None:
            ref, origin = inputs.config.evaluation.ground_truth, 'run_configuration'
        else:
            companion = Path(inputs.config.scene_manifest.path).with_suffix('.ground_truth.json')
            if companion.is_file():
                ref, origin = _ref(companion, 'gt_descriptor'), 'scene_companion'
        if ref is None:
            return {'status': 'not_evaluable', 'reason': 'labels_present_mapping_missing'
                    if inputs.scene.ground_truth is not None else 'ground_truth_missing', 'origin': origin}, None
        spec, labels, roi, refs = load_ground_truth(ref, inputs.run, inputs.scene, inputs.taxonomy)
        require(spec.unlabeled_id == inputs.config.evaluation.unlabeled_id, 'GT unlabelled ID differs from evaluation config')
        info = {'status': 'available', 'reason': None, 'origin': origin,
                'role': 'held_out' if spec.scene_id == spec.held_out_scene_id else 'development',
                'held_out_scene_id': spec.held_out_scene_id, 'development_scene_ids': list(spec.development_scene_ids),
                'split_verification': 'declared', 'neural_training_lineage': {}}
        # Read registered training descriptions only. Never open weights or fit anything to GT.
        algorithms = set()
        for key in inputs.maps:
            if key.startswith('lwnet'):
                algorithms.add('lwnet')
            if key.endswith('denoised'):
                algorithms.add('n2n4m')
        for algorithm in sorted(algorithms):
            previous_inputs = dict(inputs.inputs)
            try:
                bundle_ref = getattr(inputs.config.resources, algorithm + '_bundle')
                bundle = decode_contract(ModelBundle, inputs.json(bundle_ref, inputs.run))
                training = inputs.json(bundle.training_manifest, Path(bundle_ref.path).parent)
                keys = ('training_scene_ids', 'validation_scene_ids')
                known = isinstance(training, dict) and all(isinstance(training.get(key), list)
                        and all(isinstance(scene, str) for scene in training[key]) for key in keys)
                if known:
                    overlap = spec.held_out_scene_id in training[keys[0]] + training[keys[1]]
                    info['neural_training_lineage'][algorithm] = 'held_out_overlap' if overlap else 'declared_disjoint'
                else:
                    info['neural_training_lineage'][algorithm] = 'unverified'
            except (OSError, ValueError, TypeError, KeyError):
                # Lineage is useful evidence, but not a dependency of label/map comparison.
                inputs.inputs = previous_inputs
                info['neural_training_lineage'][algorithm] = 'unverified'
        if 'held_out_overlap' in info['neural_training_lineage'].values():
            info['split_verification'] = 'overlap_detected_diagnostic_only'
        elif 'unverified' in info['neural_training_lineage'].values():
            info['split_verification'] = 'lineage_unverified'
        return info, (spec, labels, roi, refs)
    except (OSError, ValueError, TypeError, KeyError, NotImplementedError, EOFError) as error:
        return {'status': 'not_evaluable', 'reason': 'ground_truth_invalid', 'origin': origin,
                'error': f'{type(error).__name__}: {error}',
                'descriptor': contract_dict(ref) if ref else None}, None


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _agreement(left, right, roi, classes):
    values = pair_agreement(left, right, roi, classes)
    joint = roi & (left >= 0) & (right >= 0)
    values['macro_f1_on_joint'] = gt_scores(left, right, joint, classes)['macro_f1']
    return values


def _partition_agreement(left, right, roi):
    """Compare groupings on joint coverage without pairing native IDs with minerals."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    joint = roi & (left >= 0) & (right >= 0)
    n, n_joint = int(roi.sum()), int(joint.sum())
    a, b = left[joint], right[joint]
    return dict(status='different_taxonomy', reason='incompatible_class_names',
                n_roi=n, n_joint=n_joint, coverage_joint=n_joint / n if n else None,
                adjusted_rand=float(adjusted_rand_score(a, b)) if n_joint >= 2 else None,
                normalized_mutual_info=float(normalized_mutual_info_score(a, b)) if n_joint >= 2 else None)


def analyze_run(run_path, *, ground_truth_manifest=None):
    """Analyze a saved v2 run into a new revision; never run inference or change it."""
    run_path = Path(run_path).resolve()
    with open_analysis_inputs(run_path) as inputs:
        report = analysis_directory(run_path)
        document = {'schema_version': 1, 'kind': 'whole_cube_comparison', 'run_id': run_path.name,
                    'analysis_id': report.name, 'status': 'running',
                    'source_run': contract_dict(inputs.manifest_ref), 'artifacts': []}
        write_manifest(report / 'manifest.json', document)
        artifacts = []

        def save_json(name, value):
            write_manifest(report / name, value)
            ref = _ref(report / name, Path(name).stem, report)
            artifacts.append(ref)
            return ref

        def save_array(name, value):
            np.save(report / name, value, allow_pickle=False)
            ref = array_reference(report / name, Path(name).stem, report)
            artifacts.append(ref)
            return ref

        try:
            k = len(inputs.taxonomy.class_names)
            valid = inputs.valid_pixels
            valid_ref = save_array('valid_pixels.npy', valid)
            gt_info, ground_truth = _ground_truth(inputs, ground_truth_manifest)
            gt_ref, metrics_ref, gt_mask = None, None, None
            gt_input_refs = []
            if ground_truth is not None:
                spec, labels, roi, gt_input_refs = ground_truth
                gt_ref = save_json('ground_truth.json', contract_dict(spec))
                save_array('gt_labels.npy', labels)
                gt_mask = roi & valid & (labels >= 0)
                mask_ref = save_array('gt_evaluation_mask.npy', gt_mask)
                gt_info.update(labeled_pixels=int((labels >= 0).sum()),
                               labeled_empty_pixels=int(((labels >= 0) & ~valid).sum()),
                               labeled_outside_roi=int(((labels >= 0) & ~roi).sum()),
                               evaluated_pixels=int(gt_mask.sum()), evaluation_mask=contract_dict(mask_ref))
                if not gt_mask.any():
                    gt_info.update(status='not_evaluable', reason='no_labeled_pixels_in_fixed_roi')
            evaluation = {'protocol': inputs.config.evaluation.protocol, 'taxonomy': contract_dict(inputs.taxonomy),
                          'branch_taxonomies': {key: contract_dict(value) for key, value in inputs.taxonomies.items()},
                          'standard_roi': contract_dict(valid_ref),
                          'standard_roi_rule': 'original_valid_samples_any_channel',
                          'gt_roi_rule': 'declared_roi_and_original_valid_pixels_and_labeled_gt',
                          'abstention_policy': 'prediction_minus_one_is_wrong_on_fixed_labeled_roi',
                          'macro_class_rule': 'classes_with_gt_support_in_fixed_roi',
                          'sam_is_ground_truth': False, 'ground_truth': gt_info}
            evaluation_ref = save_json('evaluation.json', evaluation)
            coverage, summary, metrics = {}, [], {}
            # Keep all four rows, including skipped/failed branches without invented maps or zeros.
            for outcome in inputs.outcomes:
                key = outcome.branch_id
                row = {'branch_id': key, 'status': outcome.status, 'reason_code': outcome.reason_code,
                       'coverage_all': None, 'coverage_valid': None, 'predicted_pixels': None,
                       'compactness_px': None, 'salt_pepper_fraction': None,
                       'gt_oa': None, 'gt_macro_f1': None, 'gt_kappa': None,
                       'gt_prediction_coverage': None, 'gt_n_evaluated': None}
                if key in inputs.maps:
                    values = coverage_summary(inputs.maps[key], valid, len(inputs.taxonomies[key].class_names), inputs.reasons[key])
                    from modules.metrics import compute_salt_pepper, compute_spatial_compactness
                    values.update(compactness_px=compute_spatial_compactness(inputs.maps[key]),
                                  salt_pepper_fraction=compute_salt_pepper(inputs.maps[key]))
                    coverage[key] = {**row, **values}
                    row.update({name: values[name] for name in ('coverage_all', 'coverage_valid', 'predicted_pixels',
                                                              'compactness_px', 'salt_pepper_fraction')})
                else:
                    coverage[key] = dict(row)
                if gt_mask is not None and gt_mask.any():
                    if key in inputs.maps and inputs.taxonomies[key] != inputs.taxonomy:
                        score = _partition_agreement(labels, inputs.maps[key], gt_mask)
                        score.update(coverage=score['coverage_joint'], n_evaluated=score['n_roi'])
                        row.update(gt_adjusted_rand=score['adjusted_rand'],
                                   gt_normalized_mutual_info=score['normalized_mutual_info'],
                                   gt_prediction_coverage=score['coverage'], gt_n_evaluated=score['n_evaluated'])
                    elif key in inputs.maps:
                        score = gt_scores(labels, inputs.maps[key], gt_mask, k)
                        confusion_ref = save_array(key + '_confusion_gt.npy', np.asarray(score['confusion'], dtype=np.int64))
                        score['confusion_ref'] = contract_dict(confusion_ref)
                        row.update(gt_oa=score['oa'], gt_macro_f1=score['macro_f1'], gt_kappa=score['kappa'],
                                   gt_prediction_coverage=score['coverage'], gt_n_evaluated=score['n_evaluated'])
                    else:
                        score = {'status': 'not_evaluable', 'reason': 'branch_' + outcome.status}
                    metrics[key] = score
                    row['gt_status'] = score['status']
                else:
                    row['gt_status'] = gt_info['reason']
                summary.append(row)
            coverage_ref = save_json('coverage.json', coverage)
            agreement = {'interpretation': 'descriptive_agreement_not_ground_truth_accuracy',
                         'roi': contract_dict(valid_ref), 'pairs': {}, 'sam_original_baseline': {}}
            def compare(left, right):
                if left not in inputs.maps or right not in inputs.maps:
                    return {'status': 'not_evaluable', 'reason': 'branch_unavailable'}
                if inputs.taxonomies[left] != inputs.taxonomies[right]:
                    return _partition_agreement(inputs.maps[left], inputs.maps[right], valid)
                return _agreement(inputs.maps[left], inputs.maps[right], valid, len(inputs.taxonomies[left].class_names))

            for left, right in combinations(BRANCHES, 2):
                pair = compare(left, right)
                agreement['pairs'][left + '__' + right] = pair
            for row in summary:
                key = row['branch_id']
                pair = (compare('sam_original', key) if key == 'sam_original' else
                        agreement['pairs']['sam_original__' + key])
                agreement['sam_original_baseline'][key] = pair
                row.update(sam_agreement_status=pair['status'], sam_agreement_on_joint=pair.get('agreement_on_joint'),
                           sam_joint_coverage=pair.get('coverage_joint'), sam_kappa_on_joint=pair.get('kappa_on_joint'),
                           sam_macro_f1_on_joint=pair.get('macro_f1_on_joint'),
                           sam_adjusted_rand=pair.get('adjusted_rand'), sam_normalized_mutual_info=pair.get('normalized_mutual_info'))
            agreement_ref = save_json('agreement.json', agreement)
            if metrics:
                metrics_ref = save_json('metrics_gt.json', {'ground_truth': gt_info, 'branches': metrics,
                    'confusion_rows': list(range(k)), 'confusion_columns': [*range(k), -1]})
                per_class = [{'branch_id': key, 'class_name': inputs.taxonomy.class_names[item['class_id']], **item}
                             for key, scores in metrics.items() for item in scores.get('per_class', [])]
                if per_class:
                    _csv(report / 'per_class_gt.csv', per_class)
                    artifacts.append(_ref(report / 'per_class_gt.csv', 'per_class_gt', report))
            _csv(report / 'summary.csv', summary)
            artifacts.append(_ref(report / 'summary.csv', 'summary', report))
            comparison = ComparisonResult(run_path.name, report.name, inputs.taxonomy, evaluation_ref,
                inputs.outcomes, coverage_ref, agreement_ref, gt_ref, metrics_ref,
                None if metrics_ref else gt_info['reason'])
            save_json('comparison.json', contract_dict(comparison))
            inputs.verify_unchanged()
            for ref in gt_input_refs:
                verify_ref(ref, run_path)
            document.update(status='completed', result=contract_dict(comparison),
                            inputs=[contract_dict(ref) for ref in (*inputs.inputs.values(), *gt_input_refs)],
                            artifacts=[contract_dict(ref) for ref in artifacts], visualization='not_run')
            write_manifest(report / 'manifest.json', document)
            return report
        except BaseException as error:
            document.update(status='failed', error=f'{type(error).__name__}: {error}',
                            artifacts=[contract_dict(ref) for ref in artifacts])
            write_manifest(report / 'manifest.json', document)
            raise
