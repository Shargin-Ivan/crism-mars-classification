"""Fixed-ROI metrics for canonical maps; abstentions never remove GT support."""
import numpy as np

from .pipeline_contracts import RejectionReason, integer, require


def _labels(value, name, num_classes, shape=None):
    integer(num_classes, 'num_classes', minimum=1)
    require(isinstance(value, np.ndarray) and value.ndim == 2
            and value.dtype.kind in 'iu', f'{name}: expected integer H,W array')
    require(shape is None or value.shape == shape, f'{name}: shape mismatch')
    require(np.all((value >= -1) & (value < num_classes)), f'{name}: invalid class ID')
    return value


def _mask(value, shape, name):
    require(isinstance(value, np.ndarray) and value.dtype == bool
            and value.shape == shape, f'{name}: expected bool mask with matching shape')
    return value


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _kappa(confusion):
    n = int(confusion.sum())
    if not n:
        return None
    k = confusion.shape[0]
    observed = float(np.trace(confusion)) / n
    expected = float(np.dot(confusion.sum(axis=1).astype(float),
                            confusion[:, :k].sum(axis=0).astype(float))) / n ** 2
    return (observed - expected) / (1 - expected) if expected < 1 else None


def gt_scores(gt, pred, roi, num_classes):
    """Score roi & labeled GT, retaining rejected predictions in every denominator.

    The caller must construct roi independently of predictions. Canonical class
    zero is a regular class; only -1 denotes unlabeled GT or abstention.
    """
    _labels(gt, 'gt', num_classes)
    _labels(pred, 'pred', num_classes, gt.shape)
    selected = _mask(roi, gt.shape, 'roi') & (gt >= 0)
    truth, predicted = gt[selected], pred[selected]
    n = int(truth.size)
    n_predicted = int(np.count_nonzero(predicted >= 0))
    columns = np.where(predicted >= 0, predicted, num_classes).astype(np.int64)
    confusion = np.bincount(truth.astype(np.int64) * (num_classes + 1) + columns,
                            minlength=num_classes * (num_classes + 1))
    confusion = confusion.reshape(num_classes, num_classes + 1)
    per_class = []
    for class_id in range(num_classes):
        tp = int(confusion[class_id, class_id])
        support = int(confusion[class_id].sum())
        count = int(confusion[:, class_id].sum())
        rejected = int(confusion[class_id, -1])
        fp, fn = count - tp, support - tp
        per_class.append(dict(class_id=class_id, support=support, predicted=count,
                              tp=tp, fp=fp, fn=fn, precision=_ratio(tp, count),
                              recall=_ratio(tp, support), f1=_ratio(2 * tp, 2 * tp + fp + fn),
                              rejected=rejected, rejection_rate=_ratio(rejected, support)))
    macro_ids = [c['class_id'] for c in per_class if c['support'] > 0]
    correct = int(np.trace(confusion))
    return dict(status='evaluated' if n else 'not_evaluable',
                reason=None if n else 'no_labeled_pixels_in_roi',
                n_evaluated=n, n_predicted=n_predicted, coverage=_ratio(n_predicted, n),
                oa=_ratio(correct, n),
                covered_accuracy=_ratio(correct, n_predicted),
                macro_f1=(float(np.mean([per_class[i]['f1'] for i in macro_ids]))
                          if macro_ids else None),
                kappa=_kappa(confusion), macro_class_ids=macro_ids,
                per_class=per_class, confusion=confusion.tolist(),
                confusion_columns=list(range(num_classes)) + [-1])


def pair_agreement(left, right, roi, num_classes):
    """Separate joint prediction agreement from coverage on the fixed ROI."""
    _labels(left, 'left', num_classes)
    _labels(right, 'right', num_classes, left.shape)
    _mask(roi, left.shape, 'roi')
    joint = roi & (left >= 0) & (right >= 0)
    n_roi, n_joint = int(roi.sum()), int(joint.sum())
    matches = int(np.count_nonzero(joint & (left == right)))
    confusion = np.bincount(left[joint].astype(np.int64) * num_classes
                            + right[joint].astype(np.int64),
                            minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return dict(status='evaluated' if n_joint else 'not_evaluable',
                reason=None if n_joint else 'no_joint_predictions_in_roi',
                n_roi=n_roi, n_joint=n_joint, n_matching=matches,
                coverage_joint=_ratio(n_joint, n_roi),
                agreement_on_joint=_ratio(matches, n_joint),
                kappa_on_joint=_kappa(confusion),
                matching_predictions_over_roi=_ratio(matches, n_roi))


def coverage_summary(labels, valid_pixels, num_classes, reasons=None):
    """Summarize support; no class may be predicted at an empty input pixel."""
    _labels(labels, 'labels', num_classes)
    _mask(valid_pixels, labels.shape, 'valid_pixels')
    predicted = labels >= 0
    require(not np.any(predicted & ~valid_pixels), 'Empty pixels cannot receive classes')
    reason_counts = {}
    if reasons is not None:
        require(isinstance(reasons, np.ndarray) and reasons.shape == labels.shape
                and reasons.dtype.kind in 'iu', 'reasons: expected integer H,W array')
        require(np.isin(reasons, [int(r) for r in RejectionReason]).all(),
                'Unknown rejection reason')
        require(np.array_equal(reasons == 0, predicted),
                'Rejection reasons and predictions disagree')
        reason_counts = {r.name: int(np.count_nonzero(reasons == r)) for r in RejectionReason}
    total, valid = int(labels.size), int(valid_pixels.sum())
    count = int(predicted.sum())
    predicted_valid = int(np.count_nonzero(predicted & valid_pixels))
    return dict(total_pixels=total, valid_pixels=valid, predicted_pixels=count,
                predicted_valid_pixels=predicted_valid, coverage_all=_ratio(count, total),
                coverage_valid=_ratio(predicted_valid, valid),
                class_counts=np.bincount(labels[predicted].astype(np.int64),
                                         minlength=num_classes).tolist(),
                reason_counts=reason_counts)
