import numpy as np


def compute_binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    pred = pred_mask > 0
    gt = gt_mask > 0

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    eps = 1e-8

    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp_px": int(tp),
        "fp_px": int(fp),
        "fn_px": int(fn),
        "tn_px": int(tn),
        "gt_area_px": int(gt.sum()),
        "pred_area_px": int(pred.sum()),
        "false_positive_area_px": int(fp),
        "false_negative_area_px": int(fn),
    }

def compute_multiclass_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, class_g_value: int) -> dict:
    pred = pred_mask > 0
    #slight change means that we can just use the class_id(set as the grayscale intensity value)
    gt = gt_mask == class_g_value

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    eps = 1e-8

    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp_px": int(tp),
        "fp_px": int(fp),
        "fn_px": int(fn),
        "tn_px": int(tn),
        "gt_area_px": int(gt.sum()),
        "pred_area_px": int(pred.sum()),
        "false_positive_area_px": int(fp),
        "false_negative_area_px": int(fn),
    }