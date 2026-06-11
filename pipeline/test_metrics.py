"""Calculate mAP@IoU% for predictions"""

import numpy as np

def calculate_iou():
    """Calculates the IoU threshold of a clip"""
    # need the ground truth processing function
    return

def calculate_ap(positives, num_gt):
    """
    Calculates the average precision for a class

    Args:
        positives (bool list): sorted boolean array to indicate TP/FP
        num_gt (int): number of ground truth labels

    Returns:
        average precision
    """
    tp = np.cumsum(positives)
    fp = np.cumsum([not p for p in positives])
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / num_gt

    # 11-point interpolated AP (PASCAL VOC)
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        p_at_r = precision[recall >= thr].max() if (recall >= thr).any() else 0.0
        ap += p_at_r / 11
    return ap

def calculate_map_iou():
    """
    Primary function that will call the rest

    Args:
        preds: prediction matrix
        gt_path: path to the ground truth tsv
        conf_threshold: confidence threshold for a true prediction
        iou_threshold: IoU threshold for a true positive
    
    Returns:
        
    """

    """
    for L in labels
        for C in clips
            sort C by index L values (least to greatest)
            drop any values below the confidence threshold
            determine if C is TP or FP based on IoU value
                call calculate_IoU()
                save TP/FP order
            calculate average precision for L
                call calculate_ap()
                save results in a list
    calculate mean average precision using average precision results
    display mAP and individual APs
    """
    return

def main():
    arr = [True, True, False, True]
    gts = 3
    print(calculate_ap(arr, gts))

main()