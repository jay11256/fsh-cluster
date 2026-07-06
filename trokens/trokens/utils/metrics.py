#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

import torch
import numpy as np

def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]



def multitask_topks_correct(preds, labels, ks=(1,)):
    """
    Args:
        preds: tuple(torch.FloatTensor), each tensor should be of shape
            [batch_size, class_count], class_count can vary on a per task basis, i.e.
            outputs[i].shape[1] can be different to outputs[j].shape[j].
        labels: tuple(torch.LongTensor), each tensor should be of shape [batch_size]
        ks: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    max_k = int(np.max(ks))
    task_count = len(preds)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
    all_correct = all_correct.to(preds[0].device)
    for output, label in zip(preds, labels):
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # Flip batch_size, class_count as .view doesn't work on non-contiguous
        max_k_idx = max_k_idx.t()
        correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        all_correct.add_(correct_for_task)

    multitask_topks_correct = [
        torch.ge(all_correct[:k].float().sum(0), task_count).float().sum(0) for k in ks
    ]

    return multitask_topks_correct


def multilabel_f1(logits, labels, threshold=0.5):
    """Macro F1 over classes with at least one positive label. Returns [0, 100]."""
    stats = multilabel_per_class_stats(logits, labels, threshold=threshold)
    f1_values = [s["f1"] for s in stats if s["support"] > 0]
    if len(f1_values) == 0:
        return torch.tensor(0.0, device=logits.device)
    return torch.tensor(f1_values, device=logits.device).mean() * 100.0
def multilabel_hamming_score(logits, labels, threshold=0.5):
    """Fraction of correct label decisions. Returns [0, 100]."""
    labels = labels.float()
    preds = (torch.sigmoid(logits) >= threshold).float()
    return (preds == labels).float().mean() * 100.0
def multilabel_exact_match(logits, labels, threshold=0.5):
    """Fraction of samples with all labels correct. Returns [0, 100]."""
    labels = labels.float()
    preds = (torch.sigmoid(logits) >= threshold).float()
    return (preds == labels).all(dim=1).float().mean() * 100.0
def multilabel_per_class_stats(logits, labels, threshold=0.5, class_names=None):
    """Per-class TP/TN/FP/FN, precision, recall, F1."""
    labels = labels.float()
    preds = (torch.sigmoid(logits) >= threshold).float()
    num_classes = labels.shape[1]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    stats = []
    for class_idx in range(num_classes):
        y_true = labels[:, class_idx]
        y_pred = preds[:, class_idx]
        tp = ((y_pred == 1) & (y_true == 1)).sum().item()
        fp = ((y_pred == 1) & (y_true == 0)).sum().item()
        fn = ((y_pred == 0) & (y_true == 1)).sum().item()
        tn = ((y_pred == 0) & (y_true == 0)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        stats.append({
            "class_idx": class_idx,
            "class_name": class_names[class_idx],
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "support": int((y_true == 1).sum().item()),
            "precision": precision, "recall": recall, "f1": f1,
        })
    return stats
def format_multilabel_metrics_report(stats):
    lines = [
        "Per-class multi-label metrics", "=" * 80,
        f"{'class':<24} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6} "
        f"{'precision':>10} {'recall':>10} {'f1':>10}",
        "-" * 80,
    ]
    for row in stats:
        lines.append(
            f"{row['class_name']:<24} "
            f"{row['tp']:>6} {row['tn']:>6} {row['fp']:>6} {row['fn']:>6} "
            f"{row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1']:>10.4f}"
        )
    return "\n".join(lines) + "\n"


def multitask_topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
   """
    num_multitask_topks_correct = multitask_topks_correct(preds, labels, ks)
    return [(x / preds[0].size(0)) * 100.0 for x in num_multitask_topks_correct]