import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

def compute_metrics(y_true, y_pred, y_proba):
    """Works for binary and multi-class (weighted averages)."""
    n_classes = y_proba.shape[1]
    if n_classes == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1]); avg = "binary"
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"); avg = "weighted"
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
