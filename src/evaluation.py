import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import CLASS_ORDER, N_CLASSES, SEED

ATTACK_START = 1


def confusion(y_true, y_pred, k=N_CLASSES):
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def per_class(cm):
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0)
    actual = cm.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def macro_f1(y_true, y_pred):
    return per_class(confusion(y_true, y_pred))[2].mean()


def detection_scores(cm):
    attacks = cm[ATTACK_START:]
    detection = 1 - attacks[:, 0].sum() / attacks.sum()
    false_alarm = 1 - cm[0, 0] / cm[0].sum()
    return detection, false_alarm


def expected_calibration_error(y_true, proba, bins=15):
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return ece


def macro_roc_auc(y_true, proba):
    aucs = [roc_auc_score(y_true == k, proba[:, k]) for k in range(N_CLASSES)
            if 0 < (y_true == k).sum() < len(y_true)]
    return float(np.mean(aucs))


def full_proba(proba, classes):
    out = np.zeros((proba.shape[0], N_CLASSES))
    out[:, classes] = proba
    return out


def evaluate(y_true, proba):
    y_pred = proba.argmax(axis=1)
    cm = confusion(y_true, y_pred)
    precision, recall, f1 = per_class(cm)
    present = cm.sum(axis=1) > 0
    detection, false_alarm = detection_scores(cm)
    scores = {
        "accuracy": np.trace(cm) / cm.sum(),
        "balanced_accuracy": recall[present].mean(),
        "macro_precision": precision.mean(),
        "macro_f1": f1.mean(),
        "roc_auc": macro_roc_auc(y_true, proba),
        "brier": np.mean(np.sum((proba - np.eye(N_CLASSES)[y_true]) ** 2, axis=1)),
        "ece": expected_calibration_error(y_true, proba),
        "detection_rate": detection,
        "false_alarm_rate": false_alarm,
    }
    scores.update({f"f1_{c}": v for c, v in zip(CLASS_ORDER, f1)})
    scores.update({f"recall_{c}": v for c, v in zip(CLASS_ORDER, recall)})
    return scores, cm


def fit_evaluate(model, X_train, y_train, X_eval, y_eval):
    model.fit(X_train, y_train)
    proba = full_proba(model.predict_proba(X_eval), model.classes_)
    scores, cm = evaluate(y_eval, proba)
    scores.update({"fit_s": model.fit_seconds, "predict_ms_per_1k": model.predict_ms_per_1k})
    return scores, cm, proba


def bootstrap_diff(y_true, pred_a, pred_b, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    n_rows = len(y_true)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_rows, n_rows)
        diffs[i] = macro_f1(y_true[idx], pred_b[idx]) - macro_f1(y_true[idx], pred_a[idx])
    return np.percentile(diffs, [2.5, 97.5])


def summarize(df, by, metrics):
    grouped = df.groupby(by, sort=False)[metrics]
    return grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_std"))
