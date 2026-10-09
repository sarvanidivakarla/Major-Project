"""Train and evaluate a binary logistic-regression attack detector."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABEL_COLUMN = "Label"
BENIGN_LABEL = "Benign"
BASE_DIR = Path(__file__).resolve().parent


def train_and_evaluate(input_path: Path, model_path: Path, metrics_path: Path, test_size: float) -> dict[str, float]:
    data = pd.read_csv(input_path, low_memory=False)
    if LABEL_COLUMN not in data.columns:
        raise ValueError(f"The input must contain a '{LABEL_COLUMN}' column.")

    feature_names = [column for column in data.columns if column != LABEL_COLUMN]
    features = data[feature_names].apply(pd.to_numeric, errors="coerce")
    labels = (data[LABEL_COLUMN].astype(str).str.strip() != BENIGN_LABEL).astype(np.int8)
    valid_rows = features.notna().all(axis=1) & labels.notna()
    features = features.loc[valid_rows]
    labels = labels.loc[valid_rows]

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=200, class_weight="balanced", n_jobs=-1),
    )

    train_started = time.perf_counter()
    model.fit(x_train, y_train)
    training_time = time.perf_counter() - train_started

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    inference_started = time.perf_counter()
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)
    inference_time = time.perf_counter() - inference_started

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()
    metrics = {
        "training_accuracy": float(accuracy_score(y_train, train_predictions)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "testing_accuracy": float(accuracy_score(y_test, test_predictions)),
        "training_f1": float(f1_score(y_train, train_predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "testing_f1": float(f1_score(y_test, test_predictions, zero_division=0)),
        "recall_detection_rate": float(recall_score(y_test, predictions, zero_division=0)),
        "training_recall": float(recall_score(y_train, train_predictions, zero_division=0)),
        "testing_recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "false_positive_rate": float(false_positive / (false_positive + true_negative)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "training_time_seconds": float(training_time),
        "inference_time_seconds": float(inference_time),
        "inference_time_ms_per_record": float(inference_time / len(x_test) * 1000),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "attack_rows": int(labels.sum()),
        "benign_rows": int((labels == 0).sum()),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, default=BASE_DIR / "artifacts" / "logistic_attack_model.joblib")
    parser.add_argument("--metrics", type=Path, default=BASE_DIR / "artifacts" / "metrics.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    metrics = train_and_evaluate(args.input, args.model, args.metrics, args.test_size)
    print("Training accuracy:", metrics["training_accuracy"])
    print("Testing accuracy:", metrics["testing_accuracy"])
    print("Training F1:", metrics["training_f1"])
    print("Testing F1:", metrics["testing_f1"])
    print("Training recall:", metrics["training_recall"])
    print("Testing recall:", metrics["testing_recall"])
    for name, value in metrics.items():
        if name not in {
            "training_accuracy",
            "testing_accuracy",
            "training_f1",
            "testing_f1",
            "training_recall",
            "testing_recall",
        }:
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()