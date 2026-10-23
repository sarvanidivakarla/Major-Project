"""Train and evaluate the Version 2 Random Forest intrusion detector."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
LABEL_COLUMN = "Label"
TIMESTAMP_COLUMN = "Timestamp"
BENIGN_LABEL = "Benign"
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"
WINDOW_MINUTES = 30
PARTITIONS = ("train", "validation", "test")
EXCLUDED_COLUMNS = {LABEL_COLUMN, TIMESTAMP_COLUMN, "Flow ID", "Src IP", "Dst IP"}
MODEL_CONFIG = {
    "model": "RandomForestClassifier",
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def current_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return None


def load_dataset(input_path: Path, suspicious_before: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    data = pd.read_csv(input_path, low_memory=False)
    required_columns = {LABEL_COLUMN, TIMESTAMP_COLUMN}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    parsed_timestamps = pd.to_datetime(
        data[TIMESTAMP_COLUMN], format=TIMESTAMP_FORMAT, errors="coerce"
    )
    invalid_timestamp_mask = parsed_timestamps.isna()
    suspicious_mask = parsed_timestamps.notna() & (parsed_timestamps < suspicious_before)
    feature_columns = [column for column in data.columns if column not in EXCLUDED_COLUMNS]
    numeric_features = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    finite_features = numeric_features.notna().all(axis=1) & ~numeric_features.isin(
        [np.inf, -np.inf]
    ).any(axis=1)
    labels = data[LABEL_COLUMN].astype("string").str.strip()
    valid_labels = labels.notna() & labels.ne("")
    usable_mask = parsed_timestamps.notna() & finite_features & valid_labels

    usable = data.loc[usable_mask].copy()
    usable["_Parsed Timestamp"] = parsed_timestamps.loc[usable_mask]
    usable[feature_columns] = numeric_features.loc[usable_mask].astype(float)
    usable["_Target"] = (labels.loc[usable_mask] != BENIGN_LABEL).astype("int8")
    usable["_Row ID"] = usable.index.astype("int64")

    manifest = {
        "input_path": str(input_path),
        "input_rows": int(len(data)),
        "input_columns": int(len(data.columns)),
        "invalid_or_missing_timestamp_rows": int(invalid_timestamp_mask.sum()),
        "invalid_or_missing_timestamp_percent": float(invalid_timestamp_mask.mean() * 100),
        "suspicious_timestamp_rows": int(suspicious_mask.sum()),
        "suspicious_timestamp_percent": float(suspicious_mask.mean() * 100),
        "suspicious_timestamps_excluded": False,
        "invalid_or_nonfinite_feature_rows": int((~finite_features).sum()),
        "invalid_or_missing_label_rows": int((~valid_labels).sum()),
        "usable_rows": int(len(usable)),
        "usable_columns": int(len(usable.columns)),
        "timestamp_format": TIMESTAMP_FORMAT,
        "timezone_assumption": "Naive timestamps; no timezone conversion applied",
        "label_mapping": {"Benign": 0, "non-Benign": 1},
        "class_distribution": {
            "benign": int((usable["_Target"] == 0).sum()),
            "attack": int((usable["_Target"] == 1).sum()),
        },
        "feature_columns": feature_columns,
    }
    return usable, manifest, feature_columns


def assign_windows(usable: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = usable.copy()
    records["window_start"] = records["_Parsed Timestamp"].dt.floor("30min")
    records["window_end"] = records["window_start"] + pd.Timedelta(minutes=WINDOW_MINUTES)
    grouped = records.groupby("window_start", sort=True)
    windows = grouped.agg(
        record_count=("_Target", "size"),
        benign_count=("_Target", lambda values: int((values == 0).sum())),
        attack_count=("_Target", lambda values: int((values == 1).sum())),
        first_timestamp=("_Parsed Timestamp", "min"),
        last_timestamp=("_Parsed Timestamp", "max"),
    ).reset_index()
    windows["window_end"] = windows["window_start"] + pd.Timedelta(minutes=WINDOW_MINUTES)
    windows["window_id"] = [f"window-{number:06d}" for number in range(1, len(windows) + 1)]
    return records, windows[
        [
            "window_id",
            "window_start",
            "window_end",
            "record_count",
            "benign_count",
            "attack_count",
            "first_timestamp",
            "last_timestamp",
        ]
    ]


def assign_temporal_partitions(records: pd.DataFrame, windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(windows) < 3:
        raise ValueError("At least three complete windows are required for temporal evaluation.")
    total = len(records)
    targets = np.array([total * 0.70, total * 0.15, total * 0.15])
    counts = windows["record_count"].to_numpy(dtype=np.int64)
    cumulative = np.concatenate(([0], np.cumsum(counts)))
    best = None
    for first_boundary in range(1, len(windows) - 1):
        for second_boundary in range(first_boundary + 1, len(windows)):
            actual = np.array(
                [
                    cumulative[first_boundary],
                    cumulative[second_boundary] - cumulative[first_boundary],
                    cumulative[-1] - cumulative[second_boundary],
                ],
                dtype=float,
            )
            score = (float(np.abs(actual - targets).sum()), first_boundary, second_boundary)
            if best is None or score < best[0]:
                best = (score, first_boundary, second_boundary, actual)
    assert best is not None
    _, first_boundary, second_boundary, actual = best

    windows = windows.copy()
    windows["partition"] = "test"
    windows.loc[: first_boundary - 1, "partition"] = "train"
    windows.loc[first_boundary : second_boundary - 1, "partition"] = "validation"
    window_partition = dict(zip(windows["window_id"], windows["partition"]))
    window_by_start = dict(zip(windows["window_start"], windows["window_id"]))
    records = records.copy()
    records["window_id"] = records["window_start"].map(window_by_start)
    records["partition"] = records["window_id"].map(window_partition)
    allocation = actual / total * 100
    audit = {
        "target_percentages": {name: float(target / total * 100) for name, target in zip(PARTITIONS, targets)},
        "actual_record_counts": {name: int(value) for name, value in zip(PARTITIONS, actual)},
        "actual_percentages": {name: float(value) for name, value in zip(PARTITIONS, allocation)},
        "allocation_deviation_percentage_points": {
            name: float(value - target / total * 100)
            for name, value, target in zip(PARTITIONS, allocation, targets)
        },
        "train_last_window": windows.iloc[first_boundary - 1]["window_id"],
        "validation_last_window": windows.iloc[second_boundary - 1]["window_id"],
    }
    return records, windows, audit


def verify_existing_temporal_audit(windows: pd.DataFrame, audit_path: Path) -> bool:
    if not audit_path.exists():
        return False
    existing = pd.read_csv(audit_path)
    expected_columns = ["window_id", "window_start", "window_end", "record_count", "partition"]
    if not set(expected_columns).issubset(existing.columns) or len(existing) != len(windows):
        raise ValueError("Existing Version 1.1 window audit does not match the current dataset.")
    current = windows.copy()
    current["window_start"] = current["window_start"].astype(str).str.replace("T", " ")
    current["window_end"] = current["window_end"].astype(str).str.replace("T", " ")
    for column in expected_columns:
        if not (existing[column].astype(str).to_numpy() == current[column].astype(str).to_numpy()).all():
            raise ValueError(f"Existing Version 1.1 audit differs in '{column}'.")
    return True


def partition_summary(records: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for partition in PARTITIONS:
        subset = records[records["partition"] == partition]
        subset_windows = windows[windows["partition"] == partition]
        if subset.empty:
            raise ValueError(f"The {partition} partition is empty.")
        benign = int((subset["_Target"] == 0).sum())
        attack = int((subset["_Target"] == 1).sum())
        rows.append(
            {
                "partition": partition,
                "records": int(len(subset)),
                "allocation_percent": float(len(subset) / len(records) * 100),
                "number_of_windows": int(len(subset_windows)),
                "benign_count": benign,
                "attack_count": attack,
                "benign_percent": float(benign / len(subset) * 100),
                "attack_percent": float(attack / len(subset) * 100),
                "first_timestamp": subset["_Parsed Timestamp"].min().isoformat(sep=" "),
                "last_timestamp": subset["_Parsed Timestamp"].max().isoformat(sep=" "),
                "first_window_start": subset_windows["window_start"].min().isoformat(sep=" "),
                "last_window_end": subset_windows["window_end"].max().isoformat(sep=" "),
            }
        )
    return pd.DataFrame(rows)


def validate_integrity(records: pd.DataFrame, windows: pd.DataFrame, feature_columns: list[str]) -> None:
    if set(records["partition"].dropna().unique()) != set(PARTITIONS):
        raise AssertionError("All three temporal partitions are not present.")
    if records["partition"].isna().any() or len(records) != sum((records["partition"] == name).sum() for name in PARTITIONS):
        raise AssertionError("Temporal partitions do not account for every usable record.")
    if windows.groupby("window_id")["partition"].nunique().max() != 1:
        raise AssertionError("A temporal window appears in multiple partitions.")
    if records["window_id"].nunique() != windows["window_id"].nunique():
        raise AssertionError("Some temporal windows have no assigned records.")
    partition_records = {name: records[records["partition"] == name] for name in PARTITIONS}
    if not partition_records["train"]["_Parsed Timestamp"].max() < partition_records["validation"]["_Parsed Timestamp"].min():
        raise AssertionError("Training timestamps are not before validation timestamps.")
    if not partition_records["validation"]["_Parsed Timestamp"].max() < partition_records["test"]["_Parsed Timestamp"].min():
        raise AssertionError("Validation timestamps are not before test timestamps.")
    window_groups = {name: windows[windows["partition"] == name] for name in PARTITIONS}
    if not window_groups["train"]["window_end"].max() <= window_groups["validation"]["window_start"].min():
        raise AssertionError("Train and validation windows overlap.")
    if not window_groups["validation"]["window_end"].max() <= window_groups["test"]["window_start"].min():
        raise AssertionError("Validation and test windows overlap.")
    features = records[feature_columns]
    if list(features.columns) != feature_columns:
        raise AssertionError("Feature columns are not stable or ordered.")
    if not all(pd.api.types.is_numeric_dtype(features[column]) for column in feature_columns):
        raise AssertionError("A model feature is not numeric.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise AssertionError("A model feature is not finite.")
    if set(feature_columns).intersection(EXCLUDED_COLUMNS):
        raise AssertionError("Metadata or target columns entered the model feature matrix.")


def safe_metric(metric, y_true: pd.Series, y_pred: np.ndarray) -> float | None:
    if metric is precision_score and y_pred.sum() == 0:
        return None
    if metric is recall_score and y_true.sum() == 0:
        return None
    if metric is f1_score and (y_true.sum() == 0 or y_pred.sum() == 0):
        return None
    return float(metric(y_true, y_pred, zero_division=0))


def evaluate(model: RandomForestClassifier, features: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
    started = time.perf_counter()
    probabilities = model.predict_proba(features)[:, 1]
    inference_seconds = time.perf_counter() - started
    predictions = (probabilities >= 0.5).astype("int8")
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    negative_total = true_negative + false_positive
    positive_total = true_positive + false_negative
    return {
        "records": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": safe_metric(precision_score, labels, predictions),
        "recall": safe_metric(recall_score, labels, predictions),
        "f1_score": safe_metric(f1_score, labels, predictions),
        "false_positive_rate": float(false_positive / negative_total) if negative_total else None,
        "false_negative_rate": float(false_negative / positive_total) if positive_total else None,
        "roc_auc": float(roc_auc_score(labels, probabilities)) if labels.nunique() == 2 else None,
        "pr_auc": float(average_precision_score(labels, probabilities)) if labels.sum() else None,
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "inference_time_seconds": float(inference_seconds),
        "inference_time_ms_per_record": float(inference_seconds / len(labels) * 1000),
        "threshold": 0.5,
    }


def create_model() -> RandomForestClassifier:
    parameters = {key: value for key, value in MODEL_CONFIG.items() if key != "model"}
    return RandomForestClassifier(**parameters)


def fit_model(features: pd.DataFrame, labels: pd.Series) -> tuple[RandomForestClassifier, float, int | None]:
    if labels.nunique() < 2:
        raise ValueError("Training partition contains only one target class.")
    memory_before = current_memory_bytes()
    started = time.perf_counter()
    model = create_model()
    model.fit(features, labels)
    duration = time.perf_counter() - started
    memory_after = current_memory_bytes()
    memory_delta = memory_after - memory_before if memory_before is not None and memory_after is not None else None
    return model, duration, memory_delta


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_value(payload), indent=2), encoding="utf-8")


def write_comparison_report(output_dir: Path, random_metrics: dict[str, Any], temporal_metrics: dict[str, Any]) -> None:
    lines = [
        "# Logistic Regression vs Random Forest: Version 2 Comparison",
        "",
        "Random Forest results are from Version 2. Logistic Regression values are historical baselines from the existing report.",
        "Random-split results are not a direct comparison because Version 1 used a different preprocessed dataset and feature schema.",
        "The temporal protocol uses the matching current timestamp-retaining dataset and 30-minute partitions.",
        "",
        "## Random 80/20 Protocol",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | FPR | FNR | ROC-AUC | PR-AUC | Train seconds | Inference ms/record |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Logistic Regression (historical) | 99.9646% | 99.9043% | 99.9987% | 99.9515% | 0.0550% | N/A | N/A | 99.9990% | 7.60 | 0.00099 |",
        f"| Random Forest v2 | {random_metrics['test']['accuracy']:.4%} | {random_metrics['test']['precision']:.4%} | {random_metrics['test']['recall']:.4%} | {random_metrics['test']['f1_score']:.4%} | {random_metrics['test']['false_positive_rate']:.4%} | {random_metrics['test']['false_negative_rate']:.4%} | {random_metrics['test']['roc_auc']:.4%} | {random_metrics['test']['pr_auc']:.4%} | {random_metrics['training_time_seconds']:.2f} | {random_metrics['test']['inference_time_ms_per_record']:.5f} |",
        "",
        "## Chronological Temporal-Window Protocol",
        "",
        "| Model | Partition | Accuracy | Precision | Recall | F1 | FPR | FNR | ROC-AUC | PR-AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for partition in PARTITIONS:
        metrics = temporal_metrics["partitions"][partition]
        lines.append(
            f"| Random Forest v2 | {partition} | {metrics['accuracy']:.4%} | {metrics['precision']:.4%} | {metrics['recall']:.4%} | {metrics['f1_score']:.4%} | {metrics['false_positive_rate']:.4%} | {metrics['false_negative_rate']:.4%} | {metrics['roc_auc']:.4%} | {metrics['pr_auc']:.4%} |"
        )
    lines.extend(
        [
            "| Logistic Regression v1.1 | test | 99.9561% | 99.1328% | 100.0000% | 99.5645% | 0.0462% | 0.0000% | 99.9981% | 99.9243% |",
            "",
            "The primary comparison is the chronological test result. Differences in model scores should be interpreted together with false alarms, missed attacks, temporal coverage, and computational cost.",
        ]
    )
    (output_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(input_path: Path, output_dir: Path, suspicious_before: pd.Timestamp) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    random_dir = output_dir / "random_split"
    temporal_dir = output_dir / "temporal"
    random_dir.mkdir(exist_ok=True)
    temporal_dir.mkdir(exist_ok=True)
    logs = {"random": [], "temporal": []}

    def log(protocol: str, message: str) -> None:
        print(f"[{protocol}] {message}")
        logs[protocol].append(message)

    usable, manifest, feature_columns = load_dataset(input_path, suspicious_before)
    records, windows = assign_windows(usable)
    records, windows, temporal_allocation = assign_temporal_partitions(records, windows)
    existing_audit = BASE_DIR / "artifacts" / "v1_1_temporal" / "window_audit.csv"
    matched_existing = verify_existing_temporal_audit(windows, existing_audit)
    validate_integrity(records, windows, feature_columns)
    summary = partition_summary(records, windows)

    x = usable[feature_columns]
    y = usable["_Target"]
    indices = np.arange(len(usable))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    random_model, random_training_seconds, random_memory_delta = fit_model(
        x.iloc[train_indices], y.iloc[train_indices]
    )
    random_metrics = {
        "protocol": "random_stratified_80_20",
        "split_seed": 42,
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "training_time_seconds": random_training_seconds,
        "training_memory_delta_bytes": random_memory_delta,
        "train": evaluate(random_model, x.iloc[train_indices], y.iloc[train_indices]),
        "test": evaluate(random_model, x.iloc[test_indices], y.iloc[test_indices]),
    }

    temporal_models = {}
    temporal_metrics: dict[str, Any] = {
        "protocol": "chronological_30_minute_windows",
        "window_duration_minutes": WINDOW_MINUTES,
        "training_time_seconds": None,
        "training_memory_delta_bytes": None,
        "partitions": {},
    }
    temporal_train = records[records["partition"] == "train"]
    temporal_model, temporal_training_seconds, temporal_memory_delta = fit_model(
        temporal_train[feature_columns], temporal_train["_Target"]
    )
    temporal_models["train"] = temporal_model
    temporal_metrics["training_time_seconds"] = temporal_training_seconds
    temporal_metrics["training_memory_delta_bytes"] = temporal_memory_delta
    for partition in PARTITIONS:
        subset = records[records["partition"] == partition]
        temporal_metrics["partitions"][partition] = evaluate(
            temporal_model, subset[feature_columns], subset["_Target"]
        )

    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": temporal_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    feature_importance.insert(0, "rank", np.arange(1, len(feature_importance) + 1))
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    joblib.dump(random_model, random_dir / "random_forest_model.joblib")
    joblib.dump(temporal_model, output_dir / "random_forest_model.joblib")
    reloaded_model = joblib.load(output_dir / "random_forest_model.joblib")
    if list(reloaded_model.feature_names_in_) != feature_columns:
        raise AssertionError("Reloaded model feature schema does not match the training schema.")

    model_size = (output_dir / "random_forest_model.joblib").stat().st_size
    manifest["software_versions"] = {"python": __import__("platform").python_version(), "pandas": pd.__version__, "scikit_learn": sklearn.__version__}
    manifest["random_split"] = {"train_rows": len(train_indices), "test_rows": len(test_indices), "seed": 42}
    manifest["temporal_partition"] = temporal_allocation
    manifest["existing_v1_1_window_audit_matched"] = matched_existing
    manifest["model_feature_count"] = len(feature_columns)
    manifest["model_size_bytes"] = model_size
    save_json(output_dir / "dataset_manifest.json", manifest)
    save_json(output_dir / "model_config.json", MODEL_CONFIG)
    save_json(random_dir / "evaluation_metrics.json", random_metrics)
    save_json(temporal_dir / "evaluation_metrics.json", temporal_metrics)
    save_json(random_dir / "confusion_matrix.json", {key: random_metrics[key] for key in ["train", "test"]})
    save_json(temporal_dir / "confusion_matrix.json", temporal_metrics["partitions"])
    summary.to_csv(temporal_dir / "partition_summary.csv", index=False)
    windows.to_csv(temporal_dir / "window_audit.csv", index=False)
    write_comparison_report(output_dir, random_metrics, temporal_metrics)

    for protocol in ("random", "temporal"):
        log(protocol, f"Input rows: {manifest['input_rows']}; usable rows: {manifest['usable_rows']}")
        log(protocol, f"Features: {len(feature_columns)}; suspicious timestamps: {manifest['suspicious_timestamp_rows']} (not excluded)")
        log(protocol, f"Model configuration: {json.dumps(MODEL_CONFIG, sort_keys=True)}")
    log("random", f"Train rows: {len(train_indices)}; test rows: {len(test_indices)}; seed: 42")
    log("random", f"Training seconds: {random_training_seconds}; memory delta bytes: {random_memory_delta}")
    log("random", "Integrity checks: feature schema, finite values, disjoint split, reload: passed")
    log("temporal", f"Windows: {len(windows)}; existing Version 1.1 audit matched: {matched_existing}")
    log("temporal", f"Allocation: {json.dumps(temporal_allocation, sort_keys=True)}")
    log("temporal", f"Training seconds: {temporal_training_seconds}; memory delta bytes: {temporal_memory_delta}")
    log("temporal", "Integrity checks: temporal order, indivisible windows, complete coverage, feature schema, reload: passed")
    (random_dir / "execution.log").write_text("\n".join(logs["random"]) + "\n", encoding="utf-8")
    (temporal_dir / "execution.log").write_text("\n".join(logs["temporal"]) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=BASE_DIR / "data" / "cic_preprocessed.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "artifacts" / "v2_random_forest")
    parser.add_argument("--suspicious-before", default="2000-01-01")
    args = parser.parse_args()
    run(args.input, args.output_dir, pd.Timestamp(args.suspicious_before))


if __name__ == "__main__":
    main()