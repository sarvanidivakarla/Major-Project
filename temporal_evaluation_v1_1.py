"""Run Version 1.1 chronological temporal-window evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABEL_COLUMN = "Label"
TIMESTAMP_COLUMN = "Timestamp"
BENIGN_LABEL = "Benign"
EXCLUDED_FEATURE_COLUMNS = {
    LABEL_COLUMN,
    TIMESTAMP_COLUMN,
    "_Parsed Timestamp",
    "Flow ID",
    "Src IP",
    "Dst IP",
}
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"
WINDOW_DURATION = pd.Timedelta(minutes=30)
PARTITIONS = ("train", "validation", "test")


def load_and_validate_data(input_path: Path, suspicious_before: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load data, parse timestamps explicitly, and record invalid/suspicious rows."""
    data = pd.read_csv(input_path, low_memory=False)
    required = {LABEL_COLUMN, TIMESTAMP_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    parsed_timestamps = pd.to_datetime(
        data[TIMESTAMP_COLUMN], format=TIMESTAMP_FORMAT, errors="coerce"
    )
    invalid_timestamp_mask = parsed_timestamps.isna()
    suspicious_mask = parsed_timestamps.notna() & (parsed_timestamps < suspicious_before)
    data["_Parsed Timestamp"] = parsed_timestamps

    audit = {
        "input_rows": int(len(data)),
        "invalid_or_missing_timestamp_rows": int(invalid_timestamp_mask.sum()),
        "invalid_or_missing_timestamp_percent": float(invalid_timestamp_mask.mean() * 100),
        "suspicious_timestamp_rows": int(suspicious_mask.sum()),
        "suspicious_timestamp_percent": float(suspicious_mask.mean() * 100),
        "timestamp_format": TIMESTAMP_FORMAT,
        "timezone_assumption": "Naive timestamps; no timezone conversion applied",
        "suspicious_timestamps_excluded": False,
    }
    return data, audit


def preprocess_features_and_labels(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Prepare finite numeric features without using Timestamp or Label as features."""
    feature_columns = [column for column in data.columns if column not in EXCLUDED_FEATURE_COLUMNS]
    numeric_features = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    finite_feature_mask = numeric_features.notna().all(axis=1) & ~numeric_features.isin(
        [np.inf, -np.inf]
    ).any(axis=1)

    labels = data[LABEL_COLUMN].astype("string").str.strip()
    valid_label_mask = labels.notna() & labels.ne("")
    valid_timestamp_mask = data["_Parsed Timestamp"].notna()
    usable_mask = valid_timestamp_mask & valid_label_mask & finite_feature_mask

    usable = data.loc[usable_mask, [TIMESTAMP_COLUMN, LABEL_COLUMN]].copy()
    usable["_Parsed Timestamp"] = data.loc[usable_mask, "_Parsed Timestamp"]
    usable[feature_columns] = numeric_features.loc[usable_mask].astype(float)
    usable["_Target"] = (labels.loc[usable_mask] != BENIGN_LABEL).astype("int8")

    preprocessing_audit = {
        "invalid_feature_rows": int((~finite_feature_mask).sum()),
        "invalid_or_missing_target_rows": int((~valid_label_mask).sum()),
        "rows_removed_during_preprocessing": int(len(data) - len(usable)),
        "final_usable_rows": int(len(usable)),
    }
    return usable, feature_columns, preprocessing_audit


def assign_temporal_windows(usable: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign every usable record to one aligned, half-open 30-minute window."""
    if usable.empty:
        raise ValueError("No usable records remain after preprocessing.")

    usable = usable.copy()
    usable["Window Start"] = usable["_Parsed Timestamp"].dt.floor("30min")
    usable["Window End"] = usable["Window Start"] + WINDOW_DURATION
    grouped = usable.groupby("Window Start", sort=True)
    windows = grouped.agg(
        record_count=("_Target", "size"),
        benign_count=("_Target", lambda values: int((values == 0).sum())),
        attack_count=("_Target", lambda values: int((values == 1).sum())),
        first_timestamp=("_Parsed Timestamp", "min"),
        last_timestamp=("_Parsed Timestamp", "max"),
    ).reset_index()
    windows["Window End"] = windows["Window Start"] + WINDOW_DURATION
    windows["window_id"] = [f"window-{index:06d}" for index in range(1, len(windows) + 1)]
    return usable, windows[
        [
            "window_id",
            "Window Start",
            "Window End",
            "record_count",
            "benign_count",
            "attack_count",
            "first_timestamp",
            "last_timestamp",
        ]
    ]


def create_chronological_partitions(
    usable: pd.DataFrame, windows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Choose complete-window boundaries closest to 70/15/15 record targets."""
    if len(windows) < 3:
        raise ValueError("At least three temporal windows are required for three partitions.")

    total_records = len(usable)
    targets = np.array([total_records * 0.70, total_records * 0.15, total_records * 0.15])
    counts = windows["record_count"].to_numpy(dtype=np.int64)
    prefix = np.concatenate(([0], np.cumsum(counts)))
    best = None
    for train_boundary in range(1, len(windows) - 1):
        for validation_boundary in range(train_boundary + 1, len(windows)):
            actual = np.array(
                [
                    prefix[train_boundary],
                    prefix[validation_boundary] - prefix[train_boundary],
                    prefix[-1] - prefix[validation_boundary],
                ],
                dtype=float,
            )
            objective = (float(np.abs(actual - targets).sum()), train_boundary, validation_boundary)
            if best is None or objective < best[0]:
                best = (objective, train_boundary, validation_boundary, actual)

    assert best is not None
    _, train_boundary, validation_boundary, actual = best
    window_partitions = np.full(len(windows), "test", dtype=object)
    window_partitions[:train_boundary] = "train"
    window_partitions[train_boundary:validation_boundary] = "validation"
    windows = windows.copy()
    windows["partition"] = window_partitions
    partition_by_window = dict(zip(windows["window_id"], windows["partition"]))
    window_id_by_start = dict(zip(windows["Window Start"], windows["window_id"]))
    usable = usable.copy()
    usable["window_id"] = usable["Window Start"].map(window_id_by_start)
    usable["partition"] = usable["window_id"].map(partition_by_window)

    if usable["partition"].isna().any():
        raise AssertionError("Some usable records were not assigned to a partition.")
    allocation = actual / total_records * 100
    partition_audit = {
        "target_percentages": {name: target / total_records * 100 for name, target in zip(PARTITIONS, targets)},
        "actual_record_counts": {name: int(count) for name, count in zip(PARTITIONS, actual)},
        "actual_percentages": {name: float(percent) for name, percent in zip(PARTITIONS, allocation)},
        "allocation_deviation_percentage_points": {
            name: float(percent - target / total_records * 100)
            for name, percent, target in zip(PARTITIONS, allocation, targets)
        },
        "train_boundary_window": windows.iloc[train_boundary - 1]["window_id"],
        "validation_boundary_window": windows.iloc[validation_boundary - 1]["window_id"],
    }
    return usable, windows, partition_audit


def audit_partition_class_counts(usable: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Create the requested record, time-range, and class-distribution summary."""
    rows = []
    for partition in PARTITIONS:
        records = usable[usable["partition"] == partition]
        partition_windows = windows[windows["partition"] == partition]
        if records.empty:
            raise ValueError(f"The {partition} partition is empty.")
        benign_count = int((records["_Target"] == 0).sum())
        attack_count = int((records["_Target"] == 1).sum())
        rows.append(
            {
                "partition": partition,
                "records": int(len(records)),
                "allocation_percent": float(len(records) / len(usable) * 100),
                "number_of_windows": int(len(partition_windows)),
                "benign_count": benign_count,
                "attack_count": attack_count,
                "benign_percent": float(benign_count / len(records) * 100),
                "attack_percent": float(attack_count / len(records) * 100),
                "first_timestamp": records["_Parsed Timestamp"].min().isoformat(sep=" "),
                "last_timestamp": records["_Parsed Timestamp"].max().isoformat(sep=" "),
                "first_window_start": partition_windows["Window Start"].min().isoformat(sep=" "),
                "last_window_end": partition_windows["Window End"].max().isoformat(sep=" "),
                "invalid_target_count": 0,
            }
        )
    return pd.DataFrame(rows)


def validate_partition_integrity(usable: pd.DataFrame, windows: pd.DataFrame, feature_columns: list[str]) -> None:
    """Raise clear errors if chronological or feature integrity is violated."""
    if set(usable["partition"].unique()) != set(PARTITIONS):
        raise AssertionError("The partition set is incomplete.")
    if len(usable) != sum((usable["partition"] == partition).sum() for partition in PARTITIONS):
        raise AssertionError("Partition counts do not cover all usable records.")
    if windows.groupby("window_id")["partition"].nunique().max() != 1:
        raise AssertionError("A temporal window appears in multiple partitions.")
    if usable["window_id"].nunique() != windows["window_id"].nunique():
        raise AssertionError("Some temporal windows were not assigned to records.")

    partition_records = {name: usable[usable["partition"] == name] for name in PARTITIONS}
    if not partition_records["train"]["_Parsed Timestamp"].max() < partition_records["validation"]["_Parsed Timestamp"].min():
        raise AssertionError("Training timestamps are not strictly before validation timestamps.")
    if not partition_records["validation"]["_Parsed Timestamp"].max() < partition_records["test"]["_Parsed Timestamp"].min():
        raise AssertionError("Validation timestamps are not strictly before test timestamps.")
    window_ranges = {
        name: windows[windows["partition"] == name] for name in PARTITIONS
    }
    if not window_ranges["train"]["Window End"].max() <= window_ranges["validation"]["Window Start"].min():
        raise AssertionError("Training and validation windows overlap.")
    if not window_ranges["validation"]["Window End"].max() <= window_ranges["test"]["Window Start"].min():
        raise AssertionError("Validation and test windows overlap.")

    features = usable[feature_columns]
    if list(features.columns) != feature_columns:
        raise AssertionError("Feature columns changed order.")
    if not all(pd.api.types.is_numeric_dtype(features[column]) for column in feature_columns):
        raise AssertionError("A model feature is not numeric.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise AssertionError("A model feature is not finite.")
    for partition in PARTITIONS:
        if list(usable[usable["partition"] == partition][feature_columns].columns) != feature_columns:
            raise AssertionError(f"Feature columns differ in the {partition} partition.")


def _safe_metric(function, actual: pd.Series, predicted: np.ndarray) -> float | None:
    if function in {precision_score, f1_score} and predicted.sum() == 0:
        return None
    if function in {recall_score} and actual.sum() == 0:
        return None
    return float(function(actual, predicted, zero_division=0))


def evaluate_partition(model, features: pd.DataFrame, labels: pd.Series) -> dict[str, object]:
    """Evaluate one partition with a fixed probability threshold of 0.5."""
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
        "precision": _safe_metric(precision_score, labels, predictions),
        "recall": _safe_metric(recall_score, labels, predictions),
        "f1_score": _safe_metric(f1_score, labels, predictions),
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


def save_audit_artifacts(
    output_dir: Path,
    summary: pd.DataFrame,
    windows: pd.DataFrame,
    metrics: dict[str, object],
    execution_log: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "partition_summary.csv", index=False)
    windows.rename(
        columns={"Window Start": "window_start", "Window End": "window_end"}
    ).to_csv(output_dir / "window_audit.csv", index=False)
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "execution.log").write_text("\n".join(execution_log) + "\n", encoding="utf-8")


def run_pipeline(input_path: Path, output_dir: Path, suspicious_before: pd.Timestamp) -> dict[str, object]:
    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    data, timestamp_audit = load_and_validate_data(input_path, suspicious_before)
    usable, feature_columns, preprocessing_audit = preprocess_features_and_labels(data)
    usable, windows = assign_temporal_windows(usable)
    usable, windows, partition_audit = create_chronological_partitions(usable, windows)
    validate_partition_integrity(usable, windows, feature_columns)
    summary = audit_partition_class_counts(usable, windows)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=200, class_weight="balanced", n_jobs=-1),
    )
    train_features = usable.loc[usable["partition"] == "train", feature_columns]
    train_labels = usable.loc[usable["partition"] == "train", "_Target"]
    if train_labels.nunique() < 2:
        raise ValueError("Training partition contains only one target class.")
    training_started = time.perf_counter()
    model.fit(train_features, train_labels)
    training_seconds = time.perf_counter() - training_started

    metrics = {
        "version": "1.1",
        "input_path": str(input_path),
        "timestamp_format": TIMESTAMP_FORMAT,
        "window_duration_minutes": 30,
        "window_alignment": "Timestamp floored to the half-open interval [window_start, window_start + 30 minutes)",
        "label_mapping": {"Benign": 0, "non-Benign": 1},
        "feature_columns": feature_columns,
        "timestamp_audit": timestamp_audit,
        "preprocessing_audit": preprocessing_audit,
        "partition_audit": partition_audit,
        "training_time_seconds": float(training_seconds),
        "partitions": {},
    }
    for partition in PARTITIONS:
        records = usable[usable["partition"] == partition]
        metrics["partitions"][partition] = evaluate_partition(
            model, records[feature_columns], records["_Target"]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": feature_columns, "version": "1.1"},
        output_dir / "logistic_attack_model_v1_1.joblib",
    )
    log(f"Input rows: {timestamp_audit['input_rows']}")
    log(f"Input path: {input_path}")
    log(f"Timestamp format: {TIMESTAMP_FORMAT}; window duration: 30 minutes; alignment: half-open floor")
    log(f"Excluded from model features: {sorted(EXCLUDED_FEATURE_COLUMNS)}")
    log(f"Invalid or missing timestamps: {timestamp_audit['invalid_or_missing_timestamp_rows']} ({timestamp_audit['invalid_or_missing_timestamp_percent']:.4f}%)")
    log(f"Suspicious timestamps before {suspicious_before.date()}: {timestamp_audit['suspicious_timestamp_rows']} (not excluded)")
    log(f"Rows removed during preprocessing: {preprocessing_audit['rows_removed_during_preprocessing']}")
    log(f"Final usable rows: {preprocessing_audit['final_usable_rows']}")
    log(f"Windows: {len(windows)}")
    log(f"Partition boundaries: train through {partition_audit['train_boundary_window']}; validation through {partition_audit['validation_boundary_window']}")
    for row in summary.itertuples(index=False):
        log(f"{row.partition}: {row.records} records, {row.number_of_windows} windows, {row.benign_count} benign, {row.attack_count} attack")
    for partition, partition_metrics in metrics["partitions"].items():
        log(
            f"{partition} metrics: accuracy={partition_metrics['accuracy']}, "
            f"precision={partition_metrics['precision']}, recall={partition_metrics['recall']}, "
            f"f1={partition_metrics['f1_score']}, roc_auc={partition_metrics['roc_auc']}, "
            f"pr_auc={partition_metrics['pr_auc']}"
        )
    log("Integrity assertions: passed")
    save_audit_artifacts(output_dir, summary, windows, metrics, log_lines)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        default=Path("data/cic_preprocessed.csv"),
        nargs="?",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/v1_1_temporal")
    )
    parser.add_argument(
        "--suspicious-before", type=str, default="2000-01-01",
        help="Flag, but do not exclude, timestamps before this date.",
    )
    args = parser.parse_args()
    suspicious_before = pd.Timestamp(args.suspicious_before)
    run_pipeline(args.input, args.output_dir, suspicious_before)


if __name__ == "__main__":
    main()