"""Small web interface for the Version 2 intrusion-detection model."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "v2_random_forest" / "random_forest_model.joblib"
DATA_PATH = BASE_DIR / "data" / "cic_preprocessed.csv"
FEATURES = [
    column
    for column in pd.read_csv(DATA_PATH, nrows=0).columns
    if column not in {"Timestamp", "Label", "Flow ID", "Src IP", "Dst IP"}
]

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


def _load_profiles() -> dict[str, dict[str, float]]:
    """Build small benign/attack examples without keeping the full CSV in memory."""
    samples: dict[str, dict[str, float]] = {}
    for chunk in pd.read_csv(DATA_PATH, usecols=FEATURES + ["Label"], chunksize=50_000):
        for label, group in chunk.groupby("Label"):
            key = "benign" if str(label).strip() == "Benign" else "attack"
            if key not in samples and not group.empty:
                row = group.iloc[0]
                samples[key] = {
                    feature: float(row[feature])
                    for feature in FEATURES
                    if np.isfinite(float(row[feature]))
                }
        if len(samples) == 2:
            break
    if not samples:
        raise RuntimeError(f"No usable rows found in {DATA_PATH}")
    return samples


PROFILES = _load_profiles()
PROFILE_MEDIANS = {
    feature: float(PROFILES["benign"].get(feature, 0.0)) for feature in FEATURES
}

# Values transcribed from PROJECT_REPORT.md, section 5.
MODEL_ACCURACIES = [
    {"version": "Version 1.0", "model": "Logistic Regression", "protocol": "Random 80/20", "partition": "Train", "records": "835,800", "accuracy": "99.9664%"},
    {"version": "Version 1.0", "model": "Logistic Regression", "protocol": "Random 80/20", "partition": "Test", "records": "208,951", "accuracy": "99.9646%"},
    {"version": "Version 1.1", "model": "Logistic Regression", "protocol": "Temporal windows", "partition": "Train", "records": "603,385", "accuracy": "99.9576%"},
    {"version": "Version 1.1", "model": "Logistic Regression", "protocol": "Temporal windows", "partition": "Validation", "records": "129,091", "accuracy": "99.9667%"},
    {"version": "Version 1.1", "model": "Logistic Regression", "protocol": "Temporal windows", "partition": "Test", "records": "86,650", "accuracy": "99.9561%"},
    {"version": "Version 2", "model": "Random Forest", "protocol": "Random 80/20", "partition": "Train", "records": "655,300", "accuracy": "100.0000%"},
    {"version": "Version 2", "model": "Random Forest", "protocol": "Random 80/20", "partition": "Test", "records": "163,826", "accuracy": "100.0000%"},
    {"version": "Version 2", "model": "Random Forest", "protocol": "Temporal windows", "partition": "Train", "records": "603,385", "accuracy": "100.0000%"},
    {"version": "Version 2", "model": "Random Forest", "protocol": "Temporal windows", "partition": "Validation", "records": "129,091", "accuracy": "100.0000%"},
    {"version": "Version 2", "model": "Random Forest", "protocol": "Temporal windows", "partition": "Test", "records": "86,650", "accuracy": "100.0000%"},
]


def _number(value: float) -> float | int:
    if not math.isfinite(value):
        return 0.0
    return int(round(value)) if abs(value - round(value)) < 1e-9 else round(value, 6)


def _random_row(profile: str = "random") -> dict[str, float | int]:
    selected = profile
    if selected == "random":
        selected = random.choice(tuple(PROFILES))
    source = PROFILES[selected]
    values: dict[str, float | int] = {}
    jitter = 0.05 if selected in {"benign", "attack"} else 0.35
    for feature in FEATURES:
        base = source.get(feature, PROFILE_MEDIANS.get(feature, 0.0))
        if base == 0:
            value = 0.0
        else:
            # Log jitter keeps rates and durations positive while remaining realistic.
            value = base * math.exp(random.uniform(-jitter, jitter))
        values[feature] = _number(value)

    # These fields are especially useful for making manual attack/benign tests clear.
    if selected == "benign":
        values["Dst Port"] = random.choice([53, 80, 443])
        values["Protocol"] = 6
    else:
        values["Dst Port"] = random.choice([21, 22, 23, 3389])
        values["Protocol"] = 6
    return values


def _predict(values: dict[str, Any]) -> dict[str, Any]:
    missing = [feature for feature in FEATURES if feature not in values]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing[:5])}")
    try:
        row = [float(values[feature]) for feature in FEATURES]
    except (TypeError, ValueError) as error:
        raise ValueError("Every feature must contain a numeric value.") from error
    if not np.isfinite(row).all():
        raise ValueError("Feature values must be finite numbers.")

    frame = pd.DataFrame([row], columns=FEATURES)
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = "attack" if probability >= 0.5 else "benign"

    # A one-feature counterfactual gives a useful local explanation for a tree model.
    impacts: list[dict[str, Any]] = []
    for feature in FEATURES:
        counterfactual = frame.copy()
        counterfactual.loc[0, feature] = PROFILE_MEDIANS[feature]
        reference_probability = float(model.predict_proba(counterfactual)[0, 1])
        impacts.append(
            {
                "feature": feature,
                "value": _number(float(values[feature])),
                "importance": round(float(getattr(model, "feature_importances_", [0] * len(FEATURES))[FEATURES.index(feature)]), 5),
                "impact": round(probability - reference_probability, 5),
            }
        )
    impacts.sort(key=lambda item: abs(item["impact"]), reverse=True)
    reasons = []
    for item in impacts[:5]:
        direction = "toward attack" if item["impact"] > 0 else "toward benign traffic"
        reasons.append(
            f"{item['feature']}={item['value']} pushed the score {direction} "
            f"(local effect {item['impact']:+.3f})."
        )
    return {
        "prediction": prediction,
        "attack_probability": round(probability, 5),
        "confidence": round(max(probability, 1 - probability), 5),
        "reasons": reasons,
        "top_features": impacts[:8],
        "values": {feature: _number(float(values[feature])) for feature in FEATURES},
    }


@app.get("/")
def index():
    return render_template(
        "index.html", features=FEATURES, model_accuracies=MODEL_ACCURACIES
    )


@app.get("/api/random")
def random_prediction():
    profile = request.args.get("profile", "random")
    if profile not in {"random", "benign", "attack"}:
        return jsonify({"error": "profile must be random, benign, or attack"}), 400
    return jsonify(_predict(_random_row(profile)))


@app.post("/api/predict")
def manual_prediction():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(_predict(payload))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
