"""Flask frontend for manually testing the trained attack detector."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "logistic_attack_model.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"

app = Flask(__name__)
bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}


@app.route("/", methods=["GET", "POST"])
def index():
    if bundle is None:
        return render_template(
            "index.html", error="Train the model first: run train_model.py.", feature_names=[], metrics=metrics
        )

    feature_names = bundle["feature_names"]
    result = None
    error = None
    values = {feature: "0" for feature in feature_names}

    if request.method == "POST":
        values.update({feature: request.form.get(feature, "") for feature in feature_names})
        try:
            row = pd.DataFrame([[float(values[feature]) for feature in feature_names]], columns=feature_names)
            prediction = int(bundle["model"].predict(row)[0])
            probability = float(bundle["model"].predict_proba(row)[0, 1])
            result = {
                "label": "Attack" if prediction else "Benign",
                "probability": f"{probability:.2%}",
            }
        except (TypeError, ValueError):
            error = "Every feature must contain a valid numeric value."

    return render_template(
        "index.html", feature_names=feature_names, values=values, result=result, error=error, metrics=metrics
    )


@app.get("/random")
def random_values():
    """Generate synthetic values using the fitted feature distributions."""
    if bundle is None:
        return jsonify({"error": "Train the model first."}), 503

    feature_names = bundle["feature_names"]
    scaler = bundle["model"].named_steps["standardscaler"]
    generator = np.random.default_rng()
    values = generator.normal(scaler.mean_, scaler.scale_)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    return jsonify({name: round(float(value), 6) for name, value in zip(feature_names, values)})


if __name__ == "__main__":
    app.run(debug=True)