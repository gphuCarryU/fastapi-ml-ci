"""Train the breast-cancer classifier and save it to model/.

Usage:
    python train.py

Outputs:
    model/model.joblib     - trained scikit-learn Pipeline (scaler + classifier)
    model/metadata.json    - model version, feature names, evaluation metrics
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).parent / "model"
MODEL_VERSION = "1.0.0"
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def main() -> dict:
    data = load_breast_cancer()
    X, y = data.data, data.target
    # In this dataset: 0 = malignant, 1 = benign
    target_names = [str(name) for name in data.target_names]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std": round(float(cv_scores.std()), 4),
    }

    metadata = {
        "model_version": MODEL_VERSION,
        "algorithm": "StandardScaler + LogisticRegression",
        "dataset": "sklearn.datasets.load_breast_cancer (Wisconsin Diagnostic)",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_names": [str(name) for name in data.feature_names],
        "target_names": target_names,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Model saved to {MODEL_DIR}")
    for name, value in metrics.items():
        print(f"  {name:<11} {value}")
    return metadata


if __name__ == "__main__":
    main()

