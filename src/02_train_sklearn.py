import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

import joblib


def macro_f1(res):
    return res["report"]["macro avg"]["f1-score"]


def main():
    DATASET_PATH = Path("../data/processed/dataset.csv")
    df = pd.read_csv(DATASET_PATH)

    # Asegurar tipos
    df["journal_id"] = df["journal_id"].astype(int)
    df["text"] = df["text"].fillna("").astype(str)

    X = df["text"].values
    y = df["journal_id"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    models = {
        "logreg": LogisticRegression(max_iter=2000, n_jobs=None),
        "linear_svm": LinearSVC(),
        "mnb": MultinomialNB(),
    }

    pipelines = {
        name: Pipeline([("tfidf", tfidf), ("clf", clf)])
        for name, clf in models.items()
    }

    results = {}

    for name, pipe in pipelines.items():
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "pipeline": pipe,
            "report": rep,
            "cm": cm,
        }

        print("\n===", name, "===")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion matrix:\n", cm)

    best_name = max(results.keys(), key=lambda n: macro_f1(results[n]))
    best = results[best_name]["pipeline"]

    print("Best model:", best_name, "macro F1:", macro_f1(results[best_name]))

    MODELS_DIR = Path("../models/sklearn")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best, MODELS_DIR / f"best_model_{best_name}.joblib")
    print("Saved:", MODELS_DIR / f"best_model_{best_name}.joblib")

    OUT_DIR = Path("../reports/sklearn")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Guardar métricas (JSON)
    metrics = {
        "best_model": best_name,
        "all_models_macro_f1": {k: macro_f1(v) for k, v in results.items()},
    }
    with open(OUT_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Errores del mejor modelo
    y_pred_best = best.predict(X_test)

    errors = df.iloc[idx_test].copy()
    errors["y_true"] = y_test
    errors["y_pred"] = y_pred_best
    errors = errors[errors["y_true"] != errors["y_pred"]]

    errors.to_csv(OUT_DIR / "errors_best_model.csv", index=False)
    print("Errors saved:", OUT_DIR / "errors_best_model.csv")


if __name__ == "__main__":
    main()
