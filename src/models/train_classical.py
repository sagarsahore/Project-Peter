"""Classical Bioacoustic Model Benchmarking & Demographic Fairness Evaluation.

Project P.E.T.E.R. (Perceptual Evaluation & Transparent Emotion Research)
Affective Computing & Biosignals Pipeline

This module executes zero-leakage 5-fold cross-validation on the 88 eGeMAPSv02
acoustic summary functionals:
- Establishes a non-negotiable floor baseline via DummyClassifier.
- Trains a gradient-boosted decision tree (LGBMClassifier) with balanced class weights.
- Enforces strict zero distribution leakage by fitting Scalers only on training partitions.
- Evaluates overall performance (Accuracy, Macro-F1, Macro-Recall) and demographic
  fairness across gender slices (F1_female, F1_male, Demographic Disparity Ratio).
- Exports fold-level metrics and aggregate summaries to CSV.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

# Attempt to import LightGBM; fallback gracefully to LogisticRegression
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    from sklearn.linear_model import LogisticRegression
    HAS_LIGHTGBM = False

# Configure structured logging
logger = logging.getLogger("peter.models.classical")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Ground-truth constants
EXPECTED_FOLDS = 5
KNOWN_METADATA_COLUMNS = {
    "file_path",
    "file_name",
    "filename",
    "file_size_bytes",
    "modality",
    "vocal_channel",
    "emotion_id",
    "emotion_code",
    "emotion_name",
    "intensity_id",
    "intensity_level",
    "statement_id",
    "statement_text",
    "repetition_id",
    "actor_id",
    "speaker_id",
    "gender",
    "fold",
}


def compute_demographic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    genders: np.ndarray,
) -> Dict[str, float]:
    """Computes overall and gender-sliced performance and fairness metrics.

    Args:
        y_true: Ground-truth target labels.
        y_pred: Model predicted labels.
        genders: Corresponding gender array ('male' or 'female').

    Returns:
        Dictionary containing overall accuracy, macro F1, macro recall,
        female F1, male F1, and demographic disparity ratio.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Gender-sliced evaluations
    female_mask = genders == "female"
    male_mask = genders == "male"

    if np.any(female_mask):
        f1_female = float(
            f1_score(
                y_true[female_mask],
                y_pred[female_mask],
                average="macro",
                zero_division=0,
            )
        )
    else:
        f1_female = 0.0

    if np.any(male_mask):
        f1_male = float(
            f1_score(
                y_true[male_mask],
                y_pred[male_mask],
                average="macro",
                zero_division=0,
            )
        )
    else:
        f1_male = 0.0

    # Demographic Disparity Ratio: min(F1_f, F1_m) / max(F1_f, F1_m)
    max_f1 = max(f1_female, f1_male)
    min_f1 = min(f1_female, f1_male)
    if max_f1 > 0.0:
        disparity_ratio = float(min_f1 / max_f1)
    else:
        disparity_ratio = 1.0 if min_f1 == 0.0 else 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_rec,
        "f1_female": f1_female,
        "f1_male": f1_male,
        "disparity_ratio": disparity_ratio,
    }


def instantiate_classifier(random_state: int = 42) -> Tuple[Any, str]:
    """Instantiates the primary classical model or fallback classifier.

    Args:
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (classifier instance, model name string).
    """
    if HAS_LIGHTGBM:
        clf = LGBMClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight="balanced",
            verbose=-1,
        )
        model_name = "LGBMClassifier"
    else:
        logger.warning("LightGBM not installed; falling back to LogisticRegression.")
        clf = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        )
        model_name = "LogisticRegression"
    return clf, model_name


def run_classical_benchmark(
    features_path: Union[str, Path],
    output_results_path: Union[str, Path],
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Executes 5-fold cross-validation comparing Dummy baseline and Classical Model.

    Args:
        features_path: Path to features_egemaps.parquet.
        output_results_path: Path to save benchmark metrics CSV.
        random_state: Random seed for reproducible training.

    Returns:
        Tuple of (results summary DataFrame, fold predictions DataFrame).

    Raises:
        FileNotFoundError: If features_path does not exist.
        ValueError: If fold column or feature columns are missing.
    """
    features_path = Path(features_path).resolve()
    if not features_path.is_file():
        raise FileNotFoundError(f"Features file does not exist: {features_path}")

    logger.info("Ingesting features from: %s", features_path)
    df = pd.read_parquet(features_path)
    logger.info("Loaded %d rows and %d columns.", len(df), len(df.columns))

    if "fold" not in df.columns:
        raise ValueError("Features DataFrame is missing required 'fold' column.")

    target_col = "emotion_name"
    if target_col not in df.columns:
        if "emotion_code" in df.columns:
            target_col = "emotion_code"
        else:
            raise ValueError("Target emotion column not found in features dataset.")

    # Identify acoustic feature columns
    feature_cols = [c for c in df.columns if c not in KNOWN_METADATA_COLUMNS]
    logger.info("Discovered %d acoustic feature columns for modeling.", len(feature_cols))

    unique_folds = sorted(df["fold"].unique())
    if len(unique_folds) != EXPECTED_FOLDS:
        logger.warning("Discovered %d folds; expected %d folds.", len(unique_folds), EXPECTED_FOLDS)

    fold_metric_records: List[Dict[str, Any]] = []
    prediction_records: List[pd.DataFrame] = []

    for fold_id in unique_folds:
        train_mask = df["fold"] != fold_id
        val_mask = df["fold"] == fold_id

        train_df = df[train_mask]
        val_df = df[val_mask]

        # Zero-leakage speaker isolation check
        train_speakers = set(train_df["speaker_id"])
        val_speakers = set(val_df["speaker_id"])
        leakage = train_speakers.intersection(val_speakers)
        if leakage:
            logger.critical("Speaker leakage detected in fold %s! Overlapping speakers: %s", fold_id, leakage)
            raise RuntimeError(f"Speaker leakage in fold {fold_id}: {leakage}")

        # Strict zero distribution leakage: Fit StandardScaler only on train partition
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].values)
        X_val = scaler.transform(val_df[feature_cols].values)

        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        genders_val = val_df["gender"].values

        # 1. Floor Baseline Model
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_val)

        dummy_metrics = compute_demographic_metrics(y_val, y_pred_dummy, genders_val)
        fold_metric_records.append({
            "fold": fold_id,
            "model_name": "DummyClassifier",
            "n_train": len(train_df),
            "n_val": len(val_df),
            **dummy_metrics,
        })

        # 2. Classical Benchmark Model
        clf, model_name = instantiate_classifier(random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred_clf = clf.predict(X_val)

        clf_metrics = compute_demographic_metrics(y_val, y_pred_clf, genders_val)
        fold_metric_records.append({
            "fold": fold_id,
            "model_name": model_name,
            "n_train": len(train_df),
            "n_val": len(val_df),
            **clf_metrics,
        })

        # Record validation predictions for transparency
        fold_preds = val_df[["filename", "speaker_id", "gender", "fold", target_col]].copy()
        fold_preds["y_true"] = y_val
        fold_preds["pred_dummy"] = y_pred_dummy
        fold_preds["pred_classical"] = y_pred_clf
        prediction_records.append(fold_preds)

    df_results = pd.DataFrame(fold_metric_records)
    df_preds = pd.concat(prediction_records, ignore_index=True)

    # Compute cross-validation aggregates (Mean & Std Dev)
    summary_rows: List[Dict[str, Any]] = []
    for m_name in df_results["model_name"].unique():
        sub = df_results[df_results["model_name"] == m_name]
        mean_row: Dict[str, Any] = {
            "fold": "mean",
            "model_name": m_name,
            "n_train": int(sub["n_train"].mean()),
            "n_val": int(sub["n_val"].mean()),
            "accuracy": float(sub["accuracy"].mean()),
            "macro_f1": float(sub["macro_f1"].mean()),
            "macro_recall": float(sub["macro_recall"].mean()),
            "f1_female": float(sub["f1_female"].mean()),
            "f1_male": float(sub["f1_male"].mean()),
            "disparity_ratio": float(sub["disparity_ratio"].mean()),
        }
        std_row: Dict[str, Any] = {
            "fold": "std",
            "model_name": m_name,
            "n_train": 0,
            "n_val": 0,
            "accuracy": float(sub["accuracy"].std()),
            "macro_f1": float(sub["macro_f1"].std()),
            "macro_recall": float(sub["macro_recall"].std()),
            "f1_female": float(sub["f1_female"].std()),
            "f1_male": float(sub["f1_male"].std()),
            "disparity_ratio": float(sub["disparity_ratio"].std()),
        }
        summary_rows.extend([mean_row, std_row])

    df_summary = pd.concat([df_results, pd.DataFrame(summary_rows)], ignore_index=True)

    # Save results to CSV
    output_path = Path(output_results_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(output_path, index=False)
    logger.info("Successfully exported benchmark results to: %s", output_path)

    # Save detailed prediction manifest
    preds_output_path = output_path.parent / "classical_predictions.csv"
    df_preds.to_csv(preds_output_path, index=False)
    logger.info("Successfully exported predictions to: %s", preds_output_path)

    # Print Formatted Terminal Report
    print_benchmark_report(df_summary)

    return df_summary, df_preds


def print_benchmark_report(df_summary: pd.DataFrame) -> None:
    """Prints a formatted ASCII evaluation and demographic fairness report.

    Args:
        df_summary: Combined fold and summary metrics DataFrame.
    """
    divider = "=" * 88
    subdivider = "-" * 88

    print("\n" + divider)
    print("       PROJECT P.E.T.E.R. -- CLASSICAL BIOACOUSTIC BENCHMARK REPORT       ")
    print(divider)

    models = [m for m in df_summary["model_name"].unique()]
    for model_name in models:
        print(f"\n>> Model: {model_name}")
        print(f"   {'Fold':<6} | {'Accuracy':<9} | {'Macro F1':<9} | {'Recall':<9} | "
              f"{'F1(Female)':<11} | {'F1(Male)':<9} | {'Disparity':<10}")
        print(f"   {'-'*6}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*11}-+-{'-'*9}-+-{'-'*10}")

        model_rows = df_summary[df_summary["model_name"] == model_name]
        for _, row in model_rows.iterrows():
            f_str = str(row["fold"])
            acc = f"{row['accuracy']*100:6.2f}%"
            f1 = f"{row['macro_f1']*100:6.2f}%"
            rec = f"{row['macro_recall']*100:6.2f}%"
            f1_f = f"{row['f1_female']*100:6.2f}%"
            f1_m = f"{row['f1_male']*100:6.2f}%"
            disp = f"{row['disparity_ratio']:6.3f}"
            if f_str in ("mean", "std"):
                print(f"   {f_str.upper():<6} | {acc:<9} | {f1:<9} | {rec:<9} | {f1_f:<11} | {f1_m:<9} | {disp:<10}")
            else:
                print(f"   Fold {f_str:<1} | {acc:<9} | {f1:<9} | {rec:<9} | {f1_f:<11} | {f1_m:<9} | {disp:<10}")

    print("\n" + subdivider)
    print(">> Baseline Outperformance & Parity Audit:")
    dummy_mean = df_summary[(df_summary["model_name"] == "DummyClassifier") & (df_summary["fold"] == "mean")].iloc[0]
    clf_models = [m for m in models if m != "DummyClassifier"]
    if clf_models:
        clf_mean = df_summary[(df_summary["model_name"] == clf_models[0]) & (df_summary["fold"] == "mean")].iloc[0]
        delta_f1 = clf_mean["macro_f1"] - dummy_mean["macro_f1"]
        delta_acc = clf_mean["accuracy"] - dummy_mean["accuracy"]
        print(f"   * Classical vs. Dummy Floor F1 Delta:    {delta_f1*100:+.2f}% "
              f"({'PASS (>10%)' if delta_f1 > 0.10 else 'FAIL'})")
        print(f"   * Classical vs. Dummy Accuracy Delta:    {delta_acc*100:+.2f}%")
        print(f"   * Mean Demographic Disparity Ratio:      {clf_mean['disparity_ratio']:.3f} "
              f"(Female F1: {clf_mean['f1_female']*100:.2f}%, Male F1: {clf_mean['f1_male']*100:.2f}%)")
    print(divider + "\n")


def main() -> None:
    """CLI entrypoint for running classical model benchmarks."""
    parser = argparse.ArgumentParser(
        description="Classical eGeMAPS Emotion Classification Benchmark & Demographic Audit"
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="Data/processed/features_egemaps.parquet",
        help="Path to eGeMAPSv02 Parquet dataset (default: Data/processed/features_egemaps.parquet)",
    )
    parser.add_argument(
        "--output-results",
        type=str,
        default="Data/processed/classical_benchmark_results.csv",
        help="Target path for benchmark results CSV (default: Data/processed/classical_benchmark_results.csv)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for model training reproducibility (default: 42)",
    )
    args = parser.parse_args()

    logger.info("Initiating classical bioacoustic benchmark pipeline...")
    run_classical_benchmark(
        features_path=args.features_path,
        output_results_path=args.output_results,
        random_state=args.random_state,
    )
    logger.info("Classical benchmark pipeline executed successfully.")


if __name__ == "__main__":
    main()
