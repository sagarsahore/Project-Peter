"""Automated Baseline & Classical Benchmark Verification Suite.

Project P.E.T.E.R. (Perceptual Evaluation & Transparent Emotion Research)
Affective Computing & Biosignals Pipeline

Verifies:
1. Classical model outperformance over Dummy floor baseline (Macro F1 delta > 0.10).
2. Strict zero-leakage speaker isolation across all 5 cross-validation folds.
3. Integrity and completeness of benchmark results CSV artifacts.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Set

import pandas as pd

from src.models.train_classical import (
    compute_demographic_metrics,
    instantiate_classifier,
)


class TestClassicalBaselines(unittest.TestCase):
    """Test suite verifying classical model performance, zero leakage, and artifacts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.results_csv_path = Path("Data/processed/classical_benchmark_results.csv")
        cls.features_parquet_path = Path("Data/processed/features_egemaps.parquet")

    def test_demographic_metrics_calculation(self) -> None:
        """Verifies synthetic accuracy, macro F1, and disparity ratio calculations."""
        import numpy as np

        y_true = np.array(["happy", "sad", "happy", "sad"])
        y_pred = np.array(["happy", "sad", "sad", "happy"])
        genders = np.array(["female", "female", "male", "male"])

        metrics = compute_demographic_metrics(y_true, y_pred, genders)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["f1_female"], 1.0)
        self.assertAlmostEqual(metrics["f1_male"], 0.0)
        self.assertEqual(metrics["disparity_ratio"], 0.0)

    def test_fold_actor_isolation(self) -> None:
        """Validates that no speaker in any train fold ever exists in the validation fold."""
        if not self.features_parquet_path.is_file():
            self.skipTest(f"Features file not found: {self.features_parquet_path}")

        df = pd.read_parquet(self.features_parquet_path)
        self.assertIn("fold", df.columns)
        self.assertIn("speaker_id", df.columns)

        unique_folds = sorted(df["fold"].unique())
        self.assertEqual(len(unique_folds), 5)

        for fold_id in unique_folds:
            train_speakers: Set[str] = set(df[df["fold"] != fold_id]["speaker_id"])
            val_speakers: Set[str] = set(df[df["fold"] == fold_id]["speaker_id"])

            overlap = train_speakers.intersection(val_speakers)
            self.assertEqual(
                len(overlap),
                0,
                f"Speaker leakage detected in fold {fold_id}! Overlap: {overlap}",
            )

    def test_results_artifact_integrity(self) -> None:
        """Checks that classical_benchmark_results.csv exists, has no nulls, and has 5 folds."""
        if not self.results_csv_path.is_file():
            self.skipTest(f"Results artifact not generated yet: {self.results_csv_path}")

        df_results = pd.read_csv(self.results_csv_path)

        # 1. Zero nulls check
        null_count = df_results.isna().sum().sum()
        self.assertEqual(
            null_count,
            0,
            f"Benchmark results CSV contains {null_count} null values.",
        )

        # 2. Check fold representation
        models = df_results["model_name"].unique().tolist()
        self.assertIn("DummyClassifier", models)
        self.assertGreaterEqual(len(models), 2)

        for model_name in models:
            sub = df_results[df_results["model_name"] == model_name]
            fold_values = set(sub["fold"].astype(str).tolist())
            expected_folds = {"0", "1", "2", "3", "4", "mean", "std"}
            self.assertTrue(
                expected_folds.issubset(fold_values),
                f"Model {model_name} missing expected fold entries: {expected_folds - fold_values}",
            )

    def test_baseline_outperformance(self) -> None:
        """Asserts classical model 5-fold mean Macro F1 is > 0.10 higher than Dummy floor."""
        if not self.results_csv_path.is_file():
            self.skipTest(f"Results artifact not generated yet: {self.results_csv_path}")

        df_results = pd.read_csv(self.results_csv_path)
        dummy_row = df_results[
            (df_results["model_name"] == "DummyClassifier") & (df_results["fold"] == "mean")
        ]
        self.assertFalse(dummy_row.empty, "DummyClassifier mean summary row not found.")
        dummy_f1 = float(dummy_row["macro_f1"].iloc[0])

        clf_models = [m for m in df_results["model_name"].unique() if m != "DummyClassifier"]
        self.assertTrue(clf_models, "No classical model found in results CSV.")
        primary_model = clf_models[0]

        clf_row = df_results[
            (df_results["model_name"] == primary_model) & (df_results["fold"] == "mean")
        ]
        self.assertFalse(clf_row.empty, f"{primary_model} mean summary row not found.")
        clf_f1 = float(clf_row["macro_f1"].iloc[0])

        delta_f1 = clf_f1 - dummy_f1
        self.assertGreater(
            delta_f1,
            0.10,
            f"{primary_model} Macro F1 ({clf_f1:.4f}) failed to exceed Dummy floor ({dummy_f1:.4f}) "
            f"by > 0.10 (Delta: {delta_f1:.4f})",
        )


if __name__ == "__main__":
    unittest.main()
