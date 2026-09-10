"""Automated Deep Learning Spectrogram Verification Suite.

Project P.E.T.E.R. (Perceptual Evaluation & Transparent Emotion Research)
Affective Computing & Biosignals Pipeline

Verifies:
1. Synthetic forward pass contract for EmotionCNN2D ([2, 1, 80, 256] -> [2, 8]).
2. Temporal padding and cropping collation logic across variable timeframes.
3. Demographic fairness metric calculations and parity ratios.
4. Deep benchmark results and predictions artifact completeness and integrity.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.deep_dataset import pad_or_crop_temporal, pad_spectrogram_collate_fn
from src.models.train_deep import (
    EmotionCNN2D,
    compute_demographic_metrics,
)


class TestDeepPipeline(unittest.TestCase):
    """Test suite for deep learning spectrogram model and pipeline contracts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.results_csv_path = Path("Data/processed/deep_benchmark_results.csv")
        cls.predictions_csv_path = Path("Data/processed/deep_predictions.csv")

    def test_forward_pass_contract(self) -> None:
        """Asserts input [2, 1, 80, 256] passes through EmotionCNN2D and outputs [2, 8]."""
        model = EmotionCNN2D(num_classes=8, dropout_p=0.3)
        model.eval()

        # Input: batch of 2, 1 channel, 80 mel bands, 256 time steps
        synthetic_input = torch.randn(2, 1, 80, 256)
        with torch.no_grad():
            output = model(synthetic_input)

        self.assertEqual(output.shape, torch.Size([2, 8]))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_padding_collation_logic(self) -> None:
        """Verifies shorter and longer tensors conform to target temporal shape without distortion."""
        target_T = 94

        # 1. Shorter input (T=50) -> should be symmetrically padded to 94
        short_tensor = torch.ones(1, 80, 50)
        padded = pad_or_crop_temporal(short_tensor, target_frames=target_T)
        self.assertEqual(padded.shape, torch.Size([1, 80, target_T]))
        # Center should contain the original 1.0 values
        diff = target_T - 50  # 44 -> 22 left, 22 right
        self.assertEqual(padded[0, 0, 0].item(), 0.0)      # Left padded zone
        self.assertEqual(padded[0, 0, 21].item(), 0.0)     # Left padded boundary
        self.assertEqual(padded[0, 0, 22].item(), 1.0)     # Data starts
        self.assertEqual(padded[0, 0, 71].item(), 1.0)     # Data ends (22 + 50 - 1 = 71)
        self.assertEqual(padded[0, 0, 72].item(), 0.0)     # Right padded boundary
        self.assertEqual(padded[0, 0, -1].item(), 0.0)     # Right padded end

        # 2. Longer input (T=200) -> should be center-cropped to 94
        long_tensor = torch.arange(200, dtype=torch.float32).repeat(1, 80, 1)
        cropped = pad_or_crop_temporal(long_tensor, target_frames=target_T)
        self.assertEqual(cropped.shape, torch.Size([1, 80, target_T]))
        # Center crop offset: (200 - 94) // 2 = 53
        self.assertEqual(cropped[0, 0, 0].item(), 53.0)
        self.assertEqual(cropped[0, 0, -1].item(), 53.0 + target_T - 1)

        # 3. Collate function test
        mock_batch = [
            (torch.randn(1, 80, 70), 2, {"gender": "female"}),
            (torch.randn(1, 80, 120), 4, {"gender": "male"}),
        ]
        batch_tensors, batch_labels, metadatas = pad_spectrogram_collate_fn(
            mock_batch, target_frames=target_T
        )
        self.assertEqual(batch_tensors.shape, torch.Size([2, 1, 80, target_T]))
        self.assertEqual(batch_labels.shape, torch.Size([2]))
        self.assertEqual(batch_labels.tolist(), [2, 4])
        self.assertEqual(len(metadatas), 2)

    def test_demographic_metrics_parity(self) -> None:
        """Verifies synthetic demographic metric and disparity ratio calculation integrity."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 1, 1])  # females perfect (0-3), males have 2 errors (4-7)
        genders = np.array(["female", "female", "female", "female", "male", "male", "male", "male"])

        metrics = compute_demographic_metrics(y_true, y_pred, genders)
        self.assertGreater(metrics["f1_female"], metrics["f1_male"])
        self.assertAlmostEqual(metrics["f1_female"], 1.0)
        self.assertGreaterEqual(metrics["disparity_ratio"], 0.0)
        self.assertLessEqual(metrics["disparity_ratio"], 1.0)
        self.assertAlmostEqual(
            metrics["disparity_ratio"],
            metrics["f1_male"] / metrics["f1_female"],
        )

    def test_deep_benchmark_artifact_integrity(self) -> None:
        """Checks that deep_benchmark_results.csv and deep_predictions.csv exist and have zero nulls."""
        if not self.results_csv_path.is_file():
            self.skipTest(f"Results file not found: {self.results_csv_path}")

        df_results = pd.read_csv(self.results_csv_path)

        # Check zero nulls
        null_count = df_results.isna().sum().sum()
        self.assertEqual(null_count, 0, f"Results CSV contains {null_count} nulls.")

        # Check all 5 folds + mean + std
        fold_values = set(df_results["fold"].astype(str).tolist())
        expected_folds = {"0", "1", "2", "3", "4", "mean", "std"}
        self.assertTrue(
            expected_folds.issubset(fold_values),
            f"Missing expected fold rows: {expected_folds - fold_values}",
        )

        # Check predictions artifact if present
        if self.predictions_csv_path.is_file():
            df_preds = pd.read_csv(self.predictions_csv_path)
            self.assertEqual(len(df_preds), 1440)
            self.assertEqual(df_preds.isna().sum().sum(), 0)
            self.assertIn("correct", df_preds.columns)


if __name__ == "__main__":
    unittest.main()
