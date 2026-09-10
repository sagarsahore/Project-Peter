"""Automated Verification Suite for Model Interpretability & Explanation Faithfulness.

Project P.E.T.E.R. (Perceptual Evaluation & Transparent Emotion Research)
Affective Computing & Biosignals Pipeline

Verifies:
1. Grad-CAM spatial-temporal heatmap dimension compliance ([80, T]).
2. Saliency perturbation contracts (most- vs. least-informative masking).
3. Explanation faithfulness audit artifacts and SHAP importance rankings.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from src.evaluation.gradcam import GradCAM
from src.evaluation.interpretability import apply_saliency_mask
from src.models.train_deep import EmotionCNN2D


class TestInterpretability(unittest.TestCase):
    """Test suite for Grad-CAM, perturbation faithfulness, and SHAP artifacts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.audit_csv_path = Path("Data/processed/perturbation_audit.csv")
        cls.shap_csv_path = Path("Data/processed/shap_importance.csv")

    def test_gradcam_spatial_dimensions(self) -> None:
        """Verifies Grad-CAM produces heatmaps matching input dimensions [80, T]."""
        model = EmotionCNN2D(num_classes=8, dropout_p=0.3)
        model.eval()
        gradcam = GradCAM(model)

        # 1. Standard dimension T=94
        input_std = torch.randn(1, 1, 80, 94)
        heatmap_std, pred_class, prob = gradcam.generate_heatmap(input_std)
        self.assertEqual(heatmap_std.shape, torch.Size([80, 94]))
        self.assertGreaterEqual(heatmap_std.min().item(), 0.0)
        self.assertLessEqual(heatmap_std.max().item(), 1.0)
        self.assertTrue(0 <= pred_class < 8)
        self.assertTrue(0.0 <= prob <= 1.0)

        # 2. Arbitrary temporal length T=256
        input_long = torch.randn(1, 1, 80, 256)
        heatmap_long, _, _ = gradcam.generate_heatmap(input_long)
        self.assertEqual(heatmap_long.shape, torch.Size([80, 256]))
        self.assertFalse(torch.isnan(heatmap_long).any())

        gradcam.remove_hooks()

    def test_perturbation_contract(self) -> None:
        """Verifies that masking high-salience regions correctly zeroes out target cells."""
        spectrogram = torch.ones(1, 1, 80, 100)
        # Create synthetic heatmap with clear high saliency on the left half
        heatmap = torch.zeros(80, 100)
        heatmap[:, :20] = 1.0  # Exactly 20% of cells have high saliency

        # Most informative 20% masking: should zero out the left 20 columns
        masked_salient = apply_saliency_mask(
            spectrogram, heatmap, k_percent=20.0, mode="most_informative", fill_value=0.0
        )
        self.assertEqual(masked_salient[0, 0, :, :20].sum().item(), 0.0)
        self.assertEqual(masked_salient[0, 0, :, 20:].mean().item(), 1.0)

        # Least informative 20% masking: should zero out columns from the right
        masked_least = apply_saliency_mask(
            spectrogram, heatmap, k_percent=20.0, mode="least_informative", fill_value=0.0
        )
        self.assertEqual(masked_least[0, 0, :, :20].mean().item(), 1.0)

    def test_interpretability_artifacts_integrity(self) -> None:
        """Verifies that perturbation_audit.csv and shap_importance.csv exist and are complete."""
        if not self.audit_csv_path.is_file():
            self.skipTest(f"Perturbation audit file not found: {self.audit_csv_path}")

        df_audit = pd.read_csv(self.audit_csv_path)
        self.assertEqual(df_audit.isna().sum().sum(), 0, "Perturbation audit contains nulls.")
        self.assertIn("delta_p_salient", df_audit.columns)
        self.assertIn("delta_p_least", df_audit.columns)
        self.assertIn("faithfulness_gap", df_audit.columns)

        if not self.shap_csv_path.is_file():
            self.skipTest(f"SHAP importance file not found: {self.shap_csv_path}")

        df_shap = pd.read_csv(self.shap_csv_path)
        self.assertEqual(df_shap.isna().sum().sum(), 0, "SHAP importance contains nulls.")
        self.assertIn("feature_name", df_shap.columns)
        self.assertIn("global_shap_importance", df_shap.columns)
        self.assertIn("female_shap_importance", df_shap.columns)
        self.assertIn("male_shap_importance", df_shap.columns)
        self.assertGreaterEqual(len(df_shap), 20)


if __name__ == "__main__":
    unittest.main()
