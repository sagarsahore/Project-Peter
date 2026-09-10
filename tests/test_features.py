"""Automated Feature Test Suite for Project P.E.T.E.R.

Affective Computing & Biosignals Pipeline
Verifies structural contracts, data integrity, and zero-leakage preservation
across the dual-path feature extraction pipelines:
1. Classical Bioacoustic Functionals (openSMILE eGeMAPSv02 Parquet).
2. Deep Log-Mel Spectrogram Tensors (.pt PyTorch format).
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import List

import pandas as pd
import torch

from src.features.acoustic_egemaps import (
    AUDIT_METADATA_COLUMNS,
    EXPECTED_FEATURE_COUNT,
    EXPECTED_RECORD_COUNT,
)
from src.features.spectrograms import (
    DEFAULT_N_MELS,
    DEFAULT_TARGET_SAMPLES,
    AudioTensorPipeline,
)


class TestFeaturesPipeline(unittest.TestCase):
    """Test suite verifying acoustic and deep spectrogram feature extraction contracts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.parquet_path = Path("Data/processed/features_egemaps.parquet")
        cls.spectrogram_dir = Path("Data/processed/spectrograms")

    def test_pipeline_synthetic_waveform_contracts(self) -> None:
        """Verifies AudioTensorPipeline unit logic on synthetic waveforms."""
        pipeline = AudioTensorPipeline()

        # 1. Downmix stereo -> mono
        stereo_waveform = torch.randn(2, 32000)
        mono_waveform = pipeline.downmix_mono(stereo_waveform)
        self.assertEqual(mono_waveform.shape[0], 1)
        self.assertEqual(mono_waveform.shape[1], 32000)

        # 2. Resample from 48 kHz to 16 kHz
        high_sr_waveform = torch.randn(1, 48000)
        resampled_waveform = pipeline.resample(high_sr_waveform, orig_sr=48000)
        self.assertEqual(resampled_waveform.shape[1], 16000)

        # 3. Center-pad short clip (1.0 sec = 16,000 samples -> 48,000 samples)
        short_waveform = torch.ones(1, 16000)
        padded_waveform = pipeline.center_pad_crop(short_waveform)
        self.assertEqual(padded_waveform.shape[1], DEFAULT_TARGET_SAMPLES)
        # Verify symmetric zero-padding
        self.assertEqual(padded_waveform[0, 0].item(), 0.0)
        self.assertEqual(padded_waveform[0, -1].item(), 0.0)
        self.assertEqual(padded_waveform[0, 24000].item(), 1.0)

        # 4. Center-crop long clip (4.0 sec = 64,000 samples -> 48,000 samples)
        long_waveform = torch.randn(1, 64000)
        cropped_waveform = pipeline.center_pad_crop(long_waveform)
        self.assertEqual(cropped_waveform.shape[1], DEFAULT_TARGET_SAMPLES)

        # 5. Compute Log-Mel Spectrogram contract: shape [1, 80, ~94]
        exact_waveform = torch.randn(1, DEFAULT_TARGET_SAMPLES)
        spec = pipeline.compute_log_mel_spectrogram(exact_waveform)
        self.assertEqual(spec.ndim, 3)
        self.assertEqual(spec.shape[0], 1)
        self.assertEqual(spec.shape[1], DEFAULT_N_MELS)
        self.assertAlmostEqual(spec.shape[2], 94, delta=2)
        self.assertFalse(torch.isnan(spec).any())
        self.assertFalse(torch.isinf(spec).any())

    def test_egemaps_parquet_integrity(self) -> None:
        """Verifies eGeMAPSv02 Parquet artifact exists, schema conforms, and has zero nulls."""
        if not self.parquet_path.is_file():
            self.skipTest(f"Feature artifact not generated yet: {self.parquet_path}")

        df = pd.read_parquet(self.parquet_path)

        # 1. Total clip count invariant (1,440)
        self.assertEqual(
            len(df),
            EXPECTED_RECORD_COUNT,
            f"Expected {EXPECTED_RECORD_COUNT} records, found {len(df)}.",
        )

        # 2. Verify all audit-critical metadata columns exist
        for col in AUDIT_METADATA_COLUMNS:
            self.assertIn(
                col,
                df.columns,
                f"Missing audit-critical metadata column: '{col}'",
            )

        # 3. Verify exactly 88 acoustic feature functionals exist
        # Acoustic features are non-metadata columns
        feature_cols: List[str] = [
            c for c in df.columns if c not in AUDIT_METADATA_COLUMNS and c not in [
                "file_path", "file_name", "file_size_bytes", "modality", "vocal_channel",
                "emotion_id", "intensity_id", "statement_id", "statement_text",
                "repetition_id", "actor_id",
            ]
        ]
        self.assertEqual(
            len(feature_cols),
            EXPECTED_FEATURE_COUNT,
            f"Expected {EXPECTED_FEATURE_COUNT} eGeMAPSv02 functionals, found {len(feature_cols)}.",
        )

        # 4. Zero null / NaN verification
        null_count = df.isna().sum().sum()
        self.assertEqual(
            null_count,
            0,
            f"Parquet table contains {null_count} null/NaN values.",
        )

        # 5. Cross-validation fold preservation
        self.assertIn("fold", df.columns)
        folds = set(df["fold"].unique())
        self.assertEqual(folds, {0, 1, 2, 3, 4})

        # Ensure no speaker leakage across folds
        speaker_folds = df.groupby("speaker_id")["fold"].nunique()
        self.assertTrue(
            (speaker_folds == 1).all(),
            "Speaker leakage detected in Parquet metadata!",
        )

    def test_spectrogram_tensor_contract(self) -> None:
        """Verifies cached deep spectrogram tensors conform to shape and numeric contracts."""
        if not self.spectrogram_dir.is_dir():
            self.skipTest(f"Spectrogram directory not created yet: {self.spectrogram_dir}")

        pt_files = sorted(self.spectrogram_dir.glob("*.pt"))
        if not pt_files:
            self.skipTest(f"No .pt tensor files found in: {self.spectrogram_dir}")

        # 1. Total clip count invariant (1,440)
        self.assertEqual(
            len(pt_files),
            EXPECTED_RECORD_COUNT,
            f"Expected {EXPECTED_RECORD_COUNT} .pt files, found {len(pt_files)}.",
        )

        # 2. Inspect a stratified sample across actors and folds
        # Sample every 50th tensor to ensure comprehensive checks without excessive I/O overhead
        sampled_files = pt_files[::50]
        self.assertGreaterEqual(len(sampled_files), 20)

        for pt_path in sampled_files:
            tensor = torch.load(pt_path, weights_only=True)

            # Assert tensor dimensions: [1, 80, time_steps]
            self.assertEqual(
                tensor.ndim,
                3,
                f"{pt_path.name}: Expected 3D tensor [1, 80, T], got {tensor.ndim}D.",
            )
            self.assertEqual(
                tensor.shape[0],
                1,
                f"{pt_path.name}: Expected 1 channel, got {tensor.shape[0]}.",
            )
            self.assertEqual(
                tensor.shape[1],
                DEFAULT_N_MELS,
                f"{pt_path.name}: Expected {DEFAULT_N_MELS} mel bands, got {tensor.shape[1]}.",
            )
            # Time steps should be ~94
            self.assertAlmostEqual(
                tensor.shape[2],
                94,
                delta=2,
                msg=f"{pt_path.name}: Time steps {tensor.shape[2]} not within ~94.",
            )

            # Numerical validity
            self.assertFalse(
                torch.isnan(tensor).any(),
                f"{pt_path.name}: Tensor contains NaN values.",
            )
            self.assertFalse(
                torch.isinf(tensor).any(),
                f"{pt_path.name}: Tensor contains Inf values.",
            )


if __name__ == "__main__":
    unittest.main()
