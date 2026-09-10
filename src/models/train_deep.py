"""Deep Learning Spectrogram Emotion Classification Pipeline & Demographic Audit.

Project P.E.T.E.R. (Perceptual Evaluation & Transparent Emotion Research)
Affective Computing & Biosignals Pipeline

This module trains a 2D Convolutional Neural Network (EmotionCNN2D) on 80-band
Log-Mel Spectrogram tensors across the exact same 5 speaker-disjoint folds as the
classical eGeMAPSv02 baseline.

Architecture:
- 4 Convolutional blocks (Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d).
- Adaptive average pooling (AdaptiveAvgPool2d((4, 4))) for variable temporal resilience.
- Dropout-regularized classification head.
- Sliced demographic fairness audit (F1_female, F1_male, Demographic Disparity Ratio).
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.deep_dataset import (
    DEFAULT_TARGET_TIME_FRAMES,
    INDEX_TO_EMOTION_NAME,
    LogMelSpectrogramDataset,
    pad_spectrogram_collate_fn,
)

# Configure structured logging
logger = logging.getLogger("peter.models.deep")
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
NUM_CLASSES = 8
EXPECTED_FOLDS = 5
CLASSICAL_BASELINE_F1 = 0.4597
CLASSICAL_BASELINE_DISPARITY = 0.703


# ==============================================================================
# Model Architecture
# ==============================================================================


class EmotionCNN2D(nn.Module):
    """Deep 2D Convolutional Neural Network for Log-Mel Spectrogram Emotion Recognition."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_p: float = 0.3,
    ) -> None:
        """Initializes 4 convolutional blocks and an adaptive pooling classification head.

        Args:
            num_classes: Number of target emotion categories (default: 8).
            dropout_p: Dropout probability in classification head (default: 0.3).
        """
        super().__init__()

        # Feature Extractor
        self.conv_blocks = nn.Sequential(
            # Block 1: [B, 1, 80, T] -> [B, 32, 40, T/2]
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: [B, 32, 40, T/2] -> [B, 64, 20, T/4]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: [B, 64, 20, T/4] -> [B, 128, 10, T/8]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: [B, 128, 10, T/8] -> [B, 256, 5, T/16]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Dimension-invariant adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor, adaptive pool, and classifier.

        Args:
            x: Input spectrogram tensor of shape [B, 1, 80, T].

        Returns:
            Logits tensor of shape [B, num_classes].
        """
        feats = self.conv_blocks(x)
        pooled = self.pool(feats)
        logits = self.classifier(pooled)
        return logits


# ==============================================================================
# Demographic Fairness Metrics
# ==============================================================================


def compute_demographic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    genders: np.ndarray,
) -> Dict[str, float]:
    """Computes overall and gender-sliced performance and fairness metrics.

    Args:
        y_true: Ground-truth emotion indices (0-7).
        y_pred: Model predicted emotion indices (0-7).
        genders: Array of corresponding gender strings ('male' / 'female').

    Returns:
        Dictionary containing overall accuracy, macro F1, macro recall,
        female F1, male F1, and demographic disparity ratio.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

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


# ==============================================================================
# Training and Evaluation Engine
# ==============================================================================


def train_single_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Runs a single training epoch and returns average training loss."""
    model.train()
    total_loss = 0.0

    for batch_x, batch_y, _ in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Runs inference across validation dataloader, returning predictions and metadata."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_genders: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []

    for batch_x, batch_y, metadatas in dataloader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        all_preds.extend(preds)
        all_targets.extend(batch_y.numpy().tolist())
        all_genders.extend([m["gender"] for m in metadatas])
        all_metadatas.extend(metadatas)

    return (
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_genders),
        all_metadatas,
    )


def train_and_evaluate_fold(
    fold_id: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    spectrogram_dir: Path,
    epochs: int,
    batch_size: int,
    device: torch.device,
    random_state: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Trains EmotionCNN2D on a single cross-validation fold with early checkpointing."""
    torch.manual_seed(random_state + fold_id)
    np.random.seed(random_state + fold_id)

    # Instantiate datasets
    train_dataset = LogMelSpectrogramDataset(
        manifest_or_df=train_df,
        spectrogram_dir=spectrogram_dir,
        in_memory=True,
    )
    val_dataset = LogMelSpectrogramDataset(
        manifest_or_df=val_df,
        spectrogram_dir=spectrogram_dir,
        in_memory=True,
    )

    collate_fn = partial(pad_spectrogram_collate_fn, target_frames=DEFAULT_TARGET_TIME_FRAMES)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = EmotionCNN2D(num_classes=NUM_CLASSES, dropout_p=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_macro_f1 = -1.0
    best_model_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, epochs + 1):
        train_loss = train_single_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred, genders, _ = evaluate_model(model, val_loader, device)

        epoch_metrics = compute_demographic_metrics(y_true, y_pred, genders)
        val_f1 = epoch_metrics["macro_f1"]
        scheduler.step(val_f1)

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())

    # Restore best checkpoint
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Serialize fold checkpoint to disk
    ckpt_dir = Path("Data/processed")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"emotion_cnn2d_fold{fold_id}.pt")
    if fold_id == 0:
        torch.save(model.state_dict(), ckpt_dir / "emotion_cnn2d_best.pt")

    # Final evaluation on best model
    final_y_true, final_y_pred, final_genders, final_meta = evaluate_model(
        model, val_loader, device
    )
    final_metrics = compute_demographic_metrics(final_y_true, final_y_pred, final_genders)

    fold_metrics = {
        "fold": fold_id,
        "model_name": "EmotionCNN2D",
        "n_train": len(train_df),
        "n_val": len(val_df),
        **final_metrics,
    }

    # Format sample-level predictions
    pred_rows: List[Dict[str, Any]] = []
    for idx in range(len(final_y_true)):
        t_idx = int(final_y_true[idx])
        p_idx = int(final_y_pred[idx])
        pred_rows.append({
            "filename": final_meta[idx]["filename"],
            "speaker_id": final_meta[idx]["speaker_id"],
            "gender": final_meta[idx]["gender"],
            "fold": fold_id,
            "emotion_name": INDEX_TO_EMOTION_NAME[t_idx],
            "y_true": t_idx,
            "y_pred": p_idx,
            "correct": int(t_idx == p_idx),
        })

    return fold_metrics, pd.DataFrame(pred_rows)


def run_deep_benchmark(
    manifest_path: Union[str, Path] = "Data/processed/dataset_manifest.csv",
    spectrogram_dir: Union[str, Path] = "Data/processed/spectrograms",
    output_results_path: Union[str, Path] = "Data/processed/deep_benchmark_results.csv",
    epochs: int = 15,
    batch_size: int = 32,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs 5-fold cross-validation benchmarking for EmotionCNN2D.

    Args:
        manifest_path: Path to dataset manifest CSV.
        spectrogram_dir: Folder containing .pt spectrogram tensors.
        output_results_path: Path to save benchmark metrics CSV.
        epochs: Number of training epochs per fold (default: 15).
        batch_size: DataLoader mini-batch size (default: 32).
        random_state: Master seed for reproducibility.

    Returns:
        Tuple of (results summary DataFrame, fold predictions DataFrame).
    """
    manifest_path = Path(manifest_path).resolve()
    spectrogram_dir = Path(spectrogram_dir).resolve()
    output_results_path = Path(output_results_path).resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    if not spectrogram_dir.is_dir():
        raise FileNotFoundError(f"Spectrogram directory not found: {spectrogram_dir}")

    df_manifest = pd.read_csv(manifest_path)
    logger.info("Loaded %d manifest records from: %s", len(df_manifest), manifest_path)

    # Device detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Operating compute device: %s", device)

    unique_folds = sorted(df_manifest["fold"].unique())
    fold_results: List[Dict[str, Any]] = []
    all_predictions: List[pd.DataFrame] = []

    logger.info("Starting zero-leakage 5-fold cross-validation (epochs per fold: %d)...", epochs)

    for fold_id in unique_folds:
        train_df = df_manifest[df_manifest["fold"] != fold_id].copy()
        val_df = df_manifest[df_manifest["fold"] == fold_id].copy()

        # Zero speaker leakage assertion
        train_speakers = set(train_df["speaker_id"])
        val_speakers = set(val_df["speaker_id"])
        leakage = train_speakers.intersection(val_speakers)
        if leakage:
            logger.critical("Speaker leakage detected in fold %d: %s", fold_id, leakage)
            raise RuntimeError(f"Speaker leakage in fold {fold_id}: {leakage}")

        logger.info(
            "Training Fold %d (Train: %d clips, Val: %d clips, Disjoint speakers)...",
            fold_id,
            len(train_df),
            len(val_df),
        )

        fold_metric, df_fold_preds = train_and_evaluate_fold(
            fold_id=fold_id,
            train_df=train_df,
            val_df=val_df,
            spectrogram_dir=spectrogram_dir,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            random_state=random_state,
        )

        fold_results.append(fold_metric)
        all_predictions.append(df_fold_preds)

        logger.info(
            "Fold %d Complete -- Accuracy: %.2f%%, Macro F1: %.2f%%, Disparity: %.3f",
            fold_id,
            fold_metric["accuracy"] * 100,
            fold_metric["macro_f1"] * 100,
            fold_metric["disparity_ratio"],
        )

    df_results = pd.DataFrame(fold_results)
    df_preds = pd.concat(all_predictions, ignore_index=True)

    # Compute cross-validation aggregates
    mean_row = {
        "fold": "mean",
        "model_name": "EmotionCNN2D",
        "n_train": int(df_results["n_train"].mean()),
        "n_val": int(df_results["n_val"].mean()),
        "accuracy": float(df_results["accuracy"].mean()),
        "macro_f1": float(df_results["macro_f1"].mean()),
        "macro_recall": float(df_results["macro_recall"].mean()),
        "f1_female": float(df_results["f1_female"].mean()),
        "f1_male": float(df_results["f1_male"].mean()),
        "disparity_ratio": float(df_results["disparity_ratio"].mean()),
    }
    std_row = {
        "fold": "std",
        "model_name": "EmotionCNN2D",
        "n_train": 0,
        "n_val": 0,
        "accuracy": float(df_results["accuracy"].std()),
        "macro_f1": float(df_results["macro_f1"].std()),
        "macro_recall": float(df_results["macro_recall"].std()),
        "f1_female": float(df_results["f1_female"].std()),
        "f1_male": float(df_results["f1_male"].std()),
        "disparity_ratio": float(df_results["disparity_ratio"].std()),
    }

    df_summary = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    # Export artifacts
    output_results_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(output_results_path, index=False)
    logger.info("Successfully exported deep benchmark results to: %s", output_results_path)

    preds_output_path = output_results_path.parent / "deep_predictions.csv"
    df_preds.to_csv(preds_output_path, index=False)
    logger.info("Successfully exported deep predictions to: %s", preds_output_path)

    # Print Formatted Report
    print_deep_benchmark_report(df_summary)

    return df_summary, df_preds


def print_deep_benchmark_report(df_summary: pd.DataFrame) -> None:
    """Prints a formatted ASCII evaluation report and baseline comparison."""
    divider = "=" * 88
    subdivider = "-" * 88

    print("\n" + divider)
    print("        PROJECT P.E.T.E.R. -- DEEP LEARNING (CNN2D) BENCHMARK REPORT       ")
    print(divider)

    print(f"\n>> Model: EmotionCNN2D (Log-Mel Spectrograms [1, 80, T])")
    print(f"   {'Fold':<6} | {'Accuracy':<9} | {'Macro F1':<9} | {'Recall':<9} | "
          f"{'F1(Female)':<11} | {'F1(Male)':<9} | {'Disparity':<10}")
    print(f"   {'-'*6}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*11}-+-{'-'*9}-+-{'-'*10}")

    for _, row in df_summary.iterrows():
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
    print(">> Head-to-Head Benchmark Comparison (vs. Classical LGBM Baseline):")
    deep_mean = df_summary[df_summary["fold"] == "mean"].iloc[0]
    delta_f1 = deep_mean["macro_f1"] - CLASSICAL_BASELINE_F1
    delta_disp = deep_mean["disparity_ratio"] - CLASSICAL_BASELINE_DISPARITY

    print(f"   * Classical LGBM Mean Macro F1:          {CLASSICAL_BASELINE_F1*100:.2f}%")
    print(f"   * Deep EmotionCNN2D Mean Macro F1:       {deep_mean['macro_f1']*100:.2f}% (Delta: {delta_f1*100:+.2f}%)")
    print(f"   * Classical Demographic Disparity Ratio: {CLASSICAL_BASELINE_DISPARITY:.3f}")
    print(f"   * Deep Demographic Disparity Ratio:      {deep_mean['disparity_ratio']:.3f} (Delta: {delta_disp:+.3f})")
    print(divider + "\n")


def main() -> None:
    """CLI orchestrator for deep learning emotion recognition pipeline."""
    parser = argparse.ArgumentParser(
        description="Deep 2D CNN Spectrogram Emotion Recognition Benchmark & Demographic Audit"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="Data/processed/dataset_manifest.csv",
        help="Path to dataset manifest CSV (default: Data/processed/dataset_manifest.csv)",
    )
    parser.add_argument(
        "--spectrogram-dir",
        type=str,
        default="Data/processed/spectrograms",
        help="Path to cached spectrogram tensors (default: Data/processed/spectrograms)",
    )
    parser.add_argument(
        "--output-results",
        type=str,
        default="Data/processed/deep_benchmark_results.csv",
        help="Target path for deep benchmark metrics (default: Data/processed/deep_benchmark_results.csv)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs per fold (default: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and validation (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    logger.info("Initiating deep spectrogram benchmark pipeline...")
    run_deep_benchmark(
        manifest_path=args.manifest,
        spectrogram_dir=args.spectrogram_dir,
        output_results_path=args.output_results,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.seed,
    )
    logger.info("Deep spectrogram pipeline executed successfully.")


if __name__ == "__main__":
    main()
