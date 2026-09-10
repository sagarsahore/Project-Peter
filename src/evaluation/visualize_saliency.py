"""Publication-Quality Grad-CAM Saliency Visualization for Audio Spectrograms.

Project P.E.T.E.R. (Perceptual Evaluation & Transparent Emotion Research)
Affective Computing & Biosignals Pipeline

This module loads a trained EmotionCNN2D model, computes Grad-CAM saliency
heatmaps over log-mel spectrograms for an expressive utterance, and generates
a publication-ready 3-panel figure:
1. Raw time-domain acoustic waveform (Time vs. Amplitude).
2. 80-band Log-Mel Spectrogram (Time vs. Mel Frequency in dB).
3. Log-Mel Spectrogram with transparent Grad-CAM saliency overlay (magma colormap).

All panels share synchronized time axes in seconds.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force headless Matplotlib backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio

from src.data.deep_dataset import INDEX_TO_EMOTION_NAME
from src.evaluation.gradcam import GradCAM
from src.features.spectrograms import AudioTensorPipeline
from src.models.train_deep import EmotionCNN2D

# Configure structured logging
logger = logging.getLogger("peter.evaluation.visualize")
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

DEFAULT_SAMPLE_FILENAME = "03-01-05-02-01-01-01.wav"  # Actor 01, Male, Angry, Strong


def resolve_audio_path(file_path_str: str, file_name: str, base_dir: Path) -> Optional[Path]:
    """Resolves an audio file path robustly across platforms and relative directories."""
    candidates = [
        Path(file_path_str),
        base_dir / file_path_str,
        base_dir / "Data" / "Raw" / f"Actor_{file_name.split('-')[-1].split('.')[0]}" / file_name,
        base_dir / "data" / "raw" / f"Actor_{file_name.split('-')[-1].split('.')[0]}" / file_name,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return None


def load_model_checkpoint(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> EmotionCNN2D:
    """Loads EmotionCNN2D with weights from the best available checkpoint."""
    if device is None:
        device = torch.device("cpu")

    model = EmotionCNN2D(num_classes=8, dropout_p=0.3).to(device)

    candidate_checkpoints = [
        Path(checkpoint_path) if checkpoint_path else None,
        Path("Data/processed/emotion_cnn2d_best.pt"),
        Path("Data/processed/emotion_cnn2d_fold0.pt"),
        Path("Data/processed/emotion_cnn2d_fold1.pt"),
    ]

    loaded = False
    for ckpt in candidate_checkpoints:
        if ckpt and ckpt.is_file():
            logger.info("Loading model weights from: %s", ckpt.resolve())
            state_dict = torch.load(ckpt, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            loaded = True
            break

    if not loaded:
        logger.warning("No pre-trained checkpoint found on disk. Initializing model with default weights.")

    model.eval()
    return model


def generate_attribution_figure(
    manifest_path: Union[str, Path] = "Data/processed/dataset_manifest.csv",
    spectrogram_dir: Union[str, Path] = "Data/processed/spectrograms",
    checkpoint_path: Optional[Union[str, Path]] = None,
    sample_filename: Optional[str] = None,
    sample_idx: Optional[int] = None,
    output_figure_path: Union[str, Path] = "reports/figures/gradcam_attribution_sample.png",
    dpi: int = 300,
) -> Path:
    """Generates and saves a publication-quality 3-panel Grad-CAM attribution figure.

    Args:
        manifest_path: Path to dataset manifest CSV.
        spectrogram_dir: Directory containing cached .pt spectrogram tensors.
        checkpoint_path: Path to model checkpoint .pt file.
        sample_filename: Target audio filename to visualize.
        sample_idx: Target index in manifest if filename not provided.
        output_figure_path: Destination path for rendered PNG.
        dpi: Output image resolution in dots per inch (default: 300).

    Returns:
        Resolved Path to the saved figure image.
    """
    manifest_path = Path(manifest_path).resolve()
    spectrogram_dir = Path(spectrogram_dir).resolve()
    output_figure_path = Path(output_figure_path).resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

    df_manifest = pd.read_csv(manifest_path)

    # 1. Identify target sample record
    if sample_filename:
        matches = df_manifest[df_manifest["file_name"] == sample_filename]
        if matches.empty:
            matches = df_manifest[df_manifest["file_path"].str.endswith(sample_filename)]
        if matches.empty:
            raise ValueError(f"Filename '{sample_filename}' not found in manifest.")
        record = matches.iloc[0]
    elif sample_idx is not None:
        if not (0 <= sample_idx < len(df_manifest)):
            raise IndexError(f"sample_idx {sample_idx} out of range (0 to {len(df_manifest)-1}).")
        record = df_manifest.iloc[sample_idx]
    else:
        # Default to an expressive sample
        matches = df_manifest[df_manifest["file_name"] == DEFAULT_SAMPLE_FILENAME]
        if not matches.empty:
            record = matches.iloc[0]
        else:
            record = df_manifest.iloc[0]

    filename = str(record.get("file_name", record.get("filename", "")))
    raw_path_str = str(record.get("file_path", ""))
    true_emotion = str(record.get("emotion_name", "unknown"))
    actor_id = int(record.get("actor_id", 0))
    gender = str(record.get("gender", "unknown"))
    intensity = str(record.get("intensity_level", "normal"))

    logger.info("Visualizing sample: %s (Emotion: %s, Gender: %s, Actor: %02d)", filename, true_emotion, gender, actor_id)

    # 2. Load raw audio waveform
    base_dir = manifest_path.parent.parent.resolve()
    audio_path = resolve_audio_path(raw_path_str, filename, base_dir)
    if audio_path is None or not audio_path.is_file():
        raise FileNotFoundError(f"Raw audio file not found on disk for {filename}")

    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform_np = waveform.squeeze().cpu().numpy()

    duration_sec = float(len(waveform_np) / sr)
    time_axis_audio = np.linspace(0.0, duration_sec, len(waveform_np))

    # 3. Load or compute log-mel spectrogram
    stem = Path(filename).stem
    pt_path = spectrogram_dir / f"{stem}.pt"
    if pt_path.is_file():
        spec_tensor = torch.load(pt_path, map_location="cpu", weights_only=True)
    else:
        logger.info("Precomputed tensor not found; computing on-the-fly via AudioTensorPipeline...")
        pipeline = AudioTensorPipeline(sample_rate=16000, target_duration_sec=3.0)
        spec_tensor = pipeline.process_file(audio_path)

    # Spec tensor shape: [1, 80, T]
    spec_np = spec_tensor.squeeze().cpu().numpy()  # [80, T]
    num_time_frames = spec_np.shape[1]

    # Standardized fixed window duration (3.0s as defined in preprocessing)
    spec_duration_sec = 3.0

    # 4. Load trained model and compute Grad-CAM
    device = torch.device("cpu")
    model = load_model_checkpoint(checkpoint_path, device=device)
    gradcam = GradCAM(model)

    heatmap, pred_class, confidence = gradcam.generate_heatmap(spec_tensor)
    heatmap_np = heatmap.cpu().numpy()  # [80, T]
    pred_emotion = INDEX_TO_EMOTION_NAME.get(pred_class, f"Class_{pred_class}")

    logger.info("Inference complete: Predicted '%s' (Confidence: %.1f%%, Ground Truth: '%s')",
                pred_emotion, confidence * 100, true_emotion)

    # 5. Render Publication-Quality 3-Panel Figure
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(11, 9.2),
        sharex=True,
    )

    # Styling constants
    COLOR_WAVE = "#1b4965"
    CMAP_SPEC = "viridis"
    CMAP_SALIENCY = "magma"

    # Align raw waveform to target 3.0s spectrogram analysis window
    # RAVDESS audio is resampled to 16kHz and center-padded or center-cropped to 3.0s (48,000 samples)
    target_samples = int(16000 * spec_duration_sec)
    cur_samples = len(waveform_np)
    # Resample waveform to 16kHz for precise temporal synchronization
    if sr != 16000:
        resampled_wf = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000).squeeze().cpu().numpy()
    else:
        resampled_wf = waveform_np

    if len(resampled_wf) >= target_samples:
        crop_start = (len(resampled_wf) - target_samples) // 2
        aligned_waveform = resampled_wf[crop_start : crop_start + target_samples]
    else:
        pad_left = (target_samples - len(resampled_wf)) // 2
        pad_right = target_samples - len(resampled_wf) - pad_left
        aligned_waveform = np.pad(resampled_wf, (pad_left, pad_right), mode="constant")

    time_axis_aligned = np.linspace(0.0, spec_duration_sec, len(aligned_waveform))

    # --------------------------------------------------------------------------
    # Subplot 1 (Top): Raw Waveform (Synchronized 3.0s Window)
    # --------------------------------------------------------------------------
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="3%", pad=0.12)
    cax1.set_visible(False)

    ax1.plot(time_axis_aligned, aligned_waveform, color=COLOR_WAVE, linewidth=0.8, alpha=0.92)
    ax1.set_xlim(0.0, spec_duration_sec)
    max_amp = max(0.2, float(np.max(np.abs(aligned_waveform))) * 1.15)
    ax1.set_ylim(-max_amp, max_amp)
    ax1.set_ylabel("Amplitude", fontsize=10, fontweight="bold")
    ax1.set_title("Panel A: Time-Domain Acoustic Waveform (16 kHz Standardized Window)", loc="left", fontsize=11, fontweight="bold", pad=8)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.tick_params(labelsize=9)

    # --------------------------------------------------------------------------
    # Subplot 2 (Middle): 80-band Log-Mel Spectrogram
    # --------------------------------------------------------------------------
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="3%", pad=0.12)

    im2 = ax2.imshow(
        spec_np,
        aspect="auto",
        origin="lower",
        extent=[0.0, spec_duration_sec, 0, 80],
        cmap=CMAP_SPEC,
        interpolation="nearest",
    )
    ax2.set_xlim(0.0, spec_duration_sec)
    ax2.set_ylabel("Mel Filter Bins", fontsize=10, fontweight="bold")
    ax2.set_title("Panel B: 80-Band Log-Mel Spectrogram (dB)", loc="left", fontsize=11, fontweight="bold", pad=8)
    ax2.tick_params(labelsize=9)

    # Secondary top x-axis showing STFT time frames (0 to num_time_frames)
    ax2_top = ax2.secondary_xaxis(
        "top",
        functions=(
            lambda s: s * (num_time_frames / spec_duration_sec),
            lambda f: f * (spec_duration_sec / num_time_frames),
        ),
    )
    ax2_top.set_xlabel("STFT Time Frames", fontsize=9, fontweight="semibold", labelpad=5)
    ax2_top.tick_params(labelsize=8)

    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label("Energy (dB)", fontsize=9, fontweight="semibold")
    cb2.ax.tick_params(labelsize=8)

    # --------------------------------------------------------------------------
    # Subplot 3 (Bottom): Log-Mel Spectrogram with Grad-CAM Saliency Overlay
    # --------------------------------------------------------------------------
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="3%", pad=0.12)

    # Base grayscale spectrogram
    ax3.imshow(
        spec_np,
        aspect="auto",
        origin="lower",
        extent=[0.0, spec_duration_sec, 0, 80],
        cmap="gray",
        alpha=0.60,
        interpolation="nearest",
    )
    # Grad-CAM heatmap overlay with alpha=0.50 transparency
    im3_cam = ax3.imshow(
        heatmap_np,
        aspect="auto",
        origin="lower",
        extent=[0.0, spec_duration_sec, 0, 80],
        cmap=CMAP_SALIENCY,
        alpha=0.50,
        interpolation="bilinear",
    )
    ax3.set_xlim(0.0, spec_duration_sec)
    ax3.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Mel Filter Bins", fontsize=10, fontweight="bold")
    ax3.set_title("Panel C: Grad-CAM Saliency Map Overlay (Conv4 Feature Attribution)", loc="left", fontsize=11, fontweight="bold", pad=8)
    ax3.tick_params(labelsize=9)

    cb3 = fig.colorbar(im3_cam, cax=cax3)
    cb3.set_label("Saliency Weight", fontsize=9, fontweight="semibold")
    cb3.ax.tick_params(labelsize=8)

    # --------------------------------------------------------------------------
    # Global Super Title & Metadata Annotations
    # --------------------------------------------------------------------------
    match_tag = "[CORRECT]" if true_emotion.lower() == pred_emotion.lower() else "[MISMATCH]"
    super_title = (
        f"Project P.E.T.E.R. -- Audio Emotion Saliency Attribution Map (Grad-CAM)\n"
        f"Ground Truth: {true_emotion.upper()} ({intensity.capitalize()})   |   "
        f"Predicted: {pred_emotion.upper()} ({confidence*100:.1f}%) {match_tag}   |   "
        f"Speaker: Actor {actor_id:02d} ({gender.capitalize()})   |   File: {filename}"
    )
    fig.suptitle(super_title, fontsize=12, fontweight="bold", y=0.985)

    # Fine-tune vertical spacing to ensure zero title collision
    fig.subplots_adjust(top=0.91, bottom=0.07, left=0.08, right=0.96, hspace=0.35)

    # 6. Save Figure
    output_figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved publication-quality figure to: %s (DPI: %d)", output_figure_path, dpi)
    return output_figure_path


def main() -> None:
    """CLI orchestrator for Grad-CAM spectrogram visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize Grad-CAM Audio Spectrogram Saliency for EmotionCNN2D"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="Data/processed/dataset_manifest.csv",
        help="Path to manifest CSV (default: Data/processed/dataset_manifest.csv)",
    )
    parser.add_argument(
        "--spectrogram-dir",
        type=str,
        default="Data/processed/spectrograms",
        help="Directory containing .pt spectrogram tensors (default: Data/processed/spectrograms)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Data/processed/emotion_cnn2d_best.pt",
        help="Path to EmotionCNN2D checkpoint (default: Data/processed/emotion_cnn2d_best.pt)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=DEFAULT_SAMPLE_FILENAME,
        help=f"Specific audio filename to visualize (default: {DEFAULT_SAMPLE_FILENAME})",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=None,
        help="Row index of sample in manifest if filename not specified",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/figures/gradcam_attribution_sample.png",
        help="Output PNG file path (default: reports/figures/gradcam_attribution_sample.png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution in DPI (default: 300)",
    )
    args = parser.parse_args()

    generate_attribution_figure(
        manifest_path=args.manifest,
        spectrogram_dir=args.spectrogram_dir,
        checkpoint_path=args.checkpoint,
        sample_filename=args.filename if args.sample_idx is None else None,
        sample_idx=args.sample_idx,
        output_figure_path=args.output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
