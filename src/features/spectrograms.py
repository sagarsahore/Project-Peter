"""Deep Spectrogram Feature Extraction and Caching Pipeline.

Project P.E.T.E.R. (Perceptual Evaluation & Transparent Emotion Research)
Affective Computing & Biosignals Pipeline

This module converts speech waveforms into standardized 80-band Log-Mel Spectrograms
optimized for deep learning architectures (e.g., CNNs, Conformer, CRNNs).

Preprocessing specifications:
- Downmix multi-channel / stereo audio to single-channel mono.
- Resample audio to 16,000 Hz.
- Center-pad or center-crop to a fixed 3.0-second temporal window (48,000 samples).
- 80-band Mel-filterbank with STFT (n_fft=1024, hop_length=512, power=2.0).
- Amplitude-to-decibel dynamic range compression (top_db=80.0).
- Output tensor shape: [1, 80, 94].
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

# Disable gradient calculation globally for offline feature extraction
torch.set_grad_enabled(False)

# Configure structured logging
logger = logging.getLogger("peter.features.spectrograms")
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

# Audio processing constants
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_DURATION_SEC = 3.0
DEFAULT_TARGET_SAMPLES = int(DEFAULT_SAMPLE_RATE * DEFAULT_DURATION_SEC)  # 48,000
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_MELS = 80
DEFAULT_POWER = 2.0
DEFAULT_TOP_DB = 80.0
EXPECTED_RECORD_COUNT = 1440


def resolve_audio_path(file_path_str: str, file_name: str, base_dir: Path) -> Optional[Path]:
    """Resolves audio file path across platforms and relative root directories.

    Args:
        file_path_str: Path string from manifest.
        file_name: Audio filename with extension.
        base_dir: Workspace or project root directory.

    Returns:
        Resolved Path if found, otherwise None.
    """
    path_candidates = [
        Path(file_path_str),
        base_dir / file_path_str,
        base_dir / "Data" / "Raw" / f"Actor_{file_name.split('-')[-1].split('.')[0]}" / file_name,
        base_dir / "data" / "raw" / f"Actor_{file_name.split('-')[-1].split('.')[0]}" / file_name,
    ]
    for cand in path_candidates:
        if cand.is_file():
            return cand.resolve()
    return None


class AudioTensorPipeline:
    """Production-grade audio tensor extraction and transformation pipeline."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        target_duration_sec: float = DEFAULT_DURATION_SEC,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        n_mels: int = DEFAULT_N_MELS,
        power: float = DEFAULT_POWER,
        top_db: float = DEFAULT_TOP_DB,
    ) -> None:
        """Initializes transformation layers and hyperparameter contracts.

        Args:
            sample_rate: Target audio sampling rate in Hz (default: 16,000).
            target_duration_sec: Target temporal window length in seconds (default: 3.0).
            n_fft: FFT window size (default: 1024).
            hop_length: Number of samples between successive STFT frames (default: 512).
            n_mels: Number of Mel frequency filterbanks (default: 80).
            power: Power spectrogram exponent (default: 2.0).
            top_db: Dynamic range compression threshold in dB (default: 80.0).
        """
        self.sample_rate = sample_rate
        self.target_duration_sec = target_duration_sec
        self.target_samples = int(sample_rate * target_duration_sec)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.power = power
        self.top_db = top_db

        # Mel Spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=self.power,
        )

        # Amplitude to Decibels transform (log-mel)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            top_db=self.top_db,
        )

    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        """Loads audio waveform from disk via torchaudio.

        Args:
            audio_path: Path to the .wav file.

        Returns:
            Tuple of (waveform tensor [channels, samples], original sample rate).
        """
        waveform, sr = torchaudio.load(str(audio_path))
        return waveform, sr

    @staticmethod
    def downmix_mono(waveform: torch.Tensor) -> torch.Tensor:
        """Downmixes multi-channel audio to single-channel mono.

        Args:
            waveform: Audio tensor [channels, samples].

        Returns:
            Mono audio tensor of shape [1, samples].
        """
        if waveform.shape[0] > 1:
            return torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resamples waveform to target sample rate if necessary.

        Args:
            waveform: Audio tensor [channels, samples].
            orig_sr: Native sample rate of the input waveform.

        Returns:
            Resampled audio tensor [channels, resampled_samples].
        """
        if orig_sr != self.sample_rate:
            return torchaudio.functional.resample(
                waveform,
                orig_freq=orig_sr,
                new_freq=self.sample_rate,
            )
        return waveform

    def center_pad_crop(self, waveform: torch.Tensor) -> torch.Tensor:
        """Center-pads or center-crops audio to exact target sample duration.

        Args:
            waveform: Audio tensor of shape [1, current_samples].

        Returns:
            Audio tensor of shape [1, target_samples].
        """
        cur_samples = waveform.shape[-1]
        if cur_samples < self.target_samples:
            diff = self.target_samples - cur_samples
            pad_left = diff // 2
            pad_right = diff - pad_left
            return F.pad(waveform, (pad_left, pad_right), mode="constant", value=0.0)
        elif cur_samples > self.target_samples:
            diff = cur_samples - self.target_samples
            crop_left = diff // 2
            return waveform[:, crop_left : crop_left + self.target_samples]
        return waveform

    def compute_log_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Transforms preconditioned waveform into an 80-band Log-Mel Spectrogram.

        Args:
            waveform: Mono audio tensor [1, target_samples].

        Returns:
            Log-Mel Spectrogram tensor [1, n_mels, time_steps].
        """
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.db_transform(mel_spec)
        return log_mel_spec

    @torch.inference_mode()
    def process_file(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Executes the full audio tensor extraction pipeline.

        Args:
            audio_path: Path to raw audio file.

        Returns:
            Log-Mel Spectrogram tensor of shape [1, 80, time_steps].
        """
        waveform, sr = self.load_audio(audio_path)
        waveform = self.downmix_mono(waveform)
        waveform = self.resample(waveform, orig_sr=sr)
        waveform = self.center_pad_crop(waveform)
        spectrogram = self.compute_log_mel_spectrogram(waveform)
        return spectrogram


def extract_and_cache_spectrograms(
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_sec: float = DEFAULT_DURATION_SEC,
) -> int:
    """Iterates through manifest, computes Log-Mel Spectrograms, and serializes tensors.

    Args:
        manifest_path: Path to dataset manifest CSV.
        output_dir: Output directory where .pt files are stored.
        sample_rate: Audio sampling frequency in Hz (default: 16,000).
        duration_sec: Fixed audio duration in seconds (default: 3.0).

    Returns:
        Number of successfully generated and cached spectrogram tensors.

    Raises:
        FileNotFoundError: If manifest file does not exist.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading manifest from: %s", manifest_path)
    df_manifest = pd.read_csv(manifest_path)
    logger.info("Discovered %d manifest entries.", len(df_manifest))

    logger.info("Initializing AudioTensorPipeline...")
    pipeline = AudioTensorPipeline(
        sample_rate=sample_rate,
        target_duration_sec=duration_sec,
    )

    base_dir = manifest_path.parent.parent.resolve()
    success_count = 0
    failure_count = 0

    logger.info("Processing and serializing spectrogram tensors to: %s", output_dir)
    for idx, row in tqdm(df_manifest.iterrows(), total=len(df_manifest), desc="Spectrogram Extraction"):
        raw_path = str(row.get("file_path", ""))
        file_name = str(row.get("file_name", ""))
        stem = Path(file_name).stem if file_name else Path(raw_path).stem

        resolved_path = resolve_audio_path(raw_path, file_name, base_dir)
        if resolved_path is None:
            logger.warning("Audio file not found: %s (%s)", file_name, raw_path)
            failure_count += 1
            continue

        target_file = output_dir / f"{stem}.pt"

        try:
            tensor = pipeline.process_file(resolved_path)

            # Contract checks
            if tensor.shape[1] != DEFAULT_N_MELS:
                logger.warning(
                    "File %s produced %d mel bands (expected %d)",
                    file_name,
                    tensor.shape[1],
                    DEFAULT_N_MELS,
                )
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logger.error("File %s produced tensor containing NaN or Inf values!", file_name)
                failure_count += 1
                continue

            # Serialize PyTorch tensor
            torch.save(tensor, target_file)
            success_count += 1

        except Exception as exc:
            logger.error("Failed to generate spectrogram for %s: %s", resolved_path, exc)
            failure_count += 1
            continue

        # Periodically trigger garbage collection to maintain lean memory footprint
        if idx % 100 == 0:
            gc.collect()

    logger.info("Spectrogram extraction summary:")
    logger.info("  - Successfully cached: %d tensors", success_count)
    logger.info("  - Failed / skipped:    %d files", failure_count)

    if success_count != EXPECTED_RECORD_COUNT:
        logger.warning(
            "Expected %d cached spectrograms, but generated %d.",
            EXPECTED_RECORD_COUNT,
            success_count,
        )

    return success_count


def main() -> None:
    """CLI orchestrator for deep spectrogram feature extraction and serialization."""
    parser = argparse.ArgumentParser(
        description="Extract and cache 80-band Log-Mel Spectrograms for Project P.E.T.E.R."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="Data/processed/dataset_manifest.csv",
        help="Path to dataset manifest CSV (default: Data/processed/dataset_manifest.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Data/processed/spectrograms",
        help="Target folder for serialized .pt tensor files (default: Data/processed/spectrograms)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Target audio sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SEC,
        help="Fixed temporal window length in seconds (default: 3.0)",
    )
    args = parser.parse_args()

    logger.info("Starting Log-Mel Spectrogram caching pipeline...")
    extract_and_cache_spectrograms(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        duration_sec=args.duration,
    )
    logger.info("Spectrogram caching completed successfully.")


if __name__ == "__main__":
    main()
