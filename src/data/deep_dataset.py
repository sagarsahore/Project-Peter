"""Deep Learning Spectrogram Dataset and Collation Module.

Project P.E.T.E.R. (Perceptual Evaluation & Transparent Emotion Research)
Affective Computing & Biosignals Pipeline

This module provides:
- LogMelSpectrogramDataset: PyTorch Dataset loading precomputed 80-band Log-Mel
  spectrogram tensors ([1, 80, time_frames]) from disk/cache with metadata.
- pad_spectrogram_collate_fn: Custom batch collator enforcing deterministic
  temporal length standardization ([B, 1, 80, T_target]) via symmetric padding/cropping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger("peter.data.deep_dataset")

# Standard emotion mapping (1-8 numerical code to 0-7 zero-indexed label)
EMOTION_CODE_TO_INDEX: Dict[int, int] = {
    1: 0,  # neutral
    2: 1,  # calm
    3: 2,  # happy
    4: 3,  # sad
    5: 4,  # angry
    6: 5,  # fearful
    7: 6,  # disgust
    8: 7,  # surprised
}

INDEX_TO_EMOTION_NAME: Dict[int, str] = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised",
}

DEFAULT_TARGET_TIME_FRAMES = 94


def pad_or_crop_temporal(
    tensor: torch.Tensor,
    target_frames: int = DEFAULT_TARGET_TIME_FRAMES,
) -> torch.Tensor:
    """Standardizes the temporal dimension of a spectrogram via symmetric padding or cropping.

    Args:
        tensor: Spectrogram tensor of shape [..., 80, current_frames].
        target_frames: Desired temporal length along the last dimension.

    Returns:
        Tensor of shape [..., 80, target_frames].
    """
    cur_frames = tensor.shape[-1]
    if cur_frames < target_frames:
        diff = target_frames - cur_frames
        pad_left = diff // 2
        pad_right = diff - pad_left
        return F.pad(tensor, (pad_left, pad_right), mode="constant", value=0.0)
    elif cur_frames > target_frames:
        diff = cur_frames - target_frames
        crop_left = diff // 2
        return tensor[..., crop_left : crop_left + target_frames]
    return tensor


class LogMelSpectrogramDataset(Dataset):
    """PyTorch Dataset loading precomputed 80-band Log-Mel Spectrograms."""

    def __init__(
        self,
        manifest_or_df: Union[str, Path, pd.DataFrame],
        spectrogram_dir: Union[str, Path] = "Data/processed/spectrograms",
        in_memory: bool = True,
        target_frames: Optional[int] = None,
    ) -> None:
        """Initializes the dataset with manifest records and optional in-memory caching.

        Args:
            manifest_or_df: Path to manifest CSV or pre-filtered pandas DataFrame.
            spectrogram_dir: Folder containing precomputed .pt spectrogram tensors.
            in_memory: Whether to preload all tensors into RAM (~41 MB total).
            target_frames: Optional fixed temporal length to apply per-sample.
        """
        super().__init__()
        if isinstance(manifest_or_df, (str, Path)):
            manifest_path = Path(manifest_or_df).resolve()
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
            self.df = pd.read_csv(manifest_path)
        else:
            self.df = manifest_or_df.copy().reset_index(drop=True)

        self.spectrogram_dir = Path(spectrogram_dir).resolve()
        self.in_memory = in_memory
        self.target_frames = target_frames

        # Cache storage
        self._tensor_cache: Dict[int, torch.Tensor] = {}

        if self.in_memory:
            self._preload_tensors()

    def _preload_tensors(self) -> None:
        """Preloads all spectrogram tensors into memory for zero-I/O epoch iteration."""
        for idx in range(len(self.df)):
            self._tensor_cache[idx] = self._load_tensor_from_disk(idx)

    def _load_tensor_from_disk(self, idx: int) -> torch.Tensor:
        """Loads a single .pt spectrogram tensor from the filesystem."""
        row = self.df.iloc[idx]
        file_name = str(row.get("file_name", row.get("filename", "")))
        stem = Path(file_name).stem

        pt_path = self.spectrogram_dir / f"{stem}.pt"
        if not pt_path.is_file():
            raise FileNotFoundError(f"Spectrogram tensor missing: {pt_path}")

        tensor = torch.load(pt_path, weights_only=True)
        if self.target_frames is not None:
            tensor = pad_or_crop_temporal(tensor, self.target_frames)
        return tensor

    def __len__(self) -> int:
        """Returns the total number of audio samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """Retrieves spectrogram tensor, mapped emotion index, and metadata.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (spectrogram [1, 80, T], label_idx [0-7], metadata_dict).
        """
        row = self.df.iloc[idx]

        if self.in_memory and idx in self._tensor_cache:
            spectrogram = self._tensor_cache[idx]
        else:
            spectrogram = self._load_tensor_from_disk(idx)

        # Extract numerical code and map to 0-7
        code = int(row.get("emotion_code", row.get("emotion_id", 1)))
        label_idx = EMOTION_CODE_TO_INDEX.get(code, 0)

        meta = {
            "filename": str(row.get("filename", row.get("file_name", ""))),
            "speaker_id": str(row.get("speaker_id", "")),
            "gender": str(row.get("gender", "")),
            "fold": int(row.get("fold", -1)),
            "emotion_name": str(row.get("emotion_name", INDEX_TO_EMOTION_NAME[label_idx])),
            "emotion_code": code,
            "label_idx": label_idx,
        }

        return spectrogram, label_idx, meta


def pad_spectrogram_collate_fn(
    batch: List[Tuple[torch.Tensor, int, Dict[str, Any]]],
    target_frames: int = DEFAULT_TARGET_TIME_FRAMES,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Collate function standardizing temporal dimensions across a batch.

    Args:
        batch: List of tuples from LogMelSpectrogramDataset.__getitem__.
        target_frames: Fixed temporal length (default: 94).

    Returns:
        Tuple of:
          - Batched spectrograms tensor [B, 1, 80, target_frames].
          - Batched label tensor [B].
          - List of metadata dictionaries.
    """
    spectrograms, labels, metadatas = zip(*batch)

    standardized_specs: List[torch.Tensor] = []
    for spec in spectrograms:
        # Ensure tensor is [1, 80, target_frames]
        standardized_specs.append(pad_or_crop_temporal(spec, target_frames))

    batch_tensors = torch.stack(standardized_specs, dim=0)  # [B, 1, 80, target_frames]
    batch_labels = torch.tensor(labels, dtype=torch.long)   # [B]

    return batch_tensors, batch_labels, list(metadatas)
