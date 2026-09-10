"""Classical Bioacoustic Feature Extraction using openSMILE eGeMAPSv02.

Project P.E.T.E.R. (Perceptual Evaluation & Transparent Emotion Research)
Affective Computing & Biosignals Pipeline

This module extracts the standard 88 acoustic summary functionals defined by
the Geneva Minimalistic Acoustic Parameter Set (eGeMAPSv02) for speech emotion
recognition and affective biosignal analysis.

Features include:
- Frequency parameters: Pitch (F0), Formants (F1-F3), Jitter.
- Energy/Amplitude parameters: Shimmer, Loudness, HNR.
- Spectral parameters: Alpha ratio, Hammarberg index, Spectral slope, MFCCs 1-4.
- Functionals: Mean, std dev, percentiles, slopes, arithmetic moments.

All audit-critical demographic, affective, and zero-leakage cross-validation
metadata are preserved alongside features in the final Parquet artifact.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Union

import opensmile
import pandas as pd
from tqdm import tqdm

# Configure structured logging
logger = logging.getLogger("peter.features.egemaps")
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
EXPECTED_FEATURE_COUNT = 88
EXPECTED_RECORD_COUNT = 1440
AUDIT_METADATA_COLUMNS = [
    "filename",
    "speaker_id",
    "gender",
    "emotion_name",
    "emotion_code",
    "intensity_level",
    "fold",
]


def resolve_audio_path(file_path_str: str, file_name: str, base_dir: Path) -> Optional[Path]:
    """Resolves an audio file path robustly across platforms and relative directories.

    Args:
        file_path_str: Absolute or relative file path stored in manifest.
        file_name: Base filename (e.g., '03-01-01-01-01-01-01.wav').
        base_dir: Workspace or repository root directory.

    Returns:
        Resolved Path if the file exists on disk, otherwise None.
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


def extract_egemaps_features(
    manifest_path: Union[str, Path],
    output_parquet_path: Union[str, Path],
) -> pd.DataFrame:
    """Extracts 88 eGeMAPSv02 functional features from audio files in the manifest.

    Args:
        manifest_path: Path to the dataset manifest CSV.
        output_parquet_path: Path to save the extracted features Parquet file.

    Returns:
        Consolidated pandas DataFrame containing eGeMAPSv02 features and metadata.

    Raises:
        FileNotFoundError: If manifest_path cannot be found.
        ValueError: If no valid records could be processed.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

    logger.info("Loading manifest from: %s", manifest_path)
    df_manifest = pd.read_csv(manifest_path)
    logger.info("Found %d records in manifest.", len(df_manifest))

    # Initialize openSMILE extractor for eGeMAPSv02 Functionals
    logger.info("Initializing openSMILE extractor (eGeMAPSv02 Functionals)...")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    base_dir = manifest_path.parent.parent.resolve()
    feature_rows: List[pd.DataFrame] = []
    valid_indices: List[int] = []

    logger.info("Extracting bioacoustic features with progress tracking...")
    for idx, row in tqdm(df_manifest.iterrows(), total=len(df_manifest), desc="eGeMAPSv02 Extraction"):
        raw_path = str(row.get("file_path", ""))
        file_name = str(row.get("file_name", ""))

        resolved_path = resolve_audio_path(raw_path, file_name, base_dir)
        if resolved_path is None:
            logger.warning("Record index %d: Audio file not found for %s (%s)", idx, file_name, raw_path)
            continue

        try:
            feats = smile.process_file(str(resolved_path))
            if feats.shape[1] != EXPECTED_FEATURE_COUNT:
                logger.warning(
                    "Record %s: Unexpected feature count %d (expected %d)",
                    file_name,
                    feats.shape[1],
                    EXPECTED_FEATURE_COUNT,
                )
                continue
            feature_rows.append(feats)
            valid_indices.append(idx)
        except Exception as exc:
            logger.error("Failed to extract eGeMAPSv02 features from %s: %s", resolved_path, exc)
            continue

    if not feature_rows:
        raise ValueError("Failed to extract features from any audio clip in the manifest.")

    # Combine extracted acoustic features
    df_features = pd.concat(feature_rows, ignore_index=True)
    df_meta = df_manifest.iloc[valid_indices].reset_index(drop=True)

    # Standardize audit-critical metadata columns
    if "filename" not in df_meta.columns and "file_name" in df_meta.columns:
        df_meta["filename"] = df_meta["file_name"]
    if "emotion_code" not in df_meta.columns and "emotion_id" in df_meta.columns:
        df_meta["emotion_code"] = df_meta["emotion_id"]

    # Verify presence of all audit-critical metadata
    missing_meta = [col for col in AUDIT_METADATA_COLUMNS if col not in df_meta.columns]
    if missing_meta:
        logger.error("Missing required metadata columns: %s", missing_meta)
        raise KeyError(f"Missing audit-critical metadata columns: {missing_meta}")

    # Merge metadata and acoustic functionals
    df_consolidated = pd.concat([df_meta, df_features], axis=1)

    # Assert integrity invariants
    feature_cols = df_features.columns.tolist()
    total_nulls = df_consolidated.isna().sum().sum()
    logger.info("Consolidated dataframe shape: %s", df_consolidated.shape)
    logger.info("Acoustic features count: %d", len(feature_cols))
    logger.info("Total null values across all cells: %d", total_nulls)

    if total_nulls > 0:
        logger.warning("Detected %d null values in consolidated feature matrix!", total_nulls)

    if len(df_consolidated) != EXPECTED_RECORD_COUNT:
        logger.warning(
            "Extracted %d records; expected %d for full RAVDESS speech dataset.",
            len(df_consolidated),
            EXPECTED_RECORD_COUNT,
        )

    # Save to Parquet with Snappy compression
    output_path = Path(output_parquet_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting consolidated features to Parquet: %s", output_path)
    df_consolidated.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    logger.info("Successfully wrote %d rows to %s", len(df_consolidated), output_path)

    return df_consolidated


def main() -> None:
    """CLI orchestrator for eGeMAPS acoustic feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract openSMILE eGeMAPSv02 Summary Functionals for RAVDESS Speech Corpus."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="Data/processed/dataset_manifest.csv",
        help="Path to the dataset manifest CSV (default: Data/processed/dataset_manifest.csv)",
    )
    parser.add_argument(
        "--output-parquet",
        type=str,
        default="Data/processed/features_egemaps.parquet",
        help="Destination path for the output Parquet file (default: Data/processed/features_egemaps.parquet)",
    )
    args = parser.parse_args()

    logger.info("Starting eGeMAPSv02 feature extraction pipeline...")
    extract_egemaps_features(
        manifest_path=args.manifest,
        output_parquet_path=args.output_parquet,
    )
    logger.info("eGeMAPSv02 feature extraction completed successfully.")


if __name__ == "__main__":
    main()
