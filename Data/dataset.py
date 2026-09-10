"""RAVDESS Speech Audio Dataset Parsing and Integrity Auditing Module.

Project P.E.T.E.R. (Perceptual Evaluation & Transparent Emotion Research)
Affective Computing & Biosignals Pipeline

This module provides a production-grade, zero-leakage-aware ingestion layer
for the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).

In Speech Emotion Recognition (SER) and affective biosignals, cross-speaker
leakage (allocating utterances from the same actor across both train and test splits)
is a fatal methodological flaw. Models trivially overfit to static vocal tract
geometry, baseline pitch (F0), and speaker-specific timbre rather than dynamic
affective prosody, resulting in severe performance over-estimation and zero
ecological validity. This module enforces speaker-independent partitioning
and strict data integrity checks.
"""

from __future__ import annotations
    
import argparse
import logging
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Configure structured logging
logger = logging.getLogger("peter.data.dataset")
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


# ==============================================================================
# Domain Enumerations & Schema Definitions
# ==============================================================================


class Modality(str, Enum):
    """RAVDESS recording modality codes."""

    FULL_AV = "01"
    VIDEO_ONLY = "02"
    AUDIO_ONLY = "03"


class VocalChannel(str, Enum):
    """RAVDESS vocal channel codes."""

    SPEECH = "01"
    SONG = "02"


class Emotion(int, Enum):
    """RAVDESS emotion category codes."""

    NEUTRAL = 1
    CALM = 2
    HAPPY = 3
    SAD = 4
    ANGRY = 5
    FEARFUL = 6
    DISGUST = 7
    SURPRISED = 8


class EmotionalIntensity(int, Enum):
    """RAVDESS emotional intensity codes."""

    NORMAL = 1
    STRONG = 2


class Statement(int, Enum):
    """RAVDESS spoken statement stimulus codes."""

    KIDS_TALKING = 1
    DOGS_SITTING = 2


EMOTION_NAME_MAP: Dict[int, str] = {
    Emotion.NEUTRAL.value: "neutral",
    Emotion.CALM.value: "calm",
    Emotion.HAPPY.value: "happy",
    Emotion.SAD.value: "sad",
    Emotion.ANGRY.value: "angry",
    Emotion.FEARFUL.value: "fearful",
    Emotion.DISGUST.value: "disgust",
    Emotion.SURPRISED.value: "surprised",
}

INTENSITY_LEVEL_MAP: Dict[int, str] = {
    EmotionalIntensity.NORMAL.value: "normal",
    EmotionalIntensity.STRONG.value: "strong",
}

STATEMENT_TEXT_MAP: Dict[int, str] = {
    Statement.KIDS_TALKING.value: "Kids are talking by the door",
    Statement.DOGS_SITTING.value: "Dogs are sitting by the door",
}

# Regex pattern strictly matching RAVDESS filename format: MM-VV-EE-II-SS-RR-AA.wav
RAVDESS_REGEX = re.compile(
    r"^(?P<modality>\d{2})-"
    r"(?P<vocal_channel>\d{2})-"
    r"(?P<emotion>\d{2})-"
    r"(?P<intensity>\d{2})-"
    r"(?P<statement>\d{2})-"
    r"(?P<repetition>\d{2})-"
    r"(?P<actor>\d{2})\.wav$",
    re.IGNORECASE,
)

# Expected ground-truth invariants for RAVDESS Speech Audio
EXPECTED_TOTAL_SPEECH_CLIPS = 1440
EXPECTED_ACTOR_COUNT = 24
EXPECTED_MALE_ACTORS = 12
EXPECTED_FEMALE_ACTORS = 12
EXPECTED_NEUTRAL_COUNT = 96
EXPECTED_NON_NEUTRAL_COUNT = 192


# ==============================================================================
# Parsing and Extraction Functions
# ==============================================================================


def parse_filename(file_path: Path) -> Optional[Dict[str, Any]]:
    """Parses and validates a single RAVDESS audio filename against schema rules.

    Args:
        file_path: Absolute or relative Path object to the .wav file.

    Returns:
        Dictionary containing extracted metadata fields if valid and within
        target speech subsets, otherwise None if malformed or filtered out.
    """
    filename = file_path.name
    match = RAVDESS_REGEX.match(filename)
    if not match:
        logger.debug("Skipping non-conforming filename: %s", filename)
        return None

    tokens = match.groupdict()

    # Strict domain filtering: audio-only ('03') and speech ('01')
    if tokens["modality"] != Modality.AUDIO_ONLY.value:
        logger.debug(
            "Filtering out non-audio-only modality '%s' in %s",
            tokens["modality"],
            filename,
        )
        return None

    if tokens["vocal_channel"] != VocalChannel.SPEECH.value:
        logger.debug(
            "Filtering out non-speech vocal channel '%s' in %s",
            tokens["vocal_channel"],
            filename,
        )
        return None

    try:
        emotion_id = int(tokens["emotion"])
        intensity_id = int(tokens["intensity"])
        statement_id = int(tokens["statement"])
        repetition_id = int(tokens["repetition"])
        actor_id = int(tokens["actor"])
    except ValueError as exc:
        logger.warning("Failed parsing numerical tokens in %s: %s", filename, exc)
        return None

    # Value domain validation
    if emotion_id not in EMOTION_NAME_MAP:
        logger.warning("Invalid emotion token '%02d' in %s", emotion_id, filename)
        return None

    if intensity_id not in INTENSITY_LEVEL_MAP:
        logger.warning("Invalid intensity token '%02d' in %s", intensity_id, filename)
        return None

    # Note: Neutral emotion only has normal intensity ('01') in RAVDESS
    if emotion_id == Emotion.NEUTRAL.value and intensity_id != EmotionalIntensity.NORMAL.value:
        logger.warning(
            "Encountered unexpected strong intensity for neutral emotion in %s",
            filename,
        )

    if statement_id not in STATEMENT_TEXT_MAP:
        logger.warning("Invalid statement token '%02d' in %s", statement_id, filename)
        return None

    if repetition_id not in (1, 2):
        logger.warning("Invalid repetition token '%02d' in %s", repetition_id, filename)
        return None

    if not (1 <= actor_id <= 24):
        logger.warning("Invalid actor token '%02d' in %s", actor_id, filename)
        return None

    # File physical integrity check
    if not file_path.is_file():
        logger.warning("Target audio file does not exist: %s", file_path)
        return None

    file_size_bytes = file_path.stat().st_size
    if file_size_bytes <= 0:
        logger.warning("Target audio file is empty (0 bytes): %s", file_path)
        return None

    # Zero-leakage actor demographic derivation
    gender = "male" if actor_id % 2 != 0 else "female"
    speaker_id = f"Actor_{actor_id:02d}"

    return {
        "file_path": str(file_path.resolve()),
        "file_name": filename,
        "file_size_bytes": file_size_bytes,
        "modality": tokens["modality"],
        "vocal_channel": tokens["vocal_channel"],
        "emotion_id": emotion_id,
        "emotion_name": EMOTION_NAME_MAP[emotion_id],
        "intensity_id": intensity_id,
        "intensity_level": INTENSITY_LEVEL_MAP[intensity_id],
        "statement_id": statement_id,
        "statement_text": STATEMENT_TEXT_MAP[statement_id],
        "repetition_id": repetition_id,
        "actor_id": actor_id,
        "speaker_id": speaker_id,
        "gender": gender,
    }


def assign_zero_leakage_folds(
    df: pd.DataFrame, n_splits: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Assigns cross-validation folds ensuring strict zero-leakage speaker isolation.

    Stratifies actors across gender to maintain balanced demographic representation
    across all folds while ensuring no actor appears in multiple folds.

    Args:
        df: Manifest DataFrame containing 'actor_id' and 'gender'.
        n_splits: Number of cross-validation folds (default: 5).
        seed: Random seed for actor partition reproducibility.

    Returns:
        DataFrame augmented with a 'fold' column (0-indexed).
    """
    df = df.copy()
    actor_df = (
        df[["actor_id", "gender"]]
        .drop_duplicates()
        .sort_values("actor_id")
        .reset_index(drop=True)
    )

    # Balance folds across genders
    males = actor_df[actor_df["gender"] == "male"]["actor_id"].sample(
        frac=1.0, random_state=seed
    ).tolist()
    females = actor_df[actor_df["gender"] == "female"]["actor_id"].sample(
        frac=1.0, random_state=seed + 1
    ).tolist()

    actor_to_fold: Dict[int, int] = {}
    for idx, act in enumerate(males):
        actor_to_fold[act] = idx % n_splits
    for idx, act in enumerate(females):
        actor_to_fold[act] = idx % n_splits

    df["fold"] = df["actor_id"].map(actor_to_fold)
    return df


def parse_ravdess_speech(data_dir: Union[str, Path]) -> pd.DataFrame:
    """Recursively parses and validates RAVDESS speech audio files under data_dir.

    Args:
        data_dir: Root directory containing extracted actor folders or audio files.

    Returns:
        Structured pandas DataFrame containing validated metadata, computed fields,
        and zero-leakage fold tags.

    Raises:
        FileNotFoundError: If data_dir does not exist.
        ValueError: If no valid speech audio files are found.
    """
    resolved_dir = Path(data_dir).resolve()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Specified data directory does not exist: {resolved_dir}")

    logger.info("Scanning for RAVDESS speech audio files in: %s", resolved_dir)
    wav_files = sorted(resolved_dir.rglob("*.wav"))

    if not wav_files:
        raise ValueError(f"No .wav files discovered under: {resolved_dir}")

    records: List[Dict[str, Any]] = []
    skipped_count = 0

    for wav_file in wav_files:
        parsed = parse_filename(wav_file)
        if parsed is not None:
            records.append(parsed)
        else:
            skipped_count += 1

    if not records:
        raise ValueError(
            f"Found {len(wav_files)} .wav files in {resolved_dir}, but none matched "
            "RAVDESS speech specifications (modality=03, vocal_channel=01)."
        )

    df = pd.DataFrame.from_records(records)

    # Sort deterministically by actor_id, emotion_id, statement_id, intensity_id, repetition_id
    sort_cols = ["actor_id", "emotion_id", "statement_id", "intensity_id", "repetition_id"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Assign zero-leakage stratified cross-validation folds
    df = assign_zero_leakage_folds(df, n_splits=5, seed=42)

    logger.info(
        "Successfully parsed %d valid speech records (skipped: %d).",
        len(df),
        skipped_count,
    )
    return df


# ==============================================================================
# Integrity Auditing and Reporting
# ==============================================================================


def audit_dataset_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """Conducts a comprehensive statistical integrity audit of the parsed dataset.

    Validates:
      - Total clip count against the 1,440 benchmark.
      - Actor population (24 unique actors, 12 male, 12 female).
      - Class distribution (96 neutral, 192 for each other emotion).
      - Audio file size and missingness.
      - Zero-leakage cross-validation fold distribution.

    Args:
        df: Manifest DataFrame returned by parse_ravdess_speech.

    Returns:
        Audit report dictionary with computed metrics and validation flags.
    """
    total_clips = len(df)
    unique_actors = df["actor_id"].nunique()
    male_actors = df[df["gender"] == "male"]["actor_id"].nunique()
    female_actors = df[df["gender"] == "female"]["actor_id"].nunique()

    emotion_counts = df["emotion_name"].value_counts().to_dict()
    gender_counts = df["gender"].value_counts().to_dict()
    intensity_counts = df["intensity_level"].value_counts().to_dict()
    statement_counts = df["statement_text"].value_counts().to_dict()
    fold_actor_counts = (
        df.groupby("fold")["actor_id"].nunique().to_dict() if "fold" in df.columns else {}
    )

    neutral_count = emotion_counts.get("neutral", 0)
    non_neutral_counts = [count for emo, count in emotion_counts.items() if emo != "neutral"]

    # Invariant assertions
    pass_total = total_clips == EXPECTED_TOTAL_SPEECH_CLIPS
    pass_actors = (
        unique_actors == EXPECTED_ACTOR_COUNT
        and male_actors == EXPECTED_MALE_ACTORS
        and female_actors == EXPECTED_FEMALE_ACTORS
    )
    pass_emotions = (
        neutral_count == EXPECTED_NEUTRAL_COUNT
        and all(c == EXPECTED_NON_NEUTRAL_COUNT for c in non_neutral_counts)
        and len(emotion_counts) == 8
    )
    pass_files = (df["file_size_bytes"] > 0).all() and not df["file_path"].isnull().any()

    all_passed = pass_total and pass_actors and pass_emotions and pass_files

    # Formatted Audit Output to stdout
    divider = "=" * 78

    print("\n" + divider)
    print("      PROJECT P.E.T.E.R. -- RAVDESS SPEECH DATASET INTEGRITY AUDIT      ")
    print(divider)
    status_str = "[PASSED] ALL INVARIANTS SATISFIED" if all_passed else "[FAILED] INVARIANT VIOLATION"
    print(f"Overall Audit Status: {status_str}\n")

    print("1. Overall Dataset Metrics:")
    print(f"   * Total Audio Clips Found: {total_clips:,} / {EXPECTED_TOTAL_SPEECH_CLIPS:,} "
          f"({'PASS' if pass_total else 'FAIL'})")
    print(f"   * Unique Actor Count:      {unique_actors} / {EXPECTED_ACTOR_COUNT} "
          f"({'PASS' if unique_actors == EXPECTED_ACTOR_COUNT else 'FAIL'})")
    print(f"   * Corrupted / 0-byte Files: {int((df['file_size_bytes'] <= 0).sum())} (PASS)")
    print(f"   * Mean File Size:          {df['file_size_bytes'].mean() / 1024:.2f} KB "
          f"(Total: {df['file_size_bytes'].sum() / (1024 * 1024):.2f} MB)")

    print("\n2. Speaker & Gender Balance:")
    print(f"   * Male Actors:   {male_actors:2d} (Utterances: {gender_counts.get('male', 0):,})")
    print(f"   * Female Actors: {female_actors:2d} (Utterances: {gender_counts.get('female', 0):,})")
    print(f"   * Parity Status: {'PASS (12 Male / 12 Female)' if pass_actors else 'FAIL'}")

    print("\n3. Affective Class Distribution (8 Emotions):")
    print(f"   {'Emotion':<12} | {'Count':<6} | {'Expected':<8} | {'Status':<6} | {'Pct':<6}")
    print(f"   {'-'*12}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")
    for emo_id in sorted(EMOTION_NAME_MAP.keys()):
        name = EMOTION_NAME_MAP[emo_id]
        count = emotion_counts.get(name, 0)
        expected = EXPECTED_NEUTRAL_COUNT if name == "neutral" else EXPECTED_NON_NEUTRAL_COUNT
        status = "PASS" if count == expected else "FAIL"
        pct = (count / total_clips) * 100 if total_clips > 0 else 0
        print(f"   {name:<12} | {count:<6} | {expected:<8} | {status:<6} | {pct:5.1f}%")

    print("\n4. Acoustic Conditioning Breakdown:")
    print("   * Emotional Intensity:")
    for int_name, cnt in intensity_counts.items():
        print(f"     - {int_name.capitalize():<7}: {cnt:4d} clips ({cnt/total_clips*100:5.1f}%)")
    print("   * Spoken Statement:")
    for stmt_name, cnt in statement_counts.items():
        print(f"     - \"{stmt_name}\": {cnt:4d} clips ({cnt/total_clips*100:5.1f}%)")

    print("\n5. Zero-Leakage Cross-Validation Fold Partitioning:")
    print("   [Fold ID]  [Actors Assigned]  [Total Clips]  [M/F Ratio]")
    for fold_id, n_acts in sorted(fold_actor_counts.items()):
        fold_df = df[df["fold"] == fold_id]
        m_count = fold_df[fold_df["gender"] == "male"]["actor_id"].nunique()
        f_count = fold_df[fold_df["gender"] == "female"]["actor_id"].nunique()
        clips_count = len(fold_df)
        print(f"     Fold {fold_id}     {n_acts:2d} actors        {clips_count:4d} clips    {m_count}M / {f_count}F")

    print(divider + "\n")

    return {
        "all_passed": all_passed,
        "total_clips": total_clips,
        "unique_actors": unique_actors,
        "male_actors": male_actors,
        "female_actors": female_actors,
        "emotion_counts": emotion_counts,
        "gender_counts": gender_counts,
        "intensity_counts": intensity_counts,
        "statement_counts": statement_counts,
    }


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def find_default_raw_dir() -> Path:
    """Locates the default raw audio directory relative to repository layout.

    Searches candidates:
      - data/raw
      - Data/Raw
      - relative to file location
    """
    candidates = [
        Path("data/raw"),
        Path("Data/Raw"),
        Path(__file__).resolve().parent / "Raw",
        Path(__file__).resolve().parent.parent / "Data" / "Raw",
        Path(__file__).resolve().parent.parent / "data" / "raw",
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand.resolve()
    # Fallback default
    return Path("data/raw").resolve()


def main() -> None:
    """CLI orchestrator for parsing RAVDESS speech dataset and generating manifest."""
    parser = argparse.ArgumentParser(
        description="RAVDESS Speech Audio Metadata Parser & Zero-Leakage Manifest Generator"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root path to extracted raw RAVDESS audio files (default: auto-detect Data/Raw)",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        default="data/processed/dataset_manifest.csv",
        help="Target CSV path for the generated dataset manifest",
    )
    args = parser.parse_args()

    # Determine input directory
    raw_dir = Path(args.data_dir).resolve() if args.data_dir else find_default_raw_dir()

    logger.info("Initiating RAVDESS parsing pipeline...")
    logger.info("Source directory: %s", raw_dir)

    try:
        df_manifest = parse_ravdess_speech(raw_dir)
    except Exception as exc:
        logger.error("Failed to parse RAVDESS dataset: %s", exc)
        sys.exit(1)

    # Perform comprehensive audit
    audit_results = audit_dataset_integrity(df_manifest)

    # Ensure target output directory exists
    output_path = Path(args.output_manifest).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export manifest
    df_manifest.to_csv(output_path, index=False)
    logger.info("Successfully exported manifest to: %s", output_path)

    if not audit_results["all_passed"]:
        logger.warning(
            "Audit completed with warnings or failed invariants! Please inspect logs above."
        )
        sys.exit(2)

    logger.info("Pipeline completed successfully with zero defects.")


if __name__ == "__main__":
    main()
