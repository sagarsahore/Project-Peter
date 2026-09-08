"""Unit tests for RAVDESS speech dataset parsing and validation module.

Project P.E.T.E.R.
Affective Computing & Biosignals Pipeline
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Data.dataset import (
    EXPECTED_ACTOR_COUNT,
    EXPECTED_FEMALE_ACTORS,
    EXPECTED_MALE_ACTORS,
    EXPECTED_NEUTRAL_COUNT,
    EXPECTED_NON_NEUTRAL_COUNT,
    EXPECTED_TOTAL_SPEECH_CLIPS,
    Emotion,
    EmotionalIntensity,
    Modality,
    Statement,
    VocalChannel,
    assign_zero_leakage_folds,
    audit_dataset_integrity,
    parse_filename,
    parse_ravdess_speech,
)


class TestRavdessDatasetParser(unittest.TestCase):
    """Test suite for RAVDESS parsing, validation, and zero-leakage assignment."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_parse_valid_filename(self) -> None:
        """Verify parsing of standard conforming filename."""
        wav_file = self.base_path / "03-01-05-02-01-01-13.wav"
        wav_file.write_bytes(b"RIFFdummywavdata")

        record = parse_filename(wav_file)
        self.assertIsNotNone(record)
        self.assertEqual(record["modality"], "03")
        self.assertEqual(record["vocal_channel"], "01")
        self.assertEqual(record["emotion_name"], "angry")
        self.assertEqual(record["intensity_level"], "strong")
        self.assertEqual(record["statement_text"], "Kids are talking by the door")
        self.assertEqual(record["repetition_id"], 1)
        self.assertEqual(record["actor_id"], 13)
        self.assertEqual(record["speaker_id"], "Actor_13")
        self.assertEqual(record["gender"], "male")  # odd = male
        self.assertEqual(record["file_size_bytes"], len(b"RIFFdummywavdata"))

    def test_parse_female_actor(self) -> None:
        """Verify actor gender mapping for even actor IDs."""
        wav_file = self.base_path / "03-01-03-01-02-02-24.wav"
        wav_file.write_bytes(b"RIFFfemalevoice")

        record = parse_filename(wav_file)
        self.assertIsNotNone(record)
        self.assertEqual(record["actor_id"], 24)
        self.assertEqual(record["gender"], "female")  # even = female
        self.assertEqual(record["emotion_name"], "happy")
        self.assertEqual(record["intensity_level"], "normal")
        self.assertEqual(record["statement_text"], "Dogs are sitting by the door")
        self.assertEqual(record["repetition_id"], 2)

    def test_filter_non_audio_or_song(self) -> None:
        """Verify strict exclusion of non-audio-only or song channels."""
        # Video modality (01)
        video_wav = self.base_path / "01-01-01-01-01-01-01.wav"
        video_wav.write_bytes(b"data")
        self.assertIsNone(parse_filename(video_wav))

        # Song channel (02)
        song_wav = self.base_path / "03-02-01-01-01-01-01.wav"
        song_wav.write_bytes(b"data")
        self.assertIsNone(parse_filename(song_wav))

    def test_malformed_filename_and_ranges(self) -> None:
        """Verify handling of invalid tokens, out-of-range values, and corrupt files."""
        # Bad token format
        bad_name = self.base_path / "invalid_audio_file.wav"
        bad_name.write_bytes(b"data")
        self.assertIsNone(parse_filename(bad_name))

        # Invalid emotion (99)
        bad_emo = self.base_path / "03-01-99-01-01-01-01.wav"
        bad_emo.write_bytes(b"data")
        self.assertIsNone(parse_filename(bad_emo))

        # Invalid actor (25)
        bad_act = self.base_path / "03-01-01-01-01-01-25.wav"
        bad_act.write_bytes(b"data")
        self.assertIsNone(parse_filename(bad_act))

        # Zero-byte empty file
        empty_wav = self.base_path / "03-01-01-01-01-01-01.wav"
        empty_wav.write_bytes(b"")
        self.assertIsNone(parse_filename(empty_wav))

    def test_zero_leakage_cross_validation_folds(self) -> None:
        """Verify cross-validation fold assignments guarantee zero speaker leakage."""
        manifest_path = Path("Data/processed/dataset_manifest.csv")
        if not manifest_path.exists():
            self.skipTest("Manifest file not generated yet")

        df = pd.read_csv(manifest_path)
        self.assertIn("fold", df.columns)

        # Check that no speaker appears in more than 1 fold
        speaker_folds = df.groupby("actor_id")["fold"].nunique()
        self.assertTrue(
            (speaker_folds == 1).all(),
            "Speaker leakage detected! At least one actor is assigned to multiple folds.",
        )

        # Check demographic balance in folds
        fold_gender_dist = df.groupby(["fold", "gender"])["actor_id"].nunique().unstack()
        self.assertTrue((fold_gender_dist["male"] > 0).all())
        self.assertTrue((fold_gender_dist["female"] > 0).all())

    def test_full_ravdess_speech_audit(self) -> None:
        """Test full parsing and auditing against the actual local dataset."""
        raw_dir = Path("Data/Raw")
        if not raw_dir.exists():
            self.skipTest("Data/Raw does not exist")

        df = parse_ravdess_speech(raw_dir)
        audit_results = audit_dataset_integrity(df)

        self.assertTrue(audit_results["all_passed"])
        self.assertEqual(audit_results["total_clips"], EXPECTED_TOTAL_SPEECH_CLIPS)
        self.assertEqual(audit_results["unique_actors"], EXPECTED_ACTOR_COUNT)
        self.assertEqual(audit_results["male_actors"], EXPECTED_MALE_ACTORS)
        self.assertEqual(audit_results["female_actors"], EXPECTED_FEMALE_ACTORS)
        self.assertEqual(audit_results["emotion_counts"]["neutral"], EXPECTED_NEUTRAL_COUNT)

        for emo in ["calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]:
            self.assertEqual(audit_results["emotion_counts"][emo], EXPECTED_NON_NEUTRAL_COUNT)


if __name__ == "__main__":
    unittest.main()
