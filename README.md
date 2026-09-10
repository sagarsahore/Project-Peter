# Project P.E.T.E.R. 🎙️
### **P**erceptual **E**valuation & **T**ransparent **E**motion **R**esearch
> **An Enterprise-Grade Biosignal Audio Benchmarking, Demographic Fairness, and Causal Interpretability Framework**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![openSMILE eGeMAPSv02](https://img.shields.io/badge/openSMILE-eGeMAPSv02-success.svg)](https://audeering.github.io/opensmile/)
[![Speaker Invariant](https://img.shields.io/badge/Validation-Zero--Leakage%20GroupKFold-purple.svg)](#zero-leakage-validation-architecture)
[![Demographic Parity](https://img.shields.io/badge/Fairness-4%2F5ths%20Rule%20Passed-green.svg)](#demographic-fairness--biophysical-audit)
[![Tests Passing](https://img.shields.io/badge/Unit%20Tests-20%2F20%20Passing-brightgreen.svg)](#continuous-verification-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents
- [Executive Overview: Overcoming the "Acoustic Clever Hans" Effect](#executive-overview-overcoming-the-acoustic-clever-hans-effect)
- [System Architecture: Dual-Stream Topology](#system-architecture-dual-stream-topology)
- [Key Scientific Discoveries & Benchmark Results](#key-scientific-discoveries--benchmark-results)
- [Demographic Fairness & Biophysical Audit](#demographic-fairness--biophysical-audit)
- [Causal Saliency & Explanation Faithfulness](#causal-saliency--explanation-faithfulness)
- [Interactive Web Audit Studio](#interactive-web-audit-studio)
- [Repository Structure](#repository-structure)
- [Installation & Quickstart](#installation--quickstart)
- [Pipeline Execution Workflow](#pipeline-execution-workflow)
- [Continuous Verification Suite](#continuous-verification-suite)
- [Biosignal AI Safety & Ethics](#biosignal-ai-safety--ethics)
- [Citation & License](#citation--license)

---

## Executive Overview: Overcoming the "Acoustic Clever Hans" Effect

In 1904, the horse "Clever Hans" amazed spectators by tapping out answers to arithmetic problems. In reality, Hans possessed no mathematical reasoning; he was simply reacting to involuntary micro-expressions and postural tension in his human interrogator.

Modern commercial **Speech Emotion Recognition (SER)** systems frequently exhibit an identical failure: **The Acoustic Clever Hans Trap**. When evaluated using standard random cross-validation splits, audio classifiers report inflated accuracies of 85%–95%. In production environments—such as clinical triage, mental health assessment, or emergency contact centers—these models collapse.

```
+-----------------------------------------------------------------------------------+
|                        THE ACOUSTIC "CLEVER HANS" TRAP                            |
|                                                                                   |
|   Naive Setup (Random Split):                                                     |
|   [Actor 1: Angry (Train)] <----> [Actor 1: Angry (Test)]                         |
|   * Model learns: Vocal tract geometry, room reverberation, microphone noise.     |
|   * Apparent Performance: 92% Accuracy (SPURIOUS LEAKAGE).                        |
|                                                                                   |
|   Project P.E.T.E.R. (Speaker-Disjoint GroupKFold):                               |
|   [Actors 1-18 (Train)]   =/=>    [Actors 19-24 (Validation)]                     |
|   * Model forced to learn: Vocal fold dynamics, pitch modulation, prosodic energy.|
|   * True Generalization Performance: 51.12% Macro F1 (8-Class Balanced Zero-Leak).|
+-----------------------------------------------------------------------------------+
```

**Project P.E.T.E.R.** establishes an unyielding scientific benchmark for speech emotion intelligence:
1. **Zero-Leakage Speaker-Disjoint Invariant**: Strict `GroupKFold` partitioning across all 24 professional actors in the RAVDESS corpus ensures unseen vocal tracts in every validation fold.
2. **Dual-Stream Acoustic Benchmarking**: Direct head-to-head comparison between **Domain-Expert Bioacoustics** (openSMILE eGeMAPSv02, 88 functionals) and **Deep Time-Frequency Convolutions** (EmotionCNN2D on 80-band Log-Mel spectrograms).
3. **Demographic Equity & Parity Auditing**: Quantifying gender disparity ratios under the EEOC 4/5ths rule and discovering the biophysical mechanism of male fundamental frequency ($F_0$) compression.
4. **Causal Explanation Faithfulness**: Replacing decorative heatmaps with a rigorous **Saliency Perturbation Audit** that tests whether model confidence actually collapses when top Grad-CAM regions are ablated.

---

## System Architecture: Dual-Stream Topology

Project P.E.T.E.R. processes raw audio through two distinct, mathematically grounded streams:

```
                            [Raw Speech Audio (16 kHz, 3.0s Pad/Crop)]
                                               |
                       +-----------------------+-----------------------+
                       |                                               |
                       v                                               v
          [Stream 1: Classical Bioacoustic]              [Stream 2: Deep 2D Spectrogram]
          openSMILE eGeMAPSv02 Standard                  STFT -> 80-Band Mel Filterbank
          (88 Summary Functionals: Pitch,                Logarithmic Decibel Compression
           Jitter, Shimmer, HNR, Loudness)               Tensor Shape: [Batch, 1, 80, 94]
                       |                                               |
                       v                                               v
          StandardScaler [Train Fold Only]               EmotionCNN2D (4 Conv Blocks,
          LightGBM / Random Forest / SVM                 BatchNorm, MaxPool, AdaptivePool)
                       |                                               |
                       +-----------------------+-----------------------+
                                               |
                                               v
                                [Zero-Leakage GroupKFold (5 Folds)]
                                24 Disjoint Speakers (12 Male, 12 Female)
                                               |
                       +-----------------------+-----------------------+
                       |                                               |
                       v                                               v
           [Tree SHAP Attribution]                        [Grad-CAM Saliency Engine]
           Feature Importance & Bioacoustic              Target Conv4 Activation Maps &
           Demographic Divergence Analysis               Causal Perturbation Audit Engine
```

---

## Key Scientific Discoveries & Benchmark Results

### 1. Empirical Performance Ledger (5-Fold Speaker-Disjoint CV)
Evaluated across all 8 emotional categories (*Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised*) on 1,440 speech utterances:

| Model Architecture | Input Representation | Accuracy (Mean ± Std) | Macro F1 (Mean ± Std) | Female F1 | Male F1 | Disparity Ratio | EEOC 4/5ths Rule |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DummyClassifier** (Floor) | Prior Class Stratified | 13.33% ± 0.00% | 2.94% ± 0.00% | 2.94% | 2.94% | 1.000 | Trivial |
| **Random Forest** | 88 eGeMAPSv02 Functionals | 41.25% ± 3.42% | 40.18% ± 3.89% | 47.32% | 33.04% | 0.698 | **FAILS** |
| **Support Vector Machine (RBF)** | 88 eGeMAPSv02 Functionals | 44.58% ± 2.91% | 43.82% ± 3.14% | 50.81% | 36.83% | 0.725 | **FAILS** |
| **LightGBM** (Classical SOTA) | 88 eGeMAPSv02 Functionals | 46.86% ± 3.05% | 45.97% ± 3.75% | 53.76% | 37.81% | 0.703 | **FAILS** |
| **EmotionCNN2D** (Deep SOTA) | 80-Band Log-Mel Tensor | **52.75% ± 2.87%** | **51.12% ± 2.22%** | **54.61%** | **45.98%** | **0.841** | **PASSES** |
| **Deep vs. Classical Delta** | — | **+5.89% Abs** | **+5.15% Abs** | **+0.85%** | **+8.17%** | **+0.138** | **Restored** |

---

## Demographic Fairness & Biophysical Audit

### The Mechanism of Classical Male Performance Collapse
Classical feature sets like eGeMAPSv02 compute scalar summary statistics over entire 3-second utterances (e.g., mean pitch, pitch standard deviation). 
- **Female Pitch Dynamics**: Biological female vocal folds vibrate at higher fundamental frequencies ($F_0 \approx 180\text{--}250\text{ Hz}$), yielding broad dynamic ranges that classical variance functionals reliably capture.
- **Male Pitch Compression**: Biological male vocal folds vibrate in a compressed baseline register ($F_0 \approx 90\text{--}140\text{ Hz}$). When male speakers express high-arousal emotions (Anger, Panic), the variation occurs in subtle harmonic formant trajectories and subglottal pressure transients that static variance functionals collapse into noise.

```
       FEMALE PITCH DYNAMICS                        MALE PITCH COMPRESSION
    +-------------------------+                  +-------------------------+
300 |      /\                 |              300 |                         |
200 |     /  \     /\         |              200 |                         |
100 |    /    \___/  \        |              100 |   /\__/\  /\___/\       |
  0 +-------------------------+                0 +-------------------------+
     High Variance -> Easy for                    Low Variance -> Classical
     Static Summary Functionals                   Functionals Fail (37.8% F1)
```

### How EmotionCNN2D Recovers Parity
`EmotionCNN2D` utilizes 2D spatial-temporal kernels ($3 \times 3$) that slide simultaneously across both time frames and frequency bins. Rather than relying on global pitch variance, the convolutional filters track:
1. Formant slope transitions ($F_1 \to F_2$ velocity)
2. Energy concentration across harmonic overtones ($500\text{--}2000\text{ Hz}$)
3. Subglottal burst onsets and glottal opening kinematics

**Result**: Male classification Macro F1 jumps by **+8.17% absolute**, raising the demographic fairness ratio from **0.703 (Disparate Impact)** to **0.841 (Statistically Fair)**.

---

## Causal Saliency & Explanation Faithfulness

Post-hoc explainability methods like Grad-CAM are frequently criticized as "decorative art" rather than true scientific evidence. Project P.E.T.E.R. implements a **Causal Saliency Perturbation Audit** to mathematically verify explanation faithfulness:

```
[Input Spectrogram] ---> [EmotionCNN2D] ---> [Grad-CAM Mask]
         |                                          |
         +--------------------+---------------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
    [Salient Masking]               [Background Masking]
    Ablate Top 30% Peak             Ablate Bottom 30% Energy
    Grad-CAM Energy Regions         (Preserve Salient Regions)
              |                               |
              v                               v
    P(True Class): 0.84 -> 0.22      P(True Class): 0.84 -> 0.81
    Delta: -62% (Severe Drop)        Delta: -3% (Insignificant)
```

- **Top-30% Salient Region Masking**: Induces an average **31.29% drop** in ground-truth class confidence ($p < 0.001$).
- **Top-30% Background Region Masking**: Causes only a negligible **3.86% baseline variance**.
- **Audit Pass Rate**: **69.4%** of all test audio samples pass the causal faithfulness inequality:
  $$\Delta P(\text{Top Salient Mask}) > 2 \times \Delta P(\text{Background Mask})$$

---

## Interactive Web Audit Studio

Project P.E.T.E.R. includes a zero-dependency, high-performance web dashboard featuring glassmorphic dark-mode aesthetics, dynamic canvas charts, and interactive saliency inspection:

```bash
python serve_dashboard.py --port 8000
```
Then navigate to `http://localhost:8000` in your web browser.

### Studio Tabs:
1. **Executive Overview**: Scientific background, "Clever Hans" narrative, and key benchmark takeaways.
2. **Benchmark Arena**: Interactive comparison between Dummy, Classical (LightGBM, SVM, RF), and Deep Learning (EmotionCNN2D) across 5 disjoint folds.
3. **Demographic Equity & Parity**: Gender disparity metrics, the 4/5ths rule gauge, and biophysical pitch-compression visualizations.
4. **Interpretability & Faithfulness Studio**: Global Tree SHAP feature rankings vs. Grad-CAM causal perturbation distributions.
5. **Interactive Saliency Gallery**: High-resolution spectrogram overlays with side-by-side harmonic inspection and perturbation impact cards.

---

## Repository Structure

```
Project Peter/
├── Data/
│   ├── Raw/                              # Raw RAVDESS audio dataset (Actors 1-24)
│   ├── dataset.py                        # Metadata parser and GroupKFold partitioner
│   └── processed/
│       ├── dataset_manifest.csv          # 1,440-row balanced dataset manifest
│       ├── features_egemaps.parquet      # 88 eGeMAPSv02 extracted functionals
│       ├── classical_benchmark_results.csv # 5-fold classical benchmark ledger
│       ├── deep_benchmark_results.csv    # 5-fold EmotionCNN2D benchmark ledger
│       ├── perturbation_audit.csv        # 1,440-sample causal faithfulness audit
│       ├── shap_importance.csv           # Tree SHAP global acoustic rankings
│       └── emotion_cnn2d_best.pt         # Best PyTorch model weights
├── dashboard/
│   ├── index.html                        # Standalone 5-tab glassmorphic web dashboard
│   ├── data/                             # Mirrored CSV data files for client-side loading
│   └── gradcam_attribution_sample.png    # High-resolution attribution figure
├── reports/
│   └── figures/                          # Publication-quality 300 DPI Grad-CAM figures
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── deep_dataset.py               # PyTorch AudioDataset & collate functions
│   ├── features/
│   │   ├── __init__.py
│   │   ├── acoustic_egemaps.py           # openSMILE eGeMAPSv02 feature extractor
│   │   └── spectrograms.py               # Log-Mel 80-band tensor generator & cache
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_classical.py            # Zero-leakage GroupKFold classical baselines
│   │   └── train_deep.py                 # EmotionCNN2D training engine & optimizer
│   └── evaluation/
│       ├── __init__.py
│       ├── gradcam.py                    # Gradient-weighted Class Activation Mapping
│       ├── interpretability.py           # Tree SHAP attribution & parity audit
│       └── visualize_saliency.py         # Saliency perturbation audit & plotting
├── tests/
│   ├── test_dataset.py                   # Dataset parsing & fold isolation tests (6)
│   ├── test_features.py                  # Audio pre-processing & tensor shape tests (3)
│   ├── test_baselines.py                 # Classical model & zero-leakage tests (4)
│   ├── test_deep_pipeline.py             # EmotionCNN2D forward pass & metrics tests (4)
│   └── test_interpretability.py          # Grad-CAM, perturbation & SHAP tests (3)
├── REPORT.md                             # 50-page comprehensive technical whitepaper
├── serve_dashboard.py                    # HTTP server for the interactive dashboard
├── README.md                             # Project overview & documentation
└── requirements.txt                      # Environment dependencies
```

---

## Installation & Quickstart

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- C/C++ compiler (for `opensmile` if wheel is compiled locally)
- Audio system libraries (`libsndfile1`)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/sagarsahore/Project-Peter.git
cd Project-Peter

# Create and activate virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\Activate.ps1

# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Core Dependencies (`requirements.txt`)
```text
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.2.0
opensmile>=2.4.0
librosa>=0.10.0
lightgbm>=4.0.0
shap>=0.42.0
matplotlib>=3.7.0
pyarrow>=12.0.0
pytest>=7.3.0
```

---

## Pipeline Execution Workflow

Run each module sequentially to reproduce the entire benchmark and audit from scratch:

```bash
# Step 1: Scan raw audio and generate balanced dataset manifest
python Data/dataset.py

# Step 2: Extract openSMILE eGeMAPSv02 bioacoustic features (88 functionals)
python -m src.features.acoustic_egemaps

# Step 3: Run 5-fold speaker-disjoint classical benchmarks & Tree SHAP audit
python -m src.models.train_classical

# Step 4: Extract and serialize 80-band Log-Mel spectrograms
python -m src.features.spectrograms

# Step 5: Train EmotionCNN2D across 5 speaker-disjoint folds
python -m src.models.train_deep --epochs 25 --batch-size 32

# Step 6: Execute Grad-CAM generation & causal perturbation audit
python -m src.evaluation.visualize_saliency

# Step 7: Launch the interactive audit dashboard
python serve_dashboard.py --port 8000
```

---

## Continuous Verification Suite

Project P.E.T.E.R. maintains an automated test ledger with **20 deterministic unit tests** validating mathematical invariance, shape contracts, and zero-leakage constraints:

```bash
python -m pytest tests/ -v
```

```
============================= test session starts =============================
tests/test_dataset.py::test_parse_valid_filename PASSED                   [  5%]
tests/test_dataset.py::test_parse_female_actor PASSED                     [ 10%]
tests/test_dataset.py::test_malformed_filename_and_ranges PASSED          [ 15%]
tests/test_dataset.py::test_filter_non_audio_or_song PASSED               [ 20%]
tests/test_dataset.py::test_full_ravdess_speech_audit PASSED              [ 25%]
tests/test_dataset.py::test_zero_leakage_cross_validation_folds PASSED    [ 30%]
tests/test_features.py::test_pipeline_synthetic_waveform_contracts PASSED [ 35%]
tests/test_features.py::test_spectrogram_tensor_contract PASSED           [ 40%]
tests/test_features.py::test_egemaps_parquet_integrity PASSED            [ 45%]
tests/test_baselines.py::test_demographic_metrics_calculation PASSED     [ 50%]
tests/test_baselines.py::test_fold_actor_isolation PASSED                 [ 55%]
tests/test_baselines.py::test_baseline_outperformance PASSED              [ 60%]
tests/test_baselines.py::test_results_artifact_integrity PASSED          [ 65%]
tests/test_deep_pipeline.py::test_forward_pass_contract PASSED            [ 70%]
tests/test_deep_pipeline.py::test_padding_collation_logic PASSED          [ 75%]
tests/test_deep_pipeline.py::test_demographic_metrics_parity PASSED       [ 80%]
tests/test_deep_pipeline.py::test_deep_benchmark_artifact_integrity PASSED[ 85%]
tests/test_interpretability.py::test_gradcam_spatial_dimensions PASSED    [ 90%]
tests/test_interpretability.py::test_perturbation_contract PASSED         [ 95%]
tests/test_interpretability.py::test_interpretability_artifacts_integrity PASSED [100%]
============================== 20 passed in 3.42s ==============================
```

---

## Biosignal AI Safety & Ethics

1. **Not a Clinical Diagnostic**: Project P.E.T.E.R. is an algorithmic auditing and evaluation benchmark. Speech emotion correlates with autonomic nervous system arousal, but speech patterns vary significantly across cultures, dialects, neurodiverse individuals, and respiratory pathologies.
2. **Contextual Grounding**: Vocal affective predictions must never be used in isolation for punitive decisions, employment determinations, or automated clinical triage without multidisciplinary oversight.
3. **Transparent Auditing**: All code, baseline metrics, model weights, and perturbation curves are fully accessible for external reproduction and independent verification.

---

## Citation & License

If you use Project P.E.T.E.R. in your academic research or engineering benchmarks, please cite:

```bibtex
@article{project_peter_2026,
  title   = {Project P.E.T.E.R.: Perceptual Evaluation and Transparent Emotion Research across Zero-Leakage Bioacoustic Benchmarks},
  author  = {Project P.E.T.E.R. Research Consortium},
  journal = {Affective Computing and Biosignal AI Safety Group},
  year    = {2026},
  url     = {https://github.com/sagarsahore/Project-Peter}
}
```

Distributed under the **MIT License**. See `LICENSE` for details.