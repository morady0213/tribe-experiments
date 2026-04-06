# Neural Reward Signals from Brain Foundation Models

**Can predicted fMRI activations augment RLHF reward modeling with richer, multi-dimensional feedback?**

This project explores Meta's TRIBE v2 brain foundation model as a source of neuroscience-principled quality dimensions for LLM alignment. Instead of augmenting binary preference labels with hand-designed rubrics, we let brain network organization define the dimensions.

---

## Key Findings

| Finding | Result |
|---------|--------|
| Brain detects sycophancy (no labels, no training) | **100% accuracy** |
| Brain detects clarity differences | **100% accuracy** |
| Two independent quality axes discovered | Comprehension (d=1.35) + Confusion (d=-2.11, r=-0.14) |
| Signal is not length-driven | Quality explains 73% of variance after length control |
| Perfect reproducibility | r = 1.0000 across runs |

---

## The Core Idea

Standard RLHF compresses human judgment into a single bit: *A or B*.

Multi-dimensional approaches like [ArmoRM](https://arxiv.org/abs/2406.12845) (19 dims) and [SteerLM](https://arxiv.org/abs/2310.05344) (5 dims) improve on this with hand-designed rubrics.

This project takes a different approach: let brain networks — which evolved over millions of years to evaluate information — define the quality dimensions. Two signals emerged that don't exist in standard reward rubrics:

1. **Comprehension axis** — how deeply the brain processes the text's meaning
2. **Confusion axis** — whether the brain's error-detection network activates

These are **orthogonal** (r = -0.14), measuring genuinely different aspects of quality.

---

## How It Works

```
Text Input
    |
gTTS (Text-to-Speech)
    |
LLaMA 3.2 3B   +   Wav2Vec-BERT 2.0
(text features)     (audio features)
         |
      TRIBE v2
   (Transformer)
         |
  20,484 predicted fMRI activations
         |
  Schaefer Atlas (1000 parcels)
         |
  5 Cognitive Dimensions
  +------------------------------------+
  | C: Comprehension  (Default A+B)    |
  | M: Memory         (Limbic)         |
  | A: Attention      (Frontoparietal) |
  | X: Confusion      (Ventral Attn)   |
  | D: DMN Suppression(-Default C)     |
  +------------------------------------+
         |
  Reward Signal for RLHF
```

---

## Experiments

### Experiment 0: Smoke Test
Verified pipeline end-to-end. All models download and run correctly.

### Experiment 1: Text Quality Probes
10 text pairs testing one quality dimension each.

**Signal strength (max activation divergence):**

| Probe | Max Diff | Signal |
|-------|----------|--------|
| Sycophancy | 0.442 | Strong |
| Engagement | 0.344 | Strong |
| Formality | 0.323 | Strong |
| Clarity | 0.339 | Strong |
| Depth | 0.304 | Strong |
| Memorability | 0.314 | Strong |
| Concision | 0.243 | Moderate |
| Emotional warmth | 0.176 | Weak |
| Accuracy | 0.174 | Weak |
| Structure | 0.138 | Weak |

Failures are neuroscience-consistent: structure requires paragraph-level perception, accuracy requires world knowledge, emotional warmth requires prosody.

### Reproducibility Test
- Same text run 3x: correlation = 1.000000, max diff = 0.0
- Length control: quality explains 73% of signal variance after matching word counts

### Experiment 2: Dimension Explorer
30 texts across 3 quality levels. Discovered the two independent axes.

Correlation matrix:
```
              Comp    Mem    Attn   Conf   DMN
Comp          1.00
Mem           0.89   1.00
Attn          0.59   0.68   1.00
Conf         -0.14  -0.11   0.54   1.00      <- Confusion is INDEPENDENT
DMN          -0.84  -0.66  -0.69  -0.23   1.00
```

### Experiment 3: Reward Augmentation Test
30 human-rated prompt-response pairs across 6 categories.

**Brain-as-Judge accuracy (no training, no labels):**

| Category | Accuracy |
|----------|----------|
| Sycophancy | 100% |
| Clarity | 100% |
| Depth | 80% |
| Mixed | 80% |
| Coherence | 40% |
| Accuracy | 20% |
| **Overall** | **70%** |

---

## Comparison to SOTA

| Approach | Dimensions | Source | Speed |
|----------|-----------|--------|-------|
| ArmoRM | 19 | Hand-designed + GPT-4 | ~50ms |
| SteerLM | 5 | Hand-designed + human | ~50ms |
| **Ours** | **5 (2 independent)** | **Brain networks** | **~90s** |

**Novel angle:** Dimension *discovery* via neuroscience, not design. The Confusion dimension (orthogonal error detection) doesn't appear in any hand-designed reward rubric.

---

## How Brain Dims Augment RLHF

```
Standard RLHF:
  Human labels (binary) -> Reward model -> PPO/DPO -> LLM

Brain-Augmented RLHF:
  Human labels (binary)  |
  Comprehension score    +-> Multi-dim reward model -> PPO/DPO -> LLM
  Confusion score        |
  Memory score           |
```

The brain dimensions add signal that hand-designed rubrics may miss — especially the Confusion axis for detecting subtle sycophancy and hidden incoherence.

---

## Setup

### Requirements
- Python 3.11, CUDA GPU (~12GB VRAM), ~8GB disk for weights

### Installation

```bash
git clone https://github.com/morady0213/tribe-experiments
cd tribe-experiments

python -m venv tribe_env
tribe_env\Scripts\activate  # Windows

pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

pip install transformers accelerate nilearn scipy scikit-learn matplotlib gtts spacy
python -m spacy download en_core_web_lg
```

### Run Experiments

```bash
python scripts/tribe_wrapper.py --test                          # Smoke test
python scripts/text_probes.py                                   # Experiment 1
python scripts/dimension_explorer.py                            # Experiment 2

python scripts/experiment3_augmentation.py --generate           # Step 1
python scripts/experiment3_augmentation.py --infer              # Step 2 (~1.5 hrs)
python scripts/experiment3_augmentation.py --rate               # Step 3 (interactive)
python scripts/experiment3_augmentation.py --analyze            # Step 4

python scripts/generate_hero_visuals.py                         # Figures
```

---

## Project Structure

```
tribe-experiments/
├── scripts/
│   ├── tribe_wrapper.py              # TRIBE v2 interface
│   ├── roi_extractor.py              # Schaefer atlas -> 5 dimensions
│   ├── text_probes.py                # Experiment 1
│   ├── neural_scorer.py              # Calibration MLP
│   ├── dimension_explorer.py         # Experiment 2
│   ├── reproducibility_test.py       # Validation
│   ├── experiment3_augmentation.py   # Experiment 3
│   └── generate_hero_visuals.py      # Publication figures
├── results/                          # All JSON outputs
├── visualizations/                   # All figures
├── TECHNICAL_REPORT.md               # Complete deep-dive
├── MEDIUM_ARTICLE.md                 # Full writeup
└── README.md
```

---

## Limitations

- **Speed:** 90s/text vs 50ms for ArmoRM — research only
- **Text-only:** One of three modalities TRIBE supports
- **Sample size:** 30 pairs is exploratory, not conclusive

---

## License

Code: MIT | TRIBE v2 weights: CC-BY-NC-4.0 (Meta FAIR, research only)

---

*Built with TRIBE v2 (Meta FAIR) | Schaefer 2018 atlas | Python 3.11 | RTX 4070*

## What This Is

This repo contains everything you need to:
1. Get TRIBE v2 running and understand its outputs
2. Probe how it responds to different text quality levels
3. Build ROI extraction and neural scoring functions
4. Run a mini calibration experiment
5. Generate neural preference labels for DPO

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (A100 80GB recommended, but experiments 0-3 work on 24GB)
- ~50GB disk space (for model weights)
- Claude Code (for agentic acceleration)

## Quick Start with Claude Code

Open your terminal in this directory and tell Claude Code:

```
"Read the README.md and setup.sh, then help me get TRIBE v2 running.
Start with Experiment 0 from the experiments guide."
```

Claude Code will:
1. Run setup.sh to install everything
2. Download TRIBE v2 weights from HuggingFace
3. Run the first inference test
4. Walk you through each experiment

## Project Structure

```
tribe-experiments/
├── README.md                  # You are here
├── setup.sh                   # One-shot environment setup
├── EXPERIMENTS.md             # Detailed guide for experiments 0-5
├── scripts/
│   ├── tribe_wrapper.py       # Clean interface around TRIBE v2
│   ├── roi_extractor.py       # Schaefer atlas → cognitive dimensions
│   ├── neural_scorer.py       # Calibration MLP + scoring function
│   ├── text_probes.py         # Experiment 1: probe text-only pathway
│   ├── dimension_explorer.py  # Experiment 2: visualize cognitive dims
│   ├── mini_calibration.py    # Experiment 3: calibrate on your ratings
│   ├── preference_labeler.py  # Experiment 4: generate neural preference labels
│   ├── neural_dpo.py          # Experiment 5: one round of Neural DPO
│   └── visualize_brain.py     # Brain map visualization utilities
├── configs/
│   ├── roi_networks.yaml      # Schaefer parcel → Yeo network mappings
│   └── experiment_config.yaml # Hyperparameters for all experiments
├── results/                   # Auto-populated by experiments
└── visualizations/            # Brain maps and plots
```

## License Note

TRIBE v2 is CC-BY-NC-4.0. Everything in this repo is for research/experimentation only.
Do not use TRIBE v2 weights in any commercial product or service.
```
