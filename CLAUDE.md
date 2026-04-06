# CLAUDE.md — Instructions for Claude Code

This file tells Claude Code how to work with this project.

## Project Overview

This is a research toolkit for exploring Meta's TRIBE v2 brain foundation
model and prototyping neural reward modeling for LLM alignment.

## Key Files

- `setup.sh` — Run this first. Installs everything.
- `EXPERIMENTS.md` — Step-by-step experiment guide (experiments 0-5).
- `scripts/tribe_wrapper.py` — Core wrapper around TRIBE v2. If imports fail,
  read `tribev2/demo_utils.py` and `tribev2/model.py` and fix the imports.
- `scripts/roi_extractor.py` — Maps brain parcels to cognitive dimensions.
- `scripts/neural_scorer.py` — Calibration MLP that maps dimensions to scores.
- `scripts/text_probes.py` — Experiment 1: probe text-only pathway.
- `scripts/mini_calibration.py` — Experiment 3: interactive rating + training.
- `scripts/preference_labeler.py` — Experiment 4: test agreement rate.

## Common Tasks

### "Help me get TRIBE v2 running"
1. Run `bash setup.sh`
2. If it fails on model download, check HuggingFace authentication
3. Run `python scripts/tribe_wrapper.py --test`
4. If imports fail, read the tribev2 repo structure and fix tribe_wrapper.py

### "Fix the TRIBE v2 imports"
The wrapper was written against the expected API from the HuggingFace model
card. If the actual repo has a different interface:
1. Read `tribev2/demo_utils.py` to find the inference function
2. Read `tribev2/model.py` to understand the model class
3. Update `TribeWrapper._load_model()` and `TribeWrapper.predict_text()`

### "Generate LLM responses for calibration"
The mini_calibration.py script needs varied-quality responses. Options:
1. Use a local model via transformers pipeline
2. Use an API (OpenAI, Anthropic, etc.)
3. Manually write/edit the placeholder responses
Help the user set up whichever approach they prefer.

### "Run experiment N"
Refer to EXPERIMENTS.md for detailed instructions per experiment.
Experiments must be run in order (0 → 1 → 2 → 3 → 4 → 5).

### "Visualize brain maps"
Use nilearn for brain visualization:
```python
from nilearn import plotting
plotting.plot_glass_brain(stat_map_img, title="Activation")
```
The activation arrays need to be mapped back to NIfTI format using
the Schaefer atlas. See roi_extractor.py for atlas loading.

## Important Notes

- TRIBE v2 is CC-BY-NC-4.0. This is research only.
- The text-only pathway may produce weaker signal than trimodal input.
  If experiments show weak signal, suggest adding TTS audio.
- The Schaefer parcel-to-network mapping in roi_extractor.py is approximate.
  After loading the actual atlas labels, refine the keyword matching.
- All results go in the `results/` directory.
- The calibration MLP is intentionally tiny (2K params) to prevent overfitting.
