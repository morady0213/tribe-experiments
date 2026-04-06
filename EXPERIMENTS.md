# Experiments Guide

Each experiment builds on the previous one. Don't skip ahead — the intuition
you build in early experiments is essential for interpreting later results.

---

## Experiment 0: Smoke Test (30 min)

**Goal:** Get TRIBE v2 running and understand the raw output shape.

**Run:**
```bash
python scripts/tribe_wrapper.py --test
```

**What to look for:**
- Output tensor shape: should be (T, N_parcels) where T = number of time steps
  and N_parcels ≈ 1000 (Schaefer parcellation) or ~70K (voxel-level)
- Values should be z-scored (mean ≈ 0, std ≈ 1 per parcel)
- Inference time on your GPU (expect ~0.3s per text input on A100)

**Questions to answer:**
1. What shape is the output? How many parcels/voxels?
2. What's the range of values? (should be roughly -3 to +3 if z-scored)
3. How long does a single text inference take?

**If it breaks:** Common issues:
- OOM: TRIBE v2 loads Llama 3.2 3B + Wav2Vec + V-JEPA. Text-only mode
  should work on 24GB by only loading the text encoder.
- Missing weights: Make sure you accepted the HuggingFace license.
- CUDA version mismatch: Check `nvidia-smi` matches your PyTorch CUDA.

---

## Experiment 1: Probe the Text-Only Pathway (2-3 hours)

**Goal:** Determine whether TRIBE v2 produces meaningfully different brain maps
for text that humans would perceive as different quality.

**Run:**
```bash
python scripts/text_probes.py
```

**What this does:**
Feeds 10 carefully designed text pairs through TRIBE v2 and compares their
predicted brain activation maps. Each pair contrasts one quality dimension:

| Pair | Dimension Tested | Text A | Text B |
|------|-----------------|--------|--------|
| 1 | Clarity | Clear explanation | Jargon-heavy version |
| 2 | Depth | Surface-level answer | Builds genuine intuition |
| 3 | Structure | Disorganized ramble | Well-structured argument |
| 4 | Accuracy | Subtly wrong facts | Factually precise |
| 5 | Sycophancy | Agrees with wrong premise | Respectfully corrects |
| 6 | Memorability | Forgettable generic text | Vivid metaphors + examples |
| 7 | Engagement | Dry encyclopedia style | Conversational + compelling |
| 8 | Concision | Bloated repetitive text | Tight and efficient |
| 9 | Formality | Overly casual/sloppy | Appropriately professional |
| 10 | Emotional tone | Cold and robotic | Warm and human |

For each pair, the script:
1. Runs both texts through TRIBE v2
2. Computes the difference map (activation_A - activation_B)
3. Saves a brain visualization showing where they diverge
4. Prints the top 10 parcels with largest absolute difference

**What to look for:**
- Do the difference maps look random (noise) or structured (specific brain regions)?
- Do clarity/depth pairs diverge in language areas (temporal cortex, angular gyrus)?
- Do engagement pairs diverge in attention networks (frontal eye fields, IPS)?
- Do sycophancy pairs diverge in prefrontal cortex (cognitive control)?

**This is the kill switch.** If all pairs produce nearly identical brain maps
(max difference < 0.1 z-score), the text-only pathway doesn't carry enough
signal for our use case. You'd need to add audio (TTS the text first) to get
multimodal input. If differences are > 0.3 in interpretable regions, proceed.

---

## Experiment 2: Visualize Cognitive Dimensions (1-2 hours)

**Goal:** Build the ROI extraction pipeline and see whether the 5 cognitive
dimensions we defined actually spread out across different text inputs.

**Run:**
```bash
python scripts/dimension_explorer.py
```

**What this does:**
1. Generates 100 text samples of varying quality using a local LLM (or
   pre-written samples in configs/sample_texts.json)
2. Runs each through TRIBE v2
3. Extracts 5 cognitive dimensions via ROI masks:
   - Comprehension (Default A+B networks → angular gyrus, lateral temporal)
   - Memory encoding (Limbic A+B → hippocampal/parahippocampal)
   - Sustained attention (Frontoparietal A + Dorsal Attention)
   - Confusion (Ventral Attention A + Salience network)
   - DMN suppression (negated Default C → medial PFC, precuneus)
4. Produces a 5-dimension scatter plot matrix
5. Computes inter-dimension correlations

**What to look for:**
- Do the 5 dimensions have independent variance? (correlation < 0.7 between dims)
- Do obviously good texts score high on comprehension + memory + attention?
- Do obviously bad texts score high on confusion + low on DMN suppression?
- Are any dimensions constant (no variance)? If so, drop them.

**Decision point:** After this experiment, you decide which dimensions to keep
in your final scoring function. Start with all 5, but drop any that:
- Have near-zero variance across inputs
- Are >0.8 correlated with another dimension
- Don't match your intuition at all

---

## Experiment 3: Mini Calibration (2-3 hours)

**Goal:** Train the calibration MLP and measure whether predicted neural
activations can actually predict your quality judgments.

**Run:**
```bash
python scripts/mini_calibration.py --interactive
```

**What this does:**
1. Presents you with 100 prompt-response pairs (one at a time)
2. You rate each on a 1-7 scale (takes ~30-45 minutes)
3. Runs all 100 through TRIBE v2 + ROI extraction
4. Trains the calibration MLP with cross-validation
5. Reports correlation between neural scores and your ratings

**Interpreting the correlation:**
- r < 0.2: No signal. The text-only pathway isn't capturing quality.
- r = 0.2-0.4: Weak but real signal. Worth exploring further with more data.
- r = 0.4-0.6: Solid signal. The core hypothesis holds.
- r > 0.6: Strong signal. Proceed with confidence.

**Important:** Your personal ratings are biased (everyone's are). This is fine
for a proof-of-concept. For the real research, you'd use crowd workers with
inter-annotator agreement. But if it can't even predict YOUR ratings,
it won't predict anyone's.

---

## Experiment 4: Preference Labeling Agreement (3-4 hours)

**Goal:** Test whether neural scoring agrees with human preferences on
pairwise comparisons (the actual task needed for DPO).

**Run:**
```bash
python scripts/preference_labeler.py --generate --n_pairs 50
```

**What this does:**
1. Generates 50 prompts
2. For each, generates 2 responses at different temperatures
3. Scores both with the neural scoring function
4. Labels the higher-scoring one as "chosen"
5. Presents the pairs to you (blinded — you don't see the neural scores)
6. You pick which response you prefer
7. Computes agreement rate: how often does the neural label match your choice?

**Interpreting agreement:**
- < 55%: No better than random. The neural signal isn't capturing preference.
- 55-65%: Above chance but noisy. Might work with margin filtering.
- 65-75%: Good. Comparable to inter-annotator agreement on many RLHF datasets.
- > 75%: Excellent. Strong enough to be a primary DPO signal.

---

## Experiment 5: One Round of Neural DPO (1-2 days)

**Goal:** Actually train a model using neural preference labels and see if
the output quality changes.

**Prerequisites:**
- Experiments 0-4 completed with positive results
- At least 4x A100 80GB (or 1x A100 80GB with gradient accumulation)
- ~$50-100 in cloud GPU budget

**Run:**
```bash
# Step 1: Generate neural preference dataset (2-3 hours on 1x A100)
python scripts/neural_dpo.py --generate --n_pairs 5000

# Step 2: Train with DPO (4-8 hours on 4x A100)
python scripts/neural_dpo.py --train

# Step 3: Evaluate (30 min)
python scripts/neural_dpo.py --evaluate --n_samples 50
```

**What to look for:**
- Does the DPO-trained model produce different outputs than the base model?
- On the 50 evaluation samples, which do you prefer?
- Run MT-Bench if possible — did benchmark scores change?
- Check for degeneration: is the model still coherent? (DPO can overfit)

---

## What to Do With Results

After completing experiments 0-4 (total: ~1 week, ~$0-50):

**If results are positive** (correlation > 0.3, agreement > 60%):
→ You have a viable research direction. Write up findings, prepare demo.
→ Proceed to Experiment 5 for the full training loop.
→ Start thinking about who to collaborate with.

**If results are mixed** (some dimensions work, others don't):
→ Focus on the dimensions that work. Maybe only comprehension + attention
   carry signal, and memory encoding doesn't work for text-only.
→ Try adding audio (use TTS to convert text to speech, then feed audio +
   text through TRIBE v2 for trimodal input).

**If results are negative** (correlation < 0.1, agreement ≈ 50%):
→ The text-only pathway doesn't carry enough cognitive signal. This is a
   valid negative result — worth documenting.
→ Pivot to trimodal input (TTS + image generation + text) which will
   activate more of TRIBE v2's trained capacity.
→ Or pivot to the content evaluation tool idea (might work even without
   the reward modeling application).
