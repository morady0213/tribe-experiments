# LinkedIn Post

## THE HOOK (Use this — short, punchy, human)

---

I ran AI responses through a brain model to see if it could tell when the AI was just telling people what they want to hear.

It could. 100% accuracy.

Here's what I built — and what it means for how we train AI.

---

When you rate an AI response as "good" or "bad", you're compressing something complex into a single bit.

Did you prefer it because it was clear? Accurate? Honest? Because it agreed with you even when it shouldn't have?

That one bit — A or B — is what most RLHF systems are trained on.

I wanted to see if brain activity could give us a richer signal.

---

**The setup:**

Meta released TRIBE v2 — a model trained on real fMRI data that predicts brain activations from text.

Feed it a sentence. It outputs predicted neural firing across 20,484 points on the cortical surface.

I mapped those activations to 5 cognitive dimensions from brain science:
- Comprehension (Default Network)
- Memory encoding (Limbic)
- Attention (Frontoparietal)
- Confusion (Ventral Attention — the brain's error detector)
- DMN suppression (are you engaged, or daydreaming?)

Then I tested: can these dimensions tell good AI responses from bad ones?

---

**What I found:**

Two independent signals emerged — not one.

Comprehension and Confusion are uncorrelated (r = -0.14). They're measuring different things.

When tested on 30 human-rated response pairs:

- Sycophancy detection: **100% accuracy**
- Clarity detection: **100% accuracy**
- Depth detection: **80% accuracy**
- Factual accuracy: **20% accuracy** (expected — the brain doesn't fact-check)

The failure is as informative as the success. The brain model couldn't detect factual errors because it predicts *perception*, not *knowledge*. That's not a bug — it's neuroscience.

---

**Why this matters for RLHF:**

The goal isn't to replace reward models. It's to augment them.

Current approaches like ArmoRM use 19 hand-designed dimensions (helpfulness, coherence, verbosity...). They work well. But the dimensions were picked by researchers guessing what matters.

Brain-derived dimensions come from how neural networks evolved to process information. The Confusion signal — independently detecting when something seems wrong — doesn't exist in any standard reward model.

Could adding 2-3 brain dimensions to existing reward models improve sycophancy detection on hard cases where text-based features look fine? That's the real question.

I don't have a definitive answer yet. But 100% accuracy on sycophancy pairs with a model that had no labels, no training, just brain network math — that's worth taking seriously.

---

**The honest part:**

This is exploratory research, not a production system.

3 minutes per text vs 50ms for ArmoRM. Not practical at scale.

30 pairs is a small sample.

But this is the kind of cross-domain thinking that occasionally finds something real. Neuroscience → ML alignment is a direction worth exploring.

Full code, results, and methodology on GitHub (link in comments).

---

*TRIBE v2 by Meta FAIR • Schaefer 2018 atlas • 4-model pipeline: LLaMA 3.2 3B + Wav2Vec-BERT + TRIBE v2 + CalibrationMLP • RTX 4070*

---

## IMAGE ORDER FOR THE POST:
1. hero_brain_sycophancy.png — Lead with this (the brain image is the hook)
2. hero_two_axes.png — The discovery
3. hero_category_results.png — The honest results
4. hero_rlhf_augmentation.png — Why it matters

## POSTING TIPS:
- Post on a Tuesday or Wednesday morning (10am Cairo time = 8am UTC)
- First comment: link to GitHub + Medium article
- Tags: #MachineLearning #RLHF #AI #Neuroscience #LLM #AIAlignment #Research
- Don't edit the post after posting (kills reach in first hour)

---

## REDDIT VERSION

### Title options (pick one based on subreddit):
- **r/MachineLearning**: `[Project] I used Meta's TRIBE v2 brain model to detect AI sycophancy — 100% accuracy with zero training`
- **r/artificial**: `I ran AI responses through a brain scanner model. It could tell when the AI was just agreeing with you — 100% of the time.`
- **r/LLMAlignment**: `Brain-derived reward signals: using predicted fMRI activations to detect sycophancy in LLM responses`

### Post body (r/MachineLearning):

**[Project] Brain-derived reward signals for LLM alignment — detecting sycophancy with predicted fMRI**

TL;DR: Used Meta's TRIBE v2 (brain foundation model) to predict neural activations from AI responses, mapped them to 5 cognitive dimensions, and tested whether these could discriminate response quality. Sycophancy detection: 100% accuracy with no labels, no training.

---

**Motivation**

Standard RLHF compresses human judgment into a single binary bit (A > B). This loses the *reason* for preference. A response can look fluent, confident, and helpful — and still be sycophantic. Text-based reward models struggle with this because sycophantic text and honest text look similar on the surface.

Neuroscience has a different angle: the brain processes sycophancy vs honesty differently at the network level. The Ventral Attention Network activates when something seems wrong. The Default Mode Network drives deep semantic processing. These are independent axes.

**Method**

4-model pipeline:
1. LLaMA 3.2 3B → text embeddings
2. Wav2Vec-BERT → prosody features (via TTS simulation)
3. TRIBE v2 → predicted fMRI activations (20,484 fsaverage5 vertices)
4. CalibrationMLP → 5 cognitive dimension scores

Schaefer 2018 atlas maps activations to networks:
- Comprehension = Default A + B parcels
- Memory = Limbic
- Attention = Frontoparietal + Dorsal Attention
- Confusion = Ventral Attention (error detection)
- DMN Suppression = negative Default C (engagement proxy)

Tested on 30 hand-rated prompt-response pairs across 6 categories.

**Results**

| Category | Brain-as-Judge Accuracy |
|---|---|
| Sycophancy | 100% |
| Clarity | 100% |
| Depth | 80% |
| Coherence | 60% |
| Factual accuracy | 20% |
| Mixed | 60% |
| **Overall** | **70%** |

The failure on factual accuracy is expected and informative: the brain model predicts *perception*, not *ground truth*. A fluent false statement activates comprehension just as well as a fluent true one.

The two key dimensions — Comprehension (effect size d=1.35) and Confusion (d=2.11) — are nearly uncorrelated (r=-0.14), suggesting they capture independent quality axes.

**Limitations**

- n=30 pairs, single rater for most categories
- 3 min/text inference time (vs 50ms for ArmoRM)
- Augmented logistic regression showed no improvement over baseline at n=30 (majority class problem)
- Text-only pathway — trimodal TRIBE input (text+audio+image) would likely perform better

**Code + full writeup**: [GitHub link] | [Medium article](https://medium.com/p/5b717488071d)

Happy to answer questions on methodology, the TRIBE model, or the ROI mapping approach.

---

## TWITTER/X THREAD

**Tweet 1 (hook):**
I ran AI responses through a brain model.

It could tell when the AI was just telling you what you want to hear.

100% accuracy. Zero training. No labels.

Here's how 🧵

---

**Tweet 2:**
The problem with RLHF:

When you rate a response "good" or "bad", you compress everything into 1 bit.

Was it good because it was clear? Accurate? Or just because it agreed with you?

That 1 bit is what reward models train on.

---

**Tweet 3:**
Meta released TRIBE v2 — a model trained on real fMRI data.

Feed it text → it predicts brain activations across 20,484 points on the cortex.

I mapped those to 5 cognitive dimensions from neuroscience:
• Comprehension
• Memory encoding
• Attention
• Confusion (error detection)
• Engagement

---

**Tweet 4:**
Then I tested: can brain activations tell good AI responses from bad ones?

Results on 30 rated pairs:

✅ Sycophancy: 100%
✅ Clarity: 100%
🟡 Depth: 80%
❌ Factual accuracy: 20%

The failure is as interesting as the success.

---

**Tweet 5:**
Why did it fail on facts?

The brain model predicts *perception*, not *knowledge*.

A fluent false statement activates comprehension just as well as a fluent true one.

That's not a bug — it's neuroscience.

---

**Tweet 6:**
Two independent signals emerged:

Comprehension and Confusion are nearly uncorrelated (r = -0.14)

They're measuring different things. The brain has separate systems for "do I understand this" and "does something seem wrong"

Current reward models don't have this separation.

---

**Tweet 7:**
This isn't a replacement for RLHF.

It's a missing dimension.

ArmoRM uses 19 hand-designed quality axes. Brain-derived dimensions come from how neural networks evolved to process language.

The Confusion signal (error detection) doesn't exist in any standard reward model.

---

**Tweet 8:**
Honest limitations:
• 30 pairs only
• 3 min/text (not production-ready)
• Augmented model showed no improvement at n=30

This is exploratory. But 100% sycophancy detection with zero labels is worth investigating.

---

**Tweet 9:**
Full code, results, methodology:
🔗 GitHub: [link]
📄 Medium: medium.com/p/5b717488071d

Built with: TRIBE v2 (Meta FAIR) • LLaMA 3.2 3B • Schaefer atlas • RTX 4070

If you work on RLHF or alignment, I'd love to hear your thoughts.
