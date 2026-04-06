"""
Experiment 3: Mini Calibration

Interactive experiment where you rate 100 prompt-response pairs,
then train the calibration MLP on your ratings.

Usage:
    python mini_calibration.py --interactive    # Rate + train
    python mini_calibration.py --from-file ratings.json  # Use existing ratings
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from tribe_wrapper import TribeWrapper
from roi_extractor import ROIExtractor
from neural_scorer import NeuralScorer

# Sample prompts for calibration. Designed to elicit varied-quality responses.
CALIBRATION_PROMPTS = [
    "Explain how the internet works to a 10-year-old.",
    "What causes seasons on Earth?",
    "How do computers store information?",
    "Why do we dream?",
    "How does GPS know where you are?",
    "What makes a rainbow appear?",
    "How do antibiotics work?",
    "Explain compound interest.",
    "Why is the sky blue?",
    "How does a refrigerator keep food cold?",
    "What causes earthquakes?",
    "How do planes fly?",
    "Why do we need sleep?",
    "How does a search engine find results so fast?",
    "What makes ice slippery?",
    "How do solar panels generate electricity?",
    "Why can't you tickle yourself?",
    "How does noise-canceling work in headphones?",
    "What causes ocean tides?",
    "How does your phone know which way is up?",
]


def generate_varied_responses(prompts: list[str], n_per_prompt: int = 5) -> list[dict]:
    """Generate responses of varying quality for calibration.

    Strategy: Use different "quality levels" to simulate what different
    LLMs or temperatures would produce. In practice, you'd use actual
    LLM calls — this generates templates for manual editing or LLM generation.
    """
    samples = []

    for prompt in prompts:
        for quality_level in range(n_per_prompt):
            sample = {
                "prompt": prompt,
                "response": f"[PLACEHOLDER — Generate a quality-{quality_level+1}/5 response to: '{prompt}']",
                "expected_quality": quality_level + 1,
                "rating": None,  # To be filled by human
            }
            samples.append(sample)

    return samples


def generate_with_llm_api(prompts: list[str]) -> list[dict]:
    """Generate actual varied-quality responses using LLM API calls.

    This attempts to use a local LLM or API. If unavailable, falls back
    to placeholder text. Claude Code can help you connect to your
    preferred LLM API.
    """
    samples = []

    # Quality-inducing system prompts
    quality_prompts = {
        1: "Answer very briefly and vaguely. Be as unhelpful as possible while still answering.",
        2: "Give a mediocre answer. Include some correct info but be disorganized and repetitive.",
        3: "Give a decent answer. Clear but not particularly insightful or memorable.",
        4: "Give a good answer. Well-structured, accurate, with a helpful example.",
        5: "Give an excellent answer. Build genuine intuition with vivid metaphors, precise yet accessible. Make it memorable.",
    }

    try:
        # Try using transformers for local generation
        from transformers import pipeline
        generator = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct",
                           device_map="auto", max_new_tokens=200)

        for prompt in prompts:
            for quality, system in quality_prompts.items():
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
                output = generator(messages, max_new_tokens=200)
                response = output[0]["generated_text"][-1]["content"]

                samples.append({
                    "prompt": prompt,
                    "response": response,
                    "intended_quality": quality,
                    "rating": None,
                })

        print(f"Generated {len(samples)} responses using local LLM")
        return samples

    except Exception as e:
        print(f"Local LLM not available ({e})")
        print("Falling back to placeholder responses.")
        print("To generate real responses, tell Claude Code:")
        print('  "Help me generate varied-quality LLM responses for calibration"')
        return generate_varied_responses(prompts)


def interactive_rating(samples: list[dict], start_idx: int = 0) -> list[dict]:
    """Present samples one at a time for human rating."""

    print("\n" + "=" * 60)
    print("  INTERACTIVE RATING SESSION")
    print("=" * 60)
    print("\nRate each response on a 1-7 scale:")
    print("  1 = Terrible (wrong, confusing, unhelpful)")
    print("  2 = Bad")
    print("  3 = Below average")
    print("  4 = Average (acceptable but not good)")
    print("  5 = Good")
    print("  6 = Very good")
    print("  7 = Excellent (clear, insightful, memorable)")
    print("\nType 'q' to quit (progress is saved), 's' to skip")
    print()

    rated_count = 0
    for i, sample in enumerate(samples):
        if i < start_idx:
            continue
        if sample["rating"] is not None:
            continue

        print(f"\n--- Sample {i+1}/{len(samples)} ---")
        print(f"PROMPT: {sample['prompt']}")
        print(f"\nRESPONSE:\n{sample['response']}")
        print()

        while True:
            try:
                inp = input("Rating (1-7, q=quit, s=skip): ").strip().lower()
                if inp == "q":
                    print(f"\nRated {rated_count} samples this session.")
                    return samples
                if inp == "s":
                    break
                rating = int(inp)
                if 1 <= rating <= 7:
                    sample["rating"] = rating
                    rated_count += 1
                    break
                else:
                    print("Please enter 1-7")
            except ValueError:
                print("Please enter a number 1-7")

    print(f"\nRated {rated_count} samples this session.")
    return samples


def train_on_ratings(
    wrapper: TribeWrapper,
    samples: list[dict],
    save: bool = True,
) -> tuple:
    """Run TRIBE v2 on rated samples and train the calibration MLP."""

    # Filter to rated samples
    rated = [s for s in samples if s["rating"] is not None]
    print(f"\nTraining on {len(rated)} rated samples...")

    if len(rated) < 20:
        print("WARNING: Very few samples. Need at least 20 for meaningful cross-validation.")
        print("Continue rating to get better results.")

    # Extract features
    print("Running TRIBE v2 inference...")
    extractor = None
    features_list = []

    for i, sample in enumerate(rated):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i+1}/{len(rated)}...")

        full_text = f"Question: {sample['prompt']}\n\nAnswer: {sample['response']}"
        activation = wrapper.predict_text(full_text)

        if extractor is None:
            extractor = ROIExtractor(n_rois=len(activation))

        dims = extractor.extract(activation)
        features_list.append(dims.to_array())

    features = np.stack(features_list)
    ratings = np.array([s["rating"] for s in rated], dtype=np.float32)

    # Train
    print("\nTraining calibration MLP...")
    scorer = NeuralScorer()
    scorer.roi_extractor = extractor
    result = scorer.train(features, ratings)

    if save:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)

        # Save scorer
        scorer.save(os.path.join(results_dir, "calibration_scorer.pt"))

        # Save results
        with open(os.path.join(results_dir, "experiment_3_results.json"), "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "n_rated": len(rated),
                "train_correlation": result.train_correlation,
                "val_correlation": result.val_correlation,
                "val_r2": result.val_r2,
                "feature_importance": result.feature_importance,
            }, f, indent=2)

    return scorer, result


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Mini Calibration")
    parser.add_argument("--interactive", action="store_true", help="Interactive rating session")
    parser.add_argument("--from-file", type=str, help="Load ratings from JSON file")
    parser.add_argument("--generate", action="store_true", help="Generate responses only (no rating)")
    parser.add_argument("--n-prompts", type=int, default=20, help="Number of prompts to use")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    samples_path = os.path.join(results_dir, "calibration_samples.json")

    if args.from_file:
        with open(args.from_file) as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} samples from {args.from_file}")
    elif os.path.exists(samples_path):
        with open(samples_path) as f:
            samples = json.load(f)
        rated = sum(1 for s in samples if s["rating"] is not None)
        print(f"Loaded {len(samples)} samples ({rated} already rated)")
    else:
        prompts = CALIBRATION_PROMPTS[:args.n_prompts]
        print(f"Generating responses for {len(prompts)} prompts...")
        samples = generate_with_llm_api(prompts)

        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"Saved {len(samples)} samples to {samples_path}")

    if args.generate:
        print("Generation complete. Run with --interactive to rate.")
        return

    if args.interactive:
        start_idx = next(
            (i for i, s in enumerate(samples) if s["rating"] is None), 0
        )
        samples = interactive_rating(samples, start_idx=start_idx)

        # Auto-save after rating
        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"Progress saved to {samples_path}")

    # Train if we have enough rated samples
    rated = [s for s in samples if s["rating"] is not None]
    if len(rated) >= 20:
        print(f"\n{len(rated)} rated samples available. Training calibration MLP...")
        wrapper = TribeWrapper(text_only=True)
        scorer, result = train_on_ratings(wrapper, samples)
    else:
        print(f"\nOnly {len(rated)} rated samples. Need at least 20 to train.")
        print("Run with --interactive to rate more samples.")


if __name__ == "__main__":
    main()
