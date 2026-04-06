"""
Experiment 4: Preference Labeling Agreement

Tests whether neural scoring agrees with human preferences on pairwise
comparisons — the actual signal needed for DPO training.

Usage:
    python preference_labeler.py --generate --n_pairs 50
    python preference_labeler.py --evaluate  # After rating
"""

import os
import sys
import json
import random
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from tribe_wrapper import TribeWrapper
from roi_extractor import ROIExtractor
from neural_scorer import NeuralScorer


EVAL_PROMPTS = [
    "Explain how encryption keeps data safe.",
    "What causes a stock market crash?",
    "How does the human eye perceive color?",
    "Why do languages evolve over time?",
    "How do electric cars work?",
    "What makes some music sound sad?",
    "How does machine learning differ from traditional programming?",
    "Why is biodiversity important?",
    "How do bridges support weight?",
    "What causes allergic reactions?",
    "Explain the concept of supply and demand.",
    "How does memory work in the human brain?",
    "Why do some materials conduct electricity?",
    "How does climate change affect ocean currents?",
    "What makes a good scientific experiment?",
    "How do social media algorithms decide what you see?",
    "Why do humans need vitamins?",
    "How does WiFi actually transmit data?",
    "What causes the northern lights?",
    "How do antibodies fight infections?",
    "Explain what makes something funny.",
    "How does 3D printing work?",
    "Why do economies experience inflation?",
    "How does CRISPR gene editing work?",
    "What makes some leaders effective?",
    "How do noise-canceling headphones work?",
    "Why do volcanoes erupt?",
    "How does blockchain technology work?",
    "What causes insomnia?",
    "How do birds navigate during migration?",
    "Explain the concept of opportunity cost.",
    "How does anesthesia make you unconscious?",
    "Why do some animals hibernate?",
    "How do contact lenses correct vision?",
    "What makes a sustainable city?",
    "How does the placebo effect work?",
    "Why do we yawn?",
    "How do submarines control their depth?",
    "What causes ocean waves?",
    "How does the immune system learn?",
    "Explain how recycling actually works.",
    "How do fireworks produce different colors?",
    "Why do prices vary between countries?",
    "How does your phone's touchscreen work?",
    "What makes soil fertile?",
    "How do elevators know where to go?",
    "Why do we get goosebumps?",
    "How does sourdough fermentation work?",
    "What causes traffic jams?",
    "How do satellites stay in orbit?",
]


def generate_response_pairs(
    prompts: list[str],
    n_pairs: int = 50,
) -> list[dict]:
    """Generate response pairs for preference evaluation.

    For a real experiment, each pair should come from the SAME model
    at different temperatures or from different random seeds, so the
    quality difference is subtle and genuine.

    If you have a local LLM, Claude Code can help you set up generation.
    Otherwise, this creates placeholder pairs for manual filling.
    """
    pairs = []
    selected = random.sample(prompts, min(n_pairs, len(prompts)))

    for prompt in selected:
        pair = {
            "prompt": prompt,
            "response_a": f"[Generate response at temperature=0.3 for: '{prompt}']",
            "response_b": f"[Generate response at temperature=1.0 for: '{prompt}']",
            "neural_scores": None,
            "neural_preferred": None,
            "human_preferred": None,
        }
        pairs.append(pair)

    return pairs


def score_pairs(
    wrapper: TribeWrapper,
    scorer: NeuralScorer,
    pairs: list[dict],
) -> list[dict]:
    """Score all pairs with the neural scoring function."""

    print(f"Scoring {len(pairs)} pairs...")
    for i, pair in enumerate(pairs):
        if (i + 1) % 10 == 0:
            print(f"  Scoring pair {i+1}/{len(pairs)}...")

        result = scorer.compare_texts(
            wrapper,
            text_a=f"Question: {pair['prompt']}\n\nAnswer: {pair['response_a']}",
            text_b=f"Question: {pair['prompt']}\n\nAnswer: {pair['response_b']}",
        )

        pair["neural_scores"] = {
            "a": result["score_a"],
            "b": result["score_b"],
            "margin": result["margin"],
        }
        pair["neural_preferred"] = result["preferred"]
        pair["neural_confident"] = result["confident"]

    return pairs


def human_evaluation(pairs: list[dict]) -> list[dict]:
    """Blinded human evaluation of response pairs."""

    print("\n" + "=" * 60)
    print("  BLINDED PREFERENCE EVALUATION")
    print("=" * 60)
    print("\nFor each pair, pick which response you prefer.")
    print("Type 'a' or 'b', or 's' to skip, 'q' to quit.")
    print("You will NOT see the neural scores until the end.\n")

    # Randomize presentation order (so 'a' isn't always the same quality)
    for pair in pairs:
        if random.random() > 0.5:
            pair["_swapped"] = True
            pair["_shown_a"] = pair["response_b"]
            pair["_shown_b"] = pair["response_a"]
        else:
            pair["_swapped"] = False
            pair["_shown_a"] = pair["response_a"]
            pair["_shown_b"] = pair["response_b"]

    evaluated = 0
    for i, pair in enumerate(pairs):
        if pair["human_preferred"] is not None:
            continue

        print(f"\n--- Pair {i+1}/{len(pairs)} ---")
        print(f"PROMPT: {pair['prompt']}\n")
        print(f"RESPONSE A:\n{pair['_shown_a']}\n")
        print(f"RESPONSE B:\n{pair['_shown_b']}\n")

        while True:
            choice = input("Preferred (a/b/s=skip/q=quit): ").strip().lower()
            if choice == "q":
                return pairs
            if choice == "s":
                break
            if choice in ("a", "b"):
                # Map back to original labels
                if pair["_swapped"]:
                    actual = "b" if choice == "a" else "a"
                else:
                    actual = choice
                pair["human_preferred"] = actual
                evaluated += 1
                break

    print(f"\nEvaluated {evaluated} pairs.")
    return pairs


def compute_agreement(pairs: list[dict]) -> dict:
    """Compute agreement rate between neural and human preferences."""

    evaluated = [
        p for p in pairs
        if p["human_preferred"] is not None and p["neural_preferred"] is not None
    ]

    if not evaluated:
        return {"error": "No pairs with both neural and human labels"}

    agrees = sum(
        1 for p in evaluated
        if p["neural_preferred"] == p["human_preferred"]
    )

    # Agreement on confident pairs only
    confident = [p for p in evaluated if p.get("neural_confident", False)]
    confident_agrees = sum(
        1 for p in confident
        if p["neural_preferred"] == p["human_preferred"]
    )

    total = len(evaluated)
    agreement = agrees / total

    result = {
        "total_evaluated": total,
        "agreements": agrees,
        "agreement_rate": agreement,
        "confident_pairs": len(confident),
        "confident_agreement": confident_agrees / len(confident) if confident else 0,
    }

    print(f"\n{'='*60}")
    print(f"  AGREEMENT RESULTS")
    print(f"{'='*60}\n")
    print(f"  Total pairs evaluated:    {total}")
    print(f"  Neural-human agreements:  {agrees}")
    print(f"  Agreement rate:           {agreement:.1%}")
    print(f"  Confident pairs:          {len(confident)}")
    print(f"  Confident agreement:      {result['confident_agreement']:.1%}")

    # Interpretation
    print()
    if agreement > 0.75:
        print("  VERDICT: Excellent agreement. Neural labels are a strong DPO signal.")
        print("  → Proceed to Experiment 5 (Neural DPO training).")
    elif agreement > 0.65:
        print("  VERDICT: Good agreement. With margin filtering, this is viable for DPO.")
        print("  → Proceed to Experiment 5 with margin threshold ≥ 0.3.")
    elif agreement > 0.55:
        print("  VERDICT: Above chance but noisy. May need more calibration data.")
        print("  → Consider running more calibration (Experiment 3) before proceeding.")
    else:
        print("  VERDICT: Near random. Neural scoring doesn't capture preferences.")
        print("  → Revisit ROI selection and calibration approach.")

    return result


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Preference Labeling")
    parser.add_argument("--generate", action="store_true", help="Generate and score pairs")
    parser.add_argument("--evaluate", action="store_true", help="Human evaluation (blinded)")
    parser.add_argument("--results", action="store_true", help="Show agreement results")
    parser.add_argument("--n_pairs", type=int, default=50, help="Number of pairs to generate")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    pairs_path = os.path.join(results_dir, "preference_pairs.json")

    if args.generate:
        # Generate pairs
        pairs = generate_response_pairs(EVAL_PROMPTS, n_pairs=args.n_pairs)

        # Load scorer from Experiment 3
        scorer_path = os.path.join(results_dir, "calibration_scorer.pt")
        if not os.path.exists(scorer_path):
            print("ERROR: No calibration scorer found.")
            print("Run Experiment 3 first: python mini_calibration.py --interactive")
            return

        wrapper = TribeWrapper(text_only=True)
        scorer = NeuralScorer()
        scorer.load(scorer_path)

        # Score pairs
        pairs = score_pairs(wrapper, scorer, pairs)

        with open(pairs_path, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"\nSaved {len(pairs)} scored pairs to {pairs_path}")
        print("Run with --evaluate for human evaluation.")

    elif args.evaluate:
        if not os.path.exists(pairs_path):
            print("ERROR: No pairs file found. Run with --generate first.")
            return

        with open(pairs_path) as f:
            pairs = json.load(f)

        pairs = human_evaluation(pairs)

        with open(pairs_path, "w") as f:
            json.dump(pairs, f, indent=2)

        # Auto-compute agreement
        result = compute_agreement(pairs)

        with open(os.path.join(results_dir, "experiment_4_results.json"), "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                **result,
            }, f, indent=2)

    elif args.results:
        if not os.path.exists(pairs_path):
            print("ERROR: No pairs file found.")
            return

        with open(pairs_path) as f:
            pairs = json.load(f)

        compute_agreement(pairs)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
