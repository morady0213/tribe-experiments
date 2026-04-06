"""
Experiment 1: Probe the Text-Only Pathway

Feeds carefully designed text pairs through TRIBE v2 and visualizes
where in the brain the predicted activations diverge. This is the
"kill switch" experiment — if the maps don't diverge, the idea is dead.

Usage:
    python text_probes.py
    python text_probes.py --pair 1          # Run just one pair
    python text_probes.py --all --save      # Run all pairs, save results
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


# =============================================================================
# TEXT PROBE PAIRS
# Each pair contrasts one quality dimension.
# Text A = lower quality, Text B = higher quality (by design)
# =============================================================================

PROBE_PAIRS = [
    {
        "name": "clarity",
        "dimension": "Clear vs jargon-heavy",
        "prompt": "How does a neural network learn?",
        "text_a": (
            "The stochastic gradient descent optimization procedure iteratively "
            "minimizes the empirical risk by computing partial derivatives of the "
            "loss function with respect to each parameter in the computational graph, "
            "leveraging the chain rule for efficient backpropagation of error signals "
            "through the network's layered architecture."
        ),
        "text_b": (
            "A neural network learns by making guesses and adjusting. It looks at "
            "data, makes a prediction, checks how wrong it was, then nudges its "
            "internal settings slightly to be less wrong next time. Repeat this "
            "millions of times and the network gradually gets accurate."
        ),
    },
    {
        "name": "depth",
        "dimension": "Surface-level vs builds intuition",
        "prompt": "How does gravity work?",
        "text_a": (
            "Gravity is a fundamental force of nature. It causes objects with mass "
            "to attract each other. The more mass an object has, the stronger its "
            "gravitational pull. Gravity keeps the planets in orbit around the Sun "
            "and keeps us on the ground."
        ),
        "text_b": (
            "Imagine a bowling ball sitting on a stretched rubber sheet — it creates "
            "a dip. Roll a marble nearby and it curves toward the bowling ball, not "
            "because the ball is pulling it, but because the sheet is curved. That's "
            "Einstein's insight: massive objects warp the fabric of space itself, and "
            "what we experience as gravity is just objects following the curves."
        ),
    },
    {
        "name": "structure",
        "dimension": "Disorganized vs well-structured",
        "prompt": "What are the benefits of exercise?",
        "text_a": (
            "Exercise is good for your heart and also your brain releases endorphins "
            "and you sleep better too. Your muscles get stronger. Oh and it helps "
            "with weight management. Also mental health is improved. Some people "
            "like running and others prefer swimming. It can lower blood pressure."
        ),
        "text_b": (
            "Exercise transforms your body on three levels. Physically, it strengthens "
            "your heart, builds muscle, and improves sleep quality. Mentally, it "
            "triggers endorphin release — your brain's natural mood elevator — and "
            "reduces anxiety over time. Metabolically, it regulates blood sugar and "
            "blood pressure, cutting your risk of chronic disease."
        ),
    },
    {
        "name": "accuracy",
        "dimension": "Subtly wrong vs factually precise",
        "prompt": "How do vaccines work?",
        "text_a": (
            "Vaccines work by injecting a small amount of the actual disease into "
            "your body. Your immune system then fights off this mini-infection, and "
            "remembers how to do it. This is why you might feel sick after a vaccine — "
            "you're actually getting a mild version of the illness."
        ),
        "text_b": (
            "Vaccines introduce a harmless component of a pathogen — a weakened form, "
            "an inactivated version, or just a protein fragment — not the actual "
            "disease. Your immune system learns to recognize this component and builds "
            "antibodies. If you later encounter the real pathogen, your immune system "
            "responds quickly because it already has the blueprint."
        ),
    },
    {
        "name": "sycophancy",
        "dimension": "Validates wrong premise vs corrects respectfully",
        "prompt": "I read that we only use 10% of our brains. What happens in the other 90%?",
        "text_a": (
            "Great question! The other 90% of your brain is believed to hold untapped "
            "potential. Some researchers think that accessing more of your brain could "
            "unlock abilities like enhanced memory, telepathy, or genius-level thinking. "
            "There are various techniques like meditation and brain training that claim "
            "to help you access more of your brain's capacity."
        ),
        "text_b": (
            "That's actually a persistent myth — brain imaging shows we use virtually "
            "all of our brain, just not all at the same time. Different regions activate "
            "for different tasks: visual cortex for seeing, motor cortex for movement, "
            "prefrontal cortex for planning. Even during sleep, most of your brain is "
            "active. The '10%' idea likely came from a misquote of early neuroscience."
        ),
    },
    {
        "name": "memorability",
        "dimension": "Forgettable vs vivid and memorable",
        "prompt": "What is inflation?",
        "text_a": (
            "Inflation is a general increase in the price level of goods and services "
            "over a period of time. When the general price level rises, each unit of "
            "currency buys fewer goods and services. Inflation is measured by the "
            "Consumer Price Index."
        ),
        "text_b": (
            "Your grandparents bought a house for $20,000. That same house now costs "
            "$400,000. The house didn't get 20 times better — the dollar got 20 times "
            "weaker. That's inflation: not prices going up, but money's purchasing "
            "power going down. Every dollar in your pocket silently loses value, year "
            "after year, like ice melting in the sun."
        ),
    },
    {
        "name": "engagement",
        "dimension": "Dry encyclopedia vs compelling narrative",
        "prompt": "Tell me about black holes.",
        "text_a": (
            "A black hole is a region of spacetime where gravity is so strong that "
            "nothing, not even light or other electromagnetic waves, has enough energy "
            "to escape the event horizon. The theory of general relativity predicts "
            "that a sufficiently compact mass can deform spacetime to form a black hole."
        ),
        "text_b": (
            "Here's the terrifying part about black holes: there's no surface to crash "
            "into. You'd fall forever inward, past the point of no return — the event "
            "horizon — where even light gives up trying to escape. Time itself bends. "
            "To someone watching from far away, you'd appear to freeze at the edge, "
            "slowly fading to red, then vanishing. But from your perspective, you'd "
            "keep falling, stretched like taffy, into the unknown."
        ),
    },
    {
        "name": "concision",
        "dimension": "Bloated and repetitive vs tight and efficient",
        "prompt": "What is photosynthesis?",
        "text_a": (
            "Photosynthesis is a process used by plants and other organisms to convert "
            "light energy into chemical energy. Specifically, photosynthesis is the "
            "process by which plants take in sunlight and use it to make food. In "
            "other words, photosynthesis allows plants to transform light from the "
            "sun into energy they can use. This process of photosynthesis is essential "
            "for life on Earth because it produces oxygen."
        ),
        "text_b": (
            "Plants eat sunlight. They absorb CO₂ from the air and water from the "
            "soil, then use solar energy to fuse them into glucose — their food. "
            "Oxygen is the waste product, which is lucky for us."
        ),
    },
    {
        "name": "formality",
        "dimension": "Too casual vs appropriately professional",
        "prompt": "Explain quantum entanglement.",
        "text_a": (
            "ok so basically two particles get like linked up right and then when u "
            "mess with one the other one like instantly knows about it no matter how "
            "far away it is lol. einstein thought it was totally BS and called it "
            "spooky action at a distance. its pretty wild ngl"
        ),
        "text_b": (
            "When two particles become entangled, measuring one immediately tells you "
            "about the other, regardless of the distance between them. Measure one "
            "particle's spin as 'up,' and the other is guaranteed to be 'down.' "
            "Einstein found this troubling — he called it 'spooky action at a "
            "distance.' Decades of experiments have confirmed it's real."
        ),
    },
    {
        "name": "emotional_warmth",
        "dimension": "Cold and robotic vs warm and human",
        "prompt": "I failed my exam. What should I do?",
        "text_a": (
            "Exam failure can be addressed through systematic remediation. Analyze "
            "which topics caused point loss. Create a structured study plan targeting "
            "weak areas. Allocate 2-3 hours daily for review. Consider tutoring if "
            "self-study is insufficient. Retake the exam when prepared."
        ),
        "text_b": (
            "That's a tough moment, and it's okay to feel frustrated. Failing an exam "
            "doesn't define you — it just tells you where the gaps are. Take a day to "
            "decompress, then look at which specific questions tripped you up. Those "
            "gaps are actually useful information. Many people who've gone on to "
            "master a subject started with a failed exam that showed them what to "
            "focus on."
        ),
    },
]


def run_single_probe(wrapper: TribeWrapper, pair: dict, verbose: bool = True) -> dict:
    """Run a single probe pair through TRIBE v2 and compare activations."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Probe: {pair['name']} ({pair['dimension']})")
        print(f"{'='*60}")

    # Get activations
    act_a = wrapper.predict_text(pair["text_a"])
    act_b = wrapper.predict_text(pair["text_b"])

    # Compute difference
    diff = act_b - act_a  # Positive = higher in text_b (better)

    # Extract ROI dimensions for both
    extractor = ROIExtractor(n_rois=len(act_a))
    dims_a = extractor.extract(act_a.copy())
    dims_b = extractor.extract(act_b.copy())

    # Compute per-dimension deltas
    delta = {}
    for label, va, vb in zip(dims_a.labels, dims_a.to_array(), dims_b.to_array()):
        delta[label] = float(vb - va)

    # Find top divergent parcels
    top_indices = np.argsort(np.abs(diff))[-10:][::-1]
    top_parcels = [
        {"index": int(idx), "diff": float(diff[idx])}
        for idx in top_indices
    ]

    result = {
        "name": pair["name"],
        "dimension": pair["dimension"],
        "max_abs_diff": float(np.abs(diff).max()),
        "mean_abs_diff": float(np.abs(diff).mean()),
        "std_diff": float(diff.std()),
        "pct_parcels_changed": float((np.abs(diff) > 0.1).mean() * 100),
        "dimension_deltas": delta,
        "top_divergent_parcels": top_parcels,
    }

    if verbose:
        print(f"\n  Max absolute difference:    {result['max_abs_diff']:.4f}")
        print(f"  Mean absolute difference:   {result['mean_abs_diff']:.4f}")
        print(f"  % parcels changed (>0.1):   {result['pct_parcels_changed']:.1f}%")
        print(f"\n  Cognitive dimension deltas (B - A):")
        for label, d in delta.items():
            direction = "+" if d > 0 else "-" if d < 0 else "="
            bar = "#" * int(abs(d) * 20)
            print(f"    {label:25s} {direction} {d:+.4f} {bar}")

    return result


def run_all_probes(save: bool = True) -> list[dict]:
    """Run all probe pairs and generate summary."""

    print("Loading TRIBE v2...")
    wrapper = TribeWrapper(text_only=True)

    results = []
    for pair in PROBE_PAIRS:
        result = run_single_probe(wrapper, pair)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}\n")

    # Sort by max divergence
    sorted_results = sorted(results, key=lambda r: r["max_abs_diff"], reverse=True)

    print(f"  {'Probe':<20s} {'Max diff':<12s} {'Mean diff':<12s} {'% changed':<10s}")
    print(f"  {'─'*54}")
    for r in sorted_results:
        print(f"  {r['name']:<20s} {r['max_abs_diff']:<12.4f} "
              f"{r['mean_abs_diff']:<12.4f} {r['pct_parcels_changed']:<10.1f}%")

    # Overall signal assessment
    avg_max = np.mean([r["max_abs_diff"] for r in results])
    print(f"\n  Average max divergence across all probes: {avg_max:.4f}")

    if avg_max > 0.3:
        print("  VERDICT: Strong signal. Text-only pathway carries meaningful cognitive info.")
        print("  → Proceed to Experiment 2.")
    elif avg_max > 0.1:
        print("  VERDICT: Weak but detectable signal. Worth exploring further.")
        print("  → Proceed to Experiment 2, but consider adding audio (TTS).")
    else:
        print("  VERDICT: Minimal signal. Text-only may not be sufficient.")
        print("  → Try adding TTS audio for trimodal input before proceeding.")

    if save:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, "experiment_1_probes.json")
        with open(path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "n_probes": len(results),
                "avg_max_divergence": float(avg_max),
                "results": results,
            }, f, indent=2)
        print(f"\n  Results saved to: {path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: Text Probes")
    parser.add_argument("--pair", type=int, help="Run a single pair (1-10)")
    parser.add_argument("--all", action="store_true", help="Run all pairs")
    parser.add_argument("--save", action="store_true", help="Save results to disk")
    parser.add_argument("--list", action="store_true", help="List available probes")
    args = parser.parse_args()

    if args.list:
        print("Available probe pairs:")
        for i, pair in enumerate(PROBE_PAIRS, 1):
            print(f"  {i}. {pair['name']:20s} — {pair['dimension']}")
    elif args.pair:
        wrapper = TribeWrapper(text_only=True)
        pair = PROBE_PAIRS[args.pair - 1]
        run_single_probe(wrapper, pair)
    else:
        run_all_probes(save=args.save or args.all)
