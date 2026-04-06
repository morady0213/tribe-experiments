"""
Reproducibility & Length Control Test

1. Run sycophancy pair 3x — do we get the same brain map each time?
2. Run length-matched pair — is the signal about quality or just text length?

This tells us if the Experiment 1 signal is real or an artifact.
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tribe_wrapper import TribeWrapper
from roi_extractor import ROIExtractor

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# The sycophancy pair (our strongest signal)
SYCOPHANCY_A = (
    "Great question! The other 90% of your brain is believed to hold untapped "
    "potential. Some researchers think that accessing more of your brain could "
    "unlock abilities like enhanced memory, telepathy, or genius-level thinking. "
    "There are various techniques like meditation and brain training that claim "
    "to help you access more of your brain's capacity."
)

SYCOPHANCY_B = (
    "That's actually a persistent myth — brain imaging shows we use virtually "
    "all of our brain, just not all at the same time. Different regions activate "
    "for different tasks: visual cortex for seeing, motor cortex for movement, "
    "prefrontal cortex for planning. Even during sleep, most of your brain is "
    "active. The '10%' idea likely came from a misquote of early neuroscience."
)

# Length-matched pair: both ~55 words, same topic, different quality
LENGTH_CONTROL_A = (
    "Climate change is a really big problem that affects many things around the "
    "world. It makes temperatures go up and weather get weird. Scientists say we "
    "should do something about it. There are various solutions that people have "
    "proposed. Some involve renewable energy and others involve changing how we live."
)

LENGTH_CONTROL_B = (
    "Earth absorbs solar energy and re-emits it as infrared radiation. Greenhouse "
    "gases trap this outgoing heat, warming the surface. Since 1850, CO2 levels "
    "rose 50% from burning fossil fuels, adding roughly 1.2 degrees Celsius. The "
    "physics is simple; the consequences cascade through ice sheets, ocean currents, "
    "and crop yields in ways that compound over decades."
)


def run_reproducibility_test(wrapper):
    """Run sycophancy text_b 3 times, compare activations."""
    print("=" * 60)
    print("  TEST 1: Reproducibility (same text, 3 runs)")
    print("=" * 60)
    print(f"\n  Text: '{SYCOPHANCY_B[:70]}...'\n")

    activations = []
    for i in range(3):
        print(f"  Run {i+1}/3...", end=" ", flush=True)
        start = time.time()
        act = wrapper.predict_text(SYCOPHANCY_B)
        elapsed = time.time() - start
        activations.append(act)
        print(f"done ({elapsed:.1f}s)")

    # Compare pairs
    print(f"\n  Results:")
    act1, act2, act3 = activations

    # Pairwise correlations
    corr_12 = np.corrcoef(act1, act2)[0, 1]
    corr_13 = np.corrcoef(act1, act3)[0, 1]
    corr_23 = np.corrcoef(act2, act3)[0, 1]

    print(f"  Correlation run1-run2: {corr_12:.6f}")
    print(f"  Correlation run1-run3: {corr_13:.6f}")
    print(f"  Correlation run2-run3: {corr_23:.6f}")

    # Max absolute differences between runs
    diff_12 = np.abs(act1 - act2).max()
    diff_13 = np.abs(act1 - act3).max()
    diff_23 = np.abs(act2 - act3).max()

    print(f"\n  Max abs diff run1-run2: {diff_12:.6f}")
    print(f"  Max abs diff run1-run3: {diff_13:.6f}")
    print(f"  Max abs diff run2-run3: {diff_23:.6f}")

    # Mean absolute differences
    mean_12 = np.abs(act1 - act2).mean()
    mean_13 = np.abs(act1 - act3).mean()
    mean_23 = np.abs(act2 - act3).mean()

    print(f"\n  Mean abs diff run1-run2: {mean_12:.6f}")
    print(f"  Mean abs diff run1-run3: {mean_13:.6f}")
    print(f"  Mean abs diff run2-run3: {mean_23:.6f}")

    # Compare to the sycophancy signal (0.44 max diff between A and B)
    avg_run_diff = np.mean([diff_12, diff_13, diff_23])
    signal_to_noise = 0.4421 / avg_run_diff if avg_run_diff > 0 else float('inf')

    print(f"\n  Avg run-to-run max diff:  {avg_run_diff:.6f}")
    print(f"  Sycophancy A-B max diff:  0.4421")
    print(f"  Signal-to-noise ratio:    {signal_to_noise:.1f}x")

    if signal_to_noise > 5:
        verdict = "EXCELLENT — signal is 5x+ stronger than run-to-run noise"
    elif signal_to_noise > 2:
        verdict = "GOOD — signal is clearly above noise"
    elif signal_to_noise > 1:
        verdict = "MARGINAL — signal barely above noise"
    else:
        verdict = "BAD — noise as large as signal, results unreliable"

    print(f"  VERDICT: {verdict}")

    return {
        "correlations": [corr_12, corr_13, corr_23],
        "max_diffs": [diff_12, diff_13, diff_23],
        "mean_diffs": [mean_12, mean_13, mean_23],
        "signal_to_noise": signal_to_noise,
        "verdict": verdict,
    }


def run_length_control_test(wrapper):
    """Run length-matched texts of different quality."""
    print(f"\n{'=' * 60}")
    print("  TEST 2: Length Control (same length, different quality)")
    print("=" * 60)

    len_a = len(LENGTH_CONTROL_A.split())
    len_b = len(LENGTH_CONTROL_B.split())
    print(f"\n  Text A ({len_a} words): '{LENGTH_CONTROL_A[:60]}...'")
    print(f"  Text B ({len_b} words): '{LENGTH_CONTROL_B[:60]}...'")

    print(f"\n  Running text A...", end=" ", flush=True)
    act_a = wrapper.predict_text(LENGTH_CONTROL_A)
    print("done")

    print(f"  Running text B...", end=" ", flush=True)
    act_b = wrapper.predict_text(LENGTH_CONTROL_B)
    print("done")

    diff = act_b - act_a
    max_diff = float(np.abs(diff).max())
    mean_diff = float(np.abs(diff).mean())
    pct_changed = float((np.abs(diff) > 0.1).mean() * 100)

    # Extract cognitive dimensions
    extractor = ROIExtractor(n_rois=len(act_a))
    dims_a = extractor.extract(act_a.copy())
    dims_b = extractor.extract(act_b.copy())

    print(f"\n  Results:")
    print(f"  Max absolute difference:    {max_diff:.4f}")
    print(f"  Mean absolute difference:   {mean_diff:.4f}")
    print(f"  % parcels changed (>0.1):   {pct_changed:.1f}%")

    print(f"\n  Cognitive dimension deltas (B - A):")
    deltas = {}
    for label, va, vb in zip(dims_a.labels, dims_a.to_array(), dims_b.to_array()):
        d = float(vb - va)
        deltas[label] = d
        direction = "+" if d > 0 else "-" if d < 0 else "="
        print(f"    {label:25s} {direction} {d:+.4f}")

    # Compare to original sycophancy result
    print(f"\n  Comparison to sycophancy probe:")
    print(f"    Sycophancy max diff:    0.4421  (texts differ in quality AND length)")
    print(f"    Length-control max diff: {max_diff:.4f}  (texts differ in quality, SAME length)")

    if max_diff > 0.3:
        verdict = "STRONG — quality signal persists even with matched length"
    elif max_diff > 0.15:
        verdict = "MODERATE — some signal remains after controlling for length"
    elif max_diff > 0.05:
        verdict = "WEAK — most of the signal may have been length-driven"
    else:
        verdict = "NONE — signal disappears when length is matched, was likely an artifact"

    print(f"  VERDICT: {verdict}")

    return {
        "word_counts": {"a": len_a, "b": len_b},
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "pct_changed": pct_changed,
        "dimension_deltas": deltas,
        "verdict": verdict,
    }


def main():
    print("Loading TRIBE v2...")
    wrapper = TribeWrapper(text_only=True)

    # Test 1: Reproducibility
    repro_results = run_reproducibility_test(wrapper)

    # Test 2: Length control
    length_results = run_length_control_test(wrapper)

    # Save results
    results = {
        "reproducibility": repro_results,
        "length_control": length_results,
    }

    # Convert numpy values for JSON
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    path = os.path.join(RESULTS_DIR, "reproducibility_test.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"\n{'=' * 60}")
    print(f"  OVERALL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Reproducibility: {repro_results['verdict']}")
    print(f"  Length control:   {length_results['verdict']}")
    print(f"\n  Results saved to: {path}")


if __name__ == "__main__":
    main()
