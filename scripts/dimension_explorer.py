"""
Experiment 2: Dimension Explorer

Runs 30 texts of varying quality through TRIBE v2, extracts 5 cognitive
dimensions, and analyzes their independence/correlation structure.

Goal: Which dimensions carry independent signal? Which are redundant?
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
VIS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# 30 texts spanning quality range: 10 low, 10 medium, 10 high
# Each has a quality_label (1-3) and a topic for grouping
SAMPLE_TEXTS = [
    # --- LOW QUALITY (vague, sycophantic, bloated, surface-level) ---
    {"text": "AI is really cool and amazing technology that will change everything in the future. Many people are excited about it. It can do lots of things.", "quality": 1, "topic": "AI"},
    {"text": "The economy is complicated. There are many factors involved. Some experts think it will go up, others think it will go down. Time will tell what happens.", "quality": 1, "topic": "economy"},
    {"text": "Great question! Exercise is super important for health. You should definitely exercise more. It helps with everything and makes you feel better overall.", "quality": 1, "topic": "exercise"},
    {"text": "Programming is when you tell a computer what to do. You write code and the computer follows instructions. There are many programming languages to choose from.", "quality": 1, "topic": "programming"},
    {"text": "Climate change is a big issue. We need to take action. Everyone should do their part. It affects the whole planet and future generations.", "quality": 1, "topic": "climate"},
    {"text": "Sleep is important for your body. You should get enough sleep every night. Not getting enough sleep is bad for you. Most people need about eight hours.", "quality": 1, "topic": "sleep"},
    {"text": "History teaches us many lessons. We can learn from the past to make better decisions in the future. Many great civilizations have risen and fallen.", "quality": 1, "topic": "history"},
    {"text": "Cooking is a useful skill to have. You can make many different dishes. Some people like cooking and some don't. There are many recipes available online.", "quality": 1, "topic": "cooking"},
    {"text": "Reading books is good for your brain. It improves vocabulary and knowledge. Everyone should read more books. There are many genres to choose from.", "quality": 1, "topic": "reading"},
    {"text": "Space is really big and interesting. There are many planets and stars. Scientists are always discovering new things about the universe. It's fascinating stuff.", "quality": 1, "topic": "space"},

    # --- MEDIUM QUALITY (correct but generic, textbook-like) ---
    {"text": "Artificial intelligence uses machine learning algorithms to identify patterns in data. Neural networks, inspired by biological neurons, process information through layers of connected nodes. Training involves adjusting connection weights to minimize prediction errors.", "quality": 2, "topic": "AI"},
    {"text": "Inflation occurs when the general price level rises, reducing purchasing power. Central banks manage inflation through interest rate adjustments. Higher rates discourage borrowing and spending, cooling demand and slowing price increases.", "quality": 2, "topic": "economy"},
    {"text": "Regular cardiovascular exercise strengthens the heart muscle, improves circulation, and reduces blood pressure. The American Heart Association recommends 150 minutes of moderate-intensity aerobic activity per week for optimal cardiovascular health.", "quality": 2, "topic": "exercise"},
    {"text": "Object-oriented programming organizes code into classes and objects. Classes define properties and methods, while objects are instances of classes. Inheritance allows classes to share behavior, reducing code duplication.", "quality": 2, "topic": "programming"},
    {"text": "The greenhouse effect occurs when atmospheric gases trap solar radiation reflected from Earth's surface. Carbon dioxide and methane are the primary greenhouse gases. Human activities have increased their concentration by approximately 50% since pre-industrial times.", "quality": 2, "topic": "climate"},
    {"text": "During sleep, the brain cycles through REM and non-REM stages approximately every 90 minutes. Non-REM sleep includes light sleep and deep slow-wave sleep, which is critical for physical recovery. REM sleep supports memory consolidation.", "quality": 2, "topic": "sleep"},
    {"text": "The Roman Republic transitioned to an Empire under Augustus in 27 BCE. The Senate retained nominal authority but real power shifted to the Emperor. This transformation was driven by decades of civil war and political instability.", "quality": 2, "topic": "history"},
    {"text": "The Maillard reaction occurs when amino acids and reducing sugars react at temperatures above 140 degrees Celsius. This chemical process creates hundreds of flavor compounds responsible for the taste of seared meat, toasted bread, and roasted coffee.", "quality": 2, "topic": "cooking"},
    {"text": "Reading comprehension involves decoding text, building mental models, and integrating new information with existing knowledge. Skilled readers automatically make inferences to fill gaps in the text, creating coherent understanding from incomplete information.", "quality": 2, "topic": "reading"},
    {"text": "Mars has a thin atmosphere composed primarily of carbon dioxide. Surface temperatures average minus 60 degrees Celsius. Evidence of ancient river channels suggests liquid water once flowed on its surface, making it a priority target for astrobiology.", "quality": 2, "topic": "space"},

    # --- HIGH QUALITY (insightful, vivid, builds genuine understanding) ---
    {"text": "Here's the uncomfortable truth about AI: it doesn't understand anything. GPT-4 can write poetry and pass bar exams, but it's pattern-matching on steroids. The real question isn't whether AI is intelligent — it's whether intelligence even requires understanding, or if sufficiently sophisticated pattern-matching is all understanding ever was.", "quality": 3, "topic": "AI"},
    {"text": "Your grandparents bought a house for $20,000. That same house costs $400,000 now. The house didn't improve — your money decayed. Inflation isn't prices rising; it's currency weakening. Every dollar in your savings account is an ice cube, slowly melting. The question is whether your investments melt slower than your cash.", "quality": 3, "topic": "economy"},
    {"text": "Your body doesn't distinguish between running from a lion and running on a treadmill. Both trigger the same ancient stress-recovery cycle: cortisol spikes, heart rate climbs, then afterwards — the crucial part — your system overshoots on recovery, building back slightly stronger. Exercise isn't punishment. It's a controlled stress that exploits your body's tendency to overcompensate.", "quality": 3, "topic": "exercise"},
    {"text": "Most people learn to code by memorizing syntax. That's like learning to write by memorizing spelling. The real skill is decomposition — breaking an overwhelming problem into pieces small enough that each one is boring. If a function makes you think hard, it's doing too much. The best code reads like a to-do list, not a puzzle.", "quality": 3, "topic": "programming"},
    {"text": "Earth has a thermostat, but we've jammed it. For millions of years, the carbon cycle balanced itself — volcanoes emit CO2, rocks and oceans absorb it. We're now emitting CO2 a hundred times faster than all volcanoes combined, and the absorption side can't keep up. It's not that the planet can't handle carbon — it's that we've overwhelmed the speed of geological recycling.", "quality": 3, "topic": "climate"},
    {"text": "Sleep isn't downtime — it's when your brain takes out the trash. During deep sleep, cerebrospinal fluid literally washes through your brain tissue, flushing out metabolic waste including amyloid-beta, the protein that accumulates in Alzheimer's. Every night of poor sleep is a night the cleanup crew couldn't finish their shift.", "quality": 3, "topic": "sleep"},
    {"text": "Rome didn't fall because barbarians were strong. It fell because Romans stopped wanting to be Roman. When citizens no longer saw the empire's survival as their personal problem — when defense became someone else's job and taxes became someone else's burden — the structure hollowed out. The legions at the border were the last thing to go, not the first.", "quality": 3, "topic": "history"},
    {"text": "Salt doesn't add flavor — it removes the barrier to tasting what's already there. Your taste receptors are suppressed by bitter compounds in most foods. Salt ions block those bitter signals, letting sweetness, umami, and aromatics through. That's why undersalted food tastes flat, not just less salty. The flavor was always there; salt just opens the door.", "quality": 3, "topic": "cooking"},
    {"text": "Speed-reading is mostly a scam. Comprehension drops off a cliff above 400 words per minute because your brain can't build mental models that fast. The bottleneck isn't your eyes — it's working memory. What actually makes you read faster over years is knowledge: the more you already know about a topic, the fewer inferences your brain needs to construct.", "quality": 3, "topic": "reading"},
    {"text": "The most unsettling fact about space isn't its size — it's its silence. We've pointed radio telescopes at thousands of nearby stars for decades and heard nothing. Either intelligent life is extraordinarily rare, or it doesn't survive long enough to signal us, or it's hiding. None of those answers are comforting.", "quality": 3, "topic": "space"},
]


def run_dimension_exploration():
    print("=" * 60)
    print("  Experiment 2: Dimension Explorer")
    print(f"  {len(SAMPLE_TEXTS)} texts (10 low / 10 medium / 10 high quality)")
    print("=" * 60)

    # Load model
    print("\nLoading TRIBE v2...")
    wrapper = TribeWrapper(text_only=True)

    # Run all texts
    all_dims = []
    all_meta = []
    extractor = None

    for i, sample in enumerate(SAMPLE_TEXTS):
        q_label = ["LOW", "MED", "HIGH"][sample["quality"] - 1]
        print(f"\n  [{i+1:2d}/30] ({q_label}) {sample['topic']:12s} — "
              f"'{sample['text'][:50]}...'", end=" ", flush=True)

        try:
            start = time.time()
            activation = wrapper.predict_text(sample["text"])
            elapsed = time.time() - start

            # Create extractor once (same for all texts)
            if extractor is None:
                extractor = ROIExtractor(n_rois=len(activation))

            dims = extractor.extract(activation.copy())
            all_dims.append(dims.to_array())
            all_meta.append({
                "index": i,
                "quality": sample["quality"],
                "topic": sample["topic"],
                "text_preview": sample["text"][:80],
                "dimensions": dims.to_dict(),
                "inference_time": elapsed,
            })
            print(f"({elapsed:.0f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    # Convert to matrix: (N, 5) — N may be < 30 if some texts failed
    dim_matrix = np.array(all_dims)
    dim_labels = ["Comprehension", "Memory", "Attention", "Confusion", "DMN suppress."]
    qualities = [m["quality"] for m in all_meta]
    print(f"\n  Successfully processed {len(all_meta)}/30 texts")

    # === ANALYSIS ===
    print(f"\n{'=' * 60}")
    print("  ANALYSIS")
    print(f"{'=' * 60}")

    # 1. Per-dimension statistics by quality level
    print("\n  Mean dimension values by quality level:")
    print(f"  {'Dimension':20s} {'Low':>8s} {'Med':>8s} {'High':>8s} {'Trend':>8s}")
    print(f"  {'─' * 52}")

    quality_groups = {1: [], 2: [], 3: []}
    for i, q in enumerate(qualities):
        quality_groups[q].append(dim_matrix[i])

    dim_trends = []
    for d in range(5):
        means = [np.mean([row[d] for row in quality_groups[q]]) for q in [1, 2, 3]]
        trend = means[2] - means[0]  # high minus low
        dim_trends.append(trend)
        arrow = "+" if trend > 0 else "-"
        print(f"  {dim_labels[d]:20s} {means[0]:+8.4f} {means[1]:+8.4f} {means[2]:+8.4f} {arrow}{abs(trend):.4f}")

    # 2. Correlation matrix
    print("\n  Inter-dimension correlation matrix:")
    corr_matrix = np.corrcoef(dim_matrix.T)
    print(f"  {'':20s}", end="")
    for label in dim_labels:
        print(f" {label[:6]:>7s}", end="")
    print()
    for i, label in enumerate(dim_labels):
        print(f"  {label:20s}", end="")
        for j in range(5):
            val = corr_matrix[i, j]
            print(f" {val:+7.3f}", end="")
        print()

    # 3. Variance per dimension
    print("\n  Per-dimension variance:")
    for d in range(5):
        var = dim_matrix[:, d].var()
        std = dim_matrix[:, d].std()
        rng = dim_matrix[:, d].max() - dim_matrix[:, d].min()
        useful = "KEEP" if std > 0.02 else "DROP (low variance)"
        print(f"  {dim_labels[d]:20s} var={var:.6f}  std={std:.4f}  range={rng:.4f}  -> {useful}")

    # 4. Quality discrimination power (effect size)
    print("\n  Quality discrimination (Cohen's d: high vs low):")
    for d in range(5):
        high_vals = [row[d] for row in quality_groups[3]]
        low_vals = [row[d] for row in quality_groups[1]]
        mean_diff = np.mean(high_vals) - np.mean(low_vals)
        pooled_std = np.sqrt((np.var(high_vals) + np.var(low_vals)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        strength = "STRONG" if abs(cohens_d) > 0.8 else "MEDIUM" if abs(cohens_d) > 0.5 else "WEAK"
        print(f"  {dim_labels[d]:20s} d={cohens_d:+.3f}  ({strength})")

    # === SAVE RESULTS ===
    results = {
        "n_texts": len(SAMPLE_TEXTS),
        "dim_labels": dim_labels,
        "quality_means": {
            "low": [float(np.mean([row[d] for row in quality_groups[1]])) for d in range(5)],
            "med": [float(np.mean([row[d] for row in quality_groups[2]])) for d in range(5)],
            "high": [float(np.mean([row[d] for row in quality_groups[3]])) for d in range(5)],
        },
        "correlation_matrix": corr_matrix.tolist(),
        "dim_trends": [float(t) for t in dim_trends],
        "per_text": all_meta,
    }

    path = os.path.join(RESULTS_DIR, "experiment_2_dimensions.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {path}")

    # === VISUALIZATIONS ===
    print("\n  Generating visualizations...")
    generate_visualizations(dim_matrix, dim_labels, qualities, corr_matrix, quality_groups, dim_trends)

    return results


def generate_visualizations(dim_matrix, dim_labels, qualities, corr_matrix, quality_groups, dim_trends):
    """Generate all Experiment 2 plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    quality_colors = {1: "#e74c3c", 2: "#f39c12", 3: "#2ecc71"}
    quality_names = {1: "Low", 2: "Medium", 3: "High"}

    # 1. Scatter matrix (5x5)
    fig, axes = plt.subplots(5, 5, figsize=(16, 16))
    fig.suptitle("Experiment 2: Cognitive Dimension Scatter Matrix\n"
                 "Red=Low quality, Orange=Medium, Green=High",
                 fontsize=14, fontweight="bold")

    for i in range(5):
        for j in range(5):
            ax = axes[i][j]
            if i == j:
                # Diagonal: histogram by quality
                for q in [1, 2, 3]:
                    vals = [dim_matrix[k, i] for k in range(len(qualities)) if qualities[k] == q]
                    ax.hist(vals, bins=8, alpha=0.6, color=quality_colors[q],
                            label=quality_names[q])
                ax.set_title(dim_labels[i], fontsize=9, fontweight="bold")
            else:
                # Off-diagonal: scatter
                for k in range(len(qualities)):
                    ax.scatter(dim_matrix[k, j], dim_matrix[k, i],
                              c=quality_colors[qualities[k]], s=30, alpha=0.7)
                # Show correlation
                r = corr_matrix[i, j]
                ax.text(0.05, 0.95, f"r={r:.2f}", transform=ax.transAxes,
                        fontsize=8, verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            if i == 4:
                ax.set_xlabel(dim_labels[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(dim_labels[i], fontsize=8)
            ax.tick_params(labelsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(VIS_DIR, "exp2_scatter_matrix.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # 2. Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(dim_labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(dim_labels, fontsize=11)
    for i in range(5):
        for j in range(5):
            color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)
    plt.colorbar(im, shrink=0.8)
    ax.set_title("Inter-Dimension Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(VIS_DIR, "exp2_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # 3. Quality group comparison (bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(5)
    width = 0.25
    for qi, (q, label) in enumerate([(1, "Low"), (2, "Medium"), (3, "High")]):
        means = [np.mean([dim_matrix[k, d] for k in range(len(qualities)) if qualities[k] == q])
                 for d in range(5)]
        stds = [np.std([dim_matrix[k, d] for k in range(len(qualities)) if qualities[k] == q])
                for d in range(5)]
        ax.bar(x + qi * width, means, width, label=label,
               color=quality_colors[q], yerr=stds, capsize=3, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(dim_labels, fontsize=11)
    ax.set_ylabel("Mean Activation", fontsize=12)
    ax.set_title("Cognitive Dimensions by Text Quality Level", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(VIS_DIR, "exp2_quality_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # 4. Dimension trend arrows (which dimensions track quality?)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if t > 0.01 else "#e74c3c" if t < -0.01 else "#95a5a6"
              for t in dim_trends]
    bars = ax.barh(dim_labels, dim_trends, color=colors, edgecolor="gray")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Trend (High quality - Low quality)", fontsize=12)
    ax.set_title("Which Dimensions Track Text Quality?", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, dim_trends):
        x_pos = val + 0.002 if val >= 0 else val - 0.002
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f"{val:+.4f}", va="center", ha=ha, fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(VIS_DIR, "exp2_dimension_trends.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    run_dimension_exploration()
