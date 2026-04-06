"""
Experiment 1 Visualizations — Brain maps + cognitive dimension comparisons.

Generates:
1. Brain surface maps showing activation differences for each probe pair
2. Cognitive dimension bar charts comparing text_a vs text_b
3. Summary dashboard of all 10 probes
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
VIS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)


def load_results():
    with open(os.path.join(RESULTS_DIR, "experiment_1_probes.json")) as f:
        return json.load(f)


def plot_summary_dashboard(data):
    """Big summary: all 10 probes ranked with their metrics."""
    results = sorted(data["results"], key=lambda r: r["max_abs_diff"], reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle("Experiment 1: Text Probe Results — Kill Switch Test",
                 fontsize=16, fontweight="bold", y=0.98)

    names = [r["name"] for r in results]
    colors = ["#2ecc71" if r["max_abs_diff"] >= 0.3 else
              "#f39c12" if r["max_abs_diff"] >= 0.2 else
              "#e74c3c" for r in results]

    # 1. Max absolute difference
    ax = axes[0]
    bars = ax.barh(range(len(names)), [r["max_abs_diff"] for r in results], color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0.3, color="green", linestyle="--", alpha=0.7, label="Strong (0.3)")
    ax.axvline(0.1, color="orange", linestyle="--", alpha=0.7, label="Weak (0.1)")
    ax.set_xlabel("Max Absolute Difference")
    ax.set_title("Signal Strength", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    # 2. Percentage of parcels changed
    ax = axes[1]
    ax.barh(range(len(names)), [r["pct_parcels_changed"] for r in results], color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("% Parcels Changed (> 0.1)")
    ax.set_title("Spatial Spread", fontsize=13, fontweight="bold")

    # 3. Cognitive dimension deltas (heatmap)
    ax = axes[2]
    dim_names = ["Comprehension", "Memory encoding", "Sustained attention",
                 "Confusion", "DMN suppression"]
    delta_matrix = np.array([
        [r["dimension_deltas"][d] for d in dim_names]
        for r in results
    ])
    im = ax.imshow(delta_matrix, cmap="RdBu_r", aspect="auto",
                   vmin=-0.2, vmax=0.2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xticks(range(len(dim_names)))
    ax.set_xticklabels(["Comp", "Mem", "Attn", "Conf", "DMN"], fontsize=10)
    ax.set_title("Cognitive Dimension Deltas (B-A)", fontsize=13, fontweight="bold")

    # Add text annotations
    for i in range(len(names)):
        for j in range(len(dim_names)):
            val = delta_matrix[i, j]
            color = "white" if abs(val) > 0.1 else "black"
            ax.text(j, i, f"{val:+.03f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(VIS_DIR, "exp1_summary_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_probe_detail(result, probe_texts):
    """Detailed view for a single probe: text comparison + dimension bars."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 1])

    name = result["name"]
    dim = result["dimension"]

    fig.suptitle(f'Probe: {name.upper()} — {dim}',
                 fontsize=15, fontweight="bold", y=0.98)

    # Top: the text pair
    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis("off")

    text_a = probe_texts[name]["text_a"]
    text_b = probe_texts[name]["text_b"]

    # Wrap text
    import textwrap
    text_a_wrapped = textwrap.fill(text_a, width=75)
    text_b_wrapped = textwrap.fill(text_b, width=75)

    quality_a = "LOWER quality" if name != "clarity" else "Jargon-heavy"
    quality_b = "HIGHER quality" if name != "clarity" else "Clear explanation"

    ax_text.text(0.02, 0.95, f"Text A ({quality_a}):", fontsize=11,
                 fontweight="bold", color="#e74c3c", transform=ax_text.transAxes,
                 verticalalignment="top")
    ax_text.text(0.02, 0.78, text_a_wrapped, fontsize=9,
                 transform=ax_text.transAxes, verticalalignment="top",
                 family="monospace", wrap=True)

    ax_text.text(0.02, 0.45, f"Text B ({quality_b}):", fontsize=11,
                 fontweight="bold", color="#2ecc71", transform=ax_text.transAxes,
                 verticalalignment="top")
    ax_text.text(0.02, 0.28, text_b_wrapped, fontsize=9,
                 transform=ax_text.transAxes, verticalalignment="top",
                 family="monospace", wrap=True)

    # Middle left: Cognitive dimension deltas
    ax_dims = fig.add_subplot(gs[1, 0])
    dim_names = list(result["dimension_deltas"].keys())
    dim_values = list(result["dimension_deltas"].values())
    dim_colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in dim_values]

    bars = ax_dims.barh(dim_names, dim_values, color=dim_colors, edgecolor="gray")
    ax_dims.axvline(0, color="black", linewidth=0.8)
    ax_dims.set_xlabel("Delta (B - A)")
    ax_dims.set_title("Cognitive Dimension Changes", fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, dim_values):
        x_pos = val + 0.005 if val >= 0 else val - 0.005
        ha = "left" if val >= 0 else "right"
        ax_dims.text(x_pos, bar.get_y() + bar.get_height()/2,
                     f"{val:+.4f}", va="center", ha=ha, fontsize=9)

    # Middle right: Key metrics
    ax_metrics = fig.add_subplot(gs[1, 1])
    ax_metrics.axis("off")

    metrics_text = (
        f"Max absolute difference:  {result['max_abs_diff']:.4f}\n"
        f"Mean absolute difference: {result['mean_abs_diff']:.4f}\n"
        f"Std of differences:       {result['std_diff']:.4f}\n"
        f"% parcels changed (>0.1): {result['pct_parcels_changed']:.1f}%\n\n"
    )

    signal = "STRONG" if result["max_abs_diff"] >= 0.3 else \
             "MODERATE" if result["max_abs_diff"] >= 0.2 else "WEAK"
    signal_color = "#2ecc71" if signal == "STRONG" else \
                   "#f39c12" if signal == "MODERATE" else "#e74c3c"

    ax_metrics.text(0.1, 0.8, "Key Metrics", fontsize=13, fontweight="bold",
                    transform=ax_metrics.transAxes, verticalalignment="top")
    ax_metrics.text(0.1, 0.65, metrics_text, fontsize=11,
                    transform=ax_metrics.transAxes, verticalalignment="top",
                    family="monospace")
    ax_metrics.text(0.1, 0.2, f"Signal: {signal}", fontsize=16,
                    fontweight="bold", color=signal_color,
                    transform=ax_metrics.transAxes)

    # Bottom: Top divergent vertices (bar chart)
    ax_parcels = fig.add_subplot(gs[2, :])
    top = result["top_divergent_parcels"]
    indices = [f"v{p['index']}" for p in top]
    diffs = [p["diff"] for p in top]
    parcel_colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in diffs]
    ax_parcels.bar(indices, diffs, color=parcel_colors, edgecolor="gray")
    ax_parcels.axhline(0, color="black", linewidth=0.8)
    ax_parcels.set_ylabel("Activation Difference (B - A)")
    ax_parcels.set_title("Top 10 Most Divergent Brain Vertices", fontweight="bold")
    ax_parcels.tick_params(axis="x", rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(VIS_DIR, f"exp1_probe_{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_brain_surface_diff(result):
    """Plot brain surface map showing activation differences for a probe."""
    try:
        from nilearn import datasets, plotting, surface
        import nibabel as nib
    except ImportError:
        print(f"  [skip brain map for {result['name']} - nilearn not available]")
        return

    name = result["name"]

    # Build a vertex-level difference map from the top divergent parcels
    # We don't have the full diff array saved, so we'll create a sparse visualization
    n_vertices_hemi = 10242  # fsaverage5
    lh_data = np.zeros(n_vertices_hemi)
    rh_data = np.zeros(n_vertices_hemi)

    for parcel in result["top_divergent_parcels"]:
        idx = parcel["index"]
        val = parcel["diff"]
        if idx < n_vertices_hemi:
            # Left hemisphere
            # Spread the value to nearby vertices for visibility
            for offset in range(-15, 16):
                neighbor = idx + offset
                if 0 <= neighbor < n_vertices_hemi:
                    weight = 1.0 - abs(offset) / 16.0
                    lh_data[neighbor] = max(lh_data[neighbor], val * weight) if val > 0 \
                        else min(lh_data[neighbor], val * weight)
        else:
            rh_idx = idx - n_vertices_hemi
            for offset in range(-15, 16):
                neighbor = rh_idx + offset
                if 0 <= neighbor < n_vertices_hemi:
                    weight = 1.0 - abs(offset) / 16.0
                    rh_data[neighbor] = max(rh_data[neighbor], val * weight) if val > 0 \
                        else min(rh_data[neighbor], val * weight)

    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    vmax = max(abs(lh_data).max(), abs(rh_data).max(), 0.1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             subplot_kw={"projection": "3d"})
    fig.suptitle(f"Brain Activation Differences: {name.upper()}\n"
                 f"Red = more active in Text B (better), Blue = more active in Text A",
                 fontsize=13, fontweight="bold")

    views = [("left", "lateral"), ("left", "medial"),
             ("right", "lateral"), ("right", "medial")]

    for ax_idx, (hemi, view) in enumerate(views):
        ax = axes[ax_idx // 2][ax_idx % 2]
        surf_mesh = fsavg[f"pial_{hemi}"]
        bg_map = fsavg[f"sulc_{hemi}"]
        stat_map = lh_data if hemi == "left" else rh_data

        plotting.plot_surf_stat_map(
            surf_mesh,
            stat_map=stat_map,
            bg_map=bg_map,
            hemi=hemi,
            view=view,
            colorbar=False,
            cmap="RdBu_r",
            vmax=vmax,
            threshold=0.02,
            axes=ax,
        )
        ax.set_title(f"{hemi.capitalize()} {view}", fontsize=10)

    # Add colorbar manually
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                                norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.4, label="Activation Difference (B - A)")

    path = os.path.join(VIS_DIR, f"exp1_brain_{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_pipeline_explainer():
    """Visual explanation of how the scoring pipeline works."""
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.axis("off")
    fig.suptitle("How TRIBE v2 Scoring Works — The Full Pipeline",
                 fontsize=16, fontweight="bold")

    # Pipeline boxes
    steps = [
        ("1. INPUT TEXT", "Your text\n(sentence or\nparagraph)", "#3498db"),
        ("2. TTS", "Text-to-Speech\nconverts to\naudio (.mp3)", "#9b59b6"),
        ("3. WHISPERX", "Transcribes +\nword-aligns\naudio", "#e67e22"),
        ("4. LLAMA 3.2", "Extracts word\nembeddings\n(3B params)", "#e74c3c"),
        ("5. WAV2VEC", "Extracts audio\nfeatures\n(w2v-bert-2.0)", "#1abc9c"),
        ("6. TRIBE v2", "Predicts brain\nactivation\n(20,484 vertices)", "#2c3e50"),
        ("7. ROI EXTRACT", "Maps to 5\ncognitive\ndimensions", "#27ae60"),
        ("8. SCORE", "Comprehension\nMemory, Attn\nConfusion, DMN", "#f39c12"),
    ]

    box_width = 0.1
    box_height = 0.35
    y_center = 0.4
    gap = 0.017

    for i, (title, desc, color) in enumerate(steps):
        x = 0.02 + i * (box_width + gap)

        # Box
        rect = plt.Rectangle((x, y_center - box_height/2), box_width, box_height,
                              facecolor=color, alpha=0.85, edgecolor="black",
                              linewidth=1.5, transform=ax.transAxes)
        ax.add_patch(rect)

        # Title
        ax.text(x + box_width/2, y_center + box_height/2 - 0.03, title,
                ha="center", va="top", fontsize=8, fontweight="bold",
                color="white", transform=ax.transAxes)

        # Description
        ax.text(x + box_width/2, y_center - 0.02, desc,
                ha="center", va="center", fontsize=7.5,
                color="white", transform=ax.transAxes)

        # Arrow
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + box_width + gap - 0.005, y_center),
                        xytext=(x + box_width + 0.005, y_center),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2),
                        transform=ax.transAxes)

    # Bottom: timing explanation
    timing_text = (
        "WHY IT TAKES ~3 MINUTES PER TEXT:\n"
        "Step 2 (TTS): ~2s  |  Step 3 (WhisperX): ~7s  |  "
        "Step 4 (Llama 3.2 3B): ~12s loading + ~2s inference  |  "
        "Step 5 (Wav2Vec): ~5s  |  Step 6 (TRIBE inference): ~30s\n"
        "TOTAL: ~60s compute + overhead = ~3 min per text.  "
        "First run is slower because models must be loaded from disk."
    )
    ax.text(0.5, 0.02, timing_text, ha="center", va="bottom", fontsize=9,
            transform=ax.transAxes, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", alpha=0.9))

    path = os.path.join(VIS_DIR, "exp1_pipeline_explainer.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    # Load probe texts from text_probes.py
    from text_probes import PROBE_PAIRS
    probe_texts = {p["name"]: p for p in PROBE_PAIRS}

    data = load_results()

    print("=" * 60)
    print("  Generating Experiment 1 Visualizations")
    print("=" * 60)

    # 1. Pipeline explainer
    print("\n[1/3] Pipeline explainer...")
    plot_pipeline_explainer()

    # 2. Summary dashboard
    print("[2/3] Summary dashboard...")
    plot_summary_dashboard(data)

    # 3. Detailed views for each probe
    print("[3/3] Individual probe details...")
    for result in data["results"]:
        plot_probe_detail(result, probe_texts)

    # 4. Brain surface maps (bonus)
    print("\n[Bonus] Brain surface maps...")
    for result in data["results"]:
        if result["max_abs_diff"] >= 0.3:  # Only for strong signals
            plot_brain_surface_diff(result)

    print(f"\nAll visualizations saved to: {VIS_DIR}")
    print(f"Total files: {len(os.listdir(VIS_DIR))}")


if __name__ == "__main__":
    main()
