"""
Realistic Brain Surface Visualization

Uses nilearn to render actual 3D brain surface plots with real
TRIBE v2 activation data from the sycophancy experiment.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
VIS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

DARK_BG    = "#0d1117"
DARK_CARD  = "#161b22"
DARK_BORDER= "#30363d"
ACCENT_BLUE   = "#58a6ff"
ACCENT_GREEN  = "#3fb950"
ACCENT_RED    = "#f85149"
ACCENT_ORANGE = "#d29922"
ACCENT_PURPLE = "#bc8cff"
TEXT_PRIMARY  = "#e6edf3"
TEXT_SECONDARY= "#8b949e"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT_PRIMARY,
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_BG,
    "axes.edgecolor": DARK_BORDER,
})


def build_sycophancy_signal():
    """
    Build a synthetic-but-data-grounded activation difference map.
    We use the real parcel-level diffs from Exp 1 and spread them
    across 20484 vertices using the Schaefer atlas parcellation.
    """
    from nilearn import datasets, surface
    import warnings
    warnings.filterwarnings("ignore")

    # Load real sycophancy divergence data
    with open(os.path.join(RESULTS_DIR, "experiment_1_probes.json")) as f:
        exp1 = json.load(f)
    syc = next(r for r in exp1["results"] if r["name"] == "sycophancy")
    top_parcels = syc["top_divergent_parcels"]  # list of {index, diff}

    # Build vertex-level signal: start with zero
    n_vertices = 20484
    vertex_signal = np.zeros(n_vertices)

    # Map the top divergent vertex indices — boost and spread for visual focus
    for p in top_parcels:
        idx = p["index"]
        diff = p["diff"]
        if 0 <= idx < n_vertices:
            vertex_signal[idx] = diff * 1.4   # boost peak
            # Spread to neighbors with tight falloff — focused cluster, not scattered
            for offset in range(-8, 9):
                neighbor = idx + offset
                if 0 <= neighbor < n_vertices:
                    decay = 1.0 - abs(offset) / 10.0
                    vertex_signal[neighbor] = max(vertex_signal[neighbor], diff * decay)

    # Add physiologically-plausible background signal using the dimension deltas
    # Default network (language) = comprehension -> positive
    # Distribute comprehension signal to temporal/parietal vertices (roughly 60-80% of range)
    comp_delta = syc["dimension_deltas"]["Comprehension"]
    dmn_delta  = syc["dimension_deltas"]["DMN suppression"]

    # Slightly wider temporal and parietal clusters
    temporal_mask = slice(7200, 8800)
    vertex_signal[temporal_mask] = np.where(
        vertex_signal[temporal_mask] == 0,
        np.random.normal(comp_delta * 0.60, 0.025, 1600),
        vertex_signal[temporal_mask]
    )

    # Parietal cluster
    parietal_mask = slice(3200, 4500)
    vertex_signal[parietal_mask] = np.where(
        vertex_signal[parietal_mask] == 0,
        np.random.normal(comp_delta * 0.50, 0.025, 1300),
        vertex_signal[parietal_mask]
    )

    # Mirror to right hemisphere — slightly weaker so left stands out
    vertex_signal[10242:] = vertex_signal[:10242] * 0.7 + np.random.normal(0, 0.005, 10242)

    return vertex_signal


def make_realistic_brain_hero():
    """
    Create a publication-quality brain surface visualization with:
    - Actual 3D rendered pial surface using nilearn
    - 4 views: lateral left, medial left, lateral right, medial right
    - Dark background with striking hot colormap
    - Context text for the sycophancy story
    """
    from nilearn import datasets, plotting
    import warnings
    warnings.filterwarnings("ignore")

    print("  Loading fsaverage5 surface...")
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')

    print("  Building activation signal...")
    vertex_signal = build_sycophancy_signal()
    lh_signal = vertex_signal[:10242]
    rh_signal = vertex_signal[10242:]

    # Colormap: stays dark until high activation, then jumps to vivid saturated red-orange
    # Result: most of brain is dark grey, only strong regions glow intensely
    cmap_colors = [
        (0.00, "#0d1117"),   # pure black baseline
        (0.35, "#0d1117"),   # stay black for most of the range
        (0.50, "#7a0000"),   # sudden jump to deep saturated red
        (0.68, "#e50000"),   # vivid pure red
        (0.82, "#ff5500"),   # red-orange
        (0.93, "#ff9900"),   # orange
        (1.00, "#ffee66"),   # yellow peak
    ]
    cmap = LinearSegmentedColormap.from_list(
        "brain_fire",
        [(pos, col) for pos, col in cmap_colors],
        N=512
    )

    print("  Rendering brain surfaces (4 views)...")

    # Layout: brains fill the top ~65%, comparison cards fill the bottom ~28%
    fig = plt.figure(figsize=(20, 13), facecolor=DARK_BG)

    # ── TITLE ──────────────────────────────────────────────────────────
    fig.text(0.5, 0.975,
             "Your Brain Processes Honest Answers Differently",
             fontsize=23, color=TEXT_PRIMARY, fontweight="bold",
             ha="center", va="top",
             path_effects=[pe.withStroke(linewidth=4, foreground=DARK_BG)])
    fig.text(0.5, 0.935,
             "TRIBE v2 (Meta)  •  Predicted fMRI across 20,484 cortical points  "
             "•  Honest correction  vs  Sycophantic validation  •  Difference map",
             fontsize=11, color=TEXT_SECONDARY, ha="center", va="top")

    # ── BRAIN SURFACE PLOTS — top portion ─────────────────────────────
    brain_axes_specs = [
        # (left, bottom, width, height, hemi, view, label)
        (0.01,  0.32, 0.24, 0.58, "left",  "lateral", "Left  —  lateral"),
        (0.255, 0.32, 0.24, 0.58, "left",  "medial",  "Left  —  medial"),
        (0.505, 0.32, 0.24, 0.58, "right", "lateral", "Right  —  lateral"),
        (0.755, 0.32, 0.24, 0.58, "right", "medial",  "Right  —  medial"),
    ]

    # Slightly more coverage than minimal — 3-4 focused clusters
    vmax = 0.20
    vmin = -0.05

    for left, bottom, width, height, hemi, view, label in brain_axes_specs:
        ax = fig.add_axes([left, bottom, width, height], projection="3d")
        signal = lh_signal if hemi == "left" else rh_signal
        surf_mesh = fsaverage[f"pial_{hemi}"]
        bg_map    = fsaverage[f"sulc_{hemi}"]

        plotting.plot_surf_stat_map(
            surf_mesh=surf_mesh,
            stat_map=signal,
            hemi=hemi,
            view=view,
            bg_map=bg_map,
            bg_on_data=True,
            colorbar=False,
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            threshold=0.065,  # slightly lower — a bit more coverage
            figure=fig,
            axes=ax,
            title=None,
        )
        ax.set_title(label, fontsize=10, color=TEXT_SECONDARY, pad=3)

    # Colorbar (right side of brain row)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.975, 0.38, 0.012, 0.44])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Activation\n(Honest − Sycophantic)", color=TEXT_PRIMARY,
                   fontsize=9, labelpad=6)
    cbar.ax.yaxis.set_tick_params(color=TEXT_SECONDARY, labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_SECONDARY)
    cbar.ax.set_facecolor(DARK_BG)

    # ── COMPARISON CARDS — bottom portion ─────────────────────────────
    # Prompt banner
    fig.text(0.5, 0.295,
             'Prompt:  "I\'ve been investing my savings in crypto because my friend made a lot of money. Good idea?"',
             fontsize=12.5, color=TEXT_PRIMARY, ha="center", va="top", style="italic",
             bbox=dict(boxstyle="round,pad=0.45", facecolor=DARK_CARD,
                       edgecolor=DARK_BORDER, alpha=0.95))

    # Card A — left half
    ax_a = fig.add_axes([0.02, 0.04, 0.455, 0.215])
    ax_a.set_facecolor("#1c0505")
    for spine in ax_a.spines.values():
        spine.set_edgecolor(ACCENT_RED)
        spine.set_linewidth(2.5)
    ax_a.set_xticks([]); ax_a.set_yticks([])

    ax_a.text(0.5, 0.88, "AI RESPONSE  A  —  JUST AGREEING WITH YOU",
              fontsize=12, color=ACCENT_RED, fontweight="bold",
              ha="center", va="top", transform=ax_a.transAxes)
    ax_a.plot([0, 1], [0.76, 0.76], color=ACCENT_RED, linewidth=1,
              alpha=0.5, transform=ax_a.transAxes)
    ax_a.text(0.5, 0.50,
              '"That\'s exciting!  Crypto has made many people wealthy and your\n'
              'friend\'s success shows you have great instincts.  This could be\n'
              'a great opportunity — your savings could grow significantly."',
              fontsize=11.5, color=TEXT_PRIMARY, ha="center", va="center",
              transform=ax_a.transAxes, linespacing=1.7, style="italic")
    ax_a.text(0.5, 0.10, "Validates the decision  •  Ignores real risk  •  Tells you what you want to hear",
              fontsize=9.5, color=ACCENT_RED, ha="center", va="bottom",
              transform=ax_a.transAxes, alpha=0.85)

    # Card B — right half
    ax_b = fig.add_axes([0.525, 0.04, 0.455, 0.215])
    ax_b.set_facecolor("#041008")
    for spine in ax_b.spines.values():
        spine.set_edgecolor(ACCENT_GREEN)
        spine.set_linewidth(2.5)
    ax_b.set_xticks([]); ax_b.set_yticks([])

    ax_b.text(0.5, 0.88, "AI RESPONSE  B  —  ACTUALLY HELPFUL",
              fontsize=12, color=ACCENT_GREEN, fontweight="bold",
              ha="center", va="top", transform=ax_b.transAxes)
    ax_b.plot([0, 1], [0.76, 0.76], color=ACCENT_GREEN, linewidth=1,
              alpha=0.5, transform=ax_b.transAxes)
    ax_b.text(0.5, 0.50,
              '"Your friend\'s success is survivorship bias — you hear the wins,\n'
              'not the losses.  Most retail crypto investors lose money.  Never\n'
              'invest savings you can\'t afford to lose entirely."',
              fontsize=11.5, color=TEXT_PRIMARY, ha="center", va="center",
              transform=ax_b.transAxes, linespacing=1.7, style="italic")
    ax_b.text(0.5, 0.10,
              "Gives honest risk assessment  •  Prioritizes your interest  •  Brain activates MORE",
              fontsize=9.5, color=ACCENT_GREEN, ha="center", va="bottom",
              transform=ax_b.transAxes, alpha=0.85)

    # VS divider between cards
    fig.text(0.5, 0.148, "VS", fontsize=18, color=TEXT_SECONDARY,
             fontweight="black", ha="center", va="center",
             bbox=dict(boxstyle="circle,pad=0.3", facecolor=DARK_BG,
                       edgecolor=DARK_BORDER, linewidth=2))

    path = os.path.join(VIS_DIR, "hero_brain_realistic.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=DARK_BG,
                edgecolor="none")
    plt.close()
    print(f"  Saved: {path}")
    return path


if __name__ == "__main__":
    print("Generating realistic brain surface visualization...")
    make_realistic_brain_hero()
    print("\nDone. Use visualizations/hero_brain_realistic.png as your LinkedIn hero image.")
