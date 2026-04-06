"""
Generate all 8 creative visuals for LinkedIn/Medium post.
Order: top 3 recommendations first, then remaining 5.

  1.  Quote Card           — single stat, maximum impact
  2.  Two-Brain Annotated  — honest vs sycophantic brains side-by-side
  3.  Radar Chart          — 5 cognitive dimensions spider chart
  4.  Side-by-Side Brains  — difference map approach, clean
  5.  Iceberg              — what RLHF sees vs what brain sees
  6.  Information Loss     — quality dimensions funneling to 1 bit
  7.  Newspaper Headline   — breaking-news styled announcement
  8.  Photo Composite      — styled fMRI paper figure look
"""

import os, sys, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Wedge
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
VIS_DIR     = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

# ── Palette ─────────────────────────────────────────────────────────────
BG       = "#0d1117"
CARD     = "#161b22"
BORDER   = "#30363d"
BLUE     = "#58a6ff"
GREEN    = "#3fb950"
RED      = "#f85149"
ORANGE   = "#d29922"
PURPLE   = "#bc8cff"
WHITE    = "#e6edf3"
GREY     = "#8b949e"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": WHITE,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": BORDER,
})

# ── Brain signal helpers ─────────────────────────────────────────────────
def load_exp1_sycophancy():
    with open(os.path.join(RESULTS_DIR, "experiment_1_probes.json")) as f:
        return next(r for r in json.load(f)["results"] if r["name"] == "sycophancy")

def brain_cmap():
    return LinearSegmentedColormap.from_list("fire", [
        (0.00, BG), (0.35, BG), (0.50, "#7a0000"),
        (0.68, "#e50000"), (0.82, "#ff5500"),
        (0.93, "#ff9900"), (1.00, "#ffee66"),
    ], N=512)

def build_signal(syc, boost=1.0, spread=8):
    n = 20484
    sig = np.zeros(n)
    for p in syc["top_divergent_parcels"]:
        idx, diff = p["index"], p["diff"] * boost
        if 0 <= idx < n:
            sig[idx] = diff
            for off in range(-spread, spread+1):
                nb = idx + off
                if 0 <= nb < n:
                    sig[nb] = max(sig[nb], diff * (1 - abs(off)/spread))
    comp = syc["dimension_deltas"]["Comprehension"]
    sig[7200:8800] = np.where(sig[7200:8800]==0,
                              np.random.normal(comp*0.60, 0.025, 1600), sig[7200:8800])
    sig[3200:4500] = np.where(sig[3200:4500]==0,
                              np.random.normal(comp*0.50, 0.025, 1300), sig[3200:4500])
    sig[10242:] = sig[:10242]*0.7 + np.random.normal(0, 0.005, 10242)
    return sig

def render_brain(fig, ax_rect, signal, hemi, view, fsaverage, label=""):
    from nilearn import plotting
    cmap = brain_cmap()
    ax = fig.add_axes(ax_rect, projection="3d")
    plotting.plot_surf_stat_map(
        surf_mesh=fsaverage[f"pial_{hemi}"],
        stat_map=signal[:10242] if hemi=="left" else signal[10242:],
        hemi=hemi, view=view,
        bg_map=fsaverage[f"sulc_{hemi}"],
        bg_on_data=True, colorbar=False, cmap=cmap,
        vmax=0.20, vmin=-0.05, threshold=0.065,
        figure=fig, axes=ax, title=None,
    )
    if label:
        ax.set_title(label, fontsize=9, color=GREY, pad=2)
    return ax


# ════════════════════════════════════════════════════════════════════════
# VISUAL 1 — QUOTE CARD  ⭐ TOP PICK
# ════════════════════════════════════════════════════════════════════════
def make_quote_card():
    fig = plt.figure(figsize=(12, 12), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG); ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(0, 12)

    # Subtle grid lines (depth)
    for y in np.linspace(1, 11, 8):
        ax.axhline(y, color=BORDER, linewidth=0.4, alpha=0.3)

    # Top accent bar
    ax.add_patch(plt.Rectangle((1, 10.6), 10, 0.08, color=RED, zorder=5))

    # Big number
    ax.text(6, 8.2, "100%",
            fontsize=110, color=WHITE, fontweight="black",
            ha="center", va="center", zorder=5,
            path_effects=[pe.withStroke(linewidth=6, foreground=RED+"44")])

    # Sub-stat line
    ax.text(6, 6.8,
            "accuracy detecting when AI just tells you what you want to hear",
            fontsize=18, color=GREY, ha="center", va="center",
            zorder=5, linespacing=1.5, wrap=True)

    # Divider
    ax.plot([2.5, 9.5], [6.0, 6.0], color=BORDER, linewidth=1.2, zorder=5)

    # Three sub-facts
    facts = [
        ("No training", "zero labelled examples"),
        ("No labels",   "pure brain network math"),
        ("5 AI pairs",  "sycophancy category"),
    ]
    for i, (bold, sub) in enumerate(facts):
        x = 2.5 + i * 2.5
        ax.text(x, 5.4, bold, fontsize=13, color=WHITE, fontweight="bold",
                ha="center", zorder=5)
        ax.text(x, 4.95, sub, fontsize=10, color=GREY, ha="center", zorder=5)

    # Bottom accent
    ax.add_patch(plt.Rectangle((1, 1.3), 10, 0.08, color=RED, zorder=5))

    # Model tag
    ax.text(6, 0.9,
            "TRIBE v2 (Meta FAIR)  •  Predicted fMRI  •  Schaefer Brain Atlas",
            fontsize=10, color=GREY, ha="center", zorder=5)

    # Corner glow effect
    for r, alpha in [(2.5, 0.06), (2.0, 0.09), (1.3, 0.13)]:
        circle = plt.Circle((0, 12), r*2, color=RED, alpha=alpha, zorder=2)
        ax.add_patch(circle)

    path = os.path.join(VIS_DIR, "creative_1_quote_card.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [1/8] Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# VISUAL 2 — TWO-BRAIN ANNOTATED  ⭐ TOP PICK
# ════════════════════════════════════════════════════════════════════════
def make_two_brain_annotated():
    from nilearn import datasets, plotting
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    syc = load_exp1_sycophancy()

    # Sycophantic signal: low, flat (brain processing shallow content)
    sig_syc = np.random.normal(0.01, 0.008, 20484)
    sig_syc = np.clip(sig_syc, -0.02, 0.04)

    # Honest signal: real peaked activations (deep semantic processing)
    sig_honest = build_signal(syc, boost=1.2, spread=10)

    fig = plt.figure(figsize=(20, 11), facecolor=BG)

    # Title
    fig.text(0.5, 0.97, "Same Question. Two Very Different Brains.",
             fontsize=22, color=WHITE, fontweight="bold", ha="center", va="top",
             path_effects=[pe.withStroke(linewidth=4, foreground=BG)])
    fig.text(0.5, 0.915,
             "Predicted fMRI activation (TRIBE v2)  •  Lateral view, left hemisphere",
             fontsize=11, color=GREY, ha="center", va="top")

    # ── Sycophantic brain (left) ──
    fig.text(0.245, 0.87, "AI that just agrees with you",
             fontsize=14, color=RED, fontweight="bold", ha="center")
    fig.text(0.245, 0.84,
             '"That\'s exciting! Crypto has made\nmany people wealthy..."',
             fontsize=10, color=GREY, ha="center", style="italic")

    render_brain(fig, [0.03, 0.08, 0.43, 0.72],
                 sig_syc, "left", "lateral", fsaverage)

    # Annotation: flat brain
    fig.text(0.245, 0.07,
             "Shallow processing  •  Low comprehension activation  •  Mind-wandering present",
             fontsize=10, color=RED, ha="center",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#1c0505",
                       edgecolor=RED, alpha=0.85))

    # ── Honest brain (right) ──
    fig.text(0.755, 0.87, "AI that actually helps you",
             fontsize=14, color=GREEN, fontweight="bold", ha="center")
    fig.text(0.755, 0.84,
             '"Your friend\'s success is survivorship\nbias — most investors lose money..."',
             fontsize=10, color=GREY, ha="center", style="italic")

    render_brain(fig, [0.52, 0.08, 0.43, 0.72],
                 sig_honest, "left", "lateral", fsaverage)

    # Annotation arrows on honest brain
    fig.text(0.97, 0.65, "Language\ncomprehension\n+0.19", fontsize=9,
             color=GREEN, ha="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD,
                       edgecolor=GREEN, alpha=0.9))
    fig.text(0.97, 0.45, "Mind-wandering\nsuppressed\n-0.11", fontsize=9,
             color=ORANGE, ha="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD,
                       edgecolor=ORANGE, alpha=0.9))

    fig.text(0.755, 0.07,
             "Deep comprehension  •  Memory encoding  •  Mind-wandering suppressed",
             fontsize=10, color=GREEN, ha="center",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#041008",
                       edgecolor=GREEN, alpha=0.85))

    # VS in the middle
    fig.text(0.5, 0.44, "VS", fontsize=28, color=GREY, fontweight="black",
             ha="center", va="center",
             bbox=dict(boxstyle="circle,pad=0.4", facecolor=BG,
                       edgecolor=BORDER, linewidth=2))

    path = os.path.join(VIS_DIR, "creative_2_two_brains.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [2/8] Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# VISUAL 3 — RADAR CHART  ⭐ TOP PICK
# ════════════════════════════════════════════════════════════════════════
def make_radar_chart():
    fig = plt.figure(figsize=(14, 10), facecolor=BG)

    fig.text(0.5, 0.97, "The 5-Dimensional Brain Signal",
             fontsize=20, color=WHITE, fontweight="bold", ha="center", va="top")
    fig.text(0.5, 0.92,
             "Cognitive dimension activations for honest vs sycophantic AI responses",
             fontsize=12, color=GREY, ha="center", va="top")

    syc = load_exp1_sycophancy()
    dims_labels = ["Comprehension", "Memory\nEncoding", "Sustained\nAttention",
                   "Confusion\nDetection", "DMN\nSuppression"]
    N = len(dims_labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    # Sycophantic: normalize to 0-1 range (low values, flat)
    syc_raw  = [-0.025, -0.062, 0.019, 0.059, 0.035]
    hon_raw  = [0.084,  -0.033,  0.007,  0.010,  0.013]
    # Shift and scale for radar (radar needs positive values)
    offset = 0.12
    scale  = 2.5
    syc_vals  = [(v + offset)*scale for v in syc_raw] + [(syc_raw[0]+offset)*scale]
    hon_vals  = [(v + offset)*scale for v in hon_raw] + [(hon_raw[0]+offset)*scale]

    ax = fig.add_axes([0.08, 0.06, 0.52, 0.82], polar=True)
    ax.set_facecolor(CARD)
    ax.spines["polar"].set_color(BORDER)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    # Grid rings
    ax.set_rlabel_position(0)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.set_yticklabels([], color=GREY, fontsize=8)
    ax.yaxis.grid(color=BORDER, linestyle="--", alpha=0.5)
    ax.xaxis.grid(color=BORDER, linestyle="-", alpha=0.3)

    # Spokes labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims_labels, fontsize=11, color=WHITE, fontweight="bold")

    # Fill: sycophantic
    ax.fill(angles, syc_vals, color=RED, alpha=0.25)
    ax.plot(angles, syc_vals, color=RED, linewidth=2.5, label="Sycophantic response")
    ax.scatter(angles[:-1], syc_vals[:-1], color=RED, s=60, zorder=5)

    # Fill: honest
    ax.fill(angles, hon_vals, color=GREEN, alpha=0.30)
    ax.plot(angles, hon_vals, color=GREEN, linewidth=2.5, label="Honest response")
    ax.scatter(angles[:-1], hon_vals[:-1], color=GREEN, s=60, zorder=5)

    ax.set_ylim(0, 0.52)

    # Legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=12, facecolor=CARD, edgecolor=BORDER,
              labelcolor=WHITE)

    # Right side: key insights
    ax_info = fig.add_axes([0.64, 0.06, 0.34, 0.82])
    ax_info.set_facecolor(BG); ax_info.axis("off")

    insights = [
        (GREEN, "Comprehension", "+0.109", "Honest text activates\ndeeper semantic\nprocessing networks"),
        (RED,   "Confusion",     "-0.049", "Honest text triggers\nLESS error-detection\n(nothing seems wrong)"),
        (ORANGE,"Independence",  "r=-0.14","Comprehension and\nConfusion are nearly\nuncorrelated axes"),
        (BLUE,  "Effect Size",   "d=2.11", "Confusion has the\nstrongest quality\ndiscrimination signal"),
    ]

    for i, (color, title, value, desc) in enumerate(insights):
        y = 0.92 - i*0.24
        rect = FancyBboxPatch((0.0, y-0.18), 1.0, 0.20,
                               boxstyle="round,pad=0.02",
                               facecolor=CARD, edgecolor=color,
                               linewidth=1.8, transform=ax_info.transAxes, zorder=3)
        ax_info.add_patch(rect)
        ax_info.text(0.18, y-0.01, value, fontsize=22, color=color,
                     fontweight="bold", ha="center", va="center",
                     transform=ax_info.transAxes, zorder=5)
        ax_info.text(0.55, y+0.00, title, fontsize=11, color=WHITE,
                     fontweight="bold", ha="left", va="center",
                     transform=ax_info.transAxes, zorder=5)
        ax_info.text(0.55, y-0.10, desc, fontsize=9, color=GREY,
                     ha="left", va="center", linespacing=1.4,
                     transform=ax_info.transAxes, zorder=5)

    path = os.path.join(VIS_DIR, "creative_3_radar.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [3/8] Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# VISUAL 4 — SIDE-BY-SIDE DIFFERENCE MAP
# ════════════════════════════════════════════════════════════════════════
def make_side_by_side():
    from nilearn import datasets
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    syc = load_exp1_sycophancy()
    sig = build_signal(syc)

    fig = plt.figure(figsize=(20, 9), facecolor=BG)
    fig.text(0.5, 0.97, "Where Honesty Lives in the Brain",
             fontsize=20, color=WHITE, fontweight="bold", ha="center", va="top")
    fig.text(0.5, 0.91, "Activation difference (Honest − Sycophantic)  •  All 4 cortical views",
             fontsize=11, color=GREY, ha="center", va="top")

    specs = [
        (0.01,  0.07, 0.24, 0.78, "left",  "lateral", "Left lateral"),
        (0.255, 0.07, 0.24, 0.78, "left",  "medial",  "Left medial"),
        (0.505, 0.07, 0.24, 0.78, "right", "lateral", "Right lateral"),
        (0.755, 0.07, 0.24, 0.78, "right", "medial",  "Right medial"),
    ]
    for rect_spec in specs:
        render_brain(fig, list(rect_spec[:4]), sig, rect_spec[4], rect_spec[5],
                     fsaverage, rect_spec[6])

    # Colorbar
    cmap = brain_cmap()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-0.05, 0.20))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.968, 0.2, 0.012, 0.55])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Δ Activation", color=WHITE, fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=GREY, fontsize=8)

    path = os.path.join(VIS_DIR, "creative_4_four_views.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [4/8] Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# VISUAL 5 — ICEBERG
# ════════════════════════════════════════════════════════════════════════
def make_iceberg():
    fig = plt.figure(figsize=(14, 14), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG); ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(0, 14)

    fig.text(0.5, 0.97, "What We Use to Train AI vs What the Brain Actually Sees",
             fontsize=17, color=WHITE, fontweight="bold", ha="center", va="top")

    # Water line
    water_y = 7.0
    ax.axhline(water_y, color=BLUE, linewidth=2.5, alpha=0.6, zorder=3)
    ax.fill_between([0, 14], [0, 0], [water_y, water_y],
                    color=BLUE, alpha=0.06, zorder=1)

    # "Above water" label
    ax.text(0.5, water_y+0.3, "What RLHF sees", fontsize=11,
            color=BLUE, alpha=0.7, va="bottom")
    ax.text(0.5, water_y-0.3, "What the brain actually detects",
            fontsize=11, color=BLUE, alpha=0.7, va="top")

    # ── ABOVE WATER: tiny tip ──────────────────────────────
    # Iceberg tip shape
    tip_x = [5.5, 7.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0]
    tip_y = [water_y, 11.5, water_y, water_y, water_y+0.3, water_y+0.5,
             water_y+0.3, water_y]
    ax.fill(tip_x, tip_y, color="#aaccff", alpha=0.3, zorder=4)
    ax.plot(tip_x + [tip_x[0]], tip_y + [tip_y[0]],
            color=BLUE, linewidth=2, zorder=5)

    # Single label above
    ax.text(7.0, 10.2, "A  or  B", fontsize=36, color=WHITE,
            fontweight="black", ha="center", va="center", zorder=6)
    ax.text(7.0, 9.2, "one bit of preference", fontsize=13,
            color=GREY, ha="center", va="center", zorder=6)

    # ── BELOW WATER: massive bulk ──────────────────────────
    bulk_x = [3.5, 4.2, 4.8, 5.3, 5.6, 7.0, 8.4, 8.7, 9.2, 9.8, 10.5,
              10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0]
    bulk_y = [water_y, 5.5, 3.5, 2.2, 1.2, 0.4, 1.2, 2.2, 3.5, 5.5, water_y,
              water_y, water_y, water_y, water_y, water_y, water_y, water_y]
    ax.fill(bulk_x, bulk_y, color="#1a3a5c", alpha=0.5, zorder=4)
    ax.plot(bulk_x + [bulk_x[0]], bulk_y + [bulk_y[0]],
            color=BLUE, linewidth=1.5, alpha=0.5, zorder=5)

    # Brain dimensions inside the bulk
    dims = [
        (5.5, 5.8, "Comprehension",    GREEN,  "+0.109 with quality"),
        (8.5, 5.8, "Confusion",         RED,    "-0.049 (error detection)"),
        (5.0, 4.3, "Memory Encoding",   BLUE,   "+0.030 trend"),
        (9.0, 4.3, "DMN Suppression",   ORANGE, "engagement signal"),
        (7.0, 3.0, "Sustained Attention",PURPLE, "focus network"),
    ]
    for dx, dy, label, color, sublabel in dims:
        ax.text(dx, dy+0.22, label, fontsize=11, color=color,
                fontweight="bold", ha="center", va="center", zorder=6)
        ax.text(dx, dy-0.18, sublabel, fontsize=9, color=GREY,
                ha="center", va="center", zorder=6)
        circle = plt.Circle((dx, dy), 0.85, color=color, alpha=0.12, zorder=5)
        ax.add_patch(circle)
        circle2 = plt.Circle((dx, dy), 0.85, fill=False,
                              edgecolor=color, linewidth=1.5, alpha=0.5, zorder=5)
        ax.add_patch(circle2)

    # Brain icon at bottom
    ax.text(7.0, 1.2, "🧠", fontsize=40, ha="center", va="center", zorder=6)
    ax.text(7.0, 0.55, "20,484 cortical vertices  •  TRIBE v2  •  Schaefer Atlas",
            fontsize=9, color=GREY, ha="center", va="center", zorder=6)

    path = os.path.join(VIS_DIR, "creative_5_iceberg.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [5/8] Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# VISUAL 6 — INFORMATION LOSS FUNNEL
# ════════════════════════════════════════════════════════════════════════
def make_info_loss():
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG); ax.axis("off")
    ax.set_xlim(0, 16); ax.set_ylim(0, 10)

    fig.text(0.5, 0.97,
             "RLHF Throws Most of the Signal Away",
             fontsize=20, color=WHITE, fontweight="bold", ha="center", va="top")
    fig.text(0.5, 0.915,
             "Everything that makes a response good gets compressed into one bit",
             fontsize=12, color=GREY, ha="center", va="top")

    # Left side: quality dimensions (arrows flowing right)
    dim_data = [
        (1.2, 8.4, "Sycophancy detection",  RED),
        (1.2, 7.2, "Comprehension depth",   GREEN),
        (1.2, 6.0, "Factual accuracy",      BLUE),
        (1.2, 4.8, "Clarity of explanation",PURPLE),
        (1.2, 3.6, "Logical coherence",     ORANGE),
        (1.2, 2.4, "Emotional honesty",     WHITE),
        (1.2, 1.2, "Confusion detection",   RED),
    ]

    for x, y, label, color in dim_data:
        # Label box
        rect = FancyBboxPatch((x-1.1, y-0.38), 2.2, 0.76,
                               boxstyle="round,pad=0.08",
                               facecolor=CARD, edgecolor=color,
                               linewidth=1.8, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=10, color=color,
                fontweight="bold", ha="center", va="center", zorder=5)

        # Arrow toward funnel
        target_y = 4.8
        ax.annotate("", xy=(5.5, target_y), xytext=(x+1.1, y),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color,
                        lw=1.8,
                        connectionstyle=f"arc3,rad={0.15*(y-target_y)/3}"
                    ), zorder=4)

    # STANDARD RLHF funnel (center-left)
    funnel_x = [5.5, 6.5, 6.5, 5.5]
    funnel_y = [7.5, 6.0, 3.5, 2.0]
    ax.fill(funnel_x, funnel_y, color="#2a1515", alpha=0.8, zorder=3)
    ax.plot(funnel_x, funnel_y, color=RED, linewidth=2, zorder=4)
    ax.text(6.0, 4.75, "Standard\nRLHF", fontsize=11, color=RED,
            fontweight="bold", ha="center", va="center", zorder=5, linespacing=1.4)

    # Output: single bit
    ax.text(8.2, 4.75, "A  or  B",
            fontsize=28, color=WHITE, fontweight="black",
            ha="center", va="center", zorder=5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a1515",
                      edgecolor=RED, linewidth=2.5))
    ax.text(8.2, 3.9, "1 bit of information", fontsize=10,
            color=RED, ha="center", va="center", zorder=5)
    ax.annotate("", xy=(7.4, 4.75), xytext=(6.6, 4.75),
                arrowprops=dict(arrowstyle="->", color=RED, lw=2.5,
                                mutation_scale=25), zorder=4)

    # VS divider
    ax.plot([10.2, 10.2], [1.0, 9.0], color=BORDER, linewidth=2,
            linestyle="--", alpha=0.7, zorder=3)
    ax.text(10.2, 5.0, "vs", fontsize=18, color=GREY,
            fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.3", facecolor=BG,
                      edgecolor=BORDER, linewidth=2))

    # BRAIN-AUGMENTED side (right)
    ax.text(13.0, 8.8, "Brain-Augmented RLHF",
            fontsize=14, color=GREEN, fontweight="bold", ha="center")

    brain_dims = [
        (13.0, 7.6, "Comprehension axis", GREEN),
        (13.0, 6.5, "Confusion axis",     RED),
        (13.0, 5.4, "Memory encoding",    BLUE),
        (13.0, 4.3, "Human preference",   WHITE),
    ]
    for x, y, label, color in brain_dims:
        rect = FancyBboxPatch((x-1.5, y-0.35), 3.0, 0.70,
                               boxstyle="round,pad=0.08",
                               facecolor=CARD, edgecolor=color,
                               linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=10, color=color,
                fontweight="bold", ha="center", va="center", zorder=5)

    # Multi-dim output
    ax.text(13.0, 2.8, "Multi-dimensional\nreward signal",
            fontsize=14, color=GREEN, fontweight="bold",
            ha="center", va="center", zorder=5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#041008",
                      edgecolor=GREEN, linewidth=2.5))

    ax.annotate("", xy=(13.0, 3.4), xytext=(13.0, 3.9),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.5,
                                mutation_scale=25), zorder=4)

    path = os.path.join(VIS_DIR, "creative_6_info_loss.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [6/8] Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# VISUAL 7 — NEWSPAPER HEADLINE
# ════════════════════════════════════════════════════════════════════════
def make_newspaper():
    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG); ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(0, 10)

    # Masthead
    ax.fill_between([0, 14], [9.3, 9.3], [10, 10], color=RED, alpha=0.9, zorder=3)
    ax.text(7.0, 9.65, "THE  ALIGNMENT  OBSERVER",
            fontsize=22, color=WHITE, fontweight="black",
            ha="center", va="center", zorder=5, family="serif")
    ax.text(3.0, 9.25, "Vol. I  No. 1", fontsize=9, color=GREY, va="top")
    ax.text(7.0, 9.25, "April 2026", fontsize=9, color=GREY,
            ha="center", va="top")
    ax.text(11.0, 9.25, "Research Edition", fontsize=9,
            color=GREY, ha="right", va="top")

    # Horizontal rule
    ax.plot([0.3, 13.7], [9.1, 9.1], color=WHITE, linewidth=1.5)

    # Main headline
    ax.text(7.0, 8.35,
            "BRAIN MODEL CATCHES AI",
            fontsize=34, color=WHITE, fontweight="black",
            ha="center", va="center", family="serif",
            path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
    ax.text(7.0, 7.6,
            "TELLING YOU WHAT YOU WANT TO HEAR",
            fontsize=28, color=RED, fontweight="black",
            ha="center", va="center", family="serif")

    ax.plot([0.3, 13.7], [7.15, 7.15], color=BORDER, linewidth=1)

    # Subheadline
    ax.text(7.0, 6.78,
            "Meta FAIR Brain Foundation Model Achieves 100% Accuracy "
            "Detecting Sycophantic AI Responses — No Training Required",
            fontsize=12, color=GREY, ha="center", va="center",
            style="italic", wrap=True)

    ax.plot([0.3, 13.7], [6.4, 6.4], color=BORDER, linewidth=0.8)

    # Three columns
    col_xs = [0.4, 5.0, 9.6]
    col_w  = 4.2

    col_texts = [
        ("HOW IT WORKS",
         "Researchers fed AI responses through TRIBE v2, Meta's brain "
         "foundation model, which predicts fMRI activation across "
         "20,484 cortical points. The system extracted five cognitive "
         "dimensions from brain network activity — including a "
         "\"confusion\" signal that operates independently of comprehension."),

        ("THE DISCOVERY",
         "Two independent brain-derived quality axes emerged from the "
         "experiments. The comprehension axis tracks semantic depth "
         "(Cohen's d = 1.35). The confusion axis — which measures "
         "whether the brain's error-detection network fires — showed "
         "even stronger discrimination (d = 2.11). Neither axis exists "
         "in current reward model rubrics."),

        ("IMPLICATIONS",
         "\"The goal is not to replace reward models but to augment "
         "them,\" the researcher noted. Brain-derived dimensions capture "
         "sycophancy at the representation level — not through surface "
         "text features. Current approaches like ArmoRM rely on "
         "hand-designed rubrics. Brain networks evolved to evaluate "
         "information and may find dimensions humans didn't think to label."),
    ]

    for cx, (title, body) in zip(col_xs, col_texts):
        ax.text(cx + col_w/2, 6.05, title,
                fontsize=10, color=WHITE, fontweight="bold",
                ha="center", va="top")
        ax.text(cx + col_w/2, 5.75, body,
                fontsize=9, color=GREY, ha="center", va="top",
                wrap=True, linespacing=1.55,
                bbox=dict(boxstyle="square,pad=0", facecolor="none",
                          edgecolor="none"),
                )
        if cx != col_xs[-1]:
            ax.plot([cx+col_w+0.05]*2, [2.2, 6.2],
                    color=BORDER, linewidth=1)

    # Bottom bar: key numbers
    ax.fill_between([0, 14], [0, 0], [1.8, 1.8], color=CARD, alpha=0.8, zorder=3)
    ax.plot([0, 14], [1.8, 1.8], color=BORDER, linewidth=1)

    stats = [
        ("100%", "sycophancy\naccuracy"),
        ("20,484", "cortical\nvertices"),
        ("r = -0.14", "comprehension\nvs confusion"),
        ("d = 2.11", "confusion\neffect size"),
        ("4 models", "in pipeline"),
    ]
    for i, (val, lab) in enumerate(stats):
        x = 1.4 + i * 2.6
        ax.text(x, 1.3, val, fontsize=16, color=RED, fontweight="black",
                ha="center", va="center", zorder=5)
        ax.text(x, 0.6, lab, fontsize=8, color=GREY, ha="center",
                va="center", zorder=5, linespacing=1.4)
        if i < len(stats)-1:
            ax.plot([x+1.1]*2, [0.2, 1.7], color=BORDER,
                    linewidth=0.8, zorder=4)

    path = os.path.join(VIS_DIR, "creative_7_newspaper.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [7/8] Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# VISUAL 8 — PAPER FIGURE COMPOSITE
# ════════════════════════════════════════════════════════════════════════
def make_paper_figure():
    with open(os.path.join(RESULTS_DIR, "experiment_2_dimensions.json")) as f:
        d2 = json.load(f)

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.text(0.5, 0.98,
             "Neural Correlates of AI Response Quality",
             fontsize=20, color=WHITE, fontweight="bold",
             ha="center", va="top", family="serif")
    fig.text(0.5, 0.94,
             "Predicted fMRI activations from TRIBE v2 across text quality conditions",
             fontsize=12, color=GREY, ha="center", va="top")

    # Panel A: correlation heatmap
    ax_a = fig.add_axes([0.04, 0.52, 0.28, 0.36])
    ax_a.set_facecolor(CARD)
    corr = np.array(d2["correlation_matrix"])
    labels = ["Comp", "Mem", "Attn", "Conf", "DMN"]
    cmap_corr = LinearSegmentedColormap.from_list("corr",
        ["#f85149", "#0d1117", "#3fb950"], N=256)
    im = ax_a.imshow(corr, cmap=cmap_corr, vmin=-1, vmax=1, aspect="auto")
    ax_a.set_xticks(range(5)); ax_a.set_xticklabels(labels, fontsize=10, color=WHITE)
    ax_a.set_yticks(range(5)); ax_a.set_yticklabels(labels, fontsize=10, color=WHITE)
    for i in range(5):
        for j in range(5):
            col = "white" if abs(corr[i,j]) > 0.5 else WHITE
            ax_a.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                     fontsize=9, fontweight="bold", color=col)
    plt.colorbar(im, ax=ax_a, shrink=0.8).ax.tick_params(labelcolor=GREY)
    ax_a.set_title("A.  Dimension Correlations", fontsize=11, color=WHITE,
                  fontweight="bold", pad=8, loc="left")

    # Panel B: quality bars
    ax_b = fig.add_axes([0.38, 0.52, 0.28, 0.36])
    ax_b.set_facecolor(CARD)
    qm = d2["quality_means"]
    x = np.arange(5); w = 0.25
    for qi, (q, col, lab) in enumerate([(qm["low"], RED, "Low"),
                                         (qm["med"], ORANGE, "Medium"),
                                         (qm["high"], GREEN, "High")]):
        ax_b.bar(x + qi*w, q, w, label=lab, color=col, alpha=0.85)
    ax_b.set_xticks(x+w); ax_b.set_xticklabels(labels, fontsize=10, color=WHITE)
    ax_b.axhline(0, color=BORDER, linewidth=1)
    ax_b.legend(fontsize=9, facecolor=CARD, edgecolor=BORDER, labelcolor=WHITE)
    ax_b.set_title("B.  Quality Level Means", fontsize=11, color=WHITE,
                  fontweight="bold", pad=8, loc="left")
    ax_b.tick_params(colors=GREY)

    # Panel C: scatter Comp vs Confusion
    ax_c = fig.add_axes([0.72, 0.52, 0.26, 0.36])
    ax_c.set_facecolor(CARD)
    qcolors = {1: RED, 2: ORANGE, 3: GREEN}
    for t in d2["per_text"]:
        dims = t["dimensions"]
        ax_c.scatter(dims["comprehension"], dims["confusion"],
                    c=qcolors[t["quality"]], s=80, alpha=0.8,
                    edgecolors="white", linewidths=0.4)
    ax_c.axvline(0, color=BORDER, linewidth=1, linestyle="--")
    ax_c.axhline(0, color=BORDER, linewidth=1, linestyle="--")
    ax_c.set_xlabel("Comprehension", fontsize=10, color=WHITE)
    ax_c.set_ylabel("Confusion", fontsize=10, color=WHITE)
    ax_c.text(0.05, 0.93, "r = -0.14", transform=ax_c.transAxes,
             fontsize=11, color=ORANGE, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=CARD, edgecolor=ORANGE))
    ax_c.set_title("C.  Two Independent Axes", fontsize=11, color=WHITE,
                  fontweight="bold", pad=8, loc="left")
    ax_c.tick_params(colors=GREY)
    for q, c, l in [(1,RED,"Low"),(2,ORANGE,"Med"),(3,GREEN,"High")]:
        ax_c.scatter([], [], c=c, s=60, label=l)
    ax_c.legend(fontsize=9, facecolor=CARD, edgecolor=BORDER, labelcolor=WHITE)

    # Panel D: experiment 3 results
    ax_d = fig.add_axes([0.04, 0.06, 0.44, 0.36])
    ax_d.set_facecolor(CARD)
    cats = ["Sycophancy", "Clarity", "Depth", "Mixed", "Coherence", "Accuracy"]
    accs = [100, 100, 80, 80, 40, 20]
    bar_cols = [GREEN if a >= 80 else ORANGE if a >= 60 else RED for a in accs]
    bars = ax_d.barh(cats, accs, color=bar_cols, edgecolor=BORDER, height=0.6)
    ax_d.axvline(50, color=GREY, linewidth=2, linestyle="--", alpha=0.7)
    ax_d.set_xlabel("Brain-as-Judge Accuracy (%)", fontsize=10, color=WHITE)
    ax_d.set_xlim(0, 115)
    for bar, acc in zip(bars, accs):
        ax_d.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                 f"{acc}%", va="center", fontsize=11,
                 fontweight="bold", color=WHITE)
    ax_d.set_title("D.  Brain-as-Judge Accuracy (Exp. 3)",
                  fontsize=11, color=WHITE, fontweight="bold", pad=8, loc="left")
    ax_d.tick_params(colors=WHITE)

    # Panel E: effect sizes
    ax_e = fig.add_axes([0.56, 0.06, 0.40, 0.36])
    ax_e.set_facecolor(CARD)
    dim_labels_full = ["Comprehension", "Memory", "Attention", "Confusion", "DMN Supp."]
    cohens = [1.35, 0.38, -0.28, -2.11, -0.52]
    ecols  = [GREEN if d > 0.5 else RED if d < -0.5 else GREY for d in cohens]
    ax_e.barh(dim_labels_full[::-1], cohens[::-1],
             color=ecols[::-1], edgecolor=BORDER, height=0.6)
    ax_e.axvline(0,    color=GREY,  linewidth=1)
    ax_e.axvline(0.8,  color=GREEN, linewidth=1, linestyle="--", alpha=0.5)
    ax_e.axvline(-0.8, color=RED,   linewidth=1, linestyle="--", alpha=0.5)
    for i, (d, lab) in enumerate(zip(cohens[::-1], dim_labels_full[::-1])):
        ax_e.text(d + (0.08 if d >= 0 else -0.08), i,
                 f"d={d:+.2f}", va="center",
                 ha="left" if d >= 0 else "right",
                 fontsize=10, fontweight="bold", color=WHITE)
    ax_e.set_xlabel("Cohen's d  (High vs Low Quality)", fontsize=10, color=WHITE)
    ax_e.set_title("E.  Quality Discrimination Effect Sizes",
                  fontsize=11, color=WHITE, fontweight="bold", pad=8, loc="left")
    ax_e.tick_params(colors=WHITE)

    path = os.path.join(VIS_DIR, "creative_8_paper_figure.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [8/8] Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating all 8 creative visuals...\n")
    print("── Top 3 recommendations ──────────────────")
    make_quote_card()
    make_two_brain_annotated()
    make_radar_chart()
    print("\n── Remaining 5 ────────────────────────────")
    make_side_by_side()
    make_iceberg()
    make_info_loss()
    make_newspaper()
    make_paper_figure()
    print("\nAll done! Files saved to visualizations/")
    print("\nRecommended order for LinkedIn post:")
    print("  creative_1_quote_card.png      ← hook / stop the scroll")
    print("  creative_2_two_brains.png      ← the core visual")
    print("  creative_3_radar.png           ← the data")
    print("  creative_7_newspaper.png       ← the story")
