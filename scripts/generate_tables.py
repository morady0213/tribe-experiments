"""
Generate publication-quality table figures for Medium article.
Dark theme, matching the project visual style.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = r"C:\Users\Genral Rady\Downloads\tribe-experiments\tribe-experiments\visualizations"
os.makedirs(OUT, exist_ok=True)

BG     = "#0d0d0d"
CARD   = "#141414"
ROW_A  = "#1a1a1a"
ROW_B  = "#141414"
WHITE  = "#f0f0f0"
GREY   = "#888888"
RED    = "#e63946"
GREEN  = "#2ecc71"
ORANGE = "#f4a261"
BLUE   = "#4fc3f7"
BORDER = "#2a2a2a"
HEADER_BG = "#1e1e1e"

def draw_table(fig, ax, headers, rows,
               col_widths, row_height=0.13,
               header_color=RED,
               col_colors=None,       # per-column header accent
               value_fmt=None):       # per-cell formatting fn(row_i, col_i, val)
    ax.set_xlim(0, 1)
    total_rows = len(rows)
    ax.set_ylim(0, (total_rows + 1) * row_height)
    ax.axis("off")

    ncols = len(headers)
    # compute x positions
    xs = [0]
    for w in col_widths[:-1]:
        xs.append(xs[-1] + w)

    def cell_bg(r):
        return ROW_A if r % 2 == 0 else ROW_B

    # --- header row ---
    y_top = total_rows * row_height
    for c, (h, w, x) in enumerate(zip(headers, col_widths, xs)):
        ax.add_patch(FancyBboxPatch(
            (x + 0.002, y_top + 0.005), w - 0.004, row_height - 0.008,
            boxstyle="round,pad=0.005",
            facecolor=HEADER_BG, edgecolor=header_color, linewidth=1.5,
            transform=ax.transData, zorder=3
        ))
        ax.text(x + w / 2, y_top + row_height / 2, h,
                color=header_color, fontsize=11, fontweight="bold",
                ha="center", va="center", transform=ax.transData, zorder=4)

    # --- data rows ---
    for r, row in enumerate(rows):
        y = (total_rows - 1 - r) * row_height
        bg = cell_bg(r)
        for c, (val, w, x) in enumerate(zip(row, col_widths, xs)):
            ax.add_patch(patches.Rectangle(
                (x, y), w, row_height,
                facecolor=bg, edgecolor=BORDER, linewidth=0.5,
                transform=ax.transData, zorder=2
            ))
            # colour logic
            color = WHITE
            fw = "normal"
            fs = 10
            if value_fmt:
                color, fw, fs = value_fmt(r, c, val)

            ax.text(x + w / 2, y + row_height / 2, val,
                    color=color, fontsize=fs, fontweight=fw,
                    ha="center", va="center", transform=ax.transData, zorder=4)

    # outer border
    ax.add_patch(patches.Rectangle(
        (0, 0), 1, (total_rows + 1) * row_height,
        fill=False, edgecolor=BORDER, linewidth=1.5,
        transform=ax.transData, zorder=5
    ))


# ══════════════════════════════════════════════════════════════
# TABLE 1 — 5 Cognitive Dimensions
# ══════════════════════════════════════════════════════════════
def make_table1():
    headers = ["Dimension", "Brain Network", "What It Measures"]
    rows = [
        ["Comprehension",       "Default A + B",            "Semantic processing depth"],
        ["Memory Encoding",     "Limbic",                   "Episodic memory engagement"],
        ["Sustained Attention", "Frontoparietal + Dorsal",  "Executive focus"],
        ["Confusion",           "Ventral Attention",         "Error detection & conflict"],
        ["DMN Suppression",     "− Default C",              "Engagement (↓ mind-wandering)"],
    ]
    col_widths = [0.26, 0.34, 0.40]

    NETWORK_COLORS = {
        "Default A + B":           "#4fc3f7",
        "Limbic":                  "#ce93d8",
        "Frontoparietal + Dorsal": "#80cbc4",
        "Ventral Attention":       "#ef9a9a",
        "− Default C":             "#a5d6a7",
    }
    DIM_COLORS = {
        "Comprehension":       "#4fc3f7",
        "Memory Encoding":     "#ce93d8",
        "Sustained Attention": "#80cbc4",
        "Confusion":           "#ef9a9a",
        "DMN Suppression":     "#a5d6a7",
    }

    def fmt(r, c, val):
        if c == 0:
            return DIM_COLORS.get(val, WHITE), "bold", 10
        if c == 1:
            return NETWORK_COLORS.get(val, GREY), "normal", 9
        return GREY, "normal", 9

    fig = plt.figure(figsize=(12, 3.8), facecolor=BG)
    fig.text(0.5, 0.97,
             "Five Cognitive Dimensions from Brain Network Parcellation",
             color=WHITE, fontsize=13, fontweight="bold",
             ha="center", va="top")
    fig.text(0.5, 0.90,
             "Schaefer 2018 atlas  •  TRIBE v2 predicted activations  •  fsaverage5",
             color=GREY, fontsize=9, ha="center", va="top")

    ax = fig.add_axes([0.02, 0.04, 0.96, 0.80])
    ax.set_facecolor(BG)
    draw_table(fig, ax, headers, rows, col_widths, row_height=0.14,
               header_color=RED, value_fmt=fmt)

    path = os.path.join(OUT, "table1_dimensions.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [1/3] Saved: {path}")


# ══════════════════════════════════════════════════════════════
# TABLE 2 — Brain-as-Judge Category Accuracy
# ══════════════════════════════════════════════════════════════
def make_table2():
    headers = ["Category", "Brain-as-Judge Accuracy"]
    rows = [
        ["Sycophancy", "100%"],
        ["Clarity",    "100%"],
        ["Depth",       "80%"],
        ["Mixed",       "80%"],
        ["Coherence",   "40%"],
        ["Factual Accuracy", "20%"],
    ]
    col_widths = [0.50, 0.50]

    ACC_COLORS = {
        "100%": GREEN,
        "80%":  ORANGE,
        "40%":  "#f4a261",
        "20%":  RED,
    }
    CAT_NOTES = {
        "Sycophancy":       "✓ Perfect — brain detects people-pleasing",
        "Clarity":          "✓ Perfect — clear vs vague text",
        "Depth":            "~ Good — surface vs nuanced answers",
        "Mixed":            "~ Good — combined quality signals",
        "Coherence":        "✗ Weak — structure hard to detect",
        "Factual Accuracy": "✗ Expected fail — brain ≠ fact-checker",
    }

    def fmt(r, c, val):
        if c == 1:
            color = ACC_COLORS.get(val, WHITE)
            return color, "bold", 12
        return WHITE, "normal", 10

    fig = plt.figure(figsize=(10, 4.5), facecolor=BG)
    fig.text(0.5, 0.97,
             "Brain-as-Judge: Accuracy by Response Category",
             color=WHITE, fontsize=13, fontweight="bold",
             ha="center", va="top")
    fig.text(0.5, 0.90,
             "30 human-rated prompt-response pairs  •  Experiment 3",
             color=GREY, fontsize=9, ha="center", va="top")

    ax = fig.add_axes([0.02, 0.04, 0.96, 0.80])
    ax.set_facecolor(BG)
    draw_table(fig, ax, headers, rows, col_widths, row_height=0.135,
               header_color=RED, value_fmt=fmt)

    # annotation strip on right
    for r, row in enumerate(rows):
        cat = row[0]
        note = CAT_NOTES.get(cat, "")
        y = (len(rows) - 1 - r) * 0.135 + 0.135 / 2
        ax.text(1.01, y, note, color=GREY, fontsize=7.5,
                va="center", ha="left", transform=ax.transData)

    path = os.path.join(OUT, "table2_accuracy.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [2/3] Saved: {path}")


# ══════════════════════════════════════════════════════════════
# TABLE 3 — Model Comparison
# ══════════════════════════════════════════════════════════════
def make_table3():
    headers = ["Model", "Accuracy", "Notes"]
    rows = [
        ["Random (majority class)", "76.7%", "Baseline ceiling at n=30"],
        ["Brain as Judge",          "70.0%", "Zero labels, zero training"],
        ["Baseline (Δ features)",   "76.7%", "5 features, logistic reg."],
        ["Augmented (all feats)",   "76.7%", "20 features — no gain yet"],
    ]
    col_widths = [0.38, 0.20, 0.42]

    def fmt(r, c, val):
        if c == 1:
            if val == "70.0%":
                return ORANGE, "bold", 11   # brain — slightly under majority
            return GREY, "bold", 11
        if c == 0:
            if "Brain" in val:
                return ORANGE, "bold", 10
            if "Augmented" in val:
                return BLUE, "normal", 10
            return WHITE, "normal", 10
        return GREY, "normal", 9

    fig = plt.figure(figsize=(12, 3.2), facecolor=BG)
    fig.text(0.5, 0.97,
             "Augmented vs Baseline Reward Model (LOO Cross-Validation)",
             color=WHITE, fontsize=13, fontweight="bold",
             ha="center", va="top")
    fig.text(0.5, 0.90,
             "n=30 pairs  •  majority class = 76.7%  •  small-sample regime",
             color=GREY, fontsize=9, ha="center", va="top")

    ax = fig.add_axes([0.02, 0.04, 0.96, 0.80])
    ax.set_facecolor(BG)
    draw_table(fig, ax, headers, rows, col_widths, row_height=0.155,
               header_color=RED, value_fmt=fmt)

    # honest callout box
    fig.text(0.5, 0.01,
             "No improvement at n=30 — majority class problem. "
             "Brain-as-judge (no training) matches augmented model.",
             color=GREY, fontsize=8, ha="center", va="bottom",
             style="italic")

    path = os.path.join(OUT, "table3_model_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [3/3] Saved: {path}")


if __name__ == "__main__":
    print("Generating table figures...")
    make_table1()
    make_table2()
    make_table3()
    print("Done! Find them in visualizations/")
