"""Make the figures used in docs/ from results/attempts.csv and figures/force_traces.json.

Run from the repository root:  python docs/make_figures.py
"""
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")

# Palette (validated: categorical, light surface). Hue = controller, fill = detector.
CLOSED, OPEN = "#2a78d6", "#eb6834"
OUTCOME_COLOURS = {"extracted": "#1baf7a", "immobile": "#eda100", "error": "#e34948", "collapse": "#6250d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SETUPS = [
    ("ground_truth", "closed_loop", "Closed loop\nground truth"),
    ("opencv", "closed_loop", "Closed loop\nOpenCV"),
    ("ground_truth", "open_loop", "Open loop\nground truth"),
    ("opencv", "open_loop", "Open loop\nOpenCV"),
]
SHORT = {("ground_truth", "closed_loop"): "CL-GT", ("opencv", "closed_loop"): "CL-CV",
         ("ground_truth", "open_loop"): "OL-GT", ("opencv", "open_loop"): "OL-CV"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
COLUMN = 3.45  # inches, one column of a two-column page
SCALE = 1.0  # figure size factor
SMALL = 6.5  # annotation font size (pt)
WIDE = 1.0  # extra width for the per-game plot on slides
NARROW = 1.0  # two-panel figures are narrower on slides so their text stays readable


def size(width, height):
    return (width * SCALE, height * SCALE)


def number(value):
    return float(value) if value not in ("", None) else None


def finite(value):
    return value is not None and math.isfinite(value)


def load_rows():
    with open(os.path.join(ROOT, "results", "attempts.csv"), encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def outcome(row):
    return "collapse" if row["tower_collapsed"] == "True" else row["outcome"]


def colour(controller):
    return CLOSED if controller == "closed_loop" else OPEN


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def successes_per_game(rows):
    fig, ax = plt.subplots(figsize=size(COLUMN * WIDE, 2.15 * WIDE))
    rng = np.random.default_rng(1)
    for index, (detector, controller, label) in enumerate(SETUPS):
        per = Counter()
        for row in rows:
            if (row["detector"], row["controller"]) == (detector, controller) and row["success"] == "True":
                per[int(row["trial"])] += 1
        values = np.array([per[t] for t in range(18)], float)
        x = index + rng.uniform(-0.17, 0.17, len(values))
        c = colour(controller)
        face = c if detector == "ground_truth" else "white"
        ax.scatter(x, values, s=14, facecolor=face, edgecolor=c, linewidth=0.9, alpha=0.85, zorder=3)
        mean = values.mean()
        ci = 2.110 * values.std(ddof=1) / math.sqrt(len(values))
        ax.errorbar(index + 0.33, mean, yerr=ci, fmt="o", ms=4.5, color=INK, capsize=2.5, lw=1, zorder=4)
        ax.text(index + 0.40, mean, f"{mean:.1f}", va="center", ha="left", fontsize=SMALL + 0.5, color=INK)
    ax.set_xticks(range(4), [s[2] for s in SETUPS])
    ax.set_xlim(-0.5, 3.8)
    ax.set_ylabel("Successful attempts per game")
    grid(ax)
    fig.savefig(os.path.join(OUT, "successes.pdf"))
    plt.close(fig)


def outcome_breakdown(rows):
    fig, ax = plt.subplots(figsize=size(COLUMN, 1.55))
    labels = []
    for index, (detector, controller, label) in enumerate(SETUPS[::-1]):
        group = [outcome(r) for r in rows if (r["detector"], r["controller"]) == (detector, controller)]
        counts = Counter(group)
        left = 0.0
        for name, c in OUTCOME_COLOURS.items():
            share = 100.0 * counts[name] / len(group)
            ax.barh(index, share, left=left, height=0.62, color=c, edgecolor="white", linewidth=1.0)
            if share >= 6:
                ax.text(left + share / 2, index, f"{share:.0f}", ha="center", va="center", fontsize=SMALL, color=INK)
            left += share
        labels.append(label.replace("\n", ", ") + f" (n={len(group)})")
    ax.set_yticks(range(4), labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of attempts (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in OUTCOME_COLOURS.values()]
    ax.legend(handles, ["extracted", "immobile (correct)", "error", "collapse"], ncol=4,
              loc="lower center", bbox_to_anchor=(0.38, 1.0), handlelength=1.0, columnspacing=0.9)
    fig.savefig(os.path.join(OUT, "outcomes.pdf"))
    plt.close(fig)


def alignment(rows):
    fig, axes = plt.subplots(1, 2, figsize=size(COLUMN * 2 * NARROW, 2.6 * NARROW), sharey=True)
    xlim, ylim = 14.5, 11.0
    for ax, controller, title in ((axes[0], "closed_loop", "Closed loop (visual servoing)"),
                                  (axes[1], "open_loop", "Open loop (IK only)")):
        ax.add_patch(Rectangle((-12.5, -7.5), 25, 15, fill=False, ec=INK2, lw=0.7))
        ax.text(-12.5, 7.8, "block face 25 x 15 mm", ha="left", va="bottom", fontsize=SMALL, color=INK2)
        ax.add_patch(Circle((0, 0), 7, fill=False, ec=INK, lw=0.8, ls="--"))
        ax.text(4.95, 4.95, "7 mm", ha="left", va="bottom", fontsize=SMALL, color=INK)
        outside = 0
        for detector, filled in (("ground_truth", True), ("opencv", False)):
            h, v = [], []
            for row in rows:
                if (row["detector"], row["controller"]) != (detector, controller):
                    continue
                hh, vv = number(row["alignment_error_h_mm"]), number(row["alignment_error_v_mm"])
                if not (finite(hh) and finite(vv)):
                    continue
                if abs(hh) > xlim or abs(vv) > ylim:
                    outside += 1
                    continue
                h.append(hh)
                v.append(vv)
            c = colour(controller)
            ax.scatter(h, v, s=7, facecolor=c if filled else "white", edgecolor=c, linewidth=0.6,
                       alpha=0.75, label="ground-truth detection" if filled else "OpenCV detection")
        ax.set_aspect("equal")
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_title(title, color=INK)
        ax.set_xlabel("Horizontal error (mm)")
        if outside:
            ax.text(xlim * 0.97, -ylim * 0.97, f"{outside} outside plot", ha="right", va="bottom",
                    fontsize=SMALL, color=INK2)
        grid(ax, "both")
    axes[0].set_ylabel("Vertical error (mm)")
    marker = dict(marker="o", linestyle="none", markersize=4.5, markeredgecolor=INK2, markeredgewidth=0.8)
    fig.legend([plt.Line2D([], [], markerfacecolor=INK2, **marker),
                plt.Line2D([], [], markerfacecolor="white", **marker)],
               ["ground-truth detection (filled)", "OpenCV detection (hollow)"],
               ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.97))
    fig.savefig(os.path.join(OUT, "alignment.pdf"))
    plt.close(fig)


def levels(rows):
    fig, axes = plt.subplots(1, 2, figsize=size(COLUMN * 2 * NARROW, 1.9 + 0.5 * (1 - NARROW)), sharey=True)
    for ax, controller, title in ((axes[0], "closed_loop", "Closed loop"), (axes[1], "open_loop", "Open loop")):
        per = defaultdict(Counter)
        for row in rows:
            if row["controller"] == controller:
                per[int(row["layer"])][outcome(row)] += 1
        layers = sorted(per)
        bottom = np.zeros(len(layers))
        for name, c in OUTCOME_COLOURS.items():
            share = np.array([100.0 * per[l][name] / sum(per[l].values()) for l in layers])
            ax.bar(layers, share, bottom=bottom, width=0.72, color=c, edgecolor="white", linewidth=0.8, label=name)
            bottom += share
        ax.set_title(title, color=INK)
        ax.set_xticks(layers, [str(l) if NARROW == 1 or l % 2 == 0 else "" for l in layers])
        ax.set_xlabel("Tower level (0 = bottom, 17 = top)" if NARROW == 1 else "Tower level (0 = bottom)")
        ax.axvline(8.5, color=INK, lw=0.8, ls=":")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Share of attempts (%)")
    axes[1].legend(ncol=4, loc="lower center", bbox_to_anchor=(-0.05, 1.13), handlelength=1.0)
    fig.savefig(os.path.join(OUT, "levels.pdf"))
    plt.close(fig)


def forces(rows):
    fig, axes = plt.subplots(1, 2, figsize=size(COLUMN * 2, 2.1), gridspec_kw={"width_ratios": [1.35, 1]})
    with open(os.path.join(ROOT, "docs", "figures", "force_traces.json"), encoding="utf-8") as stream:
        traces = json.load(stream)
    ax = axes[0]
    for trace in traces:
        profile = np.array(trace["profile"])
        if len(profile) == 0:
            continue
        t, f = profile[:, 0], profile[:, 1]
        start = np.argmax(f > 0.005) if np.any(f > 0.005) else 0
        t = t - t[start]
        c = OUTCOME_COLOURS["extracted"] if trace["outcome"] == "extracted" else OUTCOME_COLOURS["immobile"]
        ax.plot(t, f, color=c, lw=1.0, alpha=0.9)
    ax.axhline(0.32, color=INK, lw=0.8, ls="--")
    ax.text(0.15, 0.335, "threshold 0.32 N", fontsize=SMALL, color=INK)
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("Time from contact (s)")
    ax.set_ylabel("Fingertip force (N)")
    ax.set_title("Example pushes (level 9-11)", color=INK)
    ax.plot([], [], color=OUTCOME_COLOURS["extracted"], label="mobile: extracted")
    ax.plot([], [], color=OUTCOME_COLOURS["immobile"], label="immobile: stopped")
    ax.legend(loc="upper right")
    grid(ax)

    ax = axes[1]
    bins = np.linspace(0, 0.40, 41)
    labels = {"extracted": (0.055, 53, "left"), "immobile": (0.25, 45, "center")}
    for name in ("extracted", "immobile"):
        values = [number(r["peak_force_n"]) for r in rows
                  if r["controller"] == "closed_loop" and outcome(r) == name and finite(number(r["peak_force_n"]))]
        ax.hist(values, bins=bins, color=OUTCOME_COLOURS[name], alpha=0.85, edgecolor="white", linewidth=0.4)
        x, y, align = labels[name]
        ax.text(x, y, f"{name}\n(n={len(values)})", ha=align, va="bottom", fontsize=SMALL, color=INK)
    ax.set_ylim(0, 64)
    for threshold, style in ((0.32, "--"), (0.18, ":")):
        ax.axvline(threshold, color=INK, lw=0.8, ls=style)
        ax.text(threshold + 0.004, 62, f"{threshold:.2f} N", fontsize=SMALL, va="top", color=INK)
    ax.set_xlabel("Peak force in first 15 mm (N)")
    ax.set_ylabel("Attempts")
    ax.set_title("Closed loop, all games", color=INK)
    grid(ax)
    fig.savefig(os.path.join(OUT, "forces.pdf"))
    plt.close(fig)


def main():
    """`python docs/make_figures.py` for the document figures, add `slides` for the slide versions."""
    global OUT, SMALL, WIDE, NARROW
    if len(sys.argv) > 1 and sys.argv[1] == "slides":
        # Same layouts, sans-serif and larger text for projection.
        SMALL = 8.5
        WIDE = 1.3
        NARROW = 0.8
        OUT = os.path.join(OUT, "slides")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Calibri", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10.5,
            "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        })
    os.makedirs(OUT, exist_ok=True)
    rows = load_rows()
    successes_per_game(rows)
    outcome_breakdown(rows)
    alignment(rows)
    levels(rows)
    forces(rows)
    for name in ("successes", "outcomes", "alignment", "levels", "forces"):
        print(os.path.join(OUT, name + ".pdf"))


if __name__ == "__main__":
    main()
