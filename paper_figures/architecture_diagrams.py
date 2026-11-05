"""Appendix figure: block diagrams of the CNN and the WRN and where lateral inhibition is wired (conwaynet/models.py:
BaselineCNN; WideResNet with depth 40, widen_factor 2).
Box colours: learned operations (blue), fixed nonlinear operations (grey), lateral inhibition (red). A dashed red arrow
runs from the tensor that a lateral-inhibition layer reads its gates from. Writes paper_figures/out/architecture_diagrams.pdf/.png.
"""
from conwaynet import REPO_ROOT
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = REPO_ROOT
LEARN, FIXED, LI, DATA = "#cfe0f3", "#e6e6e6", "#f6c9cc", "#ffffff"
EDGE = {LEARN: "#3b6fa8", FIXED: "#7a7a7a", LI: "#c0392b", DATA: "#333333"}


def box(ax, x, y, w, h, text, col, fs=9.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=col, ec=EDGE[col], lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1, col="#333333", ls="-", rad=0.0, lw=1.1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=col, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def gate_arc(ax, x0, x1, ytop, text, rad=-0.3):
    """dashed red arc from x0 to x1 starting at the box tops (ytop), label placed just above the arc's peak."""
    arrow(ax, x0, ytop, x1, ytop, col=EDGE[LI], ls="--", rad=rad)
    peak = ytop + abs(rad) * (x1 - x0) / 2
    ax.text((x0 + x1) / 2, peak + 0.08, text, ha="center", va="bottom", fontsize=9.5, color=EDGE[LI])


def chain(ax, items, y, w, h, gap):
    """items: list of (text, colour); boxes left to right at height y; returns their x centres."""
    xs = []; x = w / 2
    for i, (t, c) in enumerate(items):
        box(ax, x, y, w, h, t, c); xs.append(x)
        if i: arrow(ax, xs[-2] + w / 2, y, x - w / 2, y)
        x += w + gap
    return xs


fig = plt.figure(figsize=(10, 7.6))
# ---------------------------------------------------------------- (a) CNN
ax = fig.add_axes([0.01, 0.69, 0.98, 0.30]); ax.set_xlim(-0.25, 14.25); ax.set_ylim(0, 3.2); ax.axis("off")
ax.text(0.0, 3.15, "(a) CNN (MNIST family, $28\\times28$)", fontsize=11.5, weight="bold", va="top")
items = [("image\n$C\\times28\\times28$", DATA), ("conv $3\\times3$\n32 ch", LEARN), ("tanh", FIXED),
         ("lateral\ninhib. 1", LI), ("conv $3\\times3$\n64 ch", LEARN), ("tanh", FIXED),
         ("max-pool\n$2\\times2$", FIXED), ("lateral\ninhib. 2", LI), ("flatten", FIXED), ("linear", LEARN)]
xs = chain(ax, items, 1.15, 1.15, 0.8, 0.27)
gate_arc(ax, xs[0], xs[3], 1.55, "gates from the image\n(per-channel image normalisation)", rad=-0.25)
gate_arc(ax, xs[3], xs[7], 1.55, "gates from the output of lateral inhibition 1\n(channel mean of the occupancy; resized $26\\to12$)", rad=-0.2)
for i, t in ((3, "$32\\times26\\times26$"), (7, "$64\\times12\\times12$")):
    ax.text(xs[i], 0.55, t, ha="center", fontsize=9, color="#555555")
ax.text(0.0, 0.12, "No normalisation layers; plain blend (no normaliser); outputs passed on as occupancies in $[0,1]$.",
        fontsize=9.5, color="#444444")

# ---------------------------------------------------------------- (b) WRN overview
ax = fig.add_axes([0.01, 0.42, 0.98, 0.24]); ax.set_xlim(-0.25, 14.25); ax.set_ylim(0, 2.6); ax.axis("off")
ax.text(0.0, 2.45, "(b) WRN (CIFAR-10, $32\\times32$): 40 layers, widening factor 2", fontsize=11.5, weight="bold", va="top")
items = [("image\n$3\\times32\\times32$", DATA), ("stem conv\n16 ch", LEARN),
         ("group 1\n6 blocks\n32 ch, $32^2$", LI), ("group 2\n6 blocks\n64 ch, $16^2$", LI),
         ("group 3\n6 blocks\n128 ch, $8^2$", LI), ("GN, tanh", LEARN), ("flatten", FIXED), ("linear", LEARN)]
xs = chain(ax, items, 1.25, 1.4, 1.0, 0.33)
ax.text(xs[1], 0.62, "no lateral inhibition\nreads the image", ha="center", va="top", fontsize=9.5, color="#555555")
ax.text((xs[2] + xs[4]) / 2, 0.62, "two lateral-inhibition layers per block (36 in all);\n"
        "groups 2 and 3 start with a stride-2 block", ha="center", va="top", fontsize=9.5, color="#555555")

# ---------------------------------------------------------------- (c) WRN block
ax = fig.add_axes([0.01, 0.0, 0.98, 0.40]); ax.set_xlim(-0.25, 14.25); ax.set_ylim(-0.3, 3.6); ax.axis("off")
ax.text(0.0, 3.55, "(c) one WRN block (pre-activation; the skip is added after lateral inhibition)", fontsize=11.5, weight="bold", va="top")
items = [("input", DATA), ("GN, tanh", LEARN), ("conv\n$3\\times3$", LEARN), ("GN, tanh", LEARN),
         ("lateral\ninhib.", LI), ("conv\n$3\\times3$", LEARN), ("GN, tanh", LEARN), ("lateral\ninhib.", LI),
         ("$+$", DATA), ("output", DATA)]
y = 1.45; xs = chain(ax, items, y, 1.08, 0.78, 0.26)
ax.text(xs[1], y - 0.62, "$\\hat{x}$", ha="center", fontsize=11)
gate_arc(ax, xs[1], xs[4], y + 0.40, "gates from the block's activated input $\\hat{x}$\n(gate-then-mean where channels change)", rad=-0.3)
gate_arc(ax, xs[4], xs[7], y + 0.40, "gates from the output of the first\nlateral-inhibition layer", rad=-0.3)
arrow(ax, xs[0], y - 0.40, xs[8], y - 0.40, col="#333333", rad=0.12)
skip_low = y - 0.40 - 0.12 * (xs[8] - xs[0]) / 2
ax.text((xs[0] + xs[8]) / 2, skip_low - 0.08, "skip: identity, or a $1\\times1$ convolution of $\\hat{x}$ where the shape changes",
        ha="center", va="top", fontsize=9.5, color="#333333")
ax.text(14.0, -0.25, "GN: group normalisation; lateral inhibition: normalised blend, output passed on as an occupancy",
        ha="right", fontsize=9, color="#555555")

for ext in ("pdf", "png"):
    fig.savefig(f"{ROOT}/paper_figures/out/architecture_diagrams.{ext}", dpi=150, bbox_inches="tight", pad_inches=0.08)
print("WROTE paper_figures/out/architecture_diagrams.pdf/.png")
