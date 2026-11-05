"""Figure (formulation appendix): the Birth and Survival gates of B5/S3 and their normalised blend, model-free.
Ported from the messy-repo viz/make_gate_fig_paper.py (run there as --rule B5S3 --gammas 15 50 200) onto the clean gate
code (conwaynet.lateral_inhibition._rule_mask). Panels: Birth gate and Survival gate for gamma 15 / 50 / 200; the
normalised blend at z~ = 0.5 for the same slopes; the normalised blend at gamma 15 for z~ = 0.25 / 0.5 / 0.75.
Also prints the gate peak at gamma 15 (0.516, quoted in the output-range appendix).
Writes paper_figures/out/fig_gates_B5S3_3panel.png. Run: PYTHONPATH=. python paper_figures/gates_figure.py
"""
from conwaynet import REPO_ROOT
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
import torch

from conwaynet.lateral_inhibition import _rule_mask, parse_rule

ROOT = REPO_ROOT; RULE, DELTA, GAMMAS = "B5/S3", 0.5, (15.0, 50.0, 200.0)
m = torch.linspace(0.0, 1.0, 1000)
births, survivals = parse_rule(RULE)
B = [_rule_mask(m, births, DELTA, g).numpy() for g in GAMMAS]
S = [_rule_mask(m, survivals, DELTA, g).numpy() for g in GAMMAS]
m = m.numpy()
print(f"gamma {GAMMAS[0]:g}: Birth peak {B[0].max():.3f}, Survival peak {S[0].max():.3f}")

plt.style.use("default")
styles = ["-", "--", ":"]
colors = [plt.get_cmap("viridis")(v) for v in np.linspace(0.15, 0.85, len(GAMMAS))]
labels = [f"γ={g:g}" for g in GAMMAS]
fig, ((axB, axS), (axM, axZ)) = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
for i in range(len(GAMMAS)):
    axB.plot(m, B[i], linestyle=styles[i], color=colors[i], label=labels[i])
    axS.plot(m, S[i], linestyle=styles[i], color=colors[i], label=labels[i])
    axM.plot(m, ((B[i] + S[i]) / 2.0) / 0.5, linestyle=styles[i], color=colors[i], label=labels[i])   # z~ = 0.5: N = 0.5
axB.set_ylabel("Birth Gate value"); axS.set_ylabel("Survive Gate value"); axM.set_ylabel(r"Blended Gate value $\tilde{{z}}=0.5$")
for z, ls, col in zip((0.25, 0.5, 0.75), ("-", "--", ":"), ("tab:blue", "tab:orange", "tab:green")):
    axZ.plot(m, ((1.0 - z) * B[0] + z * S[0]) / ((1.0 - z) ** 2 + z ** 2), linestyle=ls, color=col, label=fr"$\tilde{{z}}={z:.2f}$")
axZ.set_ylabel(r"Blended gate value $\gamma=15$"); axZ.legend(frameon=False, loc="upper right")
axM.set_xlabel("m"); axZ.set_xlabel("m")
for ax in (axB, axS, axM, axZ):
    ax.grid(True, linestyle=":", alpha=0.5)
h, l = axB.get_legend_handles_labels()
for ax in (axB, axS, axM):
    ax.legend(h, l, loc="upper left", ncol=1, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.9))
out = f"{ROOT}/paper_figures/out/fig_gates_B5S3_3panel.png"; fig.savefig(out, dpi=200, bbox_inches="tight"); print("WROTE", out)
