"""Figure (FCN activation maps and classifier weights), paper FCN on MNIST, with and without lateral inhibition.
Rows 1-2: the hidden activation maps of layers 1-10 for one digit-2 test image (the first in test order).
Rows 3-4: the classifier weight maps of the ten classes. Rendering (as the earlier depth figure): the network with
lateral inhibition is shown as occupancy (y+1)/2, so its background (y = -1) is white; the plain network's activation
and both classifiers are signed, white at zero. One colour scale per row (shared across its ten panels), colourbar per row.
Writes paper_figures/out/activation_classifier-fcn.png. Run: PYTHONPATH=. python paper_figures/fcn_depth_activation_classifier.py
"""
from conwaynet import REPO_ROOT
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from conwaynet.config import load_config
from conwaynet.models import build_model
from conwaynet.data import load_dataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"; ROOT = REPO_ROOT; HW = 28; DIGIT = 2; N = 4000
RUN = f"{ROOT}/paper_runs/fcn_mnist_tanh_ns_ki_d10"


@torch.no_grad()
def class_layer_maps(mech):
    c = load_config(f"{ROOT}/configs/paper/tanh/fcn_mnist_tanh_ns_ki_{mech}.json")
    _, _, te, meta = load_dataset(c["data_path"], 500, c["data_name"], val_frac=c["val_frac"], split_seed=c["split_seed"],
                                  data_norms=c["data_norms"], output_path=None, num_workers=2)
    m = build_model(c, meta).to(DEV); m.load_state_dict(torch.load(f"{RUN}/model_{mech}.pth", map_location=DEV)); m.eval()
    X, Y = [], []
    for xb, yb in te:
        X.append(xb); Y.append(yb)
        if sum(len(x) for x in X) >= N: break
    X, Y = torch.cat(X)[:N], torch.cat(Y)[:N]
    i = int((Y == DIGIT).nonzero()[0][0]); cur = X[i:i + 1].to(DEV).flatten(1); maps = []
    for lin, norm in zip(m.linears, m.norms):
        z = torch.tanh(norm(lin(cur)))
        if m.gate is not None:
            z = m.gate(cur, z)
        cur = z; maps.append(cur[0].reshape(HW, HW).cpu().numpy())
    return np.stack(maps), m.classifier.weight.detach().cpu().numpy().reshape(-1, HW, HW), int((Y == DIGIT).sum())


D = {mech: class_layer_maps(mech) for mech in ("ca", "none")}
rows = [((D["ca"][0] + 1.0) / 2.0, "with lateral\ninhibition\n(occupancy)"), (D["none"][0], "plain"),
        (D["ca"][1], "with lateral\ninhibition"), (D["none"][1], "plain")]
fig = plt.figure(figsize=(14, 7.0))
blocks = [fig.add_gridspec(2, 10, left=0.09, right=0.95, top=0.885, bottom=0.525, wspace=0.06, hspace=0.08),
          fig.add_gridspec(2, 10, left=0.09, right=0.95, top=0.405, bottom=0.045, wspace=0.06, hspace=0.08)]
for r, (maps, lab) in enumerate(rows):
    gs = blocks[r // 2]; vm = float(np.abs(maps).max()) or 1.0; axs = []    # one scale per row, white at zero
    for c in range(10):
        ax = fig.add_subplot(gs[r % 2, c]); a = maps[c]; axs.append(ax)
        im = ax.imshow(a, cmap="RdBu_r", vmin=-vm, vmax=vm); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if r in (0, 2):
            ax.set_title(f"layer {c + 1}" if r == 0 else f"class {c}", fontsize=10.5)
        if c == 0:
            ax.text(-0.12, 0.5, lab, transform=ax.transAxes, ha="right", va="center", fontsize=11, weight="bold")
    cb = fig.colorbar(im, ax=axs, fraction=0.012, pad=0.006); cb.ax.tick_params(labelsize=8)
fig.text(0.54, 0.965, f"Activation maps of layers 1-10 for one digit-{DIGIT} test image", ha="center", fontsize=13, weight="bold")
fig.text(0.54, 0.475, "Classifier weights", ha="center", fontsize=13, weight="bold")
out = f"{ROOT}/paper_figures/out/activation_classifier-fcn.png"; fig.savefig(out, dpi=150)
print("WROTE", out)
