"""Activation cube of the paper FCN (10 layers, MNIST), with and without lateral inhibition, for one test image: the
input and the ten hidden layers stacked as an H x W x L volume, one layer lit per frame, the two networks side by side
off the same image (left: with lateral inhibition, right: plain). Signed activations, min-maxed per layer, so the
negative-rail surround is transparent and the digit opaque. Rendering of the paper video (doi.org/10.6084/m9.figshare.
34750569); paper_figures/activation_video.py runs this once per digit and joins the clips.

Loading: the paper configs configs/paper/tanh/fcn_mnist_tanh_ns_ki_{ca,none}.json, the checkpoints
paper_runs/fcn_mnist_tanh_ns_ki_d10/model_{ca,none}.pth and that run's saved data split (never regenerated).

    python paper_figures/activation_cube.py --digit 3 --rectify signed --out paper_figures/out/cube_digit3.mp4
"""
from __future__ import annotations
from conwaynet import REPO_ROOT
import argparse, glob, json, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import proj3d
import numpy as np, torch

from conwaynet.config import load_config
from conwaynet.models import build_model
from conwaynet.data import load_dataset

ROOT = REPO_ROOT


# ---------- loading (clean repo) ----------
def _find_config(ds, mech, override=None):
    if override is not None:
        return load_config(override.replace("{mech}", mech))
    cfgs = [p for p in glob.glob(f"{ROOT}/configs/paper/*_{mech}.json")
            if json.load(open(p))["data"]["data_name"] == ds
            and json.load(open(p))["model"]["network_class"] == "FullyConnectedNet"]
    if len(cfgs) != 1:
        raise ValueError(f"expected one FCN {mech} config for {ds}, found {cfgs}")
    return load_config(cfgs[0])


def load_arm(run_dir, ds, mech, device, config_override=None):
    cfg = _find_config(ds, mech, config_override)
    tr, va, te, meta = load_dataset(cfg["data_path"], cfg["batch_size"], cfg["data_name"],
                                    val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                    data_norms=cfg["data_norms"], output_path=run_dir, num_workers=2)
    model = build_model(cfg, meta).to(device)
    model.load_state_dict(torch.load(f"{run_dir}/model_{mech}.pth", map_location=device))
    model.eval()
    return model, cfg, meta, te


def capture_maps(model, x, include_input, rectify):
    """Replicate FullyConnectedNet.forward, collecting one H x W slab per hidden layer.

    Each hidden z has width h = C*H*W laid out as the gate's (C, H, W) grid, so reshape then
    channel-mean to (H, W). Slab 0 (optional) is the input. Returns RAW (un-normalised) maps;
    normalisation (min-max, per-layer or global) happens in main().

    `rectify` ('relu' | 'signed') is the magnitude recipe and is load-bearing for li=ca:
      - 'relu'   : relu-rectify first (keeps only positive activations). WRONG for li=ca, whose
                   field sits at the negative tanh rail — relu zeros the surround AND the digit.
      - 'signed' : keep the raw signed values (the published PCT recipe, src/inference.py:232).
                   With the min-max in main(), the rail-saturated surround maps to 0 (transparent)
                   and the units near the digit map to 1 (opaque).
    """
    C, H, W = x.shape[1], x.shape[2], x.shape[3]

    def slab(vec):
        v = torch.relu(vec) if rectify == "relu" else vec
        return v.reshape(C, H, W).mean(0).detach().cpu().numpy()

    maps, labels = [], []
    if include_input:
        maps.append(slab(x.reshape(-1))); labels.append("input")
    with torch.no_grad():
        # gated-MLP stem (if present): conv0 + GroupNorm + act + deep-LI on the raw image,
        # mirroring FullyConnectedNet.forward, rendered as its own slab before the FC stack.
        cur = x.view(x.size(0), -1)
        for i, lin in enumerate(model.linears):
            z = torch.tanh(model.norms[i](lin(cur)))
            if model.gate is not None:
                z = model.gate(cur, z)
            cur = z
            maps.append(slab(z.reshape(-1))); labels.append(f"layer {i + 1}")
        # Use the real forward for the prediction so the head is handled correctly
        # (the manual trunk above matches forward).
        logits = model(x)
    return logits, maps, labels


# ---------- drawing (verbatim from PR #65) ----------
def draw_plane(ax, depth, amap, cmap, alpha_scale, alpha_gamma, alpha_threshold):
    h, w = amap.shape
    grid_x, grid_z = np.meshgrid(np.arange(w + 1, dtype=float), np.arange(h, -1, -1, dtype=float))
    grid_y = np.full_like(grid_x, float(depth))
    rgba = cmap(amap)
    alpha = np.power(np.clip(amap, 0.0, 1.0), alpha_gamma) * alpha_scale
    alpha[amap < alpha_threshold] = 0.0
    rgba[..., 3] = alpha
    facecolors = np.zeros((h + 1, w + 1, 4)); facecolors[:-1, :-1] = rgba
    ax.plot_surface(grid_x, grid_y, grid_z, facecolors=facecolors, shade=False,
                    rstride=1, cstride=1, linewidth=0, antialiased=False)


def assert_depth_order(elev, azim, n_layers):
    if math.cos(math.radians(elev)) * math.sin(math.radians(azim)) <= 0:
        raise ValueError(f"elev={elev}, azim={azim} renders layer 0 nearest (backwards); "
                         "use an azimuth with sin(azim) > 0 (e.g. 65).")


def assert_image_orientation(ax, side):
    proj = ax.get_proj(); mid = side / 2.0
    sx = lambda x: proj3d.proj_transform(x, 0.0, mid, proj)[0]
    sy = lambda z: proj3d.proj_transform(mid, 0.0, z, proj)[1]
    if sx(0.5) >= sx(side - 0.5):
        raise RuntimeError("slab renders MIRRORED left-right; flip x limits or change azim.")
    if sy(side - 0.5) <= sy(0.5):
        raise RuntimeError("slab renders FLIPPED top-bottom.")


def draw_plane_outline(ax, depth, side, color):
    ax.plot([0, side, side, 0, 0], [depth] * 5, [0, 0, side, side, 0], color=color, lw=1.2, alpha=0.9)


def draw_box(ax, side, n_layers, color):
    x0, x1, y0, y1, z0, z1 = 0, side, 0, n_layers - 1, 0, side
    for xa, xb, ya, yb, za, zb in [
        (x0, x1, y0, y0, z0, z0), (x0, x1, y1, y1, z0, z0), (x0, x1, y0, y0, z1, z1),
        (x0, x1, y1, y1, z1, z1), (x0, x0, y0, y1, z0, z0), (x1, x1, y0, y1, z0, z0),
        (x0, x0, y0, y1, z1, z1), (x1, x1, y0, y1, z1, z1), (x0, x0, y0, y0, z0, z1),
        (x1, x1, y0, y0, z0, z1), (x0, x0, y1, y1, z0, z1), (x1, x1, y1, y1, z0, z1)]:
        ax.plot([xa, xb], [ya, yb], [za, zb], color=color, lw=0.5, alpha=0.35)


def make_writer(out, fps):
    ext = os.path.splitext(out)[1].lower()
    if ext == ".gif":
        return PillowWriter(fps=fps)
    if ext == ".mp4":
        if "ffmpeg" not in animation.writers.list():
            raise RuntimeError("writing .mp4 needs ffmpeg; pass --out with a .gif extension.")
        return animation.FFMpegWriter(fps=fps, extra_args=["-pix_fmt", "yuv420p"])
    raise ValueError(f"unsupported output extension {ext!r}; use .gif or .mp4")


# ---------- main ----------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--digit", type=int, required=True, help="class label; first image of this class in test")
    p.add_argument("--out", required=True)
    p.add_argument("--run_dir", default=f"{ROOT}/paper_runs/fcn_mnist_tanh_ns_ki_d10")
    p.add_argument("--config", default=f"{ROOT}/configs/paper/tanh/fcn_mnist_tanh_ns_ki_{{mech}}.json",
                   help="config path with a {mech} placeholder")
    p.add_argument("--rectify", required=True, choices=["relu", "signed"],
                   help="REQUIRED (no default, CLAUDE.md s8). 'signed' = raw signed activations, min-maxed "
                        "so the negative-rail surround -> 0/transparent and the digit -> 1/opaque (the PCT "
                        "recipe, correct for li=ca); 'relu' = rectify first (discards negative structure).")
    p.add_argument("--norm_mode", choices=["per_layer", "global"], default="per_layer",
                   help="per_layer: each slab min-maxed by its OWN (min,max); "
                        "global: all slabs share the arm's (min,max), so a dynamic-range collapse with depth shows as fading")
    p.add_argument("--arms", nargs="+", default=["ca", "none"], choices=["ca", "none"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--elev", type=float, default=22.0)
    p.add_argument("--azim", type=float, default=65.0)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--alpha_gamma", type=float, default=0.8)
    p.add_argument("--alpha_threshold", type=float, default=0.04)
    p.add_argument("--ghost_alpha", type=float, default=0.05)
    p.add_argument("--include_input", choices=["yes", "no"], default="yes")
    args = p.parse_args()

    run_dir = args.run_dir
    if not os.path.exists(os.path.join(run_dir, "data_split_indices.json")):
        raise FileNotFoundError(f"{run_dir}/data_split_indices.json not found (refusing to regenerate a split).")
    include_input = args.include_input == "yes"

    arms, te, meta = [], None, None
    LABEL = {"ca": "with lateral inhibition (trained)", "none": "plain (trained)"}
    for spec in args.arms:
        model, cfg, meta, te = load_arm(run_dir, args.dataset, spec, args.device, args.config)
        arms.append({"li": LABEL[spec],
                     "model": model})

    # first test image of the requested class (test loader already respects the saved split)
    x = y = None
    for xb, yb in te:
        hit = (yb == args.digit).nonzero(as_tuple=True)[0]
        if len(hit):
            x = xb[hit[0]:hit[0] + 1].to(args.device); y = int(yb[hit[0]]); break
    if x is None:
        raise ValueError(f"no image of class {args.digit} in the {args.dataset} test split")

    side = x.shape[2]

    def minmax(m, lo, hi):
        return (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)

    for a in arms:
        logits, raw, labels = capture_maps(a["model"], x, include_input, args.rectify)
        a["raw"] = raw
        if args.norm_mode == "global":
            gmin = min(float(m.min()) for m in raw); gmax = max(float(m.max()) for m in raw)
            a["maps"] = [minmax(m, gmin, gmax) for m in raw]
        else:
            a["maps"] = [minmax(m, float(m.min()), float(m.max())) for m in raw]
        a["pred"] = int(logits.argmax(1)); a["prob"] = float(torch.softmax(logits, 1).max())
    n_slabs = len(arms[0]["maps"])
    assert_depth_order(args.elev, args.azim, n_slabs)
    digit_img = x[0].mean(0).detach().cpu().numpy()

    print(f"run   {run_dir}\nimage test class {y}\ncube  {side}x{side}x{n_slabs}  slabs: {', '.join(labels)}  rectify={args.rectify}")
    for a in arms:
        print(f"  {a['li']}: pred={a['pred']} ({'OK' if a['pred']==y else 'WRONG'}) p={a['prob']:.4f}")
        # raw (pre-normalisation) signed range per slab — verification handle (li=ca should be rail-pinned)
        rng = "  ".join(f"[{float(m.min()):+.2f},{float(m.max()):+.2f}]" for m in a["raw"])
        print(f"           raw signed range per slab: {rng}")

    n = len(arms)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(7.4 * n, 7.2), dpi=120); fig.patch.set_facecolor("#0d0d12")
    axes = [fig.add_subplot(1, n, k + 1, projection="3d") for k in range(n)]
    inset = fig.add_axes([0.02, 0.80, 0.11, 0.15])

    def render(lit):
        for ax, a in zip(axes, arms):
            maps = a["maps"]; ax.clear(); ax.set_facecolor("#0d0d12")
            draw_box(ax, side, n_slabs, "#8a8a99")
            for depth in range(n_slabs):
                if depth != lit:
                    draw_plane(ax, depth, maps[depth], cmap, args.ghost_alpha, args.alpha_gamma, args.alpha_threshold)
            draw_plane(ax, lit, maps[lit], cmap, 1.0, args.alpha_gamma, args.alpha_threshold)
            draw_plane_outline(ax, lit, side, "#ffd24a")
            ax.set_xlim(side, 0); ax.set_ylim(0, n_slabs - 1); ax.set_zlim(0, side)
            ax.set_box_aspect((1.0, 1.7, 1.0)); ax.view_init(elev=args.elev, azim=args.azim)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.fill = False; axis.pane.set_edgecolor("#2a2a35")
                axis.label.set_color("#b9b9c6"); axis._axinfo["grid"]["color"] = (1, 1, 1, 0.05)
            ax.tick_params(colors="#6e6e7c", labelsize=7)
            ax.set_xlabel("W", color="#b9b9c6", labelpad=-4)
            ax.set_ylabel("L  (network depth)", color="#b9b9c6", labelpad=4)
            ax.set_zlabel("H", color="#b9b9c6", labelpad=-4)
            active = float((maps[lit] > args.alpha_threshold).mean())
            ax.set_title(f"{a['li']}      pred {a['pred']} (p={a['prob']:.3f})\n"
                         f"{labels[lit]} active cells: {100*active:.1f}%", color="#e8e8f0", fontsize=11, pad=10)
        n_hidden = sum(1 for l in labels if l.startswith("layer"))
        fig.suptitle(f"FCN-{n_hidden}  {args.dataset}  class {y}   --   {labels[lit]}  ({lit+1}/{n_slabs})"
                     f"   ({args.rectify}, {args.norm_mode} normalised; alpha = activation)",
                     color="#e8e8f0", fontsize=12, y=0.975)
        inset.clear(); inset.imshow(digit_img, cmap="gray"); inset.set_xticks([]); inset.set_yticks([])
        for sp in inset.spines.values(): sp.set_color("#4a4a58")
        inset.set_title("input", color="#b9b9c6", fontsize=8, pad=2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = make_writer(args.out, args.fps)
    with writer.saving(fig, args.out, dpi=120):
        for lit in range(n_slabs):
            render(lit)
            if lit == 0:
                fig.canvas.draw(); assert_image_orientation(axes[0], side)
            writer.grab_frame(facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
