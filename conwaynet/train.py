"""Training loop: cross-entropy, gradient clipping, a constant or plain-cosine LR schedule.

The learning-rate schedule is either `constant` or a standard cosine decay over the full run; the matched-accuracy protocol (scripts/run_matched.py) adds an optional cosine
floor and an early stop at a target validation accuracy.
"""
import copy
import json
import math

import torch
import torch.nn as nn

from .augment import GPUAugMix, normalize_gpu, random_affine
from .optimizers import Muon, MuonCombo, split_params_for_muon


def build_optimizer(model, config):
    t, lr, wd = config["type"], config["lr"], config["weight_decay"]
    params = model.parameters()
    if t == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=config.get("momentum", 0.0), weight_decay=wd)
    if t == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if t == "muon":
        # `momentum` is shared with SGD: the paper CNN configs set 0.9, which is Muon's momentum there.
        # Weight decay applies to every parameter (biases and norms too; L2 for "adam").
        muon_p, adamw_p = split_params_for_muon(model)
        # print the split so a mis-routing (e.g. every layer landing in AdamW because of a
        # name match) is visible on every run
        print(f"[muon] split: {len(muon_p)} Muon tensors ({sum(p.numel() for p in muon_p):,} params) | "
              f"{len(adamw_p)} AdamW tensors ({sum(p.numel() for p in adamw_p):,} params)", flush=True)
        ratio = config.get("muon_adamw_lr_ratio", 0.1)
        muon = Muon(muon_p, lr=lr, momentum=config.get("momentum", 0.95), weight_decay=wd)
        adamw = torch.optim.AdamW(adamw_p, lr=lr * ratio, weight_decay=wd)
        return MuonCombo(muon, adamw)
    raise ValueError(f"unknown optimizer type {t!r}")


def _apply_lr_factor(optimizer, factor):
    """Scale each param group's LR by `factor` relative to its OWN configured base LR, so a
    per-group ratio (e.g. Muon's AdamW group at muon_adamw_lr_ratio) survives the schedule."""
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def train(model, train_loader, val_loader, config, meta, step_hook):
    """Train `model` in place; return (best_val_acc, last_val_acc, best_state_dict).

    Validation is the stratified val carved from train (never the held-out test), so selecting
    the best-val epoch is legitimate model selection, not test tuning. `best_state_dict` is a CPU
    copy of the weights at the best-val epoch; the caller restores it before the held-out test /
    mCA evaluation so the reported clean accuracy and the mCA describe the SAME model. The model
    left in memory after this call is the last epoch.

    When `augmix_severity > 0`, train batches arrive un-normalized in [0,1]; AugMix is applied on
    device and the result is normalized with `meta`'s mean/std before the model sees it.

    `step_hook` is None or a callable hook(step, epoch, end_of_epoch, model), called once before
    the first step (step 0), after every optimizer step, and after each epoch's validation; the
    hook decides itself whether to evaluate (scripts/run_snapshots.py uses it to save the model at fixed
    epochs). The model is put back in train mode after each call.
    """
    device = config["device"]
    model.to(device)
    optimizer = build_optimizer(model, config)
    for group in optimizer.param_groups:          # remember each group's configured LR so the
        group["base_lr"] = group["lr"]            # schedule scales it (keeps Muon's AdamW ratio);
                                                  # MuonCombo.param_groups are the inner optimizers' own dicts
    criterion = nn.CrossEntropyLoss()
    num_epochs = config["num_epochs"]
    # Matched-accuracy protocol (scripts/run_matched.py). All optional, absent = off:
    #   lr_min_factor     cosine decays to this fraction of the base LR instead of 0
    #   stop_at_val_acc   stop at the end of the first epoch whose val accuracy reaches this value (fraction);
    #                     the model left in memory is that epoch's
    #   val_history_out   write the per-epoch val accuracies to this JSON file
    _lr_min = float(config.get("lr_min_factor", 0.0))
    _stop_at = config.get("stop_at_val_acc", None)
    _val_hist = []
    max_norm = config["max_norm"]
    cosine = config["lr_schedule"] == "cosine"

    augmix = None
    if config["augmix_severity"] > 0:
        augmix = GPUAugMix(config["augmix_severity"], config["augmix_width"], config["augmix_alpha"])
    _input_checked = False                       # first batch: train inputs are [0,1] iff AugMix is on
    # Geometric affine augmentation (rotation/translate/scale/shear). Optional, OFF when the keys are
    # absent (augmentation-absent == off is the one pragmatic default here, so existing configs are
    # unaffected); the aug configs set these explicitly. Geometric-only => noise-mCA/polarity-safe.
    _aff = dict(rotate_deg=config.get("affine_rotate_deg", 0.0), translate_px=config.get("affine_translate_px", 0),
                scale_min=config.get("affine_scale_min", 1.0), scale_max=config.get("affine_scale_max", 1.0),
                shear_deg=config.get("affine_shear_deg", 0.0))
    _use_affine = not (_aff["rotate_deg"] == 0 and _aff["translate_px"] == 0 and
                       _aff["scale_min"] == 1.0 and _aff["scale_max"] == 1.0 and _aff["shear_deg"] == 0)
    best_acc = 0.0
    last_acc = 0.0
    best_state = None
    step = 0
    if step_hook is not None:
        step_hook(0, 0, False, model)
    for epoch in range(num_epochs):
        factor = 0.5 * (1.0 + math.cos(math.pi * epoch / num_epochs)) if cosine else 1.0
        factor = _lr_min + (1.0 - _lr_min) * factor if cosine else factor
        _apply_lr_factor(optimizer, factor)

        model.train()
        loss_sum = grad_sum = clipped = nb = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if not _input_checked:
                # the loader leaves train images un-normalized exactly when AugMix is on (load_dataset(augmix=...));
                # every dataset here has a negative minimum after normalisation
                if augmix is not None and x.min() < -1e-6:
                    raise RuntimeError(f"AugMix expects [0,1] input, got min={x.min().item():.3f}")
                if augmix is None and x.min() >= 0:
                    raise RuntimeError("train inputs look un-normalized (min >= 0) with AugMix off: "
                                       "load_dataset(augmix=...) and augmix_severity disagree")
                _input_checked = True
            if augmix is not None:
                x = normalize_gpu(augmix(x), meta["mean"], meta["std"])
            if _use_affine:                        # geometric aug on the (normalized) batch, border-padded
                x = random_affine(x, _aff["rotate_deg"], _aff["translate_px"],
                                  _aff["scale_min"], _aff["scale_max"], _aff["shear_deg"])
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):          # NaN/inf loss must fail loud, not train to chance
                raise RuntimeError(f"non-finite loss {loss.item()} at epoch {epoch + 1}")
            loss.backward()
            # returns the PRE-clip norm; fails at this step on a non-finite gradient. Under Muon, clipping only limits
            # the AdamW part: Muon orthogonalises its update, so its step size does not depend on the gradient's scale
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item(); grad_sum += float(gnorm); clipped += float(gnorm > max_norm); nb += 1
            step += 1
            if step_hook is not None:
                step_hook(step, epoch, False, model)
                model.train()

        acc = evaluate(model, val_loader, device)
        _val_hist.append(acc)
        last_acc = acc
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch + 1:3d}/{num_epochs}  train_loss={loss_sum/nb:.4f}  val_acc={acc:.4f}  "
              f"best={best_acc:.4f}  grad_norm={grad_sum/nb:.2f}  clip_rate={clipped/nb:.2f}", flush=True)
        if step_hook is not None:
            step_hook(step, epoch + 1, True, model)
        if _stop_at is not None and acc >= float(_stop_at):
            print(f"stop: val_acc {acc:.4f} reached the target {float(_stop_at):.4f} at epoch {epoch + 1}", flush=True)
            break

    if config.get("val_history_out") is not None:
        with open(config["val_history_out"], "w") as f:
            json.dump(_val_hist, f)
    return best_acc, last_acc, best_state
