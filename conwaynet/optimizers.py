"""Muon optimizer for matrix parameters (used with AdamW for the rest).

Muon applies a Newton-Schulz orthogonalization to the momentum update of 2-D/4-D weight
matrices; biases, norms and the classifier head go to AdamW. Kept as-is from the research
code, minus the tanh-gradient "lr trick".
"""
import math

import torch


@torch.no_grad()
def _zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Approximate G @ (G^T G)^(-1/2) via Newton-Schulz iterations."""
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.t()
    X = X / (X.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.t()
        X = a * X + (b * A + c * (A @ A)) @ X
    return (X.t() if transposed else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2-D (and reshaped 4-D) hidden-layer weight matrices."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0, eps=1e-7):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, beta = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                shape = g.shape
                if g.ndim != 2:
                    g = g.reshape(g.shape[0], -1)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1.0 - beta)
                update = g.lerp(buf, beta) if group["nesterov"] else buf
                update = _zeropower_via_newtonschulz5(update, steps=group["ns_steps"], eps=group["eps"])
                update = update * (0.2 * math.sqrt(max(update.size(0), update.size(1))))
                update = update.reshape(shape)
                if group["weight_decay"]:
                    p.mul_(1.0 - lr * group["weight_decay"])
                p.add_(update, alpha=-lr)


def split_params_for_muon(model):
    """(muon_params, adamw_params): Muon gets hidden weight matrices, AdamW the rest."""
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = any(k in name for k in ("fc", "head", "classifier"))
        (muon if (p.ndim >= 2 and not is_head) else adamw).append(p)
    return muon, adamw


class MuonCombo:
    """Drive a Muon (matrices) and an AdamW (everything else) as one optimizer."""

    def __init__(self, muon, adamw):
        self.opts = (muon, adamw)
        self.param_groups = muon.param_groups + adamw.param_groups

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()
