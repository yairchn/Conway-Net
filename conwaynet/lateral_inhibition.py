"""Cellular-automaton lateral inhibition — the LI=CA gate.

A layer is a non-local op (conv/linear) whose input is `x` and whose bounded (tanh) output is
`z`. The gate rewires the layer as `z = LI(x, z)`:

  1. read the input occupancy      x_norm = (x + 1) / 2                    in [0, 1]
  2. local neighbourhood density    m      = mean over the 3x3 Moore ring  in [0, 1]
  3. continuous CA rule             B(m), S(m) = smooth birth/survive bumps in [0, 1]
  4. state weight from the output   s      = (z + 1) / 2                    in [0, 1]
  5. convex blend                   c      = (1 - s)*B + s*S
  6. (optional) self-energy norm    c      = c / ((1 - s)^2 + s^2)
  7. map back to activation range   z_new  = 2c - 1

Gates are computed from the layer input `x`. Everything is parameter-free — the
network learns *in spite of* the gate, never by reshaping it. Setting the mechanism to `none`
returns no gate at all (the plain tanh network).
"""
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

_CHANNEL_REDUCES = ("mean", "perch_bgref")
_CHANNEL_REDUCE_STAGES = ("input", "gates")


def parse_rule(rule):
    """Parse a Game-of-Life rule string into (births, survivals) neighbour-count lists.

    Accepts "B5/S3", "B5S3", "B3/S23", comma or decimal counts (a run of digits is read as single-digit counts,
    so counts above 8 need commas). An empty set (e.g. "B/S3") makes that gate 0 everywhere: the term contributes
    nothing, so for "B/S..." an off cell stays off.
    """
    rule_u = rule.upper().replace(" ", "")
    m = re.fullmatch(r"B([^/S]*)\/?S(.*)", rule_u)
    if not m:
        raise ValueError(f"Unrecognised rule format: {rule!r} (expected e.g. 'B5/S3')")

    def to_floats(part):
        if "," in part:
            tokens = [t for t in part.split(",") if t]
        elif re.fullmatch(r"\d+", part):
            tokens = list(part)
        else:
            tokens = re.findall(r"\d+(?:\.\d+)?", part)
        return [float(t) for t in tokens]

    return to_floats(m.group(1)), to_floats(m.group(2))


def moore_mean(x):
    """Per-cell mean occupancy over the 8-cell Moore ring. The borders wrap around (torch.roll): a cell on the top
    row counts the bottom row among its neighbours. Harmless where the border is background (MNIST family); on
    CIFAR-10 feature maps an object touching one edge contributes to the density at the opposite edge."""
    s = (torch.roll(x, 1, -1) + torch.roll(x, -1, -1) +
         torch.roll(x, 1, -2) + torch.roll(x, -1, -2) +
         torch.roll(x, shifts=(1, 1), dims=(-2, -1)) +
         torch.roll(x, shifts=(1, -1), dims=(-2, -1)) +
         torch.roll(x, shifts=(-1, 1), dims=(-2, -1)) +
         torch.roll(x, shifts=(-1, -1), dims=(-2, -1)))
    return s / 8.0


def neighbour_mean(x, radius):
    """Mean occupancy over the (2r+1)x(2r+1) square around each cell, excluding the cell itself
    (circular padding). radius 1 is the 8-cell Moore ring (moore_mean); radius 2 a 24-cell 5x5 ring.
    The rule's set-points stay occupancy fractions (count/8), so a larger radius changes only the
    spatial scale the rule reads, not the rule. radius 0 has no neighbourhood: the gate reads the cell's
    own occupancy (a pointwise gate, the zero point of the neighbourhood-size series)."""
    if radius == 0:
        return x
    if 2 * radius + 1 > min(x.shape[-2:]):
        raise ValueError(f"neighbourhood radius {radius} needs a map of at least {2 * radius + 1}x{2 * radius + 1}, "
                         f"got {tuple(x.shape[-2:])} (the wrapped window would count cells twice)")
    if radius == 1:
        return moore_mean(x)
    s = torch.zeros_like(x)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy or dx:
                s = s + torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
    return s / float((2 * radius + 1) ** 2 - 1)


def _logistic_bump(m, count, delta, gamma):
    """Smooth window around `count`/8 (the density set-point for that neighbour count)."""
    t1 = (count - delta) / 8.0
    t2 = (count + delta) / 8.0
    return torch.sigmoid(gamma * (m - t1)) * (1.0 - torch.sigmoid(gamma * (m - t2)))


def _rule_mask(m, counts, delta, gamma):
    """Smooth OR of the per-count bumps: membership in the union of set-points."""
    if not counts:
        return torch.zeros_like(m)
    prod = torch.ones_like(m)
    for c in counts:
        prod = prod * (1.0 - _logistic_bump(m, c, delta, gamma))
    return 1.0 - prod


def _perch_bgref_gates(m, births, survivals, delta, gamma, eps=1e-6):
    """Per-channel background-referenced gates, for an image gate. Each input channel is an affine copy of the
    scene at its own offset, range and sign; undo that per channel: occ_c = |m_c - median(m_c)| / contrast_c with
    contrast_c = max |m_c - median(m_c)| (background -> 0, strongest object contrast -> 1, either polarity).
    B and S are computed per channel and combined weighted by contrast_c, so a channel with no contrast does not
    contribute amplified noise. Reference and scale are stop-gradient. Returns (birth, survive), (N,1,H,W)."""
    flat = m.flatten(2)
    med = flat.median(dim=-1, keepdim=True).values.detach()
    dev = (flat - med).abs()
    con = dev.amax(dim=-1, keepdim=True).detach()                             # (N,C,1) per-channel contrast
    occ = (dev / con.clamp_min(eps)).view_as(m)
    w = (con / con.sum(dim=1, keepdim=True).clamp_min(eps)).unsqueeze(-1)     # (N,C,1,1) contrast weights
    return ((_rule_mask(occ, births, delta, gamma) * w).sum(1, keepdim=True),
            (_rule_mask(occ, survivals, delta, gamma) * w).sum(1, keepdim=True))


def _check_channels(channel_reduce, channel_reduce_stage, C_in, C_out):
    if channel_reduce not in _CHANNEL_REDUCES:
        raise ValueError(f"channel_reduce must be one of {_CHANNEL_REDUCES}, got {channel_reduce!r}")
    if channel_reduce_stage not in _CHANNEL_REDUCE_STAGES:
        raise ValueError(f"channel_reduce_stage must be one of {_CHANNEL_REDUCE_STAGES}, got {channel_reduce_stage!r}")


def lateral_inhibition(x, z, *, rule, gamma, delta, normalize_gates,
                       require_zero_centring, channel_reduce, channel_reduce_stage, neighbourhood_radius):
    """LI=CA gate. `x`,`z` are 4-D (N,C,H,W); gates from `x`, applied to `z`.

    Ranges. The rule's bands are written for occupancies in [0, 1], i.e. `x` and `z` in [-1, 1]. With
    `require_zero_centring` the output is 2c - 1; without it the output is c itself, in [0, 1], and a gate that
    then reads it maps it again through (x + 1)/2, so its occupancy lies in [1/2, 1] and the survival band of
    B5/S3 (~0.31-0.44) is out of reach: the paper CNN's second gate and the WRN's second gate in every block.
    With `normalize_gates` the blend is divided by (1 - s)^2 + s^2 >= 1/2, so it can exceed 1 where the birth
    and survival bumps overlap; it is clipped to [0, 1] (zero gradient above 1; B5/S3 at gamma 15: at most 0.66,
    so the clip never acts).
    `perch_bgref` is used only when C_in != C_out, and then ignores `channel_reduce_stage`.

    Channels. When C_in == C_out output channel i is gated by input channel i (per-channel pairing; after a
    convolution the two are related only to the extent that training makes them so); when they differ the input
    channels are reduced (`channel_reduce`). In the WRN the pairing applies in every block whose width does not
    change, the reduction in the first block of each group.
    """
    m = neighbour_mean((x + 1.0) / 2.0, neighbourhood_radius)
    return gate_from_density(m, z, rule=rule, gamma=gamma, delta=delta, normalize_gates=normalize_gates,
                             require_zero_centring=require_zero_centring, channel_reduce=channel_reduce,
                             channel_reduce_stage=channel_reduce_stage)


def gate_from_density(m, z, *, rule, gamma, delta, normalize_gates, require_zero_centring, channel_reduce,
                      channel_reduce_stage):
    """Steps 3-7 of the gate, from the neighbourhood density m (N,C_in,H,W) of the input occupancy. Split from
    lateral_inhibition so that a network whose neighbourhood is not the square grid (GraphGCN: the mean over
    a node's graph neighbours) can supply its own m."""
    births, survivals = parse_rule(rule)
    C_in, C_out = m.shape[1], z.shape[1]
    _check_channels(channel_reduce, channel_reduce_stage, C_in, C_out)

    if C_in == C_out:
        # per-channel gate (each output channel gated by its own channel's rule)
        birth = _rule_mask(m, births, delta, gamma)
        survive = _rule_mask(m, survivals, delta, gamma)
    elif channel_reduce == "perch_bgref":
        birth, survive = _perch_bgref_gates(m, births, survivals, delta, gamma)
    elif channel_reduce_stage == "input":   # channel mean of the occupancy, then the rule; (N,1,H,W) broadcasts over C_out
        m_r = m.mean(dim=1, keepdim=True)
        birth = _rule_mask(m_r, births, delta, gamma)
        survive = _rule_mask(m_r, survivals, delta, gamma)
    else:  # "gates": the rule per input channel, then the channel mean of the gates
        birth = _rule_mask(m, births, delta, gamma).mean(dim=1, keepdim=True)
        survive = _rule_mask(m, survivals, delta, gamma).mean(dim=1, keepdim=True)

    # gates were computed at x's spatial resolution; resize to z's if they differ (e.g. a
    # valid conv shrank z relative to its input) so the blend is elementwise. Bilinear resizing (align_corners
    # False) stretches the field rather than cropping it to the conv's receptive-field centres, so a gate cell can
    # sit up to ~1 pixel from its unit at the map edges and ~0.5 near the centre (CNN: 28->26 and 26->12; WRN
    # stride-2 blocks: a constant half pixel). The paper networks were trained this way, identically in both arms.
    if birth.shape[-2:] != z.shape[-2:]:
        birth = F.interpolate(birth, size=z.shape[-2:], mode="bilinear", align_corners=False)
        survive = F.interpolate(survive, size=z.shape[-2:], mode="bilinear", align_corners=False)

    s = (z + 1.0) / 2.0
    c = (1.0 - s) * birth + s * survive
    if normalize_gates:
        c = c / ((1.0 - s) ** 2 + s ** 2)
    # keep the state bounded: the normalised blend can exceed 1 where the birth and survival bumps overlap
    # (rules whose sets share or neighbour a count, e.g. B3/S23 or B34/S34 at gamma 15); clipping it to the
    # occupancy range keeps every gate's input and output in the range the rules are written for. For B5/S3
    # at gamma 15 the blend stays below 0.66, so the clip never acts.
    c = c.clamp(0.0, 1.0)
    if require_zero_centring:
        c = 2.0 * c - 1.0
    return c


class LateralInhibitionCA(nn.Module):
    """Parameter-free LI=CA gate. Call as `z = gate(x, z)`.

    Convolutional callers pass 4-D tensors (N,C,H,W). Fully-connected callers pass flattened
    2-D tensors (N, C*H*W); the gate then reshapes to the spatial grid it was built with
    (`channels`, `height`, `width`) so the CA neighbourhood sees the real 2-D image geometry
    (e.g. a 784-vector -> a 1x28x28 grid), runs, and flattens back.
    """

    def __init__(self, *, rule, gamma, delta, normalize_gates,
                 require_zero_centring, channel_reduce, channel_reduce_stage, neighbourhood_radius,
                 channels=None, height=None, width=None):
        super().__init__()
        self.cfg = dict(rule=rule, gamma=gamma, delta=delta, normalize_gates=normalize_gates,
                        require_zero_centring=require_zero_centring,
                        channel_reduce=channel_reduce, channel_reduce_stage=channel_reduce_stage,
                        neighbourhood_radius=neighbourhood_radius)
        self.channels, self.height, self.width = channels, height, width

    def forward(self, x, z):
        if x.dim() == 4:
            return lateral_inhibition(x, z, **self.cfg)
        if x.dim() == 2:
            if self.channels is None:
                raise ValueError("flattened input needs the gate to be built with channels/height/width")
            N = x.shape[0]
            shape = (N, self.channels, self.height, self.width)
            out = lateral_inhibition(x.view(*shape), z.view(*shape), **self.cfg)
            return out.reshape_as(x)
        raise ValueError(f"gate expects 2-D or 4-D tensors, got {x.dim()}-D")


class GraphLateralInhibitionCA(nn.Module):
    """The gate on a graph: `z = gate(x, z)` with m = the mean input occupancy over each node's graph neighbours,
    excluding the node itself. `neighbour_mean` maps an (N,C,H,W) occupancy to that mean (GraphGCN: the 8-neighbour
    pixel graph without wraparound, so on interior pixels m is the Moore mean of the grid gate and on border pixels
    it averages the neighbours that exist instead of wrapping around)."""

    def __init__(self, *, neighbour_mean, rule, gamma, delta, normalize_gates, require_zero_centring,
                 channel_reduce, channel_reduce_stage):
        super().__init__()
        self.neighbour_mean = neighbour_mean
        self.cfg = dict(rule=rule, gamma=gamma, delta=delta, normalize_gates=normalize_gates,
                        require_zero_centring=require_zero_centring,
                        channel_reduce=channel_reduce, channel_reduce_stage=channel_reduce_stage)

    def forward(self, x, z):
        return gate_from_density(self.neighbour_mean((x + 1.0) / 2.0), z, **self.cfg)


def make_gate(mechanism, **kwargs):
    """Factory: 'ca' -> a LateralInhibitionCA gate; 'none' -> None (plain tanh network)."""
    if mechanism == "none":
        return None
    if mechanism == "ca":
        return LateralInhibitionCA(**kwargs)
    raise ValueError(f"mechanism must be 'ca' or 'none', got {mechanism!r}")
