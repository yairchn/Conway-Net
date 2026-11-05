"""The networks of the paper: FullyConnectedNet, BaselineCNN, WideResNet and GraphGCN.

Every network uses tanh activations (the CA gate needs bounded state). Lateral inhibition is
wired as `z = gate(x, z)`, with the gate's birth/survive fields computed from the layer
input `x`, at each spatial layer. With `mechanism="none"` the gate is absent and the network
is a plain tanh net (the matched control).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lateral_inhibition import GraphLateralInhibitionCA, make_gate


# --------------------------------------------------------------------------- norms & init
def _conv_norm(norm, channels):
    """2-D normalization for conv feature maps: none (CNN) or GroupNorm (WRN)."""
    if norm == "none":
        return nn.Identity()
    if norm == "gn":
        g = min(32, channels)
        while channels % g:
            g //= 2
        return nn.GroupNorm(g, channels)
    raise ValueError(f"conv norm must be 'none' or 'gn', got {norm!r}")


def init_weights(model, method, weight_init_min, weight_init_max, init_classifier="same"):
    """Initialize conv/linear weights. 'uniform' draws U(min,max); 'kaiming_tanh' uses the
    tanh-gain init appropriate for a bounded-activation trunk.

    init_classifier: how the classifier/fc HEAD is initialized AFTER the general loop.
      'same'   = leave it with `method` (the general-loop init);
      'kaiming'= small `kaiming_uniform_(nonlinearity='relu')` head. A SMALL head init makes the readout
                 grow only the weights that matter -> coherent classifier-weight templates.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if method == "uniform":
                nn.init.uniform_(m.weight, weight_init_min, weight_init_max)
            elif method == "kaiming_tanh":
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="tanh")
                else:
                    nn.init.xavier_uniform_(m.weight, gain=5.0 / 3.0)
            else:
                raise ValueError(f"unknown init method {method!r}")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    if init_classifier == "kaiming":
        head = model.classifier if isinstance(getattr(model, "classifier", None), nn.Linear) else getattr(model, "fc", None)
        if isinstance(head, nn.Linear):
            nn.init.kaiming_uniform_(head.weight, nonlinearity="relu")
            if head.bias is not None:
                nn.init.zeros_(head.bias)
    elif init_classifier != "same":
        raise ValueError(f"init_classifier must be 'same' or 'kaiming', got {init_classifier!r}")


# --------------------------------------------------------------------------- FCN
class FullyConnectedNet(nn.Module):
    """A stack of `n_hidden_layers` (Linear -> LayerNorm -> tanh -> LI) blocks, then a linear head.

    The paper calls this network the MLP (multilayer perceptron); the code and run names keep "FCN"/"fcn".

    Every hidden layer has the input's width, and the LI gate reshapes each flattened hidden vector
    back to the input's (C, H, W) grid. The first gate reads the image, so its neighbourhoods are real 2-D image
    neighbourhoods; from the second layer on, a hidden unit's neighbours are the units with adjacent grid indices,
    whose spatial order has no meaning until training gives it one (the coherent maps of the paper).
    """

    def __init__(self, *, in_channels, image_size, num_classes, n_hidden_layers, norm, li_kwargs):
        super().__init__()
        if norm != "layernorm":
            raise ValueError(f"FullyConnectedNet norm must be 'layernorm', got {norm!r}")
        h = in_channels * image_size * image_size
        self.linears = nn.ModuleList([nn.Linear(h, h) for _ in range(n_hidden_layers)])
        self.classifier = nn.Linear(h, num_classes)
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(n_hidden_layers)])
        # one shared, parameter-free gate reused at every layer (built with the input geometry)
        self.gate = make_gate(channels=in_channels, height=image_size, width=image_size, **li_kwargs)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for linear, norm in zip(self.linears, self.norms):
            z = torch.tanh(norm(linear(x)))
            if self.gate is not None:
                z = self.gate(x, z)
            x = z
        return self.classifier(x)


# --------------------------------------------------------------------------- BaselineCNN
# Site-specific gate reduction: channel_reduce = "input_site:perch_bgref" applies the per-channel
# background-referenced reduction (lateral_inhibition._perch_bgref_gates) ONLY at the gate that reads the raw
# image; every gate reading learned features keeps the channel mean at the config's channel_reduce_stage.
_INPUT_SITE_SPECS = {"perch_bgref": ("perch_bgref", "input")}


def _site_li_kwargs(li_kwargs):
    """(image-gate kwargs, feature-gate kwargs). Plain channel_reduce values apply to every gate."""
    cr = li_kwargs["channel_reduce"]
    if not str(cr).startswith("input_site:"):
        return li_kwargs, li_kwargs
    spec = cr.split(":", 1)[1]
    if spec not in _INPUT_SITE_SPECS:
        raise ValueError(f"unknown input_site spec {spec!r}; choose from {list(_INPUT_SITE_SPECS)}")
    red, stage = _INPUT_SITE_SPECS[spec]
    return dict(li_kwargs, channel_reduce=red, channel_reduce_stage=stage), dict(li_kwargs, channel_reduce="mean")


class BaselineCNN(nn.Module):
    """Two conv stages (conv -> tanh -> LI, the second max-pooled), then a linear classifier.

    li1 gates conv1's output from the raw image; li2 gates conv2's post-pool output from z1.
    """

    def __init__(self, *, in_channels, image_size, num_classes, norm, conv1_channels,
                 conv2_channels, conv_padding, li_kwargs):
        super().__init__()
        li_img_kwargs, li_feat_kwargs = _site_li_kwargs(li_kwargs)
        self.conv1 = nn.Conv2d(in_channels, conv1_channels, 3, 1, padding=conv_padding)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, 3, 1, padding=conv_padding)
        self.norm1 = _conv_norm(norm, conv1_channels)
        self.norm2 = _conv_norm(norm, conv2_channels)
        self.li1 = make_gate(**li_img_kwargs)         # reads the image
        self.li2 = make_gate(**li_feat_kwargs)

        with torch.no_grad():
            d = torch.zeros(1, in_channels, image_size, image_size)
            flat = F.max_pool2d(self.conv2(self.conv1(d)), 2).flatten(1).shape[1]
        self.classifier = nn.Linear(flat, num_classes)

    def forward(self, x):
        z1 = torch.tanh(self.norm1(self.conv1(x)))
        if self.li1 is not None:
            z1 = self.li1(x, z1)
        z2 = F.max_pool2d(torch.tanh(self.norm2(self.conv2(z1))), 2)
        if self.li2 is not None:
            z2 = self.li2(z1, z2)
        return self.classifier(torch.flatten(z2, 1))


# --------------------------------------------------------------------------- WideResNet
class _WRNBlock(nn.Module):
    """Pre-activation WRN block (norm -> tanh -> conv, twice), direct LI z = LI(pre, z).

    ca1 gates conv1's (already bounded) output; ca2 gates conv2's raw sum after a bounding
    norm+tanh, and the skip is added after the gate (main_only placement)."""

    def __init__(self, in_ch, out_ch, stride, *, norm, li_kwargs, matched_none):
        super().__init__()
        self.norm1 = _conv_norm(norm, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm2 = _conv_norm(norm, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
                         if (stride != 1 or in_ch != out_ch) else None)
        self.ca1 = make_gate(**li_kwargs)
        self.ca2 = make_gate(**li_kwargs)
        # matched_none: on the none arm, still apply the ca2 norm+tanh bounding (gate -> identity),
        # so the plain network differs from the gated one ONLY by the gate.
        self.norm_ca2 = _conv_norm(norm, out_ch) if (self.ca2 is not None or matched_none) else None

    def forward(self, x):
        pre1 = torch.tanh(self.norm1(x))
        sc = self.shortcut(pre1) if self.shortcut is not None else x
        z1 = torch.tanh(self.norm2(self.conv1(pre1)))
        if self.ca1 is not None and min(z1.shape[-2:]) >= 5:
            z1 = self.ca1(pre1, z1)
        z2 = self.conv2(z1)
        if self.norm_ca2 is not None and min(z2.shape[-2:]) >= 5:
            z2b = torch.tanh(self.norm_ca2(z2))
            z2 = self.ca2(z1, z2b) if self.ca2 is not None else z2b
        return z2 + sc


class WideResNet(nn.Module):
    """WRN (default 40-2), pre-activation, tanh, direct LI. Flatten head over the final map.

    The stem conv lifts the image to 16 channels before any gate; the stem output goes ungated into block 1."""

    def __init__(self, *, in_channels, image_size, num_classes, depth, widen_factor,
                 norm, li_kwargs, matched_none):
        super().__init__()
        if (depth - 4) % 6:
            raise ValueError(f"WRN depth must satisfy (depth-4) % 6 == 0, got {depth}")
        n = (depth - 4) // 6
        stem, chans = 16, [16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(in_channels, stem, 3, 1, padding=1, bias=False)
        groups, in_ch = [], stem
        for gi, out_ch in enumerate(chans):
            blocks = []
            for bi in range(n):
                stride = 2 if (gi > 0 and bi == 0) else 1
                blocks.append(_WRNBlock(in_ch, out_ch, stride, norm=norm, li_kwargs=li_kwargs, matched_none=matched_none))
                in_ch = out_ch
            groups.append(nn.Sequential(*blocks))
        self.groups = nn.ModuleList(groups)

        self.norm_out = _conv_norm(norm, chans[-1])
        # flatten head over the final feature map (global-avg-pool discards the spatial
        # structure the CA gate operates on, so it is deliberately not offered)
        final = image_size // 4
        self.fc = nn.Linear(chans[-1] * final * final, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for group in self.groups:
            x = group(x)
        x = torch.tanh(self.norm_out(x))
        return self.fc(x.flatten(1))


# --------------------------------------------------------------------------- GraphGCN
class GraphGCN(nn.Module):
    """GCN classifier on the pixel graph: one node per pixel, edges to the 8 Moore neighbours (no wraparound;
    border pixels have 3 or 5 neighbours), `n_hidden_layers` layers of `hidden_channels` channels per node.

    layer l:  z = tanh(A_hat h W_l + b_l),  A_hat = D^-1/2 (A + I) D^-1/2,  W_l a per-node linear map
              with lateral inhibition: z = LI(h, z), the gate's m the mean occupancy over the node's neighbours
    head:     flatten (C H W) -> linear. h_0 is the image; the first gate reads it.
    """

    def __init__(self, *, in_channels, image_size, num_classes, n_hidden_layers, hidden_channels, li_kwargs):
        super().__init__()
        S, C = image_size, hidden_channels
        self.S = S
        widths = [in_channels] + [C] * n_hidden_layers
        self.linears = nn.ModuleList([nn.Linear(widths[i], widths[i + 1], bias=False) for i in range(n_hidden_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(C)) for _ in range(n_hidden_layers)])
        self.fc = nn.Linear(C * S * S, num_classes)
        r, c = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
        src, dst = [], []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    ok = (r + dy >= 0) & (r + dy < S) & (c + dx >= 0) & (c + dx < S)
                    src.append((r * S + c)[ok]); dst.append(((r + dy) * S + (c + dx))[ok])
        src, dst = torch.cat(src), torch.cat(dst)
        deg = torch.zeros(S * S).scatter_add_(0, src, torch.ones(len(src)))
        loops = torch.arange(S * S)
        rr, cc = torch.cat([src, loops]), torch.cat([dst, loops])
        dh = (deg + 1).rsqrt()
        self.register_buffer("A_hat", torch.sparse_coo_tensor(torch.stack([rr, cc]), dh[rr] * dh[cc], (S * S, S * S)).coalesce())
        self.register_buffer("A_mean", torch.sparse_coo_tensor(torch.stack([src, dst]), 1.0 / deg[src], (S * S, S * S)).coalesce())
        if li_kwargs["mechanism"] == "ca":
            kw = {k: v for k, v in li_kwargs.items() if k not in ("mechanism", "neighbourhood_radius")}
            if li_kwargs["neighbourhood_radius"] != 1:
                raise ValueError("GraphGCN's gate reads the 8-neighbour pixel graph: neighbourhood_radius must be 1")
            self.gate = GraphLateralInhibitionCA(neighbour_mean=lambda t: self._prop(self.A_mean, t), **kw)
        elif li_kwargs["mechanism"] == "none":
            self.gate = None
        else:
            raise ValueError(f"mechanism must be 'ca' or 'none', got {li_kwargs['mechanism']!r}")

    def _prop(self, A, t):
        """Apply the (S^2, S^2) node operator A to t (N, C, S, S), keeping the layout."""
        N, Ch = t.shape[:2]
        y = torch.sparse.mm(A, t.permute(2, 3, 0, 1).reshape(self.S * self.S, N * Ch))
        return y.reshape(self.S, self.S, N, Ch).permute(2, 3, 0, 1)

    def hidden_states(self, x):
        hs, h = [], x
        for linear, b in zip(self.linears, self.biases):
            z = linear(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            z = torch.tanh(self._prop(self.A_hat, z) + b[None, :, None, None])
            if self.gate is not None:
                z = self.gate(h, z)
            h = z
            hs.append(h)
        return hs

    def forward(self, x):
        return self.fc(self.hidden_states(x)[-1].flatten(1))


# --------------------------------------------------------------------------- builder
_NETWORKS = {"FullyConnectedNet": FullyConnectedNet, "BaselineCNN": BaselineCNN, "WideResNet": WideResNet,
             "GraphGCN": GraphGCN}


def build_model(config, meta):
    """Instantiate the configured network. `meta` supplies in_channels/num_classes/image_size
    from the dataset; `config` supplies everything else."""
    if str(config["channel_reduce"]).startswith("input_site:") and config["network_class"] != "BaselineCNN":
        raise ValueError("channel_reduce='input_site:...' needs a gate that reads the image: BaselineCNN only")
    li_kwargs = dict(
        mechanism=config["mechanism"], rule=config["rule"], gamma=config["gamma"],
        delta=config["delta"], normalize_gates=config["normalize_gates"],
        require_zero_centring=config["require_zero_centring"],
        channel_reduce=config["channel_reduce"],
        channel_reduce_stage=config["channel_reduce_stage"],
        neighbourhood_radius=config["neighbourhood_radius"],
    )
    common = dict(in_channels=meta["channels"], image_size=meta["image_size"],
                  num_classes=meta["num_classes"], li_kwargs=li_kwargs)

    name = config["network_class"]
    if name == "FullyConnectedNet":
        model = FullyConnectedNet(n_hidden_layers=config["n_hidden_layers"], norm=config["norm"], **common)
    elif name == "BaselineCNN":
        model = BaselineCNN(norm=config["norm"], conv1_channels=config["conv1_channels"],
                            conv2_channels=config["conv2_channels"], conv_padding=config["conv_padding"], **common)
    elif name == "WideResNet":
        model = WideResNet(depth=config["depth"], widen_factor=config["widen_factor"], norm=config["norm"],
                           matched_none=config["matched_none"], **common)
    elif name == "GraphGCN":
        model = GraphGCN(n_hidden_layers=config["n_hidden_layers"], hidden_channels=config["hidden_channels"], **common)
    else:
        raise ValueError(f"unknown network_class {name!r}; choose from {list(_NETWORKS)}")

    init_weights(model, config["init_method"], config["weight_init_min"], config["weight_init_max"],
                 init_classifier=config["init_classifier"])
    return model
