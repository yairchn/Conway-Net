"""Single-run config: a nested JSON of sections, flattened to one dict and lightly validated.

No parameter sweeps, no plural keys — one config describes exactly one run. Missing keys fail
loud (CLAUDE.md §8): every behaviour comes from the config, nothing is silently defaulted.
"""
import json

SECTIONS = ("model", "lateral_inhibition", "optimizer", "training", "augmentation",
            "evaluation", "data", "logging")

REQUIRED = {
    "model":              ("network_class", "norm", "init_method",
                           "weight_init_min", "weight_init_max", "activation", "matched_none"),
    "lateral_inhibition": ("mechanism", "rule", "gamma", "delta", "normalize_gates",
                           "require_zero_centring",
                           "channel_reduce", "channel_reduce_stage", "neighbourhood_radius"),
    "optimizer":          ("type", "lr", "weight_decay"),
    "training":           ("num_epochs", "batch_size", "seed", "lr_schedule", "max_norm",
                           "num_workers", "device", "val_frac", "split_seed"),
    "augmentation":       ("augmix_severity", "augmix_width", "augmix_alpha"),
    "evaluation":         ("eval_mca", "mca_corruptions", "mca_n_samples"),
    "data":               ("data_name", "data_path"),
    "logging":            ("output_path",),
}

_NETWORKS = ("FullyConnectedNet", "BaselineCNN", "WideResNet", "GraphGCN")
_MECHANISMS = ("ca", "none")
_OPTIMIZERS = ("sgd", "adam", "muon")
_SCHEDULES = ("constant", "cosine")
_MCA_SETS = ("noise", "all15")

# Options removed from the code because no network in the paper uses them. Older configs (and the config copies
# in paper_runs/) still carry these keys; they load only when the key holds its "off" value, so a config that
# relied on a removed option fails loudly instead of being silently built without it.
_RETIRED = {
    "dropout": (0, 0.0), "hidden_size": (None,), "num_groups": (None,),
    "stem_lift": (False,), "stem_lc": (False,), "stem_residual": (None, False, True),
    "stem_norm": ("groupnorm", "layernorm", "none"),        # had no effect without stem_lift
    "head_bottleneck": (False,), "conv_dropout": (False,), "wrn_stem": ("conv",),
    "normalizer_exponent": (1, 1.0), "gate_peak_normalize": (False,), "ensemble_size": (1,),
    "loss": ("cross_entropy",), "classifier_mask": ("none",), "keypad_loss_weight": (0, 0.0),
    "li_skip_last": (0,), "roll_px": (0,),
    # removed 2026-10-23 (the stem appendix and the CCT exploration are no longer in the paper)
    "conv_stem": ("none",), "stem_gate": ("none",), "stem_bound": ("none",),
    "warmup_epochs": (0,), "label_smoothing": (0, 0.0),
    "require_rescaling": (True,),                          # the occupancy maps (x+1)/2 and (z+1)/2 are always applied
    # removed 2026-11-05 (the soft gate bound is not in the paper; the gate always clips)
    "gate_bound": ("clip",), "gate_softclip_beta": (None,),
}


def load_config(path):
    """Read the nested-JSON config, flatten sections into one dict, and validate."""
    with open(path) as f:
        nested = json.load(f)

    unknown = set(nested) - set(SECTIONS)
    if unknown:
        raise ValueError(f"Unknown config sections {sorted(unknown)}; allowed: {SECTIONS}")

    flat = {}
    for section in SECTIONS:
        if section not in nested:
            raise ValueError(f"Config is missing the '{section}' section")
        for key in REQUIRED[section]:
            if key not in nested[section]:
                raise ValueError(f"Config section '{section}' is missing required key '{key}'")
        for key, value in nested[section].items():
            if key in flat:
                raise ValueError(f"Duplicate config key '{key}' across sections")
            flat[key] = value

    if flat["network_class"] not in _NETWORKS:
        raise ValueError(f"network_class must be one of {_NETWORKS}, got {flat['network_class']!r}")
    if flat["activation"] != "tanh":
        raise ValueError(f"model.activation must be 'tanh', got {flat['activation']!r}")
    for key, off in _RETIRED.items():
        if key in flat and flat[key] not in off:
            raise ValueError(f"config key {key!r} = {flat[key]!r}: that option was removed from the code (allowed: {off})")
    if flat["mechanism"] not in _MECHANISMS:
        raise ValueError(f"lateral_inhibition.mechanism must be one of {_MECHANISMS}, got {flat['mechanism']!r}")
    if flat["type"] not in _OPTIMIZERS:
        raise ValueError(f"optimizer.type must be one of {_OPTIMIZERS}, got {flat['type']!r}")
    if flat["lr_schedule"] not in _SCHEDULES:
        raise ValueError(f"training.lr_schedule must be one of {_SCHEDULES}, got {flat['lr_schedule']!r}")
    if flat["mca_corruptions"] not in _MCA_SETS:
        raise ValueError(f"evaluation.mca_corruptions must be one of {_MCA_SETS}, got {flat['mca_corruptions']!r}")

    data_norms = flat.get("data_norms")
    if data_norms is not None:
        flat["data_norms"] = (tuple(data_norms[0]), tuple(data_norms[1]))
    return flat
