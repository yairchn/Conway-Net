"""Unified corruption suite for mCA-style robustness evaluation.

Single source of truth for BOTH the standard Hendrycks severities and the
*extended* severities (pushed toward maximum effect). Replaces the earlier
copy-paste split (a `_corruption_helper` + a separate `corruptions_extended`).

Design
------
Every corruption has one severity *ladder* in ``LADDERS``. The first
``STANDARD_N`` (=5) entries ARE the Hendrycks library severities; any entries
beyond that are the extension. So:

    standard severities  = LADDERS[name][:STANDARD_N]
    extended severities  = LADDERS[name]              (standard is a true prefix)

"extended" is therefore an EVAL-TIME slice, not a separate code path or a
separate set of images: generate the full ladder once (cache it), then average
severities ``[:5]`` for standard mCA or all of them for extended mCA.

Two tiers of corruption
-----------------------
* SCALAR corruptions (``EXTENDABLE``): a single scalar knob (noise std, jpeg
  quality, contrast factor, ...). Reimplemented here faithfully so the knob can
  be pushed past sev5 to maximum effect. Applied at NATIVE resolution.
* TUPLE corruptions (blur family, snow/frost/fog, elastic): their severity
  couples 2-7 parameters (some non-monotone), so there is no principled
  single-axis extension — they are STANDARD-ONLY (5 levels) and delegate to the
  ``imagecorruptions`` library. That library requires >=32px, so sub-32 inputs
  (e.g. 28px MNIST-family) are resized 28->32->corrupt->28. This resize is the
  ONLY place a resize artifact enters, and it affects tuple corruptions on
  sub-32 datasets only. Datasets >=32px (CIFAR-10) never resize.

Because scalar corruptions run natively at both standard and extended severities,
``LADDERS[name][:5]`` is a TRUE subset of the extended ladder for them (same
pixels). (The legacy exploratory code resized scalar corruptions on MNIST-family
too; this module deliberately does not, giving faithful noise magnitudes.)
"""
from __future__ import annotations
import numpy as np
from io import BytesIO
from PIL import Image

# --- numpy 2.0 / skimage compatibility shims (imagecorruptions predates them) ---
if not hasattr(np, "float_"):
    np.float_ = np.float64          # fog's plasma_fractal uses np.float_
import skimage as sk                 # noqa: E402
import skimage.filters as _skf       # noqa: E402
_orig_gaussian = _skf.gaussian
def _gaussian_compat(image, sigma, multichannel=None, channel_axis=None, **kw):
    if multichannel is not None and channel_axis is None:
        channel_axis = -1 if multichannel else None
    return _orig_gaussian(image, sigma, channel_axis=channel_axis, **kw)
import imagecorruptions.corruptions as _ic  # noqa: E402
_ic.gaussian = _gaussian_compat     # glass_blur references module-level gaussian
from imagecorruptions import corrupt as _corrupt  # noqa: E402  (top-level dispatcher)

STANDARD_N = 5

# name -> severity ladder. First STANDARD_N entries == Hendrycks library values.
# Scalar (extendable) corruptions extend past 5; tuple corruptions list only the
# 5 library tuples (informational; application delegates to the library).
LADDERS = {
    # ---- scalar, extendable (single knob) ----
    "gaussian_noise":   [.08, .12, .18, .26, .38, .50, .70, .90, 1.20],   # std
    "shot_noise":       [60, 25, 12, 5, 3, 2, 1.2, 0.7, 0.4],             # poisson scale (lower=worse)
    "impulse_noise":    [.03, .06, .09, .17, .27, .40, .55, .70, .90],    # s&p fraction
    "brightness":       [.1, .2, .3, .4, .5, .7, 1.0, 1.5, 2.5],          # +V shift
    "contrast":         [.40, .30, .20, .10, .05, .02, .01, .005, .0],    # multiplier (0=flat)
    "pixelate":         [.60, .50, .40, .30, .25, .18, .12, .08, .04],    # downscale frac
    "jpeg_compression": [25, 18, 15, 10, 7, 5, 3, 2, 1],                  # quality (lower=worse)
    "zoom_blur":        [1.11, 1.16, 1.21, 1.26, 1.31, 1.5, 2.0, 2.5, 3.0],  # max zoom (step fixed)
    # ---- tuple, standard-only (library severities 1..5) ----
    "defocus_blur":     [(3, .1), (4, .5), (6, .5), (8, .5), (10, .5)],
    "glass_blur":       [(.7, 1, 2), (.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)],
    "motion_blur":      [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)],
    "snow":             list(range(5)),   # placeholder; library owns the params
    "frost":            list(range(5)),
    "fog":              list(range(5)),
    "elastic_transform": list(range(5)),
}
EXTENDABLE = {"gaussian_noise", "shot_noise", "impulse_noise", "brightness",
              "contrast", "pixelate", "jpeg_compression", "zoom_blur"}
CORRUPTIONS_15 = ("gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
                  "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
                  "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression")
_ZOOM_STEP = 0.03


# ---------------- scalar corruption implementations (faithful, native) --------
def _gaussian_noise(x, c, rng):
    z = x / 255.0
    return (np.clip(z + rng.normal(size=z.shape, scale=c), 0, 1) * 255).astype(np.uint8)


def _shot_noise(x, c, rng):
    z = x / 255.0
    return (np.clip(rng.poisson(z * c) / float(c), 0, 1) * 255).astype(np.uint8)


def _impulse_noise(x, c, rng):
    # skimage >= 0.19 ignores numpy's global RNG and draws fresh randomness unless given its own seed, so the seed is
    # passed explicitly (seeding np.random here, as before, left impulse noise different on every call)
    z = sk.util.random_noise(x / 255.0, mode="s&p", amount=float(c), rng=int(rng.randint(1 << 30)))
    return (np.clip(z, 0, 1) * 255).astype(np.uint8)


def _brightness(x, c, rng):
    z = x / 255.0
    if z.ndim < 3 or z.shape[2] < 3:
        z = np.clip(z + c, 0, 1)
    else:
        z = sk.color.rgb2hsv(z); z[:, :, 2] = np.clip(z[:, :, 2] + c, 0, 1); z = sk.color.hsv2rgb(z)
    return (np.clip(z, 0, 1) * 255).astype(np.uint8)


def _contrast(x, c, rng):
    z = x / 255.0; m = np.mean(z, axis=(0, 1), keepdims=True)
    return (np.clip((z - m) * c + m, 0, 1) * 255).astype(np.uint8)


def _pixelate(x, c, rng):
    im = Image.fromarray(x); H, W = x.shape[:2]
    small = im.resize((max(1, int(W * c)), max(1, int(H * c))), Image.BOX)
    return np.array(small.resize((W, H), Image.NEAREST), dtype=np.uint8)


def _jpeg(x, c, rng):
    im = Image.fromarray(x); gray = im.mode != "RGB"
    if gray:
        im = im.convert("RGB")
    buf = BytesIO(); im.save(buf, "JPEG", quality=int(c)); out = Image.open(buf)
    return np.array(out.convert("L") if gray else out, dtype=np.uint8)


def _zoom_blur(x, c, rng):
    z = (x / 255.0).astype(np.float32)
    if z.ndim < 3:
        z = np.repeat(z[:, :, None], 3, axis=-1)
    facs = np.arange(1, c + 1e-9, _ZOOM_STEP); out = np.zeros_like(z)
    for zf in facs:
        out += _ic.clipped_zoom(z, zf)[:z.shape[0], :z.shape[1], :]
    z = (z + out) / (len(facs) + 1)
    return (np.clip(z, 0, 1) * 255).astype(np.uint8)


_SCALAR = {"gaussian_noise": _gaussian_noise, "shot_noise": _shot_noise,
           "impulse_noise": _impulse_noise, "brightness": _brightness,
           "contrast": _contrast, "pixelate": _pixelate, "jpeg_compression": _jpeg,
           "zoom_blur": _zoom_blur}


# ---------------- public API --------------------------------------------------
def n_levels(name, extended=True):
    """Number of severity levels for a corruption (5 if not extended / tuple)."""
    return len(LADDERS[name]) if (extended and name in EXTENDABLE) else STANDARD_N


def apply(img_u8_rgb, name, level_idx, rng=None, resize_min=32):
    """Apply ``name`` at ``LADDERS[name][level_idx]`` to an (H,W,3) uint8 image.

    Scalar corruptions run natively (no resize). Tuple corruptions delegate to
    imagecorruptions, resizing to ``resize_min`` when the image is smaller.
    ``rng`` (np.random.RandomState) seeds the stochastic scalar corruptions;
    if None a fresh default RNG is used.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    if name in _SCALAR:
        return _SCALAR[name](img_u8_rgb, LADDERS[name][level_idx], rng)
    if level_idx >= STANDARD_N:
        raise ValueError(f"{name!r} is tuple-parameterised (standard-only); level_idx must be < {STANDARD_N}")
    H, W = img_u8_rgb.shape[:2]
    if min(H, W) >= resize_min:
        return _corrupt(img_u8_rgb, corruption_name=name, severity=level_idx + 1)
    up = np.array(Image.fromarray(img_u8_rgb).resize((resize_min, resize_min), Image.BILINEAR))
    out = _corrupt(up, corruption_name=name, severity=level_idx + 1)
    return np.array(Image.fromarray(out).resize((W, H), Image.BILINEAR))


if __name__ == "__main__":
    # Self-test: shapes hold for every corruption/level on 28px (resize path) and
    # 32px (native), and deterministic scalar corruptions reproduce the library
    # at the standard severities on a >=32 image.
    rng = np.random.RandomState(0)
    for (H, W, tag) in [(28, 28, "28px"), (32, 32, "32px")]:
        img = rng.randint(0, 255, (H, W, 3), dtype=np.uint8)
        for name in CORRUPTIONS_15:
            for i in range(n_levels(name)):
                o = apply(img, name, i, np.random.RandomState(1))
                assert o.shape == img.shape, (name, i, o.shape)
        print(f"[{tag}] all corruptions x all levels OK")
    # deterministic reproduction check (contrast/brightness/pixelate/jpeg) at 32px
    img = np.random.RandomState(2).randint(0, 255, (32, 32, 3), dtype=np.uint8)
    for name in ("brightness", "contrast", "pixelate", "jpeg_compression"):
        for i in range(STANDARD_N):
            mine = apply(img, name, i)
            lib = _corrupt(img, corruption_name=name, severity=i + 1)
            if name in ("brightness", "contrast"):
                assert np.allclose(mine, lib, atol=1), f"{name} sev{i+1} mismatch vs library"
    print("deterministic scalar corruptions reproduce library at standard severities")
