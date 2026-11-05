"""GPU-side AugMix.

AugMix (Hendrycks et al. 2020): for each training image, generate K random
augmentation chains, blend them with random Dirichlet weights, then mix the
blend with the original via a Beta-distributed coefficient. The model sees
the final blended image. This trains the model to be invariant to the kind
of pixel-level variation that augmentations cause — invariance which transfers
to unseen corruption families (the CIFAR-10-C test).

How this differs from the published recipe (kept as is: the paper's WRN networks were trained with it):
- The op list is AugMix's "all ops" variant: besides autocontrast, solarize, rotate, shear and translate it
  has color, contrast, brightness and sharpness. Brightness and contrast are themselves CIFAR-10-C corruption
  types, and sharpness with a factor < 1 blends toward a 3x3 box blur, so these overlap the non-noise CIFAR-10-C
  corruptions; equalize and posterize (in the default list) are absent. There is no noise op, so the noise
  corruptions the paper evaluates (Gaussian, shot, impulse; speckle in the CIFAR-10-C table) are not trained on.
- Op choice and chain depth are drawn once per chain for the whole batch (Python `random`, seeded by
  scripts/run.py), not per image; magnitudes are drawn per image.
- Magnitudes: rotate and translate as in the reference; the shear factor (tan(30 deg x mag), about 0.16 at
  severity 3) is larger than the reference's, and the solarize threshold stays above 0.85 at severity 3.
- autocontrast uses one min/max over all channels (PIL works per channel).
- Operates on un-normalized [0, 1] tensors; apply normalize_gpu AFTER AugMix. Runs under torch.no_grad.

GPU implementation on batches in CHW float [0, 1]. Taken from the research code's training templates; identical
for RGB inputs (sharpness and color were generalised to 1-channel inputs).
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitive batched augmentations (each takes [B, C, H, W] in [0, 1])
# ---------------------------------------------------------------------------
def _batch_autocontrast(imgs: torch.Tensor) -> torch.Tensor:
    B = imgs.shape[0]
    flat = imgs.view(B, -1)
    lo = flat.min(1).values.view(B, 1, 1, 1)
    hi = flat.max(1).values.view(B, 1, 1, 1)
    return ((imgs - lo) / (hi - lo + 1e-7)).clamp(0, 1)


def _batch_solarize(imgs: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    return torch.where(imgs < thresholds.view(-1, 1, 1, 1), imgs, 1.0 - imgs)


def _batch_affine(
    imgs: torch.Tensor,
    angles: torch.Tensor | None = None,
    tx: torch.Tensor | None = None,
    ty: torch.Tensor | None = None,
    shx: torch.Tensor | None = None,
    shy: torch.Tensor | None = None,
) -> torch.Tensor:
    B, C, H, W = imgs.shape
    dev = imgs.device
    z = torch.zeros(B, device=dev)
    cos_a, sin_a = torch.ones(B, device=dev), z.clone()
    if angles is not None:
        rad = angles * (math.pi / 180.0)
        cos_a, sin_a = torch.cos(rad), torch.sin(rad)
    # note: in this matrix `shx` shears along y and `shy` along x (both are sampled symmetrically)
    shx_ = shx if shx is not None else z
    shy_ = shy if shy is not None else z
    tx_n = (2.0 * tx / W) if tx is not None else z
    ty_n = (2.0 * ty / H) if ty is not None else z
    row0 = torch.stack([cos_a - shy_ * sin_a, -sin_a - shy_ * cos_a, tx_n], 1)
    row1 = torch.stack([sin_a + shx_ * cos_a,  cos_a - shx_ * sin_a, ty_n], 1)
    grid = F.affine_grid(torch.stack([row0, row1], 1), (B, C, H, W), align_corners=False)
    return F.grid_sample(imgs, grid, mode="bilinear",
                         padding_mode="reflection", align_corners=False)


def _batch_brightness(imgs: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    return (imgs * f.view(-1, 1, 1, 1)).clamp(0, 1)


def _batch_contrast(imgs: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    return (f.view(-1, 1, 1, 1) * imgs +
            (1 - f.view(-1, 1, 1, 1)) * imgs.mean(dim=(1, 2, 3), keepdim=True)).clamp(0, 1)


def _batch_saturation(imgs: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    gray = 0.2989 * imgs[:, 0:1] + 0.5870 * imgs[:, 1:2] + 0.1140 * imgs[:, 2:3]
    return (f.view(-1, 1, 1, 1) * imgs +
            (1 - f.view(-1, 1, 1, 1)) * gray).clamp(0, 1)


def _batch_sharpness(imgs: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    C = imgs.shape[1]   # channel-agnostic (was hard-coded 3): C=3 identical, C=1 enables grayscale
    k = torch.ones(C, 1, 3, 3, device=imgs.device, dtype=imgs.dtype) / 9.0
    return (f.view(-1, 1, 1, 1) * imgs +
            (1 - f.view(-1, 1, 1, 1)) * F.conv2d(imgs, k, padding=1, groups=C)).clamp(0, 1)


_AUGMIX_OPS = [
    "autocontrast", "solarize", "rotate", "shear_x", "shear_y",
    "translate_x", "translate_y", "color", "contrast", "brightness", "sharpness",
]


class GPUAugMix:
    """Vectorised AugMix that runs entirely on the model's device.

    Parameters
    ----------
    severity : int
        Controls each primitive's magnitude. Hendrycks default = 3 (recommended).
    mixture_width : int
        K = number of chains to blend per image. Hendrycks default = 3.
    alpha : float
        Dirichlet and Beta concentration. Hendrycks default = 1.0 (≈ uniform).

    Notes
    -----
    Does NOT apply Jensen-Shannon-divergence consistency loss (an optional
    component of the published recipe). The bulk of the robustness gain comes
    from the mixing itself; JSD is an additional ~1pp.

    Inputs and outputs are un-normalized [0, 1] tensors. Normalize AFTER
    calling this. AugMix MUST see [0, 1] pixels — applying it to z-scored
    inputs makes rotate/brightness/contrast meaningless.
    """

    def __init__(self, severity: int, mixture_width: int, alpha: float) -> None:
        self.severity = int(severity)
        self.width = int(mixture_width)
        self.alpha = float(alpha)
        self._ops = _AUGMIX_OPS

    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = imgs.shape
        dev = imgs.device
        ws = torch.distributions.Dirichlet(
            torch.full((self.width,), self.alpha, device=dev)
        ).sample((B,))
        m = torch.distributions.Beta(
            torch.tensor(self.alpha, device=dev),
            torch.tensor(self.alpha, device=dev),
        ).sample((B,))
        mix = torch.zeros_like(imgs)
        for i in range(self.width):
            aug = imgs.clone()
            for _ in range(random.randint(1, 3)):
                aug = self._op(aug, random.choice(self._ops))
            mix = mix + ws[:, i].view(B, 1, 1, 1) * aug
        return (m.view(B, 1, 1, 1) * imgs + (1 - m.view(B, 1, 1, 1)) * mix).clamp(0, 1)

    def _op(self, imgs: torch.Tensor, op: str) -> torch.Tensor:
        B = imgs.shape[0]
        dev = imgs.device
        mag = torch.rand(B, device=dev) * (self.severity / 10.0)
        if op == "autocontrast":
            return _batch_autocontrast(imgs)
        if op == "solarize":
            return _batch_solarize(imgs, 1.0 - mag * 0.5)
        if op == "rotate":
            return _batch_affine(imgs, angles=(torch.rand(B, device=dev) * 2 - 1) * 30 * mag)
        if op == "shear_x":
            return _batch_affine(imgs, shx=torch.tan((torch.rand(B, device=dev) * 2 - 1) * 30 * mag * (math.pi / 180)))
        if op == "shear_y":
            return _batch_affine(imgs, shy=torch.tan((torch.rand(B, device=dev) * 2 - 1) * 30 * mag * (math.pi / 180)))
        if op == "translate_x":
            return _batch_affine(imgs, tx=(torch.rand(B, device=dev) * 2 - 1) * imgs.shape[3] * 0.33 * mag)
        if op == "translate_y":
            return _batch_affine(imgs, ty=(torch.rand(B, device=dev) * 2 - 1) * imgs.shape[2] * 0.33 * mag)
        if op == "brightness":
            return _batch_brightness(imgs, 1.0 + (torch.rand(B, device=dev) * 2 - 1) * 1.8 * mag)
        if op == "contrast":
            return _batch_contrast(imgs, 1.0 + (torch.rand(B, device=dev) * 2 - 1) * 1.8 * mag)
        if op == "color":
            # saturation needs RGB (mixes toward luma over channels 0/1/2). Grayscale has no
            # colour to jitter -> no-op for C<3 (keeps RGB identical, enables 1-channel inputs).
            return _batch_saturation(imgs, 1.0 + (torch.rand(B, device=dev) * 2 - 1) * 1.8 * mag) if imgs.shape[1] >= 3 else imgs
        if op == "sharpness":
            return _batch_sharpness(imgs, 1.0 + (torch.rand(B, device=dev) * 2 - 1) * 1.8 * mag)
        return imgs


def normalize_gpu(imgs: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
    """Apply per-channel normalize on GPU. Use AFTER GPUAugMix; the dataloader
    should return un-normalized [0, 1] tensors when AugMix is enabled.

    Parameters
    ----------
    imgs : (B, C, H, W) in [0, 1]
    mean, std : per-channel statistics, same length as C
    """
    m = torch.tensor(mean, device=imgs.device, dtype=imgs.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std,  device=imgs.device, dtype=imgs.dtype).view(1, -1, 1, 1)
    return (imgs - m) / s


def random_affine(x, rotate_deg, translate_px, scale_min, scale_max, shear_deg):
    """Per-image random geometric affine for TRAINING augmentation (rotation/translate/scale/shear).

    Operates on the batch as-is (normalized or [0,1]); `padding_mode="border"` replicates the edge
    (= background for a centred MNIST-family digit), so it is background-value- and dataset-agnostic.
    GEOMETRIC ONLY — no photometric/solarize/invert/noise ops, so it does NOT leak the noise-mCA or
    polarity-invariance evals (lateral-inhibition skill §2). Identity params -> returns x unchanged.
    """
    if rotate_deg == 0 and translate_px == 0 and scale_min == 1.0 and scale_max == 1.0 and shear_deg == 0:
        return x
    B, C, H, W = x.shape
    dev, dt = x.device, x.dtype
    rand = lambda: (torch.rand(B, device=dev, dtype=dt) * 2 - 1)
    a = rand() * rotate_deg * (math.pi / 180.0)
    sc = torch.rand(B, device=dev, dtype=dt) * (scale_max - scale_min) + scale_min
    shx = torch.tan(rand() * shear_deg * (math.pi / 180.0))
    shy = torch.tan(rand() * shear_deg * (math.pi / 180.0))
    tx = rand() * translate_px * (2.0 / W)
    ty = rand() * translate_px * (2.0 / H)
    cos_a, sin_a = torch.cos(a), torch.sin(a)
    inv = 1.0 / sc                                      # sampling-grid inverse: sc>1 zooms in
    # 2x2 sampling linear part = (1/sc) * R(a) * Shear, then translate in output space
    row0 = torch.stack([(cos_a - shy * sin_a) * inv, (-sin_a - shy * cos_a) * inv, tx], 1)
    row1 = torch.stack([(sin_a + shx * cos_a) * inv, (cos_a - shx * sin_a) * inv, ty], 1)
    theta = torch.stack([row0, row1], 1)
    grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
