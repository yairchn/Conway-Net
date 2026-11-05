"""Scale-separation index (SSI) of the FCN's hidden maps: the one definition used by every paper script.

Each hidden layer's activation, on the 28x28 grid, is band-passed by spatial frequency with radial masks centred on
the DC bin (index HW // 2 = 14 after fftshift, so every mask is conjugate symmetric and the band-passed maps are
real), radii BAND_EDGES in FFT bins: [0,2) coarse ... [13,20) fine. For band b and depth lag k,
  C_b(k) = mean over layer pairs (l, l + k) and images of the cosine between the mean-removed band-passed maps,
  tau_b  = 1/2 + sum_{k=1..K} C_b(k) / C_b(0), the sum stopped at the first k with C_b(k) <= 0,
  SSI    = tau_coarse / tau_fine,
with the same horizon K = MAX_LAG = 9 at every depth (ten-layer networks have no larger lag). Hence
1/2 <= tau_b <= K + 1/2 = 9.5 and SSI <= 19. tau_b is right-censored (`censored`) when C_b(k) > 0 at every k <= K:
the band is still correlated at the last observed lag, and tau_b is then a lower bound set by the horizon.
Computed on the first N_IMAGES = 1000 test images.
"""
import numpy as np
import torch

HW = 28; BAND_EDGES = [0, 2, 4, 6, 9, 13, 20]; MAX_LAG = 9; N_IMAGES = 1000


def band_masks(device):
    yy, xx = np.meshgrid(np.arange(HW), np.arange(HW), indexing="ij")
    r = np.sqrt((yy - HW // 2) ** 2 + (xx - HW // 2) ** 2)
    masks = [((r >= lo) & (r < hi)).astype(np.float32) for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]
    for mk in masks:                                   # conjugate symmetry: shifted index u <-> (HW - u) mod HW
        if not np.array_equal(mk, np.roll(mk[::-1, ::-1], (1, 1), axis=(0, 1))):
            raise ValueError("band mask is not conjugate symmetric")
    return [torch.tensor(mk, device=device) for mk in masks]


@torch.no_grad()
def scale_separation(maps):
    """maps: (L, N, HW, HW) hidden activations of N images at L layers. Returns C (bands x lags 0..K), tau, censored, ssi."""
    L, N = maps.shape[:2]
    if L <= MAX_LAG:
        raise ValueError(f"need more than {MAX_LAG} layers for the common horizon, got {L}")
    F = torch.fft.fftshift(torch.fft.fft2(maps.double()), dim=(-2, -1))
    C, tau, censored = [], [], []
    for mk in band_masks(maps.device):
        Mb = torch.fft.ifft2(torch.fft.ifftshift(F * mk, dim=(-2, -1)))
        v = Mb.real.reshape(L, N, -1); v = v - v.mean(-1, keepdim=True); vn = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
        c = [float((vn[:L - k] * vn[k:]).sum(-1).mean()) for k in range(MAX_LAG + 1)]
        s, cut = 0.0, False
        for k in range(1, MAX_LAG + 1):
            if c[k] <= 0:
                cut = True; break
            s += c[k] / c[0]
        C.append(c); tau.append(0.5 + s); censored.append(not cut)
    return dict(C=C, tau=tau, censored=censored, ssi=tau[0] / tau[-1])
