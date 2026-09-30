"""CAM methods: published baselines + the proposed MSF-Grad-CAM.

MSF-Grad-CAM (Multi-Scale Fusion Grad-CAM) - the "enhanced Grad-CAM":

  1. Deep layer L: standard Grad-CAM (channel weights = spatially averaged
     gradients) -> semantically reliable but coarse (e.g. 4x4 at 112 px input).
  2. Shallower layers l: element-wise positive-gradient weighting
     (Layer-CAM style), C_l = ReLU( sum_k ReLU(dy/dA_k) * A_k ), which keeps
     fine spatial detail (eyebrows, lip corners) that averaging would erase.
  3. Fusion: every map is upsampled to input size and min-max normalised, then
         E = N( C_L  *  sum_l w_l C_l / sum_l w_l )
     The deep map acts as a semantic gate: fine detail survives only where the
     class-discriminative deep evidence agrees, suppressing shallow edge noise.
  4. Flip test-time augmentation: E(x) and mirror(E(mirror(x))) are averaged,
     which reduces gradient noise on (roughly symmetric) faces.

Variants for the ablation study:
  msf         full method (fusion + flip-TTA)
  msf_notta   fusion only
  gradcam_tta deep-layer Grad-CAM + flip-TTA only
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import (EigenCAM, GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM,
                              XGradCAM)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

BASELINES = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "xgradcam": XGradCAM,
    "layercam": LayerCAM,
    "scorecam": ScoreCAM,
    "eigencam": EigenCAM,
}
PROPOSED = {"msf", "msf_notta", "gradcam_tta"}
ALL_METHODS = list(BASELINES) + sorted(PROPOSED)


def _norm(c: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample min-max normalisation of (B,H,W)."""
    b = c.shape[0]
    flat = c.reshape(b, -1)
    lo, hi = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
    return ((flat - lo) / (hi - lo + eps)).reshape_as(c)


class MSFGradCAM:
    def __init__(self, model, fusion_layers, deep_layer=None, weights=None,
                 tta: bool = True, fuse: bool = True):
        self.model = model
        self.deep = deep_layer if deep_layer is not None else fusion_layers[-1]
        self.layers = list(fusion_layers) if fuse else [self.deep]
        if self.deep not in self.layers:
            self.layers.append(self.deep)
        self.weights = weights or [1.0] * len(self.layers)
        self.tta = tta
        self._acts, self._grads, self._handles = {}, {}, []
        for i, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, i):
        def hook(_m, _inp, out):
            self._acts[i] = out.detach()
            if out.requires_grad:
                out.register_hook(lambda g: self._grads.__setitem__(i, g.detach()))
        return hook

    def release(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()

    def _single(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self._acts.clear(); self._grads.clear()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            logits.gather(1, targets.view(-1, 1)).sum().backward()
        size = x.shape[-2:]
        maps = []
        deep_idx = self.layers.index(self.deep)
        for i in range(len(self.layers)):
            a, g = self._acts[i], self._grads[i]
            if i == deep_idx:
                c = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1))
            else:
                c = F.relu((F.relu(g) * a).sum(1))
            c = F.interpolate(c.unsqueeze(1), size=size, mode="bilinear", align_corners=False)[:, 0]
            maps.append(_norm(c))
        w = torch.tensor(self.weights, device=x.device, dtype=maps[0].dtype)
        agg = sum(wi * m for wi, m in zip(w, maps)) / w.sum()
        return _norm(maps[deep_idx] * agg) if len(maps) > 1 else maps[0]

    def __call__(self, x: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        e = self._single(x, targets)
        if self.tta:
            e_flip = self._single(torch.flip(x, dims=[-1]), targets).flip(-1)
            e = _norm((e + e_flip) / 2)
        return e.cpu().numpy()


def compute_cams(method: str, model, layers: dict, x: torch.Tensor, targets: torch.Tensor,
                 scorecam_batch: int = 64) -> np.ndarray:
    """Return (B,H,W) CAMs in [0,1] for the given target class indices."""
    model.eval()
    if method in PROPOSED:
        kw = {"msf": dict(tta=True, fuse=True),
              "msf_notta": dict(tta=False, fuse=True),
              "gradcam_tta": dict(tta=True, fuse=False)}[method]
        with MSFGradCAM(model, layers["fusion"], layers["deep"], **kw) as cam:
            return cam(x, targets)
    if method not in BASELINES:
        raise ValueError(f"Unknown CAM method '{method}'. Choose from {ALL_METHODS}.")
    with BASELINES[method](model=model, target_layers=[layers["deep"]]) as cam:
        if method == "scorecam":
            cam.batch_size = scorecam_batch
        out = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(t)) for t in targets])
    out = np.nan_to_num(out.astype(np.float32))
    return _norm(torch.from_numpy(out)).numpy()
