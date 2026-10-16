"""Anime super-resolution with Real-ESRGAN's realesr-animevideov3 model
(BSD-3-Clause, https://github.com/xinntao/Real-ESRGAN), a small
SRVGGNetCompact network that is fast enough to run per frame on CPU."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

WEIGHTS = Path(__file__).resolve().parent.parent / "vendor" / "models" / "realesr-animevideov3.pth"
WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"


def _net():
    import torch
    from torch import nn
    from torch.nn import functional as F

    class SRVGGNetCompact(nn.Module):
        def __init__(self, num_feat=64, num_conv=16, upscale=4):
            super().__init__()
            self.upscale = upscale
            body = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
            for _ in range(num_conv):
                body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
            body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
            self.body = nn.ModuleList(body)
            self.upsampler = nn.PixelShuffle(upscale)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")

    net = SRVGGNetCompact()
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    net.load_state_dict(state.get("params", state))
    return net.eval()


class AnimeUpscaler:
    def __init__(self, device=None):
        import torch

        if not WEIGHTS.exists():
            WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request

            urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _net().to(self.device)

    def __call__(self, rgba: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Upscales RGBA to `size` (w, h): 4x network pass, then area-resize."""
        rgb = rgba[..., :3].astype(np.float32)
        a = rgba[..., 3:4].astype(np.float32) / 255
        # Bleed colours into transparent pixels so edges don't pick up a halo.
        pm = cv2.GaussianBlur(rgb * a, (0, 0), 3)
        wa = cv2.GaussianBlur(a[..., 0], (0, 0), 3)[..., None]
        fill = np.where(a > 0.02, rgb, pm / np.maximum(wa, 1e-3))
        x = self.torch.from_numpy(np.ascontiguousarray(fill.transpose(2, 0, 1) / 255))[None].float().to(self.device)
        with self.torch.no_grad():
            y = self.net(x)[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255
        y = cv2.resize(y, size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(rgba[..., 3], size, interpolation=cv2.INTER_CUBIC)
        return np.dstack([np.clip(y, 0, 255), np.clip(alpha, 0, 255)]).round().astype(np.uint8)
