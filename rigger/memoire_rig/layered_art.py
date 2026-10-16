"""Layered-art import: a PSD, or a folder of same-size PNGs, with named layers.

With proper layers nothing has to be guessed: parts are drawn in full
(e.g. the whole ponytail, including what the head hides), the hair behind
ornaments is painted, and closed eyes are drawn by the artist.

Recognised layer names (case-insensitive, spaces/dashes -> underscores):

    base                     the character without any separately rigged part
    <accessory name>         full art for an accessory in the rig JSON,
                             e.g. ponytail, hair_clip, leaf_ornament
    eye_closed_left/right    closed-eye sprites (image left / image right)
    eye_happy_left/right     "^" happy-eye sprites

All layers share the source picture's pixel grid; PSD layer offsets are
handled. Unknown layers are ignored; layers inside groups are flattened by
name. Missing layers fall back to the automatic cut-out.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def _norm(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def load(path: str | Path) -> dict[str, np.ndarray]:
    """Returns {layer name: RGBA uint8 array} at the source picture's size."""
    path = Path(path)
    if path.is_dir():
        from PIL import Image

        layers = {}
        for f in sorted(path.glob("*.png")):
            layers[_norm(f.stem)] = np.asarray(Image.open(f).convert("RGBA")).copy()
        sizes = {v.shape for v in layers.values()}
        if len(sizes) > 1:
            raise ValueError(f"All layer PNGs must be the same size, got {sizes}")
        return layers
    if path.suffix.lower() == ".psd":
        from psd_tools import PSDImage

        psd = PSDImage.open(path)
        W, H = psd.width, psd.height
        layers = {}
        for layer in psd.descendants():
            if layer.is_group() or not layer.is_visible():
                continue
            img = layer.composite()
            if img is None:
                continue
            canvas = np.zeros((H, W, 4), np.uint8)
            arr = np.asarray(img.convert("RGBA"))
            x0, y0 = layer.left, layer.top
            xs, ys = max(0, x0), max(0, y0)
            xe, ye = min(W, x0 + arr.shape[1]), min(H, y0 + arr.shape[0])
            if xe > xs and ye > ys:
                canvas[ys:ye, xs:xe] = arr[ys - y0:ye - y0, xs - x0:xe - x0]
            name = _norm(layer.name)
            if name in layers:  # same name twice: stack them
                a = canvas[..., 3:4] / 255.0
                layers[name] = (canvas * a + layers[name] * (1 - a)).astype(np.uint8)
            else:
                layers[name] = canvas
        return layers
    raise ValueError(f"Unsupported layered art: {path} (use a .psd or a folder of PNGs)")


def composite(layers: dict[str, np.ndarray]) -> np.ndarray:
    """Flattened picture (base under everything), used for head detection."""
    order = ["base"] + [k for k in layers if k != "base" and not k.startswith("eye_")]
    out = None
    for k in order:
        if k not in layers:
            continue
        L = layers[k].astype(np.float32)
        if out is None:
            out = L
            continue
        a = L[..., 3:4] / 255
        out = np.concatenate([L[..., :3] * a + out[..., :3] * (1 - a),
                              np.maximum(L[..., 3:4], out[..., 3:4])], -1)
    return out.astype(np.uint8)


def laid_out(layers: dict[str, np.ndarray], layout_fn) -> dict[str, np.ndarray]:
    """Applies the THA3 canvas layout to every layer."""
    return {k: layout_fn(v) for k, v in layers.items()}
