"""Crisp eye closing, drawn at output resolution.

THA3 can close the eyes itself, but at its ~128px face scale the closed eye
comes out as a soft smudge. Instead the pose engine keeps the eyes open and
this pass closes them on the final, upscaled frame:

* the eyelid slides down by squashing the (crisp) eye pixels toward the lid
  line, as a real lid would;
* near full closure the eye is covered with skin and a tapered lash stroke is
  drawn: a soft "︶" curve for blinks and sleep, a "^" arch for happy eyes;
* if the layered art supplies closed-eye sprites, those are used instead of
  the drawn stroke.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Eye:
    cx: float
    cy: float
    rx: float
    ry: float
    outer: int  # -1: outer corner on the image left, +1: on the image right

    def transformed(self, M: np.ndarray) -> "Eye":
        """Applies a 2x3 similarity transform."""
        x, y = M[:, :2] @ np.array([self.cx, self.cy]) + M[:, 2]
        s = math.hypot(M[0, 0], M[1, 0])
        return Eye(float(x), float(y), self.rx * s, self.ry * s, self.outer)


def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def lid_warp(img: np.ndarray, e: Eye, k: float, happy: bool) -> np.ndarray:
    """Slides the lid: the upper lid down for blinks, the lower lid up for
    happy squints. Works on a local patch with cv2.remap."""
    if k <= 1e-3:
        return img
    H, W = img.shape[:2]
    x0, x1 = int(max(0, e.cx - e.rx * 1.4)), int(min(W, e.cx + e.rx * 1.4))
    y0, y1 = int(max(0, e.cy - e.ry * 3.0)), int(min(H, e.cy + e.ry * 3.0))
    if x1 <= x0 or y1 <= y0:
        return img
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    u = (xs - e.cx) / (e.rx * 1.2)
    hx = np.sqrt(np.clip(1 - u * u, 0, 1))
    top = e.cy - e.ry * 1.05 * hx
    bot = e.cy + e.ry * 0.9 * hx
    src = ys.copy()
    if not happy:
        skin = top - e.ry * 1.6 * hx
        lid = top + (bot - (bot - top) * 0.12 - top) * k
        a = (ys >= skin) & (ys < lid) & (hx > 0)
        src[a] = (skin + (ys - skin) * (top - skin) / np.maximum(lid - skin, 1e-3))[a]
        b = (ys >= lid) & (ys <= bot) & (hx > 0)
        src[b] = (top + (ys - lid) * (bot - top) / np.maximum(bot - lid, 1e-3))[b]
    else:
        cheek = bot + e.ry * 1.6 * hx
        lid = bot - (bot - (top + (bot - top) * 0.12)) * k
        a = (ys <= cheek) & (ys > lid) & (hx > 0)
        src[a] = (cheek - (cheek - ys) * (cheek - bot) / np.maximum(cheek - lid, 1e-3))[a]
        b = (ys <= lid) & (ys >= top) & (hx > 0)
        src[b] = (bot - (lid - ys) * (bot - top) / np.maximum(lid - top, 1e-3))[b]
    patch = cv2.remap(img, xs, src, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    out = img.copy()
    out[y0:y1, x0:x1] = patch
    return out


def _stroke(points: np.ndarray, widths: np.ndarray, shape, ss: int = 4) -> np.ndarray:
    """Anti-aliased tapered stroke as an alpha mask (supersampled)."""
    H, W = shape
    canvas = np.zeros((H * ss, W * ss), np.uint8)
    for (p, q), (w0, w1) in zip(zip(points[:-1], points[1:]), zip(widths[:-1], widths[1:])):
        d = q - p
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        poly = np.array([p + n * w0 / 2, q + n * w1 / 2, q - n * w1 / 2, p - n * w0 / 2]) * ss
        cv2.fillConvexPoly(canvas, np.int32(np.round(poly)), 255, cv2.LINE_AA)
        cv2.circle(canvas, tuple(np.int32(np.round(q * ss))), max(1, int(w1 * ss / 2)), 255, -1, cv2.LINE_AA)
    return cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def closed_eye(img: np.ndarray, e: Eye, happy: bool, lash_rgb, amount: float, sprite: np.ndarray | None = None):
    """Covers the eye with skin and draws the closed-eye lash (or composites an
    artist's sprite). `amount` fades it in."""
    if amount <= 1e-3:
        return img
    H, W = img.shape[:2]
    out = img.astype(np.float32)
    if sprite is not None:
        a = sprite[..., 3:4].astype(np.float32) / 255 * amount
        out[..., :3] = out[..., :3] * (1 - a) + sprite[..., :3] * a
        return out.astype(np.uint8)
    # 1. skin over the eye opening, in the cheek's own tone (sampled per
    #    frame just below the eye, so shading and lighting follow along)
    y0, y1 = int(e.cy + e.ry * 1.25), int(e.cy + e.ry * 1.9)
    x0, x1 = int(e.cx - e.rx * 0.5), int(e.cx + e.rx * 0.5)
    cheek = img[max(0, y0):min(H, y1), max(0, x0):min(W, x1), :3].reshape(-1, 3).astype(np.float32)
    skin = np.median(cheek, 0) if len(cheek) else np.array([250, 235, 225], np.float32)
    mask = np.zeros((H, W), np.uint8)
    if happy:
        cv2.ellipse(mask, (int(e.cx), int(e.cy + e.ry * 0.05)), (int(e.rx * 1.12), int(e.ry * 1.12)), 0, 0, 360, 255, -1)
    else:
        cv2.ellipse(mask, (int(e.cx), int(e.cy)), (int(e.rx * 1.1), int(e.ry * 1.05)), 0, 0, 360, 255, -1)
    m = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), max(1.0, e.ry * 0.15))
    # keep fringe hair that falls over the eye: neutral, bright pixels above it
    rgb = img[..., :3].astype(np.int16)
    chroma = rgb.max(-1) - rgb.min(-1)
    hairish = ((chroma < 22) & (rgb.mean(-1) > 165)).astype(np.float32)
    rows = np.arange(H, dtype=np.float32)[:, None]
    hairish *= (rows < e.cy - e.ry * 0.2)
    hairish = cv2.GaussianBlur(hairish, (0, 0), 1.0)
    m = (m * (1 - hairish))[..., None] * amount
    out[..., :3] = out[..., :3] * (1 - m) + skin * m
    # 2. the lash line
    t = np.linspace(0, 1, 24)
    xs = e.cx + (t - 0.5) * 2 * e.rx * 0.92 * e.outer  # inner -> outer corner
    if happy:
        ys = e.cy + e.ry * 0.25 - e.ry * 0.55 * np.sin(np.pi * t)
    else:
        ys = e.cy + e.ry * 0.45 + e.ry * 0.28 * np.sin(np.pi * t)
    pts = np.stack([xs, ys], 1)
    w = e.ry * 0.17 * (0.3 + 0.7 * np.sin(np.pi * np.clip(t * 0.9 + 0.05, 0, 1)) ** 0.6) * (0.75 + 0.5 * t)
    # outer-corner lash flick
    end, dirv = pts[-1], pts[-1] - pts[-3]
    dirv = dirv / (np.linalg.norm(dirv) + 1e-9)
    up = np.array([0.0, -1.0])
    flick = np.array([end + (dirv * 0.9 + up * (0.2 if happy else 0.5)) * e.rx * f for f in (0.0, 0.18, 0.32)])
    pts = np.vstack([pts, flick[1:]])
    w = np.concatenate([w, [w[-1] * 0.6, 0.6]])
    alpha = _stroke(pts, w, (H, W))[..., None] * amount
    out[..., :3] = out[..., :3] * (1 - alpha) + np.array(lash_rgb, np.float32) * alpha
    return out.astype(np.uint8)


def lash_color(canvas_rgba: np.ndarray, eyes: list[Eye]):
    """Darkest tones along the upper lash line of the drawn eyes."""
    vals = []
    for e in eyes:
        y0, y1 = int(e.cy - e.ry), int(e.cy - e.ry * 0.3)
        x0, x1 = int(e.cx - e.rx), int(e.cx + e.rx)
        patch = canvas_rgba[y0:y1, x0:x1, :3].reshape(-1, 3).astype(np.float32)
        lum = patch.mean(1)
        vals.append(patch[lum <= np.percentile(lum, 12)])
    return np.concatenate(vals).mean(0) if vals else np.array([50, 30, 30], np.float32)


def extract_lash(rest_img: np.ndarray, e: Eye) -> np.ndarray | None:
    """Lifts the drawn upper eyelash out of the open eye: the widest very dark
    connected shape in the upper eye (the lash spans the eye; hair line-art is
    thinner and lighter, the pupil sits below), clipped per column to lash
    thickness so the pupil is left behind. Returns an RGBA sprite of the
    region x: cx +- 1.3 rx, y: cy - 1.4 ry .. cy + 0.2 ry."""
    H, W = rest_img.shape[:2]
    x0, x1 = int(e.cx - e.rx * 1.3), int(e.cx + e.rx * 1.3)
    y0, y1 = int(e.cy - e.ry * 1.4), int(e.cy + e.ry * 0.2)
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None
    reg = rest_img[y0:y1, x0:x1].astype(np.float32)
    lum = reg[..., :3].mean(-1)
    core = ((lum < 60) & (reg[..., 3] > 128)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    if n <= 1:
        return None
    best = max(range(1, n), key=lambda k: stats[k, cv2.CC_STAT_WIDTH])
    comp = lab == best
    thick = max(2, int(e.ry * 0.42))
    keep = np.zeros_like(core, np.float32)
    for c in np.nonzero(comp.any(0))[0]:
        top = np.argmax(comp[:, c])
        keep[top:top + thick, c] = comp[top:top + thick, c]
    # grow slightly and take soft alpha from darkness for clean edges
    grown = cv2.dilate(keep, np.ones((3, 3), np.uint8)) > 0
    alpha = np.clip((115 - lum) / 55, 0, 1) * grown
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 0.5)
    return np.dstack([reg[..., :3], alpha * 255]).astype(np.uint8)


def _place(img: np.ndarray, sprite: np.ndarray, x0: float, y0: float, amount: float, sy: float = 1.0) -> np.ndarray:
    """Composites a sprite with its top-left at (x0, y0), optionally scaled
    vertically about its centre."""
    h, w = sprite.shape[:2]
    M = np.float32([[1, 0, x0], [0, sy, y0 + h * (1 - sy) / 2]])
    H, W = img.shape[:2]
    warped = cv2.warpAffine(sprite, M, (W, H), flags=cv2.INTER_CUBIC, borderValue=0)
    a = warped[..., 3:4].astype(np.float32) / 255 * amount
    out = img.astype(np.float32)
    out[..., :3] = out[..., :3] * (1 - a) + warped[..., :3] * a
    return out.astype(np.uint8)


def closed_eye_from_art(img: np.ndarray, e: Eye, rest_e: Eye, lash: np.ndarray, happy: bool,
                        amount: float, cover_mask_fn) -> np.ndarray:
    """Closed eye built from the character's own lash: the eye opening is
    covered with cheek-toned skin, then the real lash is drawn as "︶"
    (flipped, lowered) for blinks or kept as an arch "^" for happy eyes."""
    if amount <= 1e-3:
        return img
    H, W = img.shape[:2]
    y0, y1 = int(e.cy + e.ry * 1.25), int(e.cy + e.ry * 1.9)
    x0, x1 = int(e.cx - e.rx * 0.5), int(e.cx + e.rx * 0.5)
    cheek = img[max(0, y0):min(H, y1), max(0, x0):min(W, x1), :3].reshape(-1, 3).astype(np.float32)
    skin = np.median(cheek, 0) if len(cheek) else np.array([250, 235, 225], np.float32)
    m = cover_mask_fn(img, e)[..., None] * amount
    out = img.astype(np.float32)
    out[..., :3] = out[..., :3] * (1 - m) + skin * m
    out = out.astype(np.uint8)
    # lash sprite is in rest-eye coordinates; follow the eye's current offset
    sx, sy0 = e.cx - rest_e.cx, e.cy - rest_e.cy
    lx = rest_e.cx - rest_e.rx * 1.3 + sx
    ly_top = rest_e.cy - rest_e.ry * 1.4 + sy0
    rows = np.nonzero(lash[..., 3].max(1) > 40)[0]
    if not len(rows):
        return out
    band_c = (rows[0] + rows[-1]) / 2
    if happy:
        target = e.cy - e.ry * 0.05  # arch sits mid-eye: "^"
        return _place(out, lash, lx, target - band_c, amount, sy=0.9)
    flipped = lash[::-1].copy()
    band_c = lash.shape[0] - 1 - band_c
    target = e.cy + e.ry * 0.35  # lowered lid line: "︶"
    return _place(out, flipped, lx, target - band_c, amount, sy=0.8)


def eye_states(p: dict[str, float]) -> dict[str, tuple[float, bool]]:
    """Per image side: (closure 0..1, happy?). THA3's "right" eye is the
    character's right, i.e. the eye on the image left."""
    out = {}
    for img_side, tha_side in (("left", "right"), ("right", "left")):
        blink = p.get(f"eye_wink_{tha_side}", 0.0)
        happy = p.get(f"eye_happy_wink_{tha_side}", 0.0)
        relaxed = p.get(f"eye_relaxed_{tha_side}", 0.0)
        k = max(blink, happy, 0.4 * relaxed)
        out[img_side] = (min(1.0, k), happy > max(blink, 0.4 * relaxed))
    return out


EYE_CLOSING_PARAMS = ("eye_wink", "eye_happy_wink", "eye_relaxed")


def strip_eye_closing(p: dict[str, float]) -> dict[str, float]:
    """Pose params with THA3's own eye closing removed (this pass does it)."""
    return {k: v for k, v in p.items() if not k.startswith(EYE_CLOSING_PARAMS)}


def cover_mask(img: np.ndarray, e: Eye) -> np.ndarray:
    """The eye opening. (White hair and eye whites can't be told apart by
    colour, so the whole opening is covered; layered art avoids this.)"""
    H, W = img.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (int(e.cx), int(e.cy + e.ry * 0.08)), (int(e.rx * 1.08), int(e.ry * 1.05)), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), max(1.0, e.ry * 0.1))


def apply(img: np.ndarray, eyes: dict[str, Eye], states, lash_rgb, sprites=None,
          lashes: dict | None = None, rest_eyes: dict | None = None) -> np.ndarray:
    for side, e in eyes.items():
        k, happy = states[side]
        # the real lid slides down over the crisp eye for partial closure...
        # (capped: past this the lash sprite takes over, and a deeper slide
        # would stretch the hair above the eye)
        img = lid_warp(img, e, min(k / 0.9, 0.62), False)
        amount = float(_smooth(0.6, 0.95, k))
        sp = (sprites or {}).get(f"eye_{'happy' if happy else 'closed'}_{side}")
        if sp is not None:
            img = closed_eye(img, e, happy, lash_rgb, amount, sp)
        elif lashes and lashes.get(side) is not None:
            # ...and the full closure is drawn with the character's own lash
            img = closed_eye_from_art(img, e, rest_eyes[side], lashes[side], happy, amount, cover_mask)
        else:
            img = closed_eye(img, e, happy, lash_rgb, amount)
    return img
