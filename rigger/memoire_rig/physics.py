"""Secondary-motion physics for accessories (hair, ornaments, ribbons).

Each accessory is a region of the character with an anchor (where it attaches)
and a tip. It is simulated as a damped pendulum chain of one or two segments,
in the head's frame of reference:

    theta1'' = -k1*theta1 - c1*theta1'           spring back to the drawn pose
               - (a_anchor . n) / L * inertia      pivot acceleration (lag)
               - gravity * g/L * sin(theta1 + roll)  hang toward world-down
               - roll''                            head rotation drags it
               + wind(t)
    theta2'' = -k2*theta2 - c2*theta2' - follow * theta1''   (whip / S-curve)

Each accessory is cut out into its own layer (the base is inpainted behind
it), like a Live2D part. A point at rest position p, a fraction s of the way
from anchor to tip, turns by theta1*b(s) about the anchor and then by
theta2*c(s) about the joint, so the bend is zero at the attachment and grows
toward the tip. The pose engine (THA3) animates the base; each layer follows
the head via a similarity transform fitted to optical flow around its anchor.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

GRAVITY = 3000.0  # px/s^2 at the 512px THA3 canvas scale (cartoon-heavy)


@dataclass
class Accessory:
    name: str
    polygon: list[list[float]]
    anchor: list[float]
    tip: list[float]
    segments: int = 1
    stiffness: float = 40.0
    damping: float = 4.0
    gravity: float = 0.0
    inertia: float = 1.0
    wind: float = 0.0
    stiffness2: float = 20.0
    damping2: float = 4.0
    follow: float = 0.6
    max_angle: float = 0.7
    feather: float = 6.0
    # axial spring: lets hanging parts stretch / compress on bounces (0 = off)
    stretch_stiffness: float = 0.0
    stretch_damping: float = 6.0
    max_stretch: float = 0.08
    ext: float = 0.0
    ext_v: float = 0.0
    # state
    th: list[float] = field(default_factory=lambda: [0.0, 0.0])
    om: list[float] = field(default_factory=lambda: [0.0, 0.0])

    @property
    def length(self) -> float:
        return float(np.hypot(self.tip[0] - self.anchor[0], self.tip[1] - self.anchor[1]))

    def step(self, dt: float, acc: np.ndarray, roll: float, roll_acc: float, t: float):
        """Advances the chain by dt with semi-implicit Euler."""
        L = max(self.length, 1.0)
        d = np.array(self.tip, float) - np.array(self.anchor, float)
        d /= np.linalg.norm(d) + 1e-9
        c, s = math.cos(self.th[0] + roll), math.sin(self.th[0] + roll)
        d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        n = np.array([-d[1], d[0]])
        wind = self.wind * (math.sin(t * 2.3) * 0.6 + math.sin(t * 5.1 + 1.3) * 0.3 + math.sin(t * 9.7) * 0.15)
        g = self.gravity * GRAVITY / L if self._hangs() else 0.0
        # wind is expressed as a deflection (radians) against the effective spring
        a1 = (-self.stiffness * self.th[0] - self.damping * self.om[0]
              - float(acc @ n) / L * self.inertia
              - g * math.sin(self.th[0] + roll)
              - roll_acc + wind * 0.12 * (self.stiffness + g))
        self.om[0] += a1 * dt
        self.th[0] += self.om[0] * dt
        if self.segments > 1:
            a2 = (-self.stiffness2 * self.th[1] - self.damping2 * self.om[1] - self.follow * a1
                  + wind * 0.08 * self.stiffness2)
            self.om[1] += a2 * dt
            self.th[1] += self.om[1] * dt
        if self.stretch_stiffness > 0:
            # acceleration along the chain axis (pivot dropping -> part stretches)
            dir_ = np.array([-n[1], n[0]])  # == d
            ae = (-self.stretch_stiffness * self.ext - self.stretch_damping * self.ext_v
                  - float(acc @ dir_) / L * self.inertia)
            self.ext_v += ae * dt
            self.ext += self.ext_v * dt
            if abs(self.ext) > self.max_stretch:
                self.ext = math.copysign(self.max_stretch, self.ext)
                self.ext_v *= -0.3
        for i in range(2):
            if abs(self.th[i]) > self.max_angle:
                self.th[i] = math.copysign(self.max_angle, self.th[i])
                self.om[i] *= -0.3

    def _hangs(self) -> bool:
        # Gravity only makes sense for parts that hang below their anchor.
        return self.tip[1] > self.anchor[1]


def load_accessories(path: str | Path) -> list[Accessory]:
    data = json.loads(Path(path).read_text())
    return [Accessory(**a) for a in data["accessories"]]


def _smooth(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def bend_field(a: Accessory, P: np.ndarray) -> np.ndarray:
    """Forward displacement of the bent chain, defined over the whole plane
    (smooth everywhere, so it can be inverted by fixed-point iteration)."""
    A = np.array(a.anchor, np.float32)
    T = np.array(a.tip, np.float32)
    L = float(np.linalg.norm(T - A))
    u = (T - A) / L
    v = P - A
    s = (v @ u) / L
    if a.segments > 1:
        b, c = _smooth(0.0, 0.5, s), _smooth(0.5, 1.0, s)
    else:
        b, c = _smooth(0.0, 1.0, s), np.zeros_like(s)
    if a.ext:
        # stretch along the axis, growing from the anchor toward the tip
        v = v + u * (v @ u)[..., None] * (a.ext * np.clip(s, 0, 1.2))[..., None]
    ang = a.th[0] * b
    cs, sn = np.cos(ang), np.sin(ang)
    p1 = np.stack([A[0] + cs * v[..., 0] - sn * v[..., 1], A[1] + sn * v[..., 0] + cs * v[..., 1]], -1)
    if a.segments > 1:
        J = A + u * L * 0.5
        cj, sj = math.cos(a.th[0]), math.sin(a.th[0])
        Jr = A + np.array([cj * (J - A)[0] - sj * (J - A)[1], sj * (J - A)[0] + cj * (J - A)[1]], np.float32)
        ang2 = a.th[1] * c
        c2, s2 = np.cos(ang2), np.sin(ang2)
        w = p1 - Jr
        p1 = np.stack([Jr[0] + c2 * w[..., 0] - s2 * w[..., 1], Jr[1] + s2 * w[..., 0] + c2 * w[..., 1]], -1)
    return (p1 - P).astype(np.float32)


class LayeredRig:
    """Live2D-style layering: every accessory is cut into its own RGBA layer,
    the base is inpainted underneath, and layers are re-composited each frame
    after following the head and bending with their physics."""

    def __init__(self, canvas_rgba: np.ndarray, accessories: list[Accessory]):
        self.acc = accessories
        size = canvas_rgba.shape[0]
        self.size = size
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
        self.grid = np.stack([xs, ys], -1)
        alpha = canvas_rgba[..., 3].astype(np.float32) / 255
        hole = np.zeros((size, size), np.uint8)
        self.layers, self.masks = [], []
        for a in accessories:
            m = np.zeros((size, size), np.uint8)
            cv2.fillPoly(m, [np.int32(np.round(a.polygon))], 255)
            soft = cv2.GaussianBlur(m.astype(np.float32) / 255, (3, 3), 0.8)
            layer = canvas_rgba.copy()
            layer[..., 3] = (alpha * soft * 255).round().astype(np.uint8)
            self.layers.append(layer)
            self.masks.append(m)
            hole |= m
        # Base: remove the accessories and inpaint what was behind them.
        hole = cv2.dilate(hole, np.ones((3, 3), np.uint8))
        rgb = np.ascontiguousarray(canvas_rgba[..., :3])
        a8 = np.ascontiguousarray(canvas_rgba[..., 3])
        base_rgb = cv2.inpaint(rgb, hole, 5, cv2.INPAINT_TELEA)
        base_a = cv2.inpaint(a8, hole, 5, cv2.INPAINT_TELEA)
        # Only keep inpainted coverage that is surrounded by the body; open
        # background stays transparent.
        base_a = np.where(hole > 0, np.where(base_a > 110, 255, 0), a8).astype(np.uint8)
        self.base = np.dstack([base_rgb, base_a]).astype(np.uint8)
        self.reference = None

    def set_reference(self, neutral_base_render: np.ndarray):
        self.reference = neutral_base_render

    def head_transform(self, a: Accessory, flow: np.ndarray | None) -> np.ndarray:
        """2x3 similarity that carries the accessory's attachment area from the
        neutral render to the current one, fitted to optical flow around the
        anchor."""
        if flow is None:
            return np.float32([[1, 0, 0], [0, 1, 0]])
        r = 28
        ax, ay = map(int, a.anchor)
        y0, y1 = max(0, ay - r), min(self.size, ay + r)
        x0, x1 = max(0, ax - r), min(self.size, ax + r)
        cur = self.grid[y0:y1:3, x0:x1:3].reshape(-1, 2)
        rest = cur + flow[y0:y1:3, x0:x1:3].reshape(-1, 2)
        M, _ = cv2.estimateAffinePartial2D(rest, cur, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        return M if M is not None else np.float32([[1, 0, 0], [0, 1, 0]])

    def compose(self, base_frame: np.ndarray, flow: np.ndarray | None, root=(0.0, 0.0)) -> np.ndarray:
        out = premultiply(base_frame)
        q = self.grid
        for a, layer in zip(self.acc, self.layers):
            M = self.head_transform(a, flow)
            Minv = cv2.invertAffineTransform(M)
            r = (q @ Minv[:, :2].T + Minv[:, 2]).astype(np.float32)  # current -> rest
            p = r.copy()
            for _ in range(6):  # invert the bend: p + F(p) = r
                p = r - bend_field(a, p)
            lp = cv2.remap(premultiply(layer), p[..., 0], p[..., 1], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            out = lp + out * (1 - lp[..., 3:4] / 255.0)
        res = unpremultiply(out)
        if root != (0.0, 0.0):
            T = np.float32([[1, 0, root[0]], [0, 1, root[1]]])
            res = unpremultiply(cv2.warpAffine(premultiply(res), T, (self.size, self.size),
                                               flags=cv2.INTER_LINEAR, borderValue=0))
        return res

    def overlay(self, canvas_rgba: np.ndarray) -> np.ndarray:
        """Debug view: tinted layers with anchor -> tip axes."""
        base = flatten(canvas_rgba, (70, 70, 70)).astype(np.float32)
        colors = [(255, 80, 80), (80, 200, 255), (255, 210, 60), (160, 255, 120), (230, 120, 255), (255, 160, 200)]
        for i, m in enumerate(self.masks):
            w = (m.astype(np.float32) / 255)[..., None] * 0.45
            base = base * (1 - w) + np.array(colors[i % len(colors)], np.float32) * w
        out = base.astype(np.uint8)
        for i, a in enumerate(self.acc):
            col = colors[i % len(colors)]
            A, T = tuple(map(int, a.anchor)), tuple(map(int, a.tip))
            cv2.line(out, A, T, col, 1, cv2.LINE_AA)
            cv2.circle(out, A, 3, (255, 255, 255), -1)
            cv2.putText(out, a.name, (A[0] + 4, A[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
        return out


def premultiply(rgba: np.ndarray) -> np.ndarray:
    f = rgba.astype(np.float32)
    f[..., :3] *= f[..., 3:4] / 255.0
    return f


def unpremultiply(f: np.ndarray) -> np.ndarray:
    a = f[..., 3:4]
    rgb = np.where(a > 1e-3, f[..., :3] * 255.0 / np.maximum(a, 1e-3), 0)
    return np.clip(np.concatenate([rgb, a], -1), 0, 255).round().astype(np.uint8)


def flatten(rgba: np.ndarray, bg=(255, 255, 255)) -> np.ndarray:
    a = rgba[..., 3:4].astype(np.float32) / 255
    return (rgba[..., :3] * a + np.array(bg, np.float32) * (1 - a)).astype(np.uint8)


def back_flow(current_rgba: np.ndarray, rest_rgba: np.ndarray) -> np.ndarray:
    """Dense flow current -> rest on grey-composited luminance."""
    g0 = cv2.cvtColor(flatten(current_rgba, (128, 128, 128)), cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(flatten(rest_rgba, (128, 128, 128)), cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 4, 21, 4, 5, 1.1, 0)
