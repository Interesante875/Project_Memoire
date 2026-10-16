"""Expression presets and "acting" for the anime rig.

An expression is a set of THA3 face parameters plus a short body-language
motion (head tilts, nods, hops, shakes). The motion is what makes the
accessory physics come alive: hops swing the ponytail, a head snap flicks the
ornaments, and so on. A timeline blends from one expression to the next.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def both(name: str, v: float) -> dict[str, float]:
    return {f"{name}_left": v, f"{name}_right": v}


FACES: dict[str, dict[str, float]] = {
    "neutral": {},
    "happy": {**both("eye_happy_wink", 1.0), **both("mouth_raised_corner", 0.9), **both("eyebrow_happy", 0.8)},
    "laugh": {**both("eye_happy_wink", 1.0), **both("eyebrow_happy", 1.0), "mouth_aaa": 0.85,
              **both("mouth_raised_corner", 0.6)},
    "surprised": {**both("eye_surprised", 1.0), **both("eyebrow_raised", 1.0), "mouth_ooo": 0.9,
                  **both("iris_small", 0.45)},
    "wink": {"eye_wink_right": 1.0, **both("mouth_raised_corner", 0.8), **both("eyebrow_happy", 0.5),
             "mouth_smirk": 0.3},
    "smug": {**both("eye_unimpressed", 0.6), "mouth_smirk": 1.0, **both("eyebrow_serious", 0.4),
             "iris_rotation_y": -0.45},
    "angry": {**both("eyebrow_angry", 1.0), **both("eye_unimpressed", 0.35), **both("mouth_lowered_corner", 0.7),
              "mouth_delta": 0.5},
    "sad": {**both("eyebrow_troubled", 1.0), **both("eye_relaxed", 0.35), **both("mouth_lowered_corner", 0.8),
            "iris_rotation_x": -0.35},
    "sleepy": {**both("eye_relaxed", 0.95), **both("eyebrow_lowered", 0.5), "mouth_uuu": 0.25},
}


def _pulse(x: float, at: float, width: float) -> float:
    """Smooth bump centred at `at`."""
    return math.exp(-((x - at) / width) ** 2)


# Acting: (t since the expression began) -> (pose params, root motion).
# Root motion: tx, ty in px of the 512 canvas, and roll in radians about the hips.
Acting = Callable[[float], tuple[dict[str, float], tuple[float, float]]]


def _glance(t: float) -> float:
    """Quick eye darts with holds: -1 / 0 / +1 steps, smoothed."""
    k = (t * 0.9) % 3.0
    target = (0.0, 0.8, -0.7)[int(k)]
    return target * min(1.0, (k % 1.0) / 0.12)


def act_neutral(t):
    return ({"head_y": 0.25 * math.sin(t * 1.3), "neck_z": 0.12 * math.sin(t * 0.9),
             "iris_rotation_y": _glance(t)}, (0.0, 0.0))


def act_happy(t):
    hop = math.sin(math.pi * 1.7 * t) ** 2  # smooth: no infinite jolt at landing
    return {"neck_z": 0.35 * math.sin(t * 3.4), "head_x": -0.1 * hop}, (0.0, -12.0 * hop)


def act_laugh(t):
    return ({"head_x": 0.18 * math.sin(t * 19), "neck_z": 0.25 * math.sin(t * 2.5) + 0.2,
             "mouth_aaa": 0.5 + 0.4 * abs(math.sin(t * 9.5))},
            (0.0, -5.0 * math.sin(t * 9.5) ** 2))


def act_surprised(t):
    jolt = _pulse(t, 0.35, 0.18)
    back = min(1.0, t / 0.35)
    return {"head_x": -0.35 * back, "body_y": 0.0}, (0.0, -13.0 * jolt)


def act_wink(t):
    snap = 1 - math.exp(-t * 9)  # quick tilt, then hold: flicks the ornaments
    return {"neck_z": -0.5 * snap, "head_y": 0.25 * snap, "body_z": -0.2 * snap}, (4.0 * snap, 0.0)


def act_smug(t):
    ease = 1 - math.exp(-t * 5)
    return {"head_y": -0.45 * ease, "neck_z": 0.25 * ease, "head_x": -0.15 * ease,
            "body_y": -0.2 * ease}, (0.0, 0.0)


def act_angry(t):
    shake = math.sin(t * 2 * math.pi * 3.2) * math.exp(-t * 0.9)
    return {"head_y": 0.55 * shake, "body_y": 0.2 * shake, "head_x": 0.12}, (6.0 * shake, 0.0)


def act_sad(t):
    ease = 1 - math.exp(-t * 2.2)
    return {"head_x": 0.35 * ease, "neck_z": 0.2 * ease + 0.06 * math.sin(t * 1.2),
            "body_z": 0.1 * ease}, (0.0, 3.0 * ease)


def act_sleepy(t):
    nod = 0.5 * (1 - math.cos(min(t, 2.2) / 2.2 * math.pi))  # drift off...
    jerk = _pulse(t, 2.45, 0.08)  # ...and snap awake
    awake = 1 - min(1.0, max(0.0, (t - 2.4) / 0.2))
    return {"head_x": 0.45 * nod * awake - 0.25 * jerk, "neck_z": 0.25 * nod * awake}, (0.0, -10.0 * jerk)


ACTING: dict[str, Acting] = {
    "neutral": act_neutral, "happy": act_happy, "laugh": act_laugh, "surprised": act_surprised,
    "wink": act_wink, "smug": act_smug, "angry": act_angry, "sad": act_sad, "sleepy": act_sleepy,
}

EYES_OPEN = {"neutral", "surprised", "smug", "angry", "sad"}


@dataclass
class Timeline:
    beats: list[tuple[str, float]] = field(default_factory=lambda: [
        ("neutral", 2.0), ("happy", 2.6), ("laugh", 2.2), ("surprised", 2.0), ("wink", 2.4),
        ("smug", 2.4), ("angry", 2.6), ("sad", 2.8), ("sleepy", 3.0), ("happy", 2.2), ("neutral", 1.6),
    ])
    blend: float = 0.35

    @property
    def duration(self) -> float:
        return sum(d for _, d in self.beats)

    def at(self, t: float) -> tuple[dict[str, float], tuple[float, float], str]:
        """Pose params, root offset and the current expression name at time t."""
        start = 0.0
        for i, (name, dur) in enumerate(self.beats):
            if t < start + dur or i == len(self.beats) - 1:
                local = t - start
                p, root = self._one(name, local, t)
                # cross-fade from the previous beat over `blend` seconds
                if i > 0 and local < self.blend:
                    prev, pdur = self.beats[i - 1]
                    pp, proot = self._one(prev, pdur + local, t)
                    k = local / self.blend
                    k = k * k * (3 - 2 * k)
                    p = {key: pp.get(key, 0.0) * (1 - k) + p.get(key, 0.0) * k for key in set(p) | set(pp)}
                    root = tuple(a * (1 - k) + b * k for a, b in zip(proot, root))
                return p, root, name
            start += dur
        raise AssertionError

    def _one(self, name: str, local: float, t: float):
        p = dict(FACES[name])
        motion, root = ACTING[name](local)
        for k, v in motion.items():
            p[k] = p.get(k, 0.0) + v
        p["breathing"] = 0.5 + 0.5 * math.sin(t * 2 * math.pi / 3.6)
        if name in EYES_OPEN:
            blink = max(_pulse(local, at, 0.07) for at in (0.9, 2.3))
            for side in ("left", "right"):
                key = f"eye_wink_{side}"
                p[key] = max(p.get(key, 0.0), blink)
        return p, root


# Head kinematics proxy for the physics: how a point on the head moves in
# canvas pixels for given THA3 pose params. Smooth and analytic, so its second
# derivative (acceleration) is clean, unlike optical flow.
NECK_PIVOT = np.array([256.0, 205.0])
ROLL_PER_UNIT = math.radians(15)


def head_point(p: dict[str, float], root: tuple[float, float], pt: np.ndarray) -> tuple[np.ndarray, float]:
    roll = ROLL_PER_UNIT * (p.get("neck_z", 0.0) + 0.6 * p.get("body_z", 0.0))
    c, s = math.cos(roll), math.sin(roll)
    v = pt - NECK_PIVOT
    w = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]]) + NECK_PIVOT
    w[0] += 18.0 * p.get("head_y", 0.0) + 10.0 * p.get("body_y", 0.0) + root[0]
    w[1] += 10.0 * p.get("head_x", 0.0) + root[1]
    return w, roll
