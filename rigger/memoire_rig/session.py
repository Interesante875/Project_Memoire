"""A live puppeteering session: performer frames + controls + idle life -> frames."""

from __future__ import annotations

import math
import random
import time
from dataclasses import replace

import numpy as np

from .engine import Controls, FaceTracker, RigEngine


class Autopilot:
    """Procedural life so the avatar never freezes: blinks, breathing, small
    head sway and glances. Applied only to channels the performer isn't
    driving."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.next_blink = 1.5
        self.blink_start = -1.0
        self.next_glance = 2.0
        self.gaze = [0.0, 0.0]
        self.gaze_target = [0.0, 0.0]

    def apply(self, c: Controls, performer: bool) -> Controls:
        t = time.monotonic() - self.t0
        if t > self.next_blink and self.blink_start < 0:
            self.blink_start = t
        blink = 0.0
        if self.blink_start >= 0:
            k = (t - self.blink_start) / 0.18
            if k < 1:
                blink = math.sin(k * math.pi)
            else:
                self.blink_start = -1
                self.next_blink = t + random.uniform(2.0, 6.0) - (1.7 if random.random() < 0.15 else 0)
        if t > self.next_glance:
            self.gaze_target = [0.0, 0.0] if random.random() < 0.5 else [random.uniform(-6, 6), random.uniform(-3, 3)]
            self.next_glance = t + random.uniform(1.0, 4.0)
        for i in range(2):
            self.gaze[i] += (self.gaze_target[i] - self.gaze[i]) * 0.25

        breath = math.sin(t * 2 * math.pi * 0.22)
        out = replace(c, pitch=c.pitch + breath * 0.8)
        if performer:
            return out  # the webcam already supplies blinks, gaze and sway
        out.yaw += math.sin(t * 0.53) * 3.0 + math.sin(t * 0.21) * 2.0
        out.pitch += math.sin(t * 0.37 + 1) * 1.5
        out.roll += math.sin(t * 0.41 + 2) * 1.5
        out.gaze_x += self.gaze[0]
        out.gaze_y += self.gaze[1]
        if blink > 0 and out.eye_open is None:
            out.eye_open = 1.0 - blink
        return out


class Session:
    def __init__(self, engine: RigEngine):
        self.engine = engine
        self.tracker = FaceTracker(engine.cropper, engine.crop_cfg)
        self.autopilot = Autopilot()
        self.controls = Controls()
        self.face_found = False

    def calibrate(self):
        self.engine.recalibrate()
        self.tracker.reset()

    def step(self, performer_rgb: np.ndarray | None = None) -> np.ndarray:
        drive = None
        if performer_rgb is not None:
            crop = self.tracker.crop(performer_rgb)
            self.face_found = crop is not None
            if crop is not None:
                drive = self.engine.driving_info(crop)
        else:
            self.face_found = False
        c = self.controls
        if c.idle:
            c = self.autopilot.apply(c, performer=drive is not None)
        return self.engine.render(drive, c)
