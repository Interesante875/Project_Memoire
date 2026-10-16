"""Anime engine: rigs a single anime character image with Talking Head Anime 3
(https://github.com/pkhungurn/talking-head-anime-3-demo, code MIT, models
CC BY 4.0 by Pramook Khungurn) and renders a looping performance.

LivePortrait is trained on real faces and usually misses anime faces; THA3 is
built for exactly this input: a cut-out character with a transparent
background. It exposes 45 rig parameters (eyebrows, eyes, irises, mouth
shapes, head/neck/body rotation, breathing).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "talking-head-anime-3-demo"

# THA3 input contract: 512x512 RGBA, head inside the 128x128 box centred in
# the top half, i.e. x 192..320, y 64..192.
CANVAS = 512
HEAD_BOX = 128
HEAD_CENTER = (256, 128)


def _import_tha3():
    if not (VENDOR / "tha3").exists():
        raise RuntimeError(f"THA3 not found at {VENDOR}. Run `python setup_liveportrait.py --anime` first.")
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))


def load_rgba(path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(path)
    if img.mode != "RGBA":
        raise ValueError("The anime engine needs a cut-out PNG with a transparent background.")
    return np.asarray(img).copy()


FACE_CASCADE = VENDOR.parent / "lbpcascade_animeface.xml"


def detect_face(rgba: np.ndarray):
    """Largest anime face (x, y, w, h) via nagadomi's lbpcascade_animeface, or
    None when the cascade is missing or finds nothing."""
    import cv2

    if not FACE_CASCADE.exists():
        return None
    a = rgba[..., 3:4].astype(np.float32) / 255
    rgb = (rgba[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    min_side = max(24, min(gray.shape) // 12)
    faces = cv2.CascadeClassifier(str(FACE_CASCADE)).detectMultiScale(gray, 1.05, 3, minSize=(min_side, min_side))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def find_head(rgba: np.ndarray) -> tuple[float, float, float]:
    """Estimates (cx, cy, size) of the head, hair included. Uses the anime face
    detector when available, else the alpha silhouette (upright bust or
    full-body shot with the head at the top)."""
    face = detect_face(rgba)
    if face is not None:
        x, y, w, h = map(float, face)
        return x + w / 2, y + h / 2 - 0.26 * h, 1.39 * w
    a = rgba[..., 3] > 16
    rows = np.where(a.any(1))[0]
    cols = np.where(a.any(0))[0]
    top, bottom = rows[0], rows[-1]
    width = cols[-1] - cols[0] + 1
    height = bottom - top + 1
    # Bust shots: head ~ 64% of the width. Full body: head ~ 1/6.5 of height.
    size = min(width * 0.64, height / 6.5) if height > width * 1.6 else width * 0.64
    band = a[top: top + int(size)]
    xs = np.nonzero(band)[1]
    cx = float(np.median(xs)) if len(xs) else (cols[0] + cols[-1]) / 2
    return cx, top + size / 2, size


def layout(rgba: np.ndarray, head: tuple[float, float, float] | None = None) -> np.ndarray:
    """Places the character onto THA3's 512x512 template canvas."""
    import cv2

    cx, cy, size = head or find_head(rgba)
    s = HEAD_BOX / size
    M = np.float32([[s, 0, HEAD_CENTER[0] - s * cx], [0, s, HEAD_CENTER[1] - s * cy]])
    out = cv2.warpAffine(rgba, M, (CANVAS, CANVAS), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    out[out[..., 3] == 0] = 0
    return out


class AnimeRig:
    def __init__(self, model: str = "separable_float", device: str | None = None):
        _import_tha3()
        import importlib

        import torch

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu" and model.endswith("_half"):
            model = model.replace("_half", "_float")
        mode = importlib.import_module(f"tha3.poser.modes.{model}")
        # THA3 defaults to cwd-relative model paths; pass absolute ones instead.
        files = {n: str(VENDOR / "data" / "models" / model / f"{n}.pt")
                 for n in ("eyebrow_decomposer", "eyebrow_morphing_combiner", "face_morpher",
                           "two_algo_face_body_rotator", "editor")}
        self.poser = mode.create_poser(self.device, module_file_names=files)
        self.dtype = self.poser.get_dtype()
        self.names = [n for g in self.poser.get_pose_parameter_groups() for n in g.get_parameter_names()]
        self.image = None

    def load(self, canvas_rgba: np.ndarray):
        from PIL import Image
        from tha3.util import extract_pytorch_image_from_PIL_image

        self.image = extract_pytorch_image_from_PIL_image(Image.fromarray(canvas_rgba, "RGBA")) \
            .to(self.device).to(self.dtype)

    def render(self, params: dict[str, float]) -> np.ndarray:
        """Returns a 512x512 RGBA uint8 frame for the given rig parameters."""
        from tha3.util import convert_output_image_from_torch_to_numpy

        pose = self.torch.zeros(len(self.names), device=self.device, dtype=self.dtype)
        for k, v in params.items():
            lo = -1.0 if k.startswith(("head_", "neck_", "body_", "iris_rotation")) else 0.0
            pose[self.names.index(k)] = min(1.0, max(lo, v))
        with self.torch.no_grad():
            out = self.poser.pose(self.image, pose)[0].float().cpu()
        rgba = (np.clip(convert_output_image_from_torch_to_numpy(out), 0, 1) * 255).round().astype(np.uint8)
        rgba[rgba[..., 3] < 12] = 0  # drop faint speckle the generator leaves in empty space
        return rgba


def _pulse(t: float, start: float, length: float, period: float) -> float:
    """0..1..0 bump of `length` seconds starting at `start`, wrapping at `period`."""
    x = (t - start) % period
    return math.sin(math.pi * x / length) if x < length else 0.0


def _window(t: float, a: float, b: float, ramp: float = 0.25) -> float:
    """Smooth 0->1->0 envelope over [a, b]."""
    if t <= a or t >= b:
        return 0.0
    k = min((t - a) / ramp, (b - t) / ramp, 1.0)
    return k * k * (3 - 2 * k)


def performance(t: float, period: float = 8.0) -> dict[str, float]:
    """A seamless loop: idle sway and breathing, blinks, a short line of
    speech, then a smile. All motions are periodic in `period` seconds."""
    w = 2 * math.pi / period
    p = {
        "breathing": 0.5 + 0.5 * math.sin(2 * w * t),
        "head_y": 0.35 * math.sin(w * t),
        "head_x": 0.12 * math.sin(2 * w * t + 1.0),
        "neck_z": 0.22 * math.sin(w * t + 0.6),
        "body_y": 0.18 * math.sin(w * t - 0.4),
        "body_z": 0.10 * math.sin(w * t + 0.2),
        "iris_rotation_y": -0.25 * math.sin(w * t),
        "iris_rotation_x": 0.1 * math.sin(3 * w * t),
    }
    blink = max(_pulse(t, s, 0.22, period) for s in (1.1, 4.4, 7.2))
    talk = _window(t, 2.0, 3.9)
    smile = _window(t, 5.0, 6.8, ramp=0.4)
    # Speech: cycle vowel shapes with a syllable-rate envelope.
    if talk:
        vowels = ["mouth_aaa", "mouth_iii", "mouth_ooo", "mouth_eee", "mouth_uuu", "mouth_aaa"]
        syl = (t - 2.0) * 6.0
        i = int(syl) % len(vowels)
        p[vowels[i]] = talk * (0.55 + 0.45 * abs(math.sin(math.pi * syl)))
        p["eyebrow_raised_left"] = p["eyebrow_raised_right"] = 0.35 * talk
    for side in ("left", "right"):
        p[f"eye_wink_{side}"] = blink * (1 - smile)
        p[f"eye_happy_wink_{side}"] = smile
        p[f"mouth_raised_corner_{side}"] = smile
        p[f"eyebrow_happy_{side}"] = smile
    return p


def render_loop(rig: AnimeRig, seconds: float = 8.0, fps: int = 24, progress=None) -> list[np.ndarray]:
    frames = []
    n = int(round(seconds * fps))
    for i in range(n):
        frames.append(rig.render(performance(i / fps, seconds)))
        if progress:
            progress(i + 1, n)
    return frames


def crop_box(frames: list[np.ndarray], margin: int = 12) -> tuple[int, int, int, int]:
    alpha = np.max([f[..., 3] for f in frames], axis=0) > 8
    ys, xs = np.nonzero(alpha)
    x0, x1 = max(0, xs.min() - margin), min(CANVAS, xs.max() + margin)
    y0, y1 = max(0, ys.min() - margin), min(CANVAS, ys.max() + margin)
    return x0, y0, x1, y1


def save_outputs(frames: list[np.ndarray], out_dir: Path, name: str, fps: int, scale: float = 2.0,
                 background=(246, 239, 228)) -> list[Path]:
    """Writes an MP4 (on a background colour), a transparent WebM for OBS
    overlays, and a GIF preview."""
    import cv2
    import imageio.v2 as imageio
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = crop_box(frames)
    w, h = int((x1 - x0) * scale) // 2 * 2, int((y1 - y0) * scale) // 2 * 2
    big = [cv2.resize(f[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_LANCZOS4) for f in frames]
    bg = np.array(background, np.float32)

    def flat(f):
        a = f[..., 3:4].astype(np.float32) / 255
        return (f[..., :3] * a + bg * (1 - a)).astype(np.uint8)

    paths = []
    mp4 = out_dir / f"{name}.mp4"
    imageio.mimwrite(mp4, [flat(f) for f in big], fps=fps, codec="libx264", quality=8, macro_block_size=2)
    paths.append(mp4)

    webm = out_dir / f"{name}_transparent.webm"
    writer = imageio.get_writer(webm, fps=fps, codec="libvpx-vp9", pixelformat="yuva420p",
                                output_params=["-auto-alt-ref", "0", "-b:v", "2M"], macro_block_size=2)
    for f in big:
        writer.append_data(f)
    writer.close()
    paths.append(webm)

    gif = out_dir / f"{name}.gif"
    half = [Image.fromarray(flat(cv2.resize(f, (w // 2, h // 2), interpolation=cv2.INTER_AREA))) for f in big]
    half[0].save(gif, save_all=True, append_images=half[1:], duration=int(1000 / fps), loop=0, optimize=True)
    paths.append(gif)
    return paths


class StreamWriter:
    """Writes frames as they are rendered: MP4 on a background colour, a
    transparent WebM (VP9 alpha) for OBS, and a half-size GIF preview."""

    def __init__(self, out_dir: Path, name: str, fps: int, box, scale=2.0, background=(246, 239, 228)):
        import imageio.v2 as imageio

        out_dir.mkdir(parents=True, exist_ok=True)
        self.box, self.scale, self.fps = box, scale, fps
        x0, y0, x1, y1 = box
        self.size = (int((x1 - x0) * scale) // 2 * 2, int((y1 - y0) * scale) // 2 * 2)
        self.bg = np.array(background, np.float32)
        self.paths = [out_dir / f"{name}.mp4", out_dir / f"{name}_transparent.webm", out_dir / f"{name}.gif"]
        self.mp4 = imageio.get_writer(self.paths[0], fps=fps, codec="libx264", quality=8, macro_block_size=2)
        self.webm = imageio.get_writer(self.paths[1], fps=fps, codec="libvpx-vp9", pixelformat="yuva420p",
                                       output_params=["-auto-alt-ref", "0", "-b:v", "2M"], macro_block_size=2)
        self.gif = []

    def add(self, rgba: np.ndarray) -> np.ndarray:
        import cv2
        from PIL import Image

        x0, y0, x1, y1 = self.box
        big = cv2.resize(rgba[y0:y1, x0:x1], self.size, interpolation=cv2.INTER_LANCZOS4)
        a = big[..., 3:4].astype(np.float32) / 255
        flat = (big[..., :3] * a + self.bg * (1 - a)).astype(np.uint8)
        self.mp4.append_data(flat)
        self.webm.append_data(big)
        if len(self.gif) < 2000:
            half = cv2.resize(flat, (self.size[0] // 2, self.size[1] // 2), interpolation=cv2.INTER_AREA)
            self.gif.append(Image.fromarray(half))
        return flat

    def close(self) -> list[Path]:
        self.mp4.close()
        self.webm.close()
        g = self.gif
        g[0].save(self.paths[2], save_all=True, append_images=g[1:], duration=int(1000 / self.fps), loop=0, optimize=True)
        return self.paths


def render_showreel(rig: AnimeRig, layered, timeline, fps: int = 24, substeps: int = 10, progress=None):
    """Yields (rgba_frame, expression_name). THA3 poses the base; accessory
    layers follow the head and are simulated with sub-stepped physics."""
    from .expressions import head_point
    from .physics import back_flow

    rig.load(layered.base)
    neutral = rig.render({})
    layered.set_reference(neutral)
    n = int(round(timeline.duration * fps))
    dt = 1.0 / fps
    h = dt / substeps

    def kin(t, pt):
        p, root, _ = timeline.at(max(0.0, t))
        return head_point(p, root, pt)

    for i in range(n):
        t = i * dt
        for k in range(substeps):
            ts = t - dt + (k + 1) * h
            if ts - h < 0:
                continue  # no history before t=0: avoid a fake kick
            for a in layered.acc:
                pt = np.array(a.anchor, float)
                (x0, r0), (x1, r1), (x2, r2) = kin(ts - h, pt), kin(ts, pt), kin(ts + h, pt)
                acc = (x2 - 2 * x1 + x0) / (h * h)
                a.step(h, acc, r1, (r2 - 2 * r1 + r0) / (h * h), ts)
        p, root, name = timeline.at(t)
        frame = rig.render(p)
        flow = back_flow(frame, neutral)
        yield layered.compose(frame, flow, root), name
        if progress:
            progress(i + 1, n)
