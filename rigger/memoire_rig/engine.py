"""The rig engine: turns one portrait photo into a puppet driven by LivePortrait.

"Rigging" here is LivePortrait's implicit-keypoint model: from the source photo
we extract a 3D appearance volume plus 21 canonical 3D keypoints, head pose,
expression offsets, eye/lip openness, and a paste-back mask. Every frame we
build new driving keypoints (from a webcam face, manual controls, the idle
autopilot, or any mix) and the generator re-renders the face.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "LivePortrait"


def _import_liveportrait():
    if not (VENDOR / "src").exists():
        raise RuntimeError(
            f"LivePortrait not found at {VENDOR}. Run `python setup_liveportrait.py` first."
        )
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))


_import_liveportrait()

import cv2  # noqa: E402
import torch  # noqa: E402

from src.config.crop_config import CropConfig  # noqa: E402
from src.config.inference_config import InferenceConfig  # noqa: E402
from src.live_portrait_wrapper import LivePortraitWrapper  # noqa: E402
from src.utils.camera import get_rotation_matrix  # noqa: E402
from src.utils.crop import (  # noqa: E402
    crop_image_by_bbox,
    parse_bbox_from_landmark,
    paste_back,
    prepare_paste_back,
)
from src.utils.cropper import Cropper  # noqa: E402
from src.utils.io import contiguous, resize_to_limit  # noqa: E402
from src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio  # noqa: E402


class NoFaceError(ValueError):
    pass


@dataclass
class Controls:
    """Rig parameters layered on top of whatever drives the face.

    Pose offsets are in degrees. Expression sliders use the ranges of
    LivePortrait's retargeting editor: smile -0.3..1.3, wink 0..39,
    eyebrow -30..30, gaze_x -30..30, gaze_y -63..63.
    """

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    smile: float = 0.0
    wink: float = 0.0
    eyebrow: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    # 0 = closed .. 1 = as in the photo. None leaves the eyes to the driver.
    eye_open: float | None = None
    # 0 = closed .. 1 = wide open. None leaves the mouth to the driver.
    mouth_open: float | None = None
    # Scales all motion coming from the driver (webcam).
    strength: float = 1.0
    idle: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "Controls":
        c = cls()
        for k, v in d.items():
            if k == "idle":
                c.idle = bool(v)
            elif k in ("eye_open", "mouth_open"):
                setattr(c, k, None if v is None else float(v))
            elif hasattr(c, k):
                setattr(c, k, float(v))
        return c


@dataclass
class SourceRig:
    img_rgb: np.ndarray
    f_s: torch.Tensor
    x_s: torch.Tensor
    x_c_s: torch.Tensor
    info: dict
    lmk_crop: np.ndarray
    M_c2o: np.ndarray
    mask_ori: np.ndarray
    eye_ratio: float
    lip_ratio: float
    lip_delta0: torch.Tensor | None
    summary: dict = field(default_factory=dict)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RigEngine:
    def __init__(self, device: str | None = None, max_dim: int = 1024):
        self.device = device or pick_device()
        cpu = self.device == "cpu"
        self.inf_cfg = InferenceConfig(flag_force_cpu=cpu, flag_use_half_precision=self.device == "cuda")
        self.crop_cfg = CropConfig(flag_force_cpu=cpu)
        self.wrapper = LivePortraitWrapper(inference_cfg=self.inf_cfg)
        self.cropper = Cropper(crop_cfg=self.crop_cfg, flag_force_cpu=cpu, image_type="human_face")
        self.max_dim = max_dim
        self.src: SourceRig | None = None
        self.driver0: dict | None = None
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ rig
    @torch.no_grad()
    def load_source(self, img_rgb: np.ndarray) -> dict:
        """Auto-rigs a portrait. Raises NoFaceError when no face is found."""
        img_rgb = contiguous(resize_to_limit(img_rgb, max_dim=self.max_dim, division=2))
        crop = self.cropper.crop_source_image(img_rgb, self.crop_cfg)
        if crop is None:
            raise NoFaceError("No face found in the source image.")
        w = self.wrapper
        I_s = w.prepare_source(crop["img_crop_256x256"])
        info = w.get_kp_info(I_s)
        f_s = w.extract_feature_3d(I_s)
        x_s = w.transform_keypoint(info)
        lmk = crop["lmk_crop"]

        eye_ratio = float(calc_eye_close_ratio(lmk[None]).mean())
        lip_ratio = float(calc_lip_close_ratio(lmk[None])[0][0])
        # Like LivePortrait's pipeline: start from a closed mouth so the
        # performer's neutral face maps to a neutral avatar.
        lip_delta0 = None
        closed = w.calc_combined_lip_ratio([0.0], lmk)
        if closed[0][0] >= self.inf_cfg.lip_normalize_threshold:
            lip_delta0 = w.retarget_lip(x_s, closed)

        M_c2o = crop["M_c2o"]
        mask_ori = prepare_paste_back(self.inf_cfg.mask_crop, M_c2o, dsize=(img_rgb.shape[1], img_rgb.shape[0]))

        with self.lock:
            self.src = SourceRig(img_rgb, f_s, x_s, info["kp"], info, lmk, M_c2o, mask_ori,
                                 eye_ratio, lip_ratio, lip_delta0)
            self.driver0 = None
            self.src.summary = self._summarize(crop)
        return self.src.summary

    def _summarize(self, crop: dict) -> dict:
        """Rig description for the UI, in source-image pixel coordinates."""
        s = self.src
        M = s.M_c2o[:2]
        to_img = lambda pts: (np.c_[pts, np.ones(len(pts))] @ M.T).round(1).tolist()  # noqa: E731
        d = self.crop_cfg.dsize
        corners = to_img(np.array([[0, 0], [d, 0], [d, d], [0, d]], dtype=np.float32))
        # Implicit keypoints live in [-1, 1] crop space (grid_sample convention).
        kp = s.x_s[0, :, :2].float().cpu().numpy()
        kp_img = to_img((kp + 1) / 2 * d)
        return {
            "width": int(s.img_rgb.shape[1]),
            "height": int(s.img_rgb.shape[0]),
            "crop": corners,
            "landmarks": to_img(crop["lmk_crop"]),
            "keypoints": kp_img,
            "pose": {k: round(float(s.info[k].item()), 1) for k in ("pitch", "yaw", "roll")},
            "eye_ratio": round(s.eye_ratio, 3),
            "lip_ratio": round(s.lip_ratio, 3),
            "device": self.device,
        }

    def recalibrate(self):
        """Makes the performer's current face the neutral pose."""
        self.driver0 = None

    # --------------------------------------------------------------- render
    @torch.no_grad()
    def driving_info(self, face_crop_256: np.ndarray) -> dict:
        return self.wrapper.get_kp_info(self.wrapper.prepare_source(face_crop_256))

    @torch.no_grad()
    def render(self, drive: dict | None, c: Controls, full_frame: bool = True) -> np.ndarray:
        """Renders one frame. `drive` is the motion-extractor output for the
        performer's face (or None for controls/idle only)."""
        with self.lock:
            s = self.src
            if s is None:
                raise RuntimeError("No source image rigged yet.")
            w, dev = self.wrapper, self.wrapper.device
            info = s.info
            delta = info["exp"].clone()
            scale = info["scale"].clone()
            t = info["t"].clone()
            R_rel = torch.eye(3, device=dev)[None]

            if drive is not None:
                if self.driver0 is None:
                    self.driver0 = {k: v.clone() for k, v in drive.items() if isinstance(v, torch.Tensor)}
                d0, k = self.driver0, c.strength
                R_d = get_rotation_matrix(drive["pitch"], drive["yaw"], drive["roll"])
                R_d0 = get_rotation_matrix(d0["pitch"], d0["yaw"], d0["roll"])
                if k != 1.0:
                    # scale the relative rotation through its Euler angles
                    R_d = get_rotation_matrix(d0["pitch"] + (drive["pitch"] - d0["pitch"]) * k,
                                              d0["yaw"] + (drive["yaw"] - d0["yaw"]) * k,
                                              d0["roll"] + (drive["roll"] - d0["roll"]) * k)
                R_rel = R_d @ R_d0.permute(0, 2, 1)
                delta = delta + (drive["exp"] - d0["exp"]) * k
                scale = scale * (1 + (drive["scale"] / d0["scale"] - 1) * k)
                t = t + (drive["t"] - d0["t"]) * k

            R_src = get_rotation_matrix(info["pitch"] + c.pitch, info["yaw"] + c.yaw, info["roll"] + c.roll)
            R_new = R_rel @ R_src
            delta = apply_expression(delta, c)
            t[..., 2] = 0
            x_d = scale * (s.x_c_s @ R_new + delta) + t

            if c.eye_open is not None and abs(c.eye_open - 1.0) > 1e-3:
                target = max(0.0, c.eye_open) * s.eye_ratio
                x_d = x_d + w.retarget_eye(s.x_s, w.calc_combined_eye_ratio([[target]], s.lmk_crop))
            if c.mouth_open is not None and c.mouth_open > 1e-3:
                target = min(1.0, c.mouth_open) * 0.6
                x_d = x_d + w.retarget_lip(s.x_s, w.calc_combined_lip_ratio([target], s.lmk_crop))

            x_d = w.stitching(s.x_s, x_d)
            if s.lip_delta0 is not None and c.mouth_open is None:
                x_d = x_d + s.lip_delta0
            out = w.warp_decode(s.f_s, s.x_s, x_d)
            face = w.parse_output(out["out"])[0]
            if not full_frame:
                return face
            return paste_back(face, s.M_c2o, s.img_rgb, s.mask_ori)


def apply_expression(delta: torch.Tensor, c: Controls) -> torch.Tensor:
    """Expression offsets on the 21 implicit keypoints, using the same keypoint
    edits as LivePortrait's retargeting editor (src/gradio_pipeline.py)."""
    d = delta
    if c.gaze_x or c.gaze_y:
        x, y = c.gaze_x, c.gaze_y
        d[0, 11, 0] += x * (0.0007 if x > 0 else 0.001)
        d[0, 15, 0] += x * (0.001 if x > 0 else 0.0007)
        d[0, 11, 1] += y * -0.001
        d[0, 15, 1] += y * -0.001
        blink = -y / 2.0
        d[0, 11, 1] += blink * -0.001
        d[0, 13, 1] += blink * 0.0003
        d[0, 15, 1] += blink * -0.001
        d[0, 16, 1] += blink * 0.0003
    if c.smile:
        s = c.smile
        d[0, 20, 1] += s * -0.01
        d[0, 14, 1] += s * -0.02
        d[0, 17, 1] += s * 0.0065
        d[0, 17, 2] += s * 0.003
        d[0, 13, 1] += s * -0.00275
        d[0, 16, 1] += s * -0.00275
        d[0, 3, 1] += s * -0.0035
        d[0, 7, 1] += s * -0.0035
    if c.wink:
        k = c.wink
        d[0, 11, 1] += k * 0.001
        d[0, 13, 1] += k * -0.0003
        d[0, 17, 0] += k * 0.0003
        d[0, 17, 1] += k * 0.0003
        d[0, 3, 1] += k * -0.0003
    if c.eyebrow:
        e = c.eyebrow
        if e > 0:
            d[0, 1, 1] += e * 0.001
            d[0, 2, 1] += e * -0.001
        else:
            d[0, 1, 0] += e * -0.001
            d[0, 2, 0] += e * 0.001
            d[0, 1, 1] += e * 0.0003
            d[0, 2, 1] += e * -0.0003
    return d


class FaceTracker:
    """Finds and follows the performer's face in webcam frames and returns the
    256x256 crop LivePortrait's motion extractor expects."""

    REDETECT_EVERY = 60

    def __init__(self, cropper: Cropper, crop_cfg: CropConfig):
        self.cropper = cropper
        self.cfg = crop_cfg
        self.lmk = None
        self.box = None
        self.frames = 0

    def reset(self):
        self.lmk = self.box = None

    def crop(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        self.frames += 1
        if self.lmk is None or self.frames % self.REDETECT_EVERY == 0:
            faces = self.cropper.face_analysis_wrapper.get(
                contiguous(frame_rgb[..., ::-1]), flag_do_landmark_2d_106=True, direction="large-small"
            )
            if not faces:
                self.reset()
                return None
            seed = faces[0].landmark_2d_106
        else:
            seed = self.lmk
        self.lmk = self.cropper.human_landmark_runner.run(frame_rgb, seed)
        bb = parse_bbox_from_landmark(
            self.lmk,
            scale=self.cfg.scale_crop_driving_video,
            vx_ratio=self.cfg.vx_ratio_crop_driving_video,
            vy_ratio=self.cfg.vy_ratio_crop_driving_video,
        )["bbox"]
        box = np.array([bb[0, 0], bb[0, 1], bb[2, 0], bb[2, 1]], dtype=np.float32)
        # Light smoothing keeps the crop (and so the avatar) from jittering.
        self.box = box if self.box is None else self.box * 0.6 + box * 0.4
        return crop_image_by_bbox(frame_rgb, self.box.tolist(), dsize=256, flag_rot=False)["img_crop"]


def decode_image(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("Could not decode image.")
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        # Flatten transparency onto white so cut-out PNGs rig cleanly.
        a = arr[..., 3:4].astype(np.float32) / 255
        arr = (arr[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    return contiguous(arr[..., ::-1])


def encode_jpeg(img_rgb: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()
