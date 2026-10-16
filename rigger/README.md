# Memoire Rig: one-photo VTuber rigging

Give it one portrait photo and it rigs it automatically. You can then puppet the avatar live from a webcam, a microphone and rig sliders, or let it idle on its own. It is built on the open-source [LivePortrait](https://github.com/KwaiVGI/LivePortrait) (MIT) neural portrait animator.

![Studio](docs/studio.jpg)

Frames below, left to right: driven by a performer video ×5; yaw + smile sliders; eyes closed + mouth open + roll sliders.

![Rendered frames](docs/demo_frames.jpg)

## What "auto-rig" does

When you upload a photo, the engine:

1. **Finds the face** using InsightFace detection plus LivePortrait's 203-point landmark model. It then crops and aligns the face.
2. **Builds the rig.** LivePortrait extracts a 3D appearance volume, 21 implicit 3D motion keypoints, head pose (yaw/pitch/roll), expression offsets, scale and translation.
3. **Measures the eye and lip openness** and normalises the mouth to closed, so that your neutral face maps to a neutral avatar.
4. **Prepares a paste-back mask** so the animated face blends into the full original picture, including hair, body and background.

Each frame, the rig is posed by any combination of:

| Driver | What it moves |
| --- | --- |
| **Webcam** | Head rotation and translation, every expression, blinks, gaze. Motion is relative to the neutral pose you calibrate. |
| **Mic lip-sync** | Mouth opening from voice loudness. |
| **Idle life** | Random blinks, breathing, gentle head sway, glances. Only blinks and breathing are added while the webcam is driving. |
| **Rig controls** | Yaw, pitch, roll, smile, wink, eyebrow, gaze, eye-open and mouth-open overrides. These use LivePortrait's own retargeting edits. |

## Setup

You need Python 3.10, git, and ideally an NVIDIA GPU. A GPU runs in real time; a CPU takes several seconds per frame.

```bash
conda create -n memoire python=3.10 -y && conda activate memoire
cd rigger
python setup_liveportrait.py
```

The setup script does three things:

- Clones LivePortrait into `vendor/` at a pinned commit.
- Installs PyTorch (CUDA 12.1, or CPU when no NVIDIA GPU is found) and the dependencies.
- Downloads the pretrained weights (about 500 MB) from Hugging Face.

On Apple Silicon it uses the macOS requirements and the MPS backend.

## Use

**Browser studio:**

```bash
python -m memoire_rig serve        # then open http://127.0.0.1:8765
```

Drop in a photo, click **Start webcam**, and hold a neutral face for a second. **Set neutral pose** recalibrates. **Record clip** saves a WebM and includes mic audio when lip-sync is on.

**Stream to OBS, Discord or Zoom as a virtual camera:**

```bash
python -m memoire_rig cam --source photo.jpg --virtualcam
```

On Windows and macOS, the virtual camera needs OBS installed. On Linux it needs `v4l2loopback`. In the preview window, press `c` to recalibrate and `q` to quit.

**Inspect a rig** (crop, pose, keypoints) as JSON:

```bash
python -m memoire_rig rig --source photo.jpg
```

## Tips

- Use a sharp, front-facing photo with one person and the full face visible, ideally with eyes open and mouth closed.
- Keep your webcam face roughly the same size and pose as when you calibrated. Motion is relative to that pose.
- LivePortrait is trained on real faces. Anime or very stylised art often fails detection. For anime characters, [Talking Head Anime](https://github.com/pkhungurn/talking-head-anime-3-demo) and [EasyVtuber](https://github.com/yuyuyzl/EasyVtuber) are better engines.

## Licensing

- LivePortrait's code is MIT licensed.
- LivePortrait uses **InsightFace detection models, which are for non-commercial research only**. Replace them before any commercial use.
- Only animate people who have consented to it.

## Layout

```
rigger/
  setup_liveportrait.py   fetch engine + weights
  memoire_rig/
    engine.py             auto-rig, per-frame posing, expression edits, face tracker
    session.py            webcam + controls + idle autopilot -> frames
    server.py             FastAPI: /api/source upload, /ws lock-step frame stream
    __main__.py           CLI: serve | cam | rig
  web/                    browser studio (index.html, app.js, style.css)
```
