"""Command line entry point.

    python -m memoire_rig serve [--host 127.0.0.1] [--port 8765]
    python -m memoire_rig cam --source me.jpg [--camera 0] [--virtualcam]
    python -m memoire_rig rig --source me.jpg       # print the auto-rig as JSON
    python -m memoire_rig anime --source char.png   # rig + animate an anime character
"""

import argparse
import json
import sys
import time

import numpy as np


def cmd_serve(args, engine):
    import uvicorn

    from .server import create_app

    print(f"Memoire Rig studio: http://{args.host}:{args.port}  (device: {engine.device})")
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")


def cmd_rig(args, engine):
    from .engine import decode_image

    with open(args.source, "rb") as f:
        rig = engine.load_source(decode_image(f.read()))
    if not args.full:
        rig.pop("landmarks")
    json.dump(rig, sys.stdout, indent=2)
    print()


def cmd_cam(args, engine):
    """Local webcam -> avatar, shown in a window and/or sent to a virtual camera
    (OBS, Discord, Zoom). Keys: c = recalibrate neutral pose, q / Esc = quit."""
    import cv2

    from .engine import decode_image
    from .session import Session

    with open(args.source, "rb") as f:
        engine.load_source(decode_image(f.read()))
    session = Session(engine)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera}")

    vcam = None
    if args.virtualcam:
        import pyvirtualcam

        h, w = engine.src.img_rgb.shape[:2]
        vcam = pyvirtualcam.Camera(width=w, height=h, fps=30)
        print(f"Virtual camera: {vcam.device}")

    print("Look at the camera with a neutral face. c = recalibrate, q = quit")
    t_last, fps = time.perf_counter(), 0.0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            out = session.step(frame_bgr[..., ::-1].copy())
            now = time.perf_counter()
            fps = fps * 0.9 + 0.1 / max(1e-3, now - t_last)
            t_last = now
            if vcam:
                vcam.send(out)
            if not args.no_window:
                view = out[..., ::-1].copy()
                label = f"{fps:4.1f} fps" + ("" if session.face_found else "  (no face - idle)")
                cv2.putText(view, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("Memoire Rig", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    session.calibrate()
    finally:
        cap.release()
        if vcam:
            vcam.close()
        cv2.destroyAllWindows()


def cmd_anime(args):
    """Rigs a transparent-background anime character with Talking Head Anime 3
    and renders a looping idle / talk / smile performance."""
    from pathlib import Path

    from PIL import Image

    from .anime import AnimeRig, find_head, layout, load_rgba, render_loop, save_outputs

    rgba = load_rgba(args.source)
    head = tuple(map(float, args.head.split(","))) if args.head else find_head(rgba)
    print(f"Head: centre ({head[0]:.0f}, {head[1]:.0f}), size {head[2]:.0f}px")
    canvas = layout(rgba, head)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.source).stem
    Image.fromarray(canvas, "RGBA").save(out_dir / f"{name}_rig_input.png")

    rig = AnimeRig(model=args.model, device=args.device)
    t = time.perf_counter()
    if args.accessories:
        return anime_showreel(args, rig, canvas, out_dir, name, t)
    rig.load(canvas)

    def progress(i, n):
        print(f"\rRendering {i}/{n} frames ({(time.perf_counter() - t) / i:.2f}s/frame)", end="", flush=True)

    frames = render_loop(rig, seconds=args.seconds, fps=args.fps, progress=progress)
    print()
    for p in save_outputs(frames, out_dir, name, args.fps, scale=args.scale):
        print(f"Wrote {p}")


def anime_showreel(args, rig, canvas, out_dir, name, t0):
    """Expressions + accessory physics, rendered as one showreel."""
    import cv2

    from .anime import StreamWriter, crop_box, render_showreel
    from .expressions import FACES, Timeline
    from .physics import LayeredRig, load_accessories

    layered = LayeredRig(canvas, load_accessories(args.accessories))
    cv2.imwrite(str(out_dir / f"{name}_layers.png"), cv2.cvtColor(layered.overlay(canvas), cv2.COLOR_RGB2BGR))
    (out_dir / f"{name}_expressions.json").write_text(json.dumps(FACES, indent=1))
    timeline = Timeline()
    x0, y0, x1, y1 = crop_box([canvas], margin=0)
    m = 28  # head motion, bounces and swinging parts need room
    box = (max(0, x0 - m), max(0, y0 - m - 10), min(512, x1 + m), min(512, y1 + 4))
    writer = StreamWriter(out_dir, f"{name}_showreel", args.fps, box, scale=args.scale)
    stills, seen = {}, None

    def progress(i, n):
        print(f"\rRendering {i}/{n} frames ({(time.perf_counter() - t0) / i:.2f}s/frame)", end="", flush=True)

    for i, (frame, expr) in enumerate(render_showreel(rig, layered, timeline, args.fps, progress=progress)):
        flat = writer.add(frame)
        if expr != seen:
            seen, start = expr, i
        if i - start == int(args.fps * 1.1) and expr not in stills:  # mid-expression still
            stills[expr] = flat
    print()
    for p in writer.close():
        print(f"Wrote {p}")
    if stills:
        h = min(f.shape[0] for f in stills.values())
        tiles = []
        for k, f in stills.items():
            f = cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h))
            cv2.putText(f, k, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 40, 40), 2, cv2.LINE_AA)
            tiles.append(f)
        sheet = out_dir / f"{name}_expressions.jpg"
        cv2.imwrite(str(sheet), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
        print(f"Wrote {sheet}")


def main():
    ap = argparse.ArgumentParser(prog="memoire_rig", description="One-photo VTuber rigging on LivePortrait.")
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], help="force a device (default: auto)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the browser rigging studio")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)

    c = sub.add_parser("cam", help="drive the avatar from a local webcam")
    c.add_argument("--source", required=True, help="portrait photo to rig")
    c.add_argument("--camera", type=int, default=0)
    c.add_argument("--virtualcam", action="store_true", help="output to a virtual webcam (needs pyvirtualcam)")
    c.add_argument("--no-window", action="store_true")

    r = sub.add_parser("rig", help="auto-rig a photo and print the rig")
    r.add_argument("--source", required=True)
    r.add_argument("--full", action="store_true", help="include all 203 landmarks")

    a = sub.add_parser("anime", help="rig + animate an anime character (Talking Head Anime 3)")
    a.add_argument("--source", required=True, help="PNG with a transparent background")
    a.add_argument("--head", help="override head as cx,cy,size in source pixels")
    a.add_argument("--accessories", help="accessory physics rig (JSON); renders the expression showreel")
    a.add_argument("--out", default="renders")
    a.add_argument("--seconds", type=float, default=8.0)
    a.add_argument("--fps", type=int, default=24)
    a.add_argument("--scale", type=float, default=2.0, help="upscale factor for the outputs")
    a.add_argument("--model", default="separable_float",
                   choices=["standard_float", "separable_float", "standard_half", "separable_half"])

    args = ap.parse_args()
    if args.cmd == "anime":
        return cmd_anime(args)
    from .engine import RigEngine

    engine = RigEngine(device=args.device)
    {"serve": cmd_serve, "cam": cmd_cam, "rig": cmd_rig}[args.cmd](args, engine)


if __name__ == "__main__":
    main()
