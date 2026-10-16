"""Command line entry point.

    python -m memoire_rig serve [--host 127.0.0.1] [--port 8765]
    python -m memoire_rig cam --source me.jpg [--camera 0] [--virtualcam]
    python -m memoire_rig rig --source me.jpg       # print the auto-rig as JSON
"""

import argparse
import json
import sys
import time


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

    args = ap.parse_args()
    from .engine import RigEngine

    engine = RigEngine(device=args.device)
    {"serve": cmd_serve, "cam": cmd_cam, "rig": cmd_rig}[args.cmd](args, engine)


if __name__ == "__main__":
    main()
