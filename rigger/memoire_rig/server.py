"""HTTP + WebSocket server for the browser rigging studio."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .engine import Controls, NoFaceError, RigEngine, decode_image, encode_jpeg
from .session import Session

WEB = Path(__file__).resolve().parent.parent / "web"


def create_app(engine: RigEngine) -> FastAPI:
    app = FastAPI(title="Memoire Rig")
    session = Session(engine)
    # One GPU, one model: serialize inference across clients.
    gpu = asyncio.Lock()

    @app.get("/api/status")
    def status():
        return {
            "device": engine.device,
            "rigged": engine.src is not None,
            "rig": engine.src.summary if engine.src else None,
        }

    @app.post("/api/source")
    async def upload_source(file: UploadFile = File(...)):
        data = await file.read()
        try:
            img = decode_image(data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        async with gpu:
            try:
                t = time.perf_counter()
                rig = await asyncio.to_thread(engine.load_source, img)
                rig["rig_ms"] = round((time.perf_counter() - t) * 1000)
            except NoFaceError as e:
                raise HTTPException(422, str(e))
        session.calibrate()
        return rig

    @app.get("/api/source.jpg")
    def source_image():
        if engine.src is None:
            raise HTTPException(404, "No source rigged yet.")
        return Response(encode_jpeg(engine.src.img_rgb, 92), media_type="image/jpeg")

    @app.websocket("/ws")
    async def drive(ws: WebSocket):
        """Lock-step protocol. Client sends either a binary webcam JPEG or a
        text message; every frame request gets one binary JPEG back.

        text: {"type": "controls", ...Controls fields}
              {"type": "calibrate"}
              {"type": "tick"}   render with no performer (idle / sliders)
        """
        await ws.accept()
        last = time.perf_counter()
        fps = 0.0
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                frame = None
                if msg.get("text") is not None:
                    data = json.loads(msg["text"])
                    kind = data.pop("type", "")
                    if kind == "controls":
                        session.controls = Controls.from_dict(data)
                        continue
                    if kind == "calibrate":
                        session.calibrate()
                        continue
                    if kind != "tick":
                        continue
                elif msg.get("bytes"):
                    frame = decode_image(msg["bytes"])
                if engine.src is None:
                    await ws.send_text(json.dumps({"type": "error", "message": "Upload a source image first."}))
                    continue
                async with gpu:
                    out = await asyncio.to_thread(session.step, frame)
                    jpg = await asyncio.to_thread(encode_jpeg, out)
                now = time.perf_counter()
                fps = fps * 0.9 + 0.1 / max(1e-3, now - last)
                last = now
                await ws.send_bytes(jpg)
                await ws.send_text(json.dumps({"type": "stats", "fps": round(fps, 1), "face": session.face_found}))
        except WebSocketDisconnect:
            pass

    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    return app
