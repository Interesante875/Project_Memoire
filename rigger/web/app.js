// Memoire Rig studio: uploads a portrait, shows the auto-rig, streams the
// performer's webcam to the LivePortrait server and draws the avatar frames.

const $ = (id) => document.getElementById(id);
const els = {
  device: $("device"), drop: $("drop"), file: $("file"), snap: $("snap"),
  rigStatus: $("rigStatus"), rigInfo: $("rigInfo"), showRig: $("showRig"),
  cam: $("cam"), calib: $("calib"), mic: $("mic"), idle: $("idle"),
  eyeOn: $("eyeOn"), mouthOn: $("mouthOn"), reset: $("reset"),
  rec: $("rec"), png: $("png"), full: $("full"),
  source: $("source"), out: $("out"), outFrame: $("outFrame"),
  fps: $("fps"), video: $("video"), log: $("log"),
};
const sliders = [...document.querySelectorAll("input[data-c]")];
const defaults = Object.fromEntries(sliders.map((s) => [s.dataset.c, s.value]));

const state = {
  rig: null,
  sourceImg: null,
  ws: null,
  camOn: false,
  inFlight: false,
  mic: null,
  recorder: null,
  controlsDirty: true,
};

const log = (msg) => (els.log.textContent = msg);

// ---------------------------------------------------------------- server
async function refreshStatus() {
  try {
    const s = await (await fetch("api/status")).json();
    els.device.textContent = `engine: ${s.device}`;
    els.device.classList.toggle("bad", s.device === "cpu");
    if (s.device === "cpu") els.device.title = "No GPU found: expect a few frames per second at most.";
    if (s.rigged) await showRig(s.rig);
  } catch {
    els.device.textContent = "server offline";
    els.device.classList.add("bad");
    log("Start the backend with:  python -m memoire_rig serve");
  }
}

function connect() {
  const url = new URL("ws", location.href);
  url.protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(url);
  ws.binaryType = "blob";
  ws.onopen = () => {
    state.ws = ws;
    state.inFlight = false;
    state.controlsDirty = true;
    pump();
  };
  ws.onmessage = async (ev) => {
    if (typeof ev.data === "string") {
      const m = JSON.parse(ev.data);
      if (m.type === "stats") {
        els.fps.textContent = `${m.fps} fps` + (state.camOn && !m.face ? " · no face" : "");
      } else if (m.type === "error") {
        state.inFlight = false;
        setTimeout(pump, 500);
      }
      return;
    }
    await drawFrame(ev.data);
    state.inFlight = false;
    pump();
  };
  ws.onclose = () => {
    state.ws = null;
    setTimeout(connect, 1500);
  };
}

async function drawFrame(blob) {
  const bmp = await createImageBitmap(blob);
  const c = els.out;
  if (c.width !== bmp.width || c.height !== bmp.height) {
    c.width = bmp.width;
    c.height = bmp.height;
  }
  c.getContext("2d").drawImage(bmp, 0, 0);
  bmp.close();
}

// ---------------------------------------------------------------- controls
function controls() {
  const c = { type: "controls", idle: els.idle.checked };
  for (const s of sliders) c[s.dataset.c] = parseFloat(s.value);
  if (!els.eyeOn.checked) c.eye_open = null;
  if (!els.mouthOn.checked) c.mouth_open = null;
  if (state.mic) {
    const level = state.mic.level();
    c.mouth_open = Math.max(c.mouth_open ?? 0, level);
    if (c.mouth_open < 0.02 && !els.mouthOn.checked) c.mouth_open = null;
  }
  return c;
}

sliders.forEach((s) => s.addEventListener("input", () => (state.controlsDirty = true)));
[els.idle, els.eyeOn, els.mouthOn].forEach((e) => e.addEventListener("change", () => (state.controlsDirty = true)));
els.reset.onclick = () => {
  for (const s of sliders) s.value = defaults[s.dataset.c];
  els.eyeOn.checked = els.mouthOn.checked = false;
  state.controlsDirty = true;
};

// One request in flight at a time: the server answers each with a frame.
const grab = document.createElement("canvas");
async function pump() {
  const ws = state.ws;
  if (!ws || state.inFlight || !state.rig) return;
  state.inFlight = true;
  if (state.controlsDirty || state.mic) {
    ws.send(JSON.stringify(controls()));
    state.controlsDirty = false;
  }
  const v = els.video;
  if (state.camOn && v.readyState >= 2) {
    const scale = Math.min(1, 640 / v.videoWidth);
    grab.width = Math.round(v.videoWidth * scale);
    grab.height = Math.round(v.videoHeight * scale);
    grab.getContext("2d").drawImage(v, 0, 0, grab.width, grab.height);
    const blob = await new Promise((r) => grab.toBlob(r, "image/jpeg", 0.8));
    ws.send(await blob.arrayBuffer());
  } else {
    ws.send(JSON.stringify({ type: "tick" }));
  }
}

// ---------------------------------------------------------------- source
async function uploadSource(blob) {
  els.rigStatus.textContent = "Detecting face and building the rig…";
  els.rigInfo.hidden = true;
  const fd = new FormData();
  fd.append("file", blob, "source.png");
  const res = await fetch("api/source", { method: "POST", body: fd });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    els.rigStatus.textContent = body.detail || `Rigging failed (${res.status}).`;
    return;
  }
  await showRig(body);
}

async function showRig(rig) {
  state.rig = rig;
  const img = new Image();
  img.src = `api/source.jpg?t=${Date.now()}`;
  await img.decode();
  state.sourceImg = img;
  drawSource();
  const p = rig.pose;
  els.rigStatus.textContent = `Rigged${rig.rig_ms ? ` in ${(rig.rig_ms / 1000).toFixed(1)} s` : ""}. Ready to perform.`;
  els.rigInfo.innerHTML = `
    <dt>Landmarks</dt><dd>${rig.landmarks.length} face points</dd>
    <dt>Motion rig</dt><dd>${rig.keypoints.length} implicit 3D keypoints</dd>
    <dt>Head pose</dt><dd>yaw ${p.yaw}°, pitch ${p.pitch}°, roll ${p.roll}°</dd>
    <dt>Eyes / lips</dt><dd>open ${rig.eye_ratio} / ${rig.lip_ratio}</dd>`;
  els.rigInfo.hidden = false;
  [els.rec, els.png, els.full].forEach((b) => (b.disabled = false));
  state.inFlight = false;
  pump();
}

function drawSource() {
  const img = state.sourceImg;
  if (!img) return;
  const c = els.source, ctx = c.getContext("2d"), rig = state.rig;
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);
  if (!els.showRig.checked || !rig) return;
  const u = Math.max(1.5, c.width / 400);
  ctx.strokeStyle = "rgba(255,255,255,0.8)";
  ctx.setLineDash([6 * u, 4 * u]);
  ctx.lineWidth = u;
  ctx.beginPath();
  rig.crop.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(255,214,120,0.9)";
  for (const [x, y] of rig.landmarks) ctx.fillRect(x - u / 2, y - u / 2, u, u);
  ctx.fillStyle = "#e3343d";
  ctx.strokeStyle = "#fff";
  for (const [x, y] of rig.keypoints) {
    ctx.beginPath();
    ctx.arc(x, y, 2.6 * u, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}
els.showRig.onchange = drawSource;

els.file.onchange = () => els.file.files[0] && uploadSource(els.file.files[0]);
els.drop.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && els.file.click());
["dragenter", "dragover"].forEach((t) =>
  els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) =>
  els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.remove("over"); }));
els.drop.addEventListener("drop", (e) => {
  const f = [...e.dataTransfer.files].find((f) => f.type.startsWith("image/"));
  if (f) uploadSource(f);
});
window.addEventListener("paste", (e) => {
  const item = [...e.clipboardData.items].find((i) => i.type.startsWith("image/"));
  if (item) uploadSource(item.getAsFile());
});

els.snap.onclick = async () => {
  await ensureCamera();
  const v = els.video;
  grab.width = v.videoWidth;
  grab.height = v.videoHeight;
  grab.getContext("2d").drawImage(v, 0, 0);
  uploadSource(await new Promise((r) => grab.toBlob(r, "image/png")));
};

// ---------------------------------------------------------------- performer
async function ensureCamera() {
  if (els.video.srcObject) return;
  els.video.srcObject = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  await els.video.play();
}

els.cam.onclick = async () => {
  if (state.camOn) {
    state.camOn = false;
    els.video.srcObject?.getTracks().forEach((t) => t.stop());
    els.video.srcObject = null;
    els.cam.textContent = "Start webcam";
    els.cam.classList.remove("on");
    els.calib.disabled = true;
    return;
  }
  try {
    await ensureCamera();
  } catch (e) {
    log(`Camera unavailable: ${e.message}`);
    return;
  }
  state.camOn = true;
  els.cam.textContent = "Stop webcam";
  els.cam.classList.add("on");
  els.calib.disabled = false;
  state.ws?.send(JSON.stringify({ type: "calibrate" }));
  log("Hold a neutral face for a moment — the first frame becomes the rest pose.");
};
els.calib.onclick = () => state.ws?.send(JSON.stringify({ type: "calibrate" }));

class Mic {
  static async start() {
    const m = new Mic();
    m.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    m.ctx = new AudioContext();
    m.analyser = m.ctx.createAnalyser();
    m.analyser.fftSize = 1024;
    m.ctx.createMediaStreamSource(m.stream).connect(m.analyser);
    m.buf = new Float32Array(m.analyser.fftSize);
    m.value = 0;
    return m;
  }
  level() {
    this.analyser.getFloatTimeDomainData(this.buf);
    let sum = 0;
    for (const v of this.buf) sum += v * v;
    const rms = Math.sqrt(sum / this.buf.length);
    const target = Math.min(1, Math.max(0, (rms - 0.015) / 0.1));
    this.value += (target - this.value) * (target > this.value ? 0.6 : 0.25);
    return this.value;
  }
  stop() {
    this.stream.getTracks().forEach((t) => t.stop());
    this.ctx.close();
  }
}

els.mic.onclick = async () => {
  if (state.mic) {
    state.mic.stop();
    state.mic = null;
    els.mic.classList.remove("on");
    state.controlsDirty = true;
    return;
  }
  try {
    state.mic = await Mic.start();
    els.mic.classList.add("on");
  } catch (e) {
    log(`Microphone unavailable: ${e.message}`);
  }
};

// ---------------------------------------------------------------- output
els.png.onclick = () => {
  els.out.toBlob((b) => download(b, "memoire-avatar.png"), "image/png");
};
els.full.onclick = () => els.outFrame.requestFullscreen?.();
els.rec.onclick = () => {
  if (state.recorder) {
    state.recorder.stop();
    return;
  }
  const stream = els.out.captureStream(30);
  if (state.mic) stream.addTrack(state.mic.stream.getAudioTracks()[0]);
  const type = ["video/webm;codecs=vp9", "video/webm", "video/mp4"].find((t) => MediaRecorder.isTypeSupported(t));
  const rec = new MediaRecorder(stream, { mimeType: type });
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = () => {
    download(new Blob(chunks, { type: rec.mimeType }), `memoire-avatar.${rec.mimeType.includes("mp4") ? "mp4" : "webm"}`);
    state.recorder = null;
    els.rec.textContent = "Record clip";
    els.rec.classList.remove("on");
  };
  rec.start();
  state.recorder = rec;
  els.rec.textContent = "Stop recording";
  els.rec.classList.add("on");
};

function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

refreshStatus();
connect();
