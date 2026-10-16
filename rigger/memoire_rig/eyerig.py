"""Detailed eye capture and a Live2D-style eye rig built from the drawing.

From the open eye in the source art (at output resolution) each eye is split
into its drawn parts:

    upper_lash   the thick lash line (widest very-dark shape, incl. its
                 coloured corner accents), clipped to lash thickness
    lower_line   the thin lower lid line
    iris         the coloured iris disc, including pupil detail
    highlights   specular dots inside the iris (stay put when the iris moves)
    sclera       eye white, kept as a per-row tone gradient (lash shadow)
    opening      the visible eye opening, column by column between the lash
                 and the lower lid
    strands      hair line-art crossing the eye, kept on top of everything

The rig then redraws the eye every frame: the lid comes down column by column
carrying the real lash, the iris (with pupil) shifts for gaze and scales for
surprise inside the opening, highlights stay fixed, strands stay on top, and
at full closure the lash settles into "︶" (blink) or "^" (happy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .eyes import Eye


def _comp_largest(mask: np.ndarray, key="area"):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    idx = {"area": cv2.CC_STAT_AREA, "width": cv2.CC_STAT_WIDTH}[key]
    best = max(range(1, n), key=lambda k: st[k, idx])
    return lab == best


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8) * 255
    h, w = m.shape
    flood = m.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return (m | cv2.bitwise_not(flood)) > 0


@dataclass
class EyeParts:
    eye: Eye                 # rest eye in output coordinates
    x0: int                  # patch origin (output coordinates)
    y0: int
    patch: np.ndarray        # rest RGBA patch
    region: np.ndarray       # float mask of what the rig redraws
    lash: np.ndarray         # bool masks, patch-sized
    lower: np.ndarray
    iris: np.ndarray
    highlights: np.ndarray
    strands: np.ndarray
    top: np.ndarray          # per column: first row of the opening (or -1)
    bot: np.ndarray          # per column: last row of the opening
    sclera_rows: np.ndarray  # per row RGB eye-white tone
    skin: np.ndarray
    iris_center: tuple[float, float] = (0.0, 0.0)
    travel: tuple[float, float] = (0.0, 0.0)
    debug: dict = field(default_factory=dict)


def _robust_poly(x: np.ndarray, y: np.ndarray, deg: int = 2, iters: int = 3):
    """Polynomial fit with iterative outlier trimming."""
    keep = np.ones(len(x), bool)
    for _ in range(iters):
        if keep.sum() <= deg + 2:
            break
        c = np.polyfit(x[keep], y[keep], deg)
        r = np.abs(np.polyval(c, x) - y)
        thr = max(1.0, 2.0 * np.median(r[keep]))
        keep = r <= thr
    return np.polyfit(x[keep], y[keep], deg) if keep.sum() > deg else np.polyfit(x, y, min(deg, len(x) - 1))


def capture(rest: np.ndarray, e: Eye) -> EyeParts | None:
    """Captures one open eye (output resolution) as clean fitted shapes:
    iris ellipse, lash underside and lower-lid curves, lash pixels above the
    underside (spikes kept), highlights inside the iris."""
    H, W = rest.shape[:2]
    x0, x1 = int(max(0, e.cx - e.rx * 1.6)), int(min(W, e.cx + e.rx * 1.6))
    y0, y1 = int(max(0, e.cy - e.ry * 1.7)), int(min(H, e.cy + e.ry * 1.5))
    P = rest[y0:y1, x0:x1].copy()
    h, w = P.shape[:2]
    rgb = P[..., :3].astype(np.float32)
    sm = cv2.GaussianBlur(rgb, (0, 0), max(0.8, e.ry * 0.04))  # de-noise for decisions
    lum = sm.mean(-1)
    hsv = cv2.cvtColor(np.clip(sm, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    sat, hue = hsv[..., 1], hsv[..., 0]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = e.cx - x0, e.cy - y0
    ell = ((xs - cx) / (e.rx * 1.2)) ** 2 + ((ys - cy) / (e.ry * 1.15)) ** 2 <= 1

    # --- iris: ellipse fitted to the dominant saturated hue
    cand = ell & (sat > 70) & (lum > 30)
    if cand.sum() < 20:
        return None
    hh = np.median(hue[cand])
    dh = np.minimum(np.abs(hue - hh), 180 - np.abs(hue - hh))
    g = _comp_largest(ell & (dh < 25) & (sat > 40))
    cnts, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts or len(max(cnts, key=len)) < 6:
        return None
    (icx, icy), (iw, ih), iang = cv2.fitEllipse(max(cnts, key=len))
    iris_ell = np.zeros((h, w), np.uint8)
    cv2.ellipse(iris_ell, ((icx, icy), (iw, ih), iang), 1, -1, cv2.LINE_AA)
    iris_ell = iris_ell > 0

    # --- lash underside: per column, bottom of the first dark run from the
    #     top; fitted with a smooth curve across the eye
    green = (dh < 25) & (sat > 45)
    reddish = (sm[..., 0] > sm[..., 1] + 18) & (lum < 130)
    dark = ((lum < 90) | reddish) & ~green & ell & (ys < cy + e.ry * 0.3)
    xs_l, ys_l = [], []
    for c in range(w):
        r = np.nonzero(dark[:, c])[0]
        if not len(r):
            continue
        end = r[0]
        while end + 1 < h and dark[end + 1, c]:
            end += 1
        xs_l.append(c)
        ys_l.append(end + 1)
    if len(xs_l) < 6:
        return None
    xs_l, ys_l = np.array(xs_l, float), np.array(ys_l, float)
    c_lash = _robust_poly(xs_l, ys_l, 2)

    # --- lower lid: per column, last non-skin pixel below the iris centre
    skinlike = (sm[..., 0] > sm[..., 2] + 6) & (lum > 190) & (sat < 60)
    xs_b, ys_b = [], []
    for c in range(w):
        col = ell[:, c] & ~skinlike[:, c] & (ys[:, c] > icy)
        r = np.nonzero(col)[0]
        if len(r):
            xs_b.append(c)
            ys_b.append(r.max())
    c_low = _robust_poly(np.array(xs_b, float), np.array(ys_b, float), 2) if len(xs_b) > 5 else None
    xx = np.arange(w, dtype=np.float32)
    top_c = np.polyval(c_lash, xx)
    bot_c = np.polyval(c_low, xx) if c_low is not None else np.full(w, icy + ih / 2)
    inside_x = (xx > xs_l.min()) & (xx < xs_l.max()) & (bot_c > top_c + 1)
    eyeball = inside_x[None, :] & (ys >= top_c[None, :]) & (ys <= bot_c[None, :])
    eyeball = cv2.morphologyEx(eyeball.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    iris = iris_ell & eyeball
    sclera = eyeball & ~iris

    # --- lash sprite: dark pixels from the top down to the underside curve,
    #     connected to the main lash (spikes included), soft alpha
    band = dark | (((lum < 120) | reddish) & ~green & (ys < top_c[None, :] + 1))
    band &= ys < top_c[None, :] + 1.5
    lash = _comp_largest(band, "width")
    lash = cv2.dilate(lash.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    lash &= (lum < 170) & ~eyeball

    highlights = iris & (lum > 215) & (sat < 45)
    lower = (lum < 150) & ~eyeball & ~lash & ell & (ys > bot_c[None, :] - 1) & (ys < bot_c[None, :] + e.ry * 0.25)

    top = np.where(inside_x, np.round(top_c), -1).astype(int)
    bot = np.where(inside_x, np.round(bot_c), -1).astype(int)

    # eye-white tone per row, from its bright pixels only (the corners carry
    # dark lash accents that must not tint the white)
    bright = sclera & (lum > 200) & (sat < 50)
    fallback = np.median(rgb[bright], 0) if bright.sum() > 5 else np.array([245, 245, 250], np.float32)
    rows = np.tile(fallback, (h, 1)).astype(np.float32)
    for r in range(h):
        sr = bright[r]
        if sr.sum() >= 3:
            rows[r] = np.median(rgb[r][sr], 0)
    if h > 7:
        rows = cv2.GaussianBlur(rows[:, None, :], (1, 9), 0).reshape(h, 3)
    cheek = rgb[int(min(h - 1, cy + e.ry * 1.25)):h, int(cx - e.rx * 0.4):int(cx + e.rx * 0.4)].reshape(-1, 3)
    skin = np.median(cheek, 0) if len(cheek) else np.array([250, 236, 226], np.float32)
    # what is behind the lash: filled from its surroundings, so the lash can
    # leave its place without a flat patch (it often overlaps hair)
    eyeball = cv2.dilate(eyeball.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    eyeball[:max(0, int(np.nanmin(np.where(inside_x, top_c, np.inf))) - 1)] = False
    iris &= eyeball
    hole = cv2.dilate((lash | eyeball).astype(np.uint8), np.ones((3, 3), np.uint8))
    behind = cv2.inpaint(np.ascontiguousarray(P[..., :3]), hole, 7, cv2.INPAINT_TELEA).astype(np.float32)
    # smooth local skin field: skin-like pixels around the eye, spread by
    # normalised convolution, so lids match the surrounding tone and shading
    near = cv2.dilate((eyeball | lash | lower).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    src = skinlike & ~near
    sig = max(2.0, e.ry * 0.5)
    num = cv2.GaussianBlur(rgb * src[..., None], (0, 0), sig)
    den = cv2.GaussianBlur(src.astype(np.float32), (0, 0), sig)[..., None]
    skin_field = np.where(den > 1e-3, num / np.maximum(den, 1e-3), skin)

    ew = inside_x.sum()
    travel = (max(1.0, (ew - iw) * 0.5 + iw * 0.1), e.ry * 0.1)
    region = cv2.dilate((eyeball | lash | lower).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    region = cv2.GaussianBlur(region, (0, 0), 1.2)
    strands = np.zeros((h, w), bool)
    return EyeParts(e, x0, y0, P, region, lash, lower, iris, highlights, strands, top, bot,
                    rows, skin, (icx, icy), travel,
                    debug={"opening": eyeball, "sclera": sclera, "eyeball": eyeball, "behind": behind,
                           "skin_field": skin_field,
                           "curves": (top_c, bot_c, inside_x), "iris_ellipse": ((icx, icy), (iw, ih), iang)})


def _shift(img: np.ndarray, dx: float, dy: float, scale: float, center) -> np.ndarray:
    cx, cy = center
    M = np.float32([[scale, 0, cx - scale * cx + dx], [0, scale, cy - scale * cy + dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def draw(parts: EyeParts, k: float, happy: bool, gaze=(0.0, 0.0), iris_scale: float = 1.0,
         skin=None) -> np.ndarray:
    """Composes the eye patch (RGBA float) for closure k, gaze in [-1, 1]^2.
    Only the eyeball (iris + white) and the lash are ever changed."""
    P = parts.patch.astype(np.float32)
    h, w = P.shape[:2]
    rgb = P[..., :3].copy()
    skin = parts.debug["skin_field"] if skin is None else skin
    eyeball = parts.debug["eyeball"]
    top, bot = parts.top.astype(np.float32), parts.bot.astype(np.float32)
    valid = np.nonzero(parts.top >= 0)[0]
    if not len(valid):
        return np.dstack([rgb, P[..., 3]])
    # columns without eyeball (e.g. lash spikes past the corners) borrow the
    # nearest eyeball column's lid travel
    near = valid[np.abs(np.arange(w)[:, None] - valid[None, :]).argmin(1)]
    span = (bot - top + 1)[near]
    kk = min(1.0, k)
    off = kk * span                       # how far the lid has come down
    lid = top[near] + off
    ys = np.arange(h, dtype=np.float32)[:, None]
    visible = eyeball & (ys >= lid[None, :])
    covered = eyeball & ~visible

    # 1. iris + pupil: lift it off (paint eye white), redraw moved/scaled
    if gaze != (0.0, 0.0) or iris_scale != 1.0:
        sclera = parts.sclera_rows[:, None, :] * np.ones((1, w, 1), np.float32)
        rgb = np.where((parts.iris & ~parts.strands)[..., None], sclera, rgb)
        dx, dy = gaze[0] * parts.travel[0], gaze[1] * parts.travel[1]
        iris_rgba = np.dstack([P[..., :3], parts.iris.astype(np.float32) * 255])
        moved = _shift(iris_rgba, dx, dy, iris_scale, parts.iris_center)
        a = (moved[..., 3:4] / 255) * (eyeball & ~parts.strands)[..., None]
        rgb = rgb * (1 - a) + moved[..., :3] * a
        # highlights stay where the light is
        hl = parts.highlights & (moved[..., 3] > 128)
        rgb = np.where(hl[..., None], P[..., :3], rgb)

    # 2. the lid: skin over the covered eyeball
    rgb = np.where(covered[..., None], skin, rgb)

    # 3. the real lash rides down with the lid, column by column
    lash_rgb, lash_a = np.zeros_like(rgb), np.zeros((h, w), np.float32)
    for c in np.nonzero(parts.lash.any(0))[0]:
        o = int(round(off[c]))
        src = parts.lash[: h - o, c]
        lash_a[o:, c] = src
        lash_rgb[o:, c] = P[: h - o, c, :3]
    settle = float(np.clip((k - 0.8) / 0.2, 0, 1))  # final shape: "︶" or "^"
    if settle > 0:
        lash_rgba = np.dstack([P[..., :3], parts.lash.astype(np.float32)])
        r = np.nonzero(parts.lash.any(1))[0]
        band = (r[0] + r[-1]) / 2
        target = parts.eye.cy - parts.y0 + parts.eye.ry * (-0.05 if happy else 0.3)
        L = lash_rgba if happy else lash_rgba[::-1].copy()
        bc = band if happy else (h - 1 - band)
        sy = 0.9 if happy else 0.8
        M = np.float32([[1, 0, 0], [0, sy, target - sy * bc]])
        fin = cv2.warpAffine(L, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=0)
        lash_rgb = lash_rgb * (1 - settle) + fin[..., :3] * settle
        lash_a = lash_a * (1 - settle) + fin[..., 3] * settle
        # closed: the whole eye becomes smooth lid skin (feathered edge)
        lid_area = cv2.dilate((eyeball | parts.lower).astype(np.uint8), np.ones((5, 3), np.uint8))
        soft = cv2.GaussianBlur(lid_area.astype(np.float32), (0, 0), 1.5)
        soft = np.maximum(soft, eyeball)[..., None] * settle
        rgb = rgb * (1 - soft) + skin * soft
    # the lash leaves its old place (revealing what is behind it), then is drawn
    if kk > 0:
        gone = cv2.dilate(parts.lash.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~eyeball
        rgb = np.where(gone[..., None], parts.debug["behind"], rgb)
    a = np.clip(lash_a, 0, 1)[..., None]
    rgb = rgb * (1 - a) + lash_rgb * a

    # 4. lower lid line (fades as the eye closes) and hair strands on top
    la = parts.lower[..., None] * (1 - settle)
    rgb = rgb * (1 - la) + P[..., :3] * la
    rgb = np.where(parts.strands[..., None], P[..., :3], rgb)
    return np.dstack([rgb, P[..., 3]])


def render(img: np.ndarray, parts: EyeParts, T: np.ndarray, k: float, happy: bool,
           gaze=(0.0, 0.0), iris_scale: float = 1.0) -> np.ndarray:
    """Draws the rigged eye into the output frame. T maps rest output
    coordinates to current output coordinates (2x3 similarity)."""
    patch = draw(parts, k, happy, gaze, iris_scale)
    H, W = img.shape[:2]
    A = np.float32([[1, 0, parts.x0], [0, 1, parts.y0]])
    M = (np.vstack([T, [0, 0, 1]]) @ np.vstack([A, [0, 0, 1]]))[:2]
    warped = cv2.warpAffine(patch, M, (W, H), flags=cv2.INTER_CUBIC, borderValue=0)
    m = cv2.warpAffine(parts.region, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
    out = img.astype(np.float32)
    out[..., :3] = out[..., :3] * (1 - m) + warped[..., :3] * m
    return np.clip(out, 0, 255).astype(np.uint8)


def debug_view(parts: EyeParts) -> np.ndarray:
    """Colour-coded capture: lash red, lower line orange, iris green,
    highlights white, sclera blue, strands magenta."""
    v = parts.patch[..., :3].astype(np.float32) * 0.35
    for m, col in ((parts.debug["sclera"], (90, 140, 255)), (parts.iris, (60, 220, 90)),
                   (parts.highlights, (255, 255, 255)), (parts.lower, (255, 160, 40)),
                   (parts.lash, (255, 50, 60)), (parts.strands, (230, 70, 230))):
        v[m] = col
    v = v.astype(np.uint8)
    if "curves" in parts.debug:
        top_c, bot_c, inside = parts.debug["curves"]
        xs = np.nonzero(inside)[0]
        for curve, col in ((top_c, (255, 255, 0)), (bot_c, (0, 255, 255))):
            pts = np.stack([xs, curve[xs]], 1).astype(np.int32)
            cv2.polylines(v, [pts], False, col, 1, cv2.LINE_AA)
        cv2.ellipse(v, parts.debug["iris_ellipse"], (255, 255, 255), 1, cv2.LINE_AA)
    return v
