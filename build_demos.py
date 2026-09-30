"""Build homepage demo clips.

Real clips: third-person iPhone video, with first-person stills inset at the
top left. Stills come from action_frames and switch at image_captured_at.
The phone video's creation_time is the timeline zero, matched against the
absolute timestamps in atomic_actions.jsonl.

Simulation clips are already ego | map | instruction and are only re-encoded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"E:\code\SimpleVLN_Homepage")
OUT = ROOT / "assets" / "demos"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
ROBOT = Path(r"E:\code\robotdata_download\extracted")
SIM_ROOT = Path(r"E:\code\pass4_videos\pass4")

OUT_W, OUT_H = 1280, 720
OUT_FPS = 15
PIP_W, PIP_H = 420, 315
PIP_RADIUS = 22
PIP_MARGIN = 0
PIP_RED = (96, 86, 255)  # BGR, coral red border
STOP_HOLD = 10
LEAD_FRAMES = 4

FONT_CLEAR = Path(r"C:\Windows\Fonts\segoeuib.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")
YOLOX_PATH = ROOT / "_models" / "yolox.onnx"
STOP_PATH = ROOT / "assets" / "stop-hand.png"
ARROW_FORWARD = ROOT / "assets" / "arrow-forward.png"
ARROW_TURN = ROOT / "assets" / "arrow-turn.png"
TEXT_WHITE = (255, 255, 255, 255)
# Highlight and arrows: #E5CCFF at the top, #CCFFFF at the bottom.
GRAD_TOP = (0xE5, 0xCC, 0xFF)
GRAD_BOT = (0xCC, 0xFF, 0xFF)

SIMS = [
    {
        "id": "sim-foyer",
        "title": "Foyer",
        "src": SIM_ROOT / "2azQ1b91cZZ" / "1_success_spl_gt0.85" / "2azQ1b91cZZ_139_trial3.mp4",
    },
    {
        "id": "sim-gallery",
        "title": "Gallery",
        "src": SIM_ROOT / "Z6MFQCViBuw" / "1_success_spl_gt0.85" / "Z6MFQCViBuw_1496_trial0.mp4",
    },
    {
        "id": "sim-kitchen",
        "title": "Kitchen",
        "src": SIM_ROOT / "zsNo4HB9uLZ" / "1_success_spl_gt0.85" / "zsNo4HB9uLZ_52_trial0.mp4",
    },
    {
        "id": "sim-living",
        "title": "Living room",
        "src": SIM_ROOT / "EU6Fwq7SyZv" / "1_success_spl_gt0.85" / "EU6Fwq7SyZv_212_trial3.mp4",
    },
]

# index is the 0-based order of MOV / run folders on that day.
REALS = [
    {
        "id": "real-pillar",
        "title": "Yellow pillar",
        "day": "0907",
        "index": 1,
        "need": "yellow pillar",
        "highlight": ["yellow pillar"],
    },
    {
        "id": "real-kfc",
        "title": "KFC entrance",
        # KFC05 is the fifth phone recording on 0912; use its explicit run.
        "day": "0912",
        "index": 4,
        "run": ROBOT / "0912result" / "trajectory_runs_0912" / "e2e-2turn_20260912_174828_KFC05",
        "need": "kfc",
        "highlight": ["KFC entrance"],
    },
    {
        "id": "real-vending",
        "title": "Vending machine",
        "day": "0912",
        "index": 1,
        "need": "vending machine",
        "highlight": ["vending machine", "orange top"],
    },
    {
        "id": "real-path",
        "title": "Outdoor path",
        "day": "0912",
        "index": 2,
        "need": "stone path",
        "highlight": ["stone path", "blue fitness area"],
    },
    {
        "id": "real-follow",
        "title": "Follow the person",
        "day": "0912",
        "index": 3,
        "need": "person in black",
        "highlight": ["person in black"],
        "trim_tail": 8.0,
        "blur_person": True,
        # Output time. The later part of this take wanders, so it ends here.
        "out_end": 67.0,
    },
    {
        "id": "real-car",
        "title": "Red car",
        "day": "0912",
        "index": 5,
        "need": "red car",
        "highlight": ["pillar", "red car"],
        "expose": 1.35,
    },
    {
        "id": "real-office",
        "title": "Office hallway",
        "day": "0912",
        "index": 0,
        "need": "first room",
        "highlight": ["first room"],
        "blur_person": True,
        "playback": 2,
    },
]


def choose_speed(duration: float) -> int:
    if duration <= 120:
        return 1
    for factor in (2, 4, 6, 8):
        if duration / factor <= 120.05:
            return factor
    return 8


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def mov_start(path: Path) -> datetime:
    proc = subprocess.run(
        [FFMPEG, "-i", str(path), "-f", "ffmetadata", "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    blob = proc.stdout + "\n" + proc.stderr
    for line in blob.splitlines():
        if "creation_time" in line and "T" in line:
            raw = line.split(":", 1)[-1].strip()
            # 2026-09-06T14:45:45.000000Z
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            return parse_time(raw)
    raise RuntimeError(f"no creation_time in {path}")


def runs_for(day: str) -> list[Path]:
    found = sorted(
        p
        for p in (ROBOT / f"{day}result").rglob("*")
        if p.is_dir() and (p / "atomic_actions.jsonl").is_file()
    )
    return found


def movs_for(day: str) -> list[Path]:
    return sorted((ROBOT / f"{day}result").rglob("*.MOV"))


def load_events(run: Path) -> tuple[str, list[tuple[float, Path]]]:
    rows = [
        json.loads(line)
        for line in (run / "atomic_actions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    instruction = (run / "instruction.txt").read_text(encoding="utf-8").strip()
    events = []
    for row in rows:
        stamp = parse_time(row["image_captured_at"])
        image = run / row["after_image"].replace("\\", "/")
        if not image.is_file():
            raise FileNotFoundError(image)
        events.append((stamp, image, row.get("action", "")))
    events.sort(key=lambda item: item[0])
    return instruction, events


def resolve_real(spec: dict) -> dict:
    runs = runs_for(spec["day"])
    movs = movs_for(spec["day"])
    run = spec.get("run", runs[spec["index"]])
    mov = movs[spec["index"]]
    instruction, events = load_events(run)
    if spec["need"].lower() not in instruction.lower():
        raise RuntimeError(f"{spec['id']} paired with unexpected instruction: {instruction}")
    start = mov_start(mov)
    cap = cv2.VideoCapture(str(mov))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    duration = frames / fps
    duration = max(1.0, duration - float(spec.get("trim_tail", 0.0)))
    timed = [
        ((stamp - start).total_seconds(), image, action)
        for stamp, image, action in events
    ]
    speed = int(spec["playback"]) if spec.get("playback") else choose_speed(duration)
    lead = LEAD_FRAMES * speed / OUT_FPS
    first_t = timed[0][0]
    last_t = timed[-1][0]
    # Keep only a few frames before the first action and first-person still.
    clip_start = 0.0 if first_t <= 0 else max(0.0, first_t - lead)
    clip_end = duration if last_t <= 0 else min(duration, max(last_t, clip_start) + lead)
    if spec.get("out_end") is not None:
        clip_end = min(clip_end, float(spec["out_end"]) * speed)
    clip_end = max(clip_end, clip_start + 1.0)
    print(
        f"{spec['id']}: {mov.name} {clip_start:.1f}-{clip_end:.1f}s -> {speed}x "
        f"= {(clip_end - clip_start) / speed:.1f}s | "
        f"ego {first_t:.1f}s..{last_t:.1f}s | {len(timed)} stills | {instruction}"
    )
    return {
        **spec,
        "mov": mov,
        "instruction": instruction,
        "events": timed,
        "duration": duration,
        "clip_start": clip_start,
        "clip_end": clip_end,
        "fps": fps,
        "speed": speed,
    }


def load_bgr(path: Path) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    return image


def trim_bars(image: np.ndarray) -> np.ndarray:
    """Drop a uniform black letterbox. Night scenes with real detail are kept."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    def edge_is_bar(strip: np.ndarray) -> bool:
        return float(strip.mean()) < 4 and float(strip.std()) < 2

    top = 0
    while top < height // 6 and edge_is_bar(gray[top : top + 1]):
        top += 1
    bottom = height
    while bottom > height * 5 // 6 and edge_is_bar(gray[bottom - 1 : bottom]):
        bottom -= 1
    left = 0
    while left < width // 6 and edge_is_bar(gray[:, left : left + 1]):
        left += 1
    right = width
    while right > width * 5 // 6 and edge_is_bar(gray[:, right - 1 : right]):
        right -= 1
    if right - left < 32 or bottom - top < 32:
        return image
    return image[top:bottom, left:right]


def cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale to fill the window. No side bars inside the inset."""
    src_h, src_w = image.shape[:2]
    scale = max(width / src_w, height / src_h)
    resized_w = max(width, int(round(src_w * scale)))
    resized_h = max(height, int(round(src_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    x = (resized_w - width) // 2
    y = (resized_h - height) // 2
    return resized[y : y + height, x : x + width]


def action_kind(text: str) -> str:
    upper = text.upper()
    if "LEFT" in upper:
        return "left"
    if "RIGHT" in upper:
        return "right"
    if "BACK" in upper:
        return "back"
    if "STOP" in upper:
        return "stop"
    return "forward"


def pip_bank(
    events: list[tuple[float, Path, str]], people: PeopleBlur | None = None
) -> tuple[list[tuple[float, np.ndarray]], list[tuple[float, str]]]:
    bank = []
    actions = []
    for t, path, action in events:
        image = trim_bars(load_bgr(path))
        if people is not None:
            people.apply(image, hold=False)
        bank.append((t, cover(image, PIP_W, PIP_H)))
        actions.append((t, action_kind(action)))
    return bank, actions


def still_at(bank: list[tuple[float, object]], t: float):
    """Latest sample at or before t. Nothing is shown before the first one."""
    chosen = None
    for stamp, image in bank:
        if stamp <= t + 1e-3:
            chosen = image
        else:
            break
    return chosen


def font(size: int) -> ImageFont.FreeTypeFont:
    path = FONT_CLEAR if FONT_CLEAR.is_file() else FONT_BOLD
    if not path.is_file():
        path = Path(r"C:\Windows\Fonts\arial.ttf")
    return ImageFont.truetype(str(path), size)


class PeopleBlur:
    """Soft blur on detected people. The robot is left sharp."""

    def __init__(self) -> None:
        sys.path.insert(0, str(ROOT / "_models"))
        from yolox import YoloX

        self.det = YoloX(str(YOLOX_PATH), confThreshold=0.4, nmsThreshold=0.45, objThreshold=0.4)
        self.prev: list[tuple[int, int, int, int]] = []
        self.miss = 0

    def boxes(self, frame: np.ndarray, hold: bool = True) -> list[tuple[int, int, int, int]]:
        height, width = frame.shape[:2]
        resized = cv2.resize(frame, (640, 640)).astype(np.float32)
        raw = self.det.infer(resized)
        found: list[tuple[int, int, int, int]] = []
        if raw is not None and len(raw):
            for row in raw:
                if int(row[5]) != 0 or float(row[4]) < 0.4:
                    continue
                x, y, bw, bh = (float(v) for v in row[:4])
                x1 = int(x / 640 * width)
                y1 = int(y / 640 * height)
                x2 = int((x + bw) / 640 * width)
                y2 = int((y + bh) / 640 * height)
                box_w, box_h = x2 - x1, y2 - y1
                if box_h < 36 or box_w < 16:
                    continue
                # Standing people are taller than they are wide. A low, wide
                # blob is the robot, not a person.
                if y1 > height * 0.58 and box_w > box_h:
                    continue
                pad_x = int(box_w * 0.45)
                pad_top = int(box_h * 0.40)
                pad_bot = int(box_h * 0.28)
                found.append((
                    max(0, x1 - pad_x),
                    max(0, y1 - pad_top),
                    min(width - 1, x2 + pad_x),
                    min(height - 1, y2 + pad_bot),
                ))
        if found:
            self.prev = found
            self.miss = 0
            return found
        if hold and self.prev and self.miss < 4:
            self.miss += 1
            return self.prev
        self.prev = []
        return []

    def apply(self, frame: np.ndarray, hold: bool = True, protect_robot: bool = False) -> None:
        regions = self.boxes(frame, hold=hold)
        if not regions:
            return
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), np.float32)
        for x1, y1, x2, y2 in regions:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ax = max(8, int((x2 - x1) * 0.55))
            ay = max(8, int((y2 - y1) * 0.55))
            cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, thickness=-1)
        # Wide, soft falloff so the blur melts into the background.
        mask = cv2.GaussianBlur(mask, (0, 0), 36)
        if protect_robot:
            shield = np.zeros((height, width), np.float32)
            cv2.ellipse(
                shield,
                (width // 2, int(height * 0.98)),
                (int(width * 0.24), int(height * 0.40)),
                0, 0, 360, 1.0, thickness=-1,
            )
            shield = cv2.GaussianBlur(shield, (0, 0), 14)
            mask *= 1.0 - shield
        mask = np.clip(mask, 0.0, 1.0)
        if float(mask.max()) < 0.04:
            return
        blurred = cv2.GaussianBlur(frame, (0, 0), 26)
        alpha = mask[:, :, None]
        mixed = frame.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha
        frame[:] = mixed.astype(np.uint8)


def segments(text: str, phrases: list[str]) -> list[tuple[str, bool]]:
    lower = text.lower()
    spans: list[tuple[int, int]] = []
    for phrase in phrases:
        needle = phrase.lower()
        start = 0
        while True:
            found = lower.find(needle, start)
            if found < 0:
                break
            spans.append((found, found + len(needle)))
            start = found + len(needle)
    spans.sort()
    parts: list[tuple[str, bool]] = []
    cursor = 0
    for begin, end in spans:
        if begin < cursor:
            continue
        if begin > cursor:
            parts.append((text[cursor:begin], False))
        parts.append((text[begin:end], True))
        cursor = end
    if cursor < len(text):
        parts.append((text[cursor:], False))
    return parts or [(text, False)]


def _paste_gradient_text(canvas: Image.Image, x: float, y: float, text: str,
                         face: ImageFont.FreeTypeFont) -> None:
    probe = ImageDraw.Draw(canvas)
    box = probe.textbbox((0, 0), text, font=face)
    width = max(1, box[2] - box[0] + 2)
    height = max(1, box[3] - box[1] + 2)
    glyph = Image.new("L", (width, height), 0)
    ImageDraw.Draw(glyph).text((-box[0], -box[1]), text, font=face, fill=255)
    alpha = np.array(glyph)
    pixels = np.zeros((height, width, 4), np.uint8)
    for row in range(height):
        mix = row / max(1, height - 1)
        red = int(GRAD_TOP[0] * (1 - mix) + GRAD_BOT[0] * mix)
        green = int(GRAD_TOP[1] * (1 - mix) + GRAD_BOT[1] * mix)
        blue = int(GRAD_TOP[2] * (1 - mix) + GRAD_BOT[2] * mix)
        pixels[row, :, 0] = red
        pixels[row, :, 1] = green
        pixels[row, :, 2] = blue
        pixels[row, :, 3] = alpha[row]
    canvas.alpha_composite(Image.fromarray(pixels, "RGBA"), (int(x + box[0]), int(y + box[1])))


def paint_chunks(canvas: Image.Image, x: float, y: float, text: str, origin: int,
                 face: ImageFont.FreeTypeFont, spans: list[tuple[int, int]]) -> None:
    """White body, with keywords in a top-to-bottom lavender-to-cyan gradient."""
    draw = ImageDraw.Draw(canvas)
    cursor = 0
    while cursor < len(text):
        hot = any(begin <= origin + cursor < end for begin, end in spans)
        nxt = cursor + 1
        while nxt < len(text):
            hot_next = any(begin <= origin + nxt < end for begin, end in spans)
            if hot_next != hot:
                break
            nxt += 1
        chunk = text[cursor:nxt]
        draw.text((x + 1, y + 1), chunk, font=face, fill=(0, 0, 0, 170))
        if hot:
            _paste_gradient_text(canvas, x, y, chunk, face)
        else:
            draw.text((x, y), chunk, font=face, fill=TEXT_WHITE)
        x += draw.textlength(chunk, font=face)
        cursor = nxt


def render_overlay(instruction: str, phrases: list[str], speed: int) -> np.ndarray:
    """Bottom caption in white and blue, with the speed badge at the top right.

    The top-left stays empty so the first-person window can sit flush there.
    """
    canvas = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    body = font(40)
    badge = font(32)
    max_w = OUT_W - 56
    words = instruction.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=body) <= max_w:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    line_h = 48
    band_top = OUT_H - 20 - len(lines) * line_h
    for yy in range(band_top - 28, OUT_H):
        fade = (yy - (band_top - 28)) / max(1, OUT_H - (band_top - 28))
        shade = int(150 * min(1.0, max(0.0, fade)))
        draw.line([(0, yy), (OUT_W, yy)], fill=(0, 0, 0, shade))
    y = band_top
    highlighted = segments(instruction, phrases)
    spans = _spans(highlighted)
    consumed = 0
    for line in lines:
        idx = instruction.find(line, consumed)
        if idx < 0:
            idx = consumed
        paint_chunks(canvas, 22, y, line, idx, body, spans)
        consumed = idx + len(line)
        y += line_h

    label = f"{speed}×"
    tw = draw.textlength(label, font=badge)
    bx, by = OUT_W - 28 - tw, 16
    draw.rounded_rectangle(
        [bx - 12, by - 6, bx + tw + 12, by + 38],
        radius=10,
        fill=(0, 0, 0, 150),
    )
    draw.text((bx, by), label, font=badge, fill=TEXT_WHITE)
    return np.array(canvas)


def _spans(parts: list[tuple[str, bool]]) -> list[tuple[int, int]]:
    spans = []
    cursor = 0
    for text, hot in parts:
        if hot:
            spans.append((cursor, cursor + len(text)))
        cursor += len(text)
    return spans


def blend(frame: np.ndarray, overlay: np.ndarray) -> None:
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    if not np.any(alpha):
        return
    base = frame.astype(np.float32)
    color = overlay[:, :, :3][:, :, ::-1].astype(np.float32)  # RGBA -> BGR
    mixed = base * (1 - alpha) + color * alpha
    frame[:] = mixed.astype(np.uint8)


def window_mask(height: int, width: int, radius: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    radius = min(radius, height // 2, width // 2)
    cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, thickness=-1)
    cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, thickness=-1)
    cv2.circle(mask, (radius, radius), radius, 255, thickness=-1)
    cv2.circle(mask, (width - radius - 1, radius), radius, 255, thickness=-1)
    cv2.circle(mask, (radius, height - radius - 1), radius, 255, thickness=-1)
    cv2.circle(mask, (width - radius - 1, height - radius - 1), radius, 255, thickness=-1)
    return mask


def place_pip(frame: np.ndarray, pip: np.ndarray) -> None:
    """First-person window flush with the top-left corner of the main frame."""
    height, width = pip.shape[:2]
    inset = 7
    outer = window_mask(height, width, PIP_RADIUS)
    inner_h, inner_w = height - 2 * inset, width - 2 * inset
    inner = np.zeros((height, width), np.uint8)
    inner[inset : inset + inner_h, inset : inset + inner_w] = window_mask(
        inner_h, inner_w, max(4, PIP_RADIUS - inset)
    )
    photo = (inner > 0).astype(np.float32)[:, :, None]
    roi = frame[0:height, 0:width].astype(np.float32)
    frame[0:height, 0:width] = (
        roi * (1.0 - photo) + pip.astype(np.float32) * photo
    ).astype(np.uint8)
    ring = (outer > 0) & (inner == 0)
    frame[0:height, 0:width][ring] = PIP_RED


_ARROW_CACHE: dict[str, np.ndarray] = {}
_STOP_SPRITE: np.ndarray | None = None


def _png_sprite(path: Path, height: int, flip_x: bool = False, flip_y: bool = False) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    scale = height / image.height
    image = image.resize((max(1, int(round(image.width * scale))), height), Image.Resampling.LANCZOS)
    if flip_x:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_y:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    array = np.array(image)
    bgr = array[:, :, :3][:, :, ::-1]
    return np.dstack([bgr, array[:, :, 3]])


def arrow_sprite(kind: str) -> np.ndarray:
    cached = _ARROW_CACHE.get(kind)
    if cached is not None:
        return cached
    if kind == "left":
        sprite = _png_sprite(ARROW_TURN, 200)
    elif kind == "right":
        sprite = _png_sprite(ARROW_TURN, 200, flip_x=True)
    elif kind == "back":
        sprite = _png_sprite(ARROW_FORWARD, 190, flip_y=True)
    else:
        sprite = _png_sprite(ARROW_FORWARD, 190)
    _ARROW_CACHE[kind] = sprite
    return sprite


def stop_sprite() -> np.ndarray:
    global _STOP_SPRITE
    if _STOP_SPRITE is None:
        image = Image.open(STOP_PATH).convert("RGBA").resize((200, 200), Image.Resampling.LANCZOS)
        array = np.array(image)
        bgr = array[:, :, :3][:, :, ::-1]
        _STOP_SPRITE = np.dstack([bgr, array[:, :, 3]])
    return _STOP_SPRITE


def blit(frame: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
    height, width = sprite.shape[:2]
    x = min(max(0, x), frame.shape[1] - width)
    y = min(max(0, y), frame.shape[0] - height)
    alpha = sprite[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y : y + height, x : x + width].astype(np.float32)
    frame[y : y + height, x : x + width] = (
        roi * (1 - alpha) + sprite[:, :, :3].astype(np.float32) * alpha
    ).astype(np.uint8)


def paint_arrow(frame: np.ndarray, kind: str | None) -> None:
    if kind in (None, "stop"):
        return
    sprite = arrow_sprite(kind)
    blit(frame, sprite, PIP_W + 18, 10)


def paint_stop(frame: np.ndarray) -> None:
    sprite = stop_sprite()
    blit(frame, sprite, OUT_W - sprite.shape[1] - 36, 70)


def lift_tone(frame: np.ndarray, alpha: float, beta: float, knee_at: float, gain: float, gamma: float) -> None:
    lifted = frame.astype(np.float32) * alpha + beta
    knee = lifted > knee_at
    lifted = np.where(knee, knee_at + (lifted - knee_at) * gain, lifted)
    frame[:] = (np.clip(lifted / 255.0, 0.0, 1.0) ** gamma * 255.0).astype(np.uint8)


def improve_frame(frame: np.ndarray, spec: dict, people: PeopleBlur | None = None) -> None:
    expose = spec.get("expose")
    if expose:
        lift_tone(frame, float(expose), 14, 190, 0.42, 0.9)
    elif spec.get("brighten"):
        frame[:] = cv2.convertScaleAbs(frame, alpha=1.12, beta=20)
    if people is not None:
        people.apply(frame, protect_robot=True)


def brighten_pip(pip: np.ndarray, spec: dict) -> np.ndarray:
    if not spec.get("expose"):
        return pip
    pip = pip.copy()
    lift_tone(pip, 2.6, 16, 200, 0.35, 0.78)
    return pip


def compose_real(spec: dict, preview: bool) -> None:
    people = PeopleBlur() if spec.get("blur_person") else None
    bank, actions = pip_bank(spec["events"], people)
    if people is not None:
        people.prev = []
        people.miss = 0
    overlay = render_overlay(spec["instruction"], spec["highlight"], spec["speed"])
    cap = cv2.VideoCapture(str(spec["mov"]))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {spec['mov']}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or spec["fps"]
    step = (src_fps * spec["speed"]) / OUT_FPS
    out_path = OUT / f"{spec['id']}.mp4"
    poster_path = OUT / f"{spec['id']}.jpg"

    clip_start = float(spec["clip_start"])
    clip_end = float(spec["clip_end"])

    def decorate(frame: np.ndarray, t: float, ending: bool = False) -> np.ndarray:
        improve_frame(frame, spec, people)
        blend(frame, overlay)
        kind = still_at(actions, t)
        if ending or kind == "stop":
            paint_stop(frame)
        else:
            paint_arrow(frame, kind)
        pip = still_at(bank, t)
        if pip is not None:
            place_pip(frame, brighten_pip(pip, spec))
        return frame

    if preview:
        cap.release()
        cap = cv2.VideoCapture(str(spec["mov"]))
        preview_dir = ROOT / "_preview"
        preview_dir.mkdir(exist_ok=True)
        times = [clip_start + 0.4, (clip_start + clip_end) * 0.5, max(clip_start, clip_end - 1.5)]
        for n, t in enumerate(times):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
            decorate(frame, t, ending=(n == len(times) - 1))
            dest = preview_dir / f"align_{spec['id']}_{n}.jpg"
            cv2.imwrite(str(dest), frame)
            print("preview", dest.name, f"t={t:.1f}s", flush=True)
        cap.release()
        return

    proc = subprocess.Popen(
        [
            FFMPEG, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}", "-r", str(OUT_FPS),
            "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast",
            "-movflags", "+faststart",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(clip_start * src_fps) - 1))
    src_i = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    next_src = clip_start * src_fps
    written = 0
    poster_saved = False
    last_base = None
    last_t = clip_end
    try:
        while True:
            if not cap.grab():
                break
            if src_i + 0.5 >= next_src:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                t = src_i / src_fps
                if t > clip_end:
                    break
                frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
                last_base = frame.copy()
                last_t = t
                decorate(frame, t)
                proc.stdin.write(frame.tobytes())
                if not poster_saved and t > min(clip_start + 4.0, clip_start + (clip_end - clip_start) * 0.25):
                    cv2.imwrite(str(poster_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    poster_saved = True
                written += 1
                next_src += step
                if written % 150 == 0:
                    print(f"  {spec['id']} {written} frames ({t:.0f}s src)", flush=True)
            src_i += 1
        if last_base is not None:
            ending = last_base
            improve_frame(ending, spec, people)
            blend(ending, overlay)
            paint_stop(ending)
            pip = still_at(bank, last_t)
            if pip is not None:
                place_pip(ending, brighten_pip(pip, spec))
            for _ in range(STOP_HOLD):
                proc.stdin.write(ending.tobytes())
                written += 1
    finally:
        cap.release()
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace")
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed for {spec['id']}: {err}")
    if not poster_saved:
        raise RuntimeError(f"no poster for {spec['id']}")
    print(f"wrote {out_path.name} ({written} frames, {written / OUT_FPS:.1f}s)")


def encode_sim(spec: dict) -> None:
    dest = OUT / f"{spec['id']}.mp4"
    poster = OUT / f"{spec['id']}.jpg"
    subprocess.run(
        [
            FFMPEG, "-y", "-loglevel", "error", "-i", str(spec["src"]),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "veryfast",
            "-movflags", "+faststart", str(dest),
        ],
        check=True,
    )
    cap = cv2.VideoCapture(str(dest))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frames // 3, 0))
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(poster), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    print("wrote", dest.name)


def _compose_job(spec: dict) -> str:
    compose_real(spec, preview=False)
    return spec["id"]


def main() -> None:
    preview = "--preview" in sys.argv
    reals_only = "--reals" in sys.argv
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [resolve_real(item) for item in REALS]
    if only:
        specs = [spec for spec in specs if spec["id"] == only]
    if preview:
        for spec in specs:
            compose_real(spec, preview=True)
        return
    for stale in ("real-window.mp4", "real-window.jpg"):
        path = OUT / stale
        if path.is_file() and (only is None or only == "real-office"):
            path.unlink()
    if not reals_only:
        for spec in SIMS:
            if not spec["src"].is_file():
                raise FileNotFoundError(spec["src"])
            encode_sim(spec)
    workers = 1 if len(specs) < 2 else max(2, min(len(specs), os.cpu_count() or 2, 4))
    print(f"encoding {len(specs)} clips on {workers} workers", flush=True)
    if workers == 1:
        for spec in specs:
            compose_real(spec, preview=False)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name in pool.map(_compose_job, specs):
            print("finished", name, flush=True)


if __name__ == "__main__":
    main()
