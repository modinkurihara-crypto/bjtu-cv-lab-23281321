from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_LABELS = ("fhy", "nm", "sy", "zx")
CANVAS_W = 1280
CANVAS_H = 720
FPS = 24


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def put_text(frame: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.7, bold: bool = False) -> None:
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (28, 34, 42),
        2 if bold else 1,
        cv2.LINE_AA,
    )


def fit_to_box(image: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(box_w / w, box_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def make_contact_montage(contact: np.ndarray) -> np.ndarray:
    panels = 9
    h, w = contact.shape[:2]
    panel_w = w // panels
    crops = [contact[:, i * panel_w : (i + 1) * panel_w] for i in range(panels)]

    tile_w, tile_h = 390, 180
    gap = 16
    top = 116
    left = (CANVAS_W - (3 * tile_w + 2 * gap)) // 2
    frame = np.full((CANVAS_H, CANVAS_W, 3), 248, dtype=np.uint8)
    put_text(frame, "Retrieval test: query and Top-8 results", (42, 48), 0.9, True)
    put_text(frame, "Green border = relevant, red border = irrelevant", (42, 82), 0.62, False)

    for idx, crop in enumerate(crops):
        row, col = divmod(idx, 3)
        x = left + col * (tile_w + gap)
        y = top + row * (tile_h + gap)
        tile = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
        resized = fit_to_box(crop, tile_w - 8, tile_h - 8)
        rh, rw = resized.shape[:2]
        ox = (tile_w - rw) // 2
        oy = (tile_h - rh) // 2
        tile[oy : oy + rh, ox : ox + rw] = resized
        cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), (212, 217, 226), 1)
        frame[y : y + tile_h, x : x + tile_w] = tile
    return frame


def make_detection_frame(detection: np.ndarray) -> np.ndarray:
    frame = np.full((CANVAS_H, CANVAS_W, 3), 248, dtype=np.uint8)
    put_text(frame, "Text detection test", (42, 48), 0.9, True)
    put_text(frame, "Yellow = detector output, green = LabelMe annotation", (42, 82), 0.62, False)
    resized = fit_to_box(detection, 940, 585)
    h, w = resized.shape[:2]
    x = (CANVAS_W - w) // 2
    y = 112
    frame[y : y + h, x : x + w] = resized
    cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (212, 217, 226), 1)
    return frame


def make_title_frame(label: str) -> np.ndarray:
    frame = np.full((CANVAS_H, CANVAS_W, 3), 248, dtype=np.uint8)
    put_text(frame, "BJTU Computer Vision Experiment 2", (330, 280), 1.0, True)
    put_text(frame, f"Demo sample: {label}", (500, 340), 0.85, False)
    put_text(frame, "Image retrieval + text detection", (420, 390), 0.72, False)
    return frame


def add_seconds(writer: cv2.VideoWriter, frame: np.ndarray, seconds: float) -> None:
    for _ in range(int(FPS * seconds)):
        writer.write(frame)


def create_video(label: str, outputs: Path, out_dir: Path, assets_dir: Path) -> Path:
    contact_candidates = sorted((outputs / "retrieval" / "contact_sheets").glob(f"{label}_*.jpg"))
    det_candidates = sorted((outputs / "text_detection" / "clear_examples").glob(f"{label}_1_*.jpg"))
    if not contact_candidates:
        raise FileNotFoundError(f"No contact sheet for {label}")
    if not det_candidates:
        raise FileNotFoundError(f"No detection example for {label}")

    contact = read_image(contact_candidates[0])
    detection = read_image(det_candidates[0])
    retrieval_frame = make_contact_montage(contact)
    detection_frame = make_detection_frame(detection)
    title_frame = make_title_frame(label)

    assets_dir.mkdir(parents=True, exist_ok=True)
    write_image(assets_dir / f"{label}_retrieval_montage.jpg", retrieval_frame)
    write_image(assets_dir / f"{label}_text_detection.jpg", detection_frame)

    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"demo_{label}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (CANVAS_W, CANVAS_H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {video_path}")

    add_seconds(writer, title_frame, 1.2)
    add_seconds(writer, retrieval_frame, 3.2)
    add_seconds(writer, detection_frame, 3.2)
    writer.release()
    return video_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create short demo videos from experiment outputs.")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("demo_videos"))
    parser.add_argument("--assets", type=Path, default=Path("docs/sample_results"))
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    args = parser.parse_args()

    for label in args.labels:
        path = create_video(label, args.outputs, args.out, args.assets)
        print(path)


if __name__ == "__main__":
    main()
