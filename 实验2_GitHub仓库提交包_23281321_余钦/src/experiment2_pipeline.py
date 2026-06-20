from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


CLASS_NAMES = {
    "sy": "Siyuan Building",
    "tsg": "Library",
    "nm": "South Gate",
    "ty": "Tianyou Hall",
    "sjz": "Century Bell",
    "zx": "Zhixing Stele",
    "mh": "Ming Lake Stele",
    "fhy": "Fanghua Garden Stele",
    "yf": "Yifu Building",
    "jx": "Mechanical Building",
    "kx": "Science Hall",
    "yk": "Welcoming Pine",
}

EVAL_KS = (20, 40, 60)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageItem:
    path: Path
    label: str
    split: str


def label_from_name(path: Path) -> str:
    return path.name.split("-", 1)[0].lower()


def read_image(path: Path, mode: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    # cv2.imread is fragile with non-ASCII parent paths on some systems.
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, mode)


def write_image(path: Path, image: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    params = []
    if ext.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def iter_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def scan_dataset(dataset: Path) -> tuple[list[ImageItem], list[ImageItem]]:
    base_bjtu = dataset / "image_retrieval" / "base" / "BJTU"
    base_util = dataset / "image_retrieval" / "base" / "util_pic"
    query_dir = dataset / "image_retrieval" / "query"

    base_items: list[ImageItem] = []
    for path in iter_images(base_bjtu):
        base_items.append(ImageItem(path=path, label=label_from_name(path), split="base"))
    for path in iter_images(base_util):
        base_items.append(ImageItem(path=path, label="util", split="base"))

    query_items = [ImageItem(path=path, label=label_from_name(path), split="query") for path in iter_images(query_dir)]
    return base_items, query_items


def l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image
    scale = max_side / side
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def extract_global_feature(path: Path) -> np.ndarray | None:
    image = read_image(path)
    if image is None:
        return None
    image = resize_max_side(image, 640)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 4], [0, 180, 0, 256, 0, 256]).flatten()
    hist = l2_normalize(hist)

    small = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
    spatial = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32).flatten() / 255.0
    spatial = l2_normalize(spatial)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    hog_feat = hog.compute(gray).flatten()
    hog_feat = l2_normalize(hog_feat)

    feature = np.concatenate([1.25 * hist, 0.55 * spatial, 1.0 * hog_feat]).astype(np.float32)
    return l2_normalize(feature)


def extract_orb(path: Path, nfeatures: int = 700) -> np.ndarray | None:
    image = read_image(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = resize_max_side(image, 960)
    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=7)
    _kp, des = orb.detectAndCompute(image, None)
    return des


def compute_or_load_features(items: list[ImageItem], cache_path: Path, rebuild: bool) -> tuple[np.ndarray, list[str], list[str]]:
    if cache_path.exists() and not rebuild:
        data = np.load(cache_path, allow_pickle=True)
        return data["features"], list(data["paths"]), list(data["labels"])

    features: list[np.ndarray] = []
    paths: list[str] = []
    labels: list[str] = []
    for item in tqdm(items, desc=f"features {cache_path.stem}"):
        feat = extract_global_feature(item.path)
        if feat is None:
            print(f"skip unreadable image: {item.path}")
            continue
        features.append(feat)
        paths.append(str(item.path))
        labels.append(item.label)

    feature_mat = np.vstack(features).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=feature_mat, paths=np.array(paths), labels=np.array(labels))
    return feature_mat, paths, labels


def compute_or_load_orb(items: list[ImageItem], cache_path: Path, rebuild: bool) -> dict[str, np.ndarray | None]:
    if cache_path.exists() and not rebuild:
        with cache_path.open("rb") as f:
            return pickle.load(f)

    descs: dict[str, np.ndarray | None] = {}
    for item in tqdm(items, desc=f"orb {cache_path.stem}"):
        descs[str(item.path)] = extract_orb(item.path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(descs, f)
    return descs


def orb_match_score(query_desc: np.ndarray | None, base_desc: np.ndarray | None) -> float:
    if query_desc is None or base_desc is None or len(query_desc) < 4 or len(base_desc) < 4:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        raw = matcher.knnMatch(query_desc, base_desc, k=2)
    except cv2.error:
        return 0.0
    good = 0
    distance_sum = 0.0
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.76 * n.distance:
            good += 1
            distance_sum += m.distance
    if good == 0:
        return 0.0
    avg_distance = distance_sum / good
    coverage = good / math.sqrt(len(query_desc) * len(base_desc))
    return float(coverage * (1.0 - min(avg_distance, 96.0) / 96.0))


def rank_with_optional_orb(
    query_features: np.ndarray,
    base_features: np.ndarray,
    query_paths: list[str],
    base_paths: list[str],
    query_orb: dict[str, np.ndarray | None] | None,
    base_orb: dict[str, np.ndarray | None] | None,
    shortlist: int,
    orb_weight: float,
) -> np.ndarray:
    sim = query_features @ base_features.T
    rankings = np.empty((len(query_paths), base_features.shape[0]), dtype=np.int32)
    for qi, qpath in enumerate(tqdm(query_paths, desc="ranking")):
        scores = sim[qi].copy()
        if query_orb is not None and base_orb is not None and orb_weight > 0:
            n = min(shortlist, len(scores))
            top = np.argpartition(-scores, n - 1)[:n]
            local = np.array(
                [orb_match_score(query_orb.get(qpath), base_orb.get(base_paths[bi])) for bi in top],
                dtype=np.float32,
            )
            if float(local.max()) > 0:
                local = local / float(local.max())
                scores[top] += orb_weight * local
        rankings[qi] = np.argsort(-scores)
    return rankings


def write_topk_results(
    out_csv: Path,
    rankings: np.ndarray,
    query_paths: list[str],
    query_labels: list[str],
    base_paths: list[str],
    base_labels: list[str],
    scores: np.ndarray,
    topk: int,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "query_label", "rank", "base", "base_label", "score", "relevant"])
        for qi, order in enumerate(rankings):
            for rank, bi in enumerate(order[:topk], start=1):
                relevant = int(base_labels[bi] == query_labels[qi])
                writer.writerow([query_paths[qi], query_labels[qi], rank, base_paths[bi], base_labels[bi], f"{scores[qi, bi]:.6f}", relevant])


def compute_precision_rows(
    rankings: np.ndarray,
    query_labels: list[str],
    base_labels: list[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    class_rows: list[dict[str, str]] = []
    overall_sums = {k: [] for k in EVAL_KS}
    labels = sorted(label for label in set(query_labels) if label in CLASS_NAMES)

    for label in labels:
        q_indices = [i for i, q_label in enumerate(query_labels) if q_label == label]
        row: dict[str, str] = {
            "label": label,
            "landmark": CLASS_NAMES[label],
            "query_count": str(len(q_indices)),
        }
        for k in EVAL_KS:
            vals = []
            for qi in q_indices:
                top = rankings[qi, :k]
                rel = np.array([base_labels[bi] == query_labels[qi] for bi in top], dtype=np.float32)
                vals.append(float(rel.mean()))
            value = float(np.mean(vals)) if vals else 0.0
            row[f"P@{k}"] = f"{value:.4f}"
            overall_sums[k].extend(vals)
        class_rows.append(row)

    overall_rows = [
        {"K": str(k), "Precision": f"{(float(np.mean(vals)) if vals else 0.0):.4f}", "query_count": str(len(query_labels))}
        for k, vals in overall_sums.items()
    ]
    return class_rows, overall_rows


def write_dict_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_precision(class_rows: list[dict[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in class_rows:
        label = row["label"]
        y = [float(row[f"P@{k}"]) for k in EVAL_KS]
        plt.figure(figsize=(5.6, 3.6))
        plt.plot(EVAL_KS, y, marker="o", linewidth=2.4, color="#1f77b4")
        plt.ylim(0, 1.02)
        plt.xticks(EVAL_KS)
        plt.xlabel("K")
        plt.ylabel("Precision")
        plt.title(f"P@K - {label} ({row['landmark']})")
        plt.grid(alpha=0.28)
        for x, value in zip(EVAL_KS, y):
            plt.text(x, min(value + 0.035, 0.98), f"{value:.2f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / f"precision_{label}.png", dpi=160)
        plt.close()


def cv_to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def make_thumb(path: str, size: tuple[int, int]) -> Image.Image:
    image = read_image(Path(path))
    if image is None:
        return Image.new("RGB", size, "white")
    pil = cv_to_pil(image)
    pil.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(pil, ((size[0] - pil.width) // 2, (size[1] - pil.height) // 2))
    return canvas


def add_caption(image: Image.Image, caption: str, border: str | None = None) -> Image.Image:
    pad = 8
    cap_h = 34
    canvas = Image.new("RGB", (image.width, image.height + cap_h), "white")
    canvas.paste(image, (0, cap_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), caption[:44], fill=(20, 20, 20))
    if border:
        draw.rectangle([1, cap_h + 1, canvas.width - 2, canvas.height - 2], outline=border, width=5)
    return canvas


def make_retrieval_contact_sheets(
    out_dir: Path,
    rankings: np.ndarray,
    query_paths: list[str],
    query_labels: list[str],
    base_paths: list[str],
    base_labels: list[str],
    per_class: int = 1,
    topn: int = 8,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(23281321)
    chosen: list[int] = []
    for label in sorted(label for label in set(query_labels) if label in CLASS_NAMES):
        idxs = [i for i, ql in enumerate(query_labels) if ql == label]
        rng.shuffle(idxs)
        chosen.extend(idxs[:per_class])

    thumb_size = (180, 128)
    for qi in chosen:
        tiles = [add_caption(make_thumb(query_paths[qi], thumb_size), f"Query: {query_labels[qi]}", "#333333")]
        for rank, bi in enumerate(rankings[qi, :topn], start=1):
            color = "#1f9d55" if base_labels[bi] == query_labels[qi] else "#d64545"
            cap = f"#{rank} {base_labels[bi]}"
            tiles.append(add_caption(make_thumb(base_paths[bi], thumb_size), cap, color))
        sheet = Image.new("RGB", (len(tiles) * thumb_size[0], thumb_size[1] + 34), "white")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, (i * thumb_size[0], 0))
        safe_name = Path(query_paths[qi]).stem.replace("/", "_")
        sheet.save(out_dir / f"{query_labels[qi]}_{safe_name}.jpg", quality=92)


def intersect_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    return max(0, x2 - x1) * max(0, y2 - y1)


def merge_boxes(boxes: list[tuple[int, int, int, int]], image_area: int) -> list[tuple[int, int, int, int]]:
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    merged: list[tuple[int, int, int, int]] = []
    for box in boxes:
        x, y, w, h = box
        if w * h > image_area * 0.45:
            continue
        consumed = False
        for i, old in enumerate(merged):
            inter = intersect_area(box, old)
            small = max(1, min(w * h, old[2] * old[3]))
            close_y = abs((y + h / 2) - (old[1] + old[3] / 2)) < max(h, old[3]) * 0.75
            close_x = abs((x + w / 2) - (old[0] + old[2] / 2)) < (w + old[2]) * 0.65
            if inter / small > 0.25 or (close_y and close_x and inter > 0):
                ox, oy, ow, oh = old
                nx, ny = min(x, ox), min(y, oy)
                nx2, ny2 = max(x + w, ox + ow), max(y + h, oy + oh)
                merged[i] = (nx, ny, nx2 - nx, ny2 - ny)
                consumed = True
                break
        if not consumed:
            merged.append(box)
    return sorted(merged, key=lambda b: (b[1], b[0]))[:10]


def detect_text_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    original_h, original_w = image.shape[:2]
    work = resize_max_side(image, 1200)
    scale_x = original_w / work.shape[1]
    scale_y = original_h / work.shape[0]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    boxes: list[tuple[int, int, int, int]] = []
    kernels = [(21, 5), (35, 9), (9, 21)]
    for invert in (False, True):
        src = 255 - gray if invert else gray
        for kw, kh in kernels:
            rect = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
            blackhat = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, rect)
            grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
            grad = np.absolute(grad)
            if grad.max() > grad.min():
                grad = ((grad - grad.min()) / (grad.max() - grad.min()) * 255).astype("uint8")
            closed = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rect)
            thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                if area < 260 or w < 18 or h < 8:
                    continue
                aspect = w / max(1, h)
                if not (0.45 <= aspect <= 18):
                    continue
                if h > work.shape[0] * 0.35 or w > work.shape[1] * 0.92:
                    continue
                boxes.append((int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)))

    try:
        mser = cv2.MSER_create(delta=5, min_area=80, max_area=max(500, int(work.shape[0] * work.shape[1] * 0.03)))
        _regions, rects = mser.detectRegions(gray)
        for x, y, w, h in rects:
            if w < 12 or h < 8:
                continue
            aspect = w / max(1, h)
            if 0.25 <= aspect <= 12:
                boxes.append((int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)))
    except cv2.error:
        pass

    return merge_boxes(boxes, original_w * original_h)


def annotation_path_for(image_path: str, annotation_dir: Path | None) -> Path | None:
    if annotation_dir is None:
        return None
    candidate = annotation_dir / f"{Path(image_path).stem}.json"
    return candidate if candidate.exists() else None


def load_labelme_boxes(image_path: str, annotation_dir: Path | None) -> list[tuple[int, int, int, int]]:
    json_path = annotation_path_for(image_path, annotation_dir)
    if json_path is None:
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    boxes: list[tuple[int, int, int, int]] = []
    for shape in data.get("shapes", []):
        points = shape.get("points") or []
        if not points:
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        x1, x2 = max(0, int(min(xs))), max(0, int(max(xs)))
        y1, y2 = max(0, int(min(ys))), max(0, int(max(ys)))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def draw_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    gt_boxes: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    canvas = image.copy()
    for x, y, w, h in gt_boxes or []:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (20, 190, 60), max(2, int(min(canvas.shape[:2]) / 420)))
    for x, y, w, h in boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 210, 255), max(2, int(min(canvas.shape[:2]) / 450)))
    cv2.putText(
        canvas,
        f"det: {len(boxes)}  gt: {len(gt_boxes or [])}",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 160, 220),
        2,
        cv2.LINE_AA,
    )
    return canvas


def make_detection_thumb(path: str, size: tuple[int, int], annotation_dir: Path | None) -> Image.Image:
    image = read_image(Path(path))
    if image is None:
        return Image.new("RGB", size, "white")
    boxes = detect_text_boxes(image)
    gt_boxes = load_labelme_boxes(path, annotation_dir)
    drawn = draw_boxes(image, boxes, gt_boxes)
    pil = cv_to_pil(drawn)
    pil.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(pil, ((size[0] - pil.width) // 2, (size[1] - pil.height) // 2))
    return canvas


def make_text_visual_samples(
    out_dir: Path,
    rankings: np.ndarray,
    query_paths: list[str],
    query_labels: list[str],
    base_paths: list[str],
    base_labels: list[str],
    annotation_dir: Path | None,
    per_class: int = 2,
    topn: int = 5,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    thumb_size = (220, 156)
    for label in sorted(label for label in set(query_labels) if label in CLASS_NAMES):
        idxs = [i for i, ql in enumerate(query_labels) if ql == label]
        for seq, qi in enumerate(idxs[:per_class], start=1):
            query_tile = add_caption(make_thumb(query_paths[qi], thumb_size), f"Query {label}", "#333333")
            top_tiles = []
            for rank, bi in enumerate(rankings[qi, :topn], start=1):
                color = "#1f9d55" if base_labels[bi] == query_labels[qi] else "#d64545"
                top_tiles.append(add_caption(make_thumb(base_paths[bi], thumb_size), f"Top{rank} {base_labels[bi]}", color))
            detect_rank = 1
            detect_bi = rankings[qi, 0]
            relevant_rank = None
            relevant_bi = None
            for rank, bi in enumerate(rankings[qi, :60], start=1):
                if base_labels[bi] == query_labels[qi]:
                    if relevant_rank is None:
                        relevant_rank = rank
                        relevant_bi = bi
                    if annotation_path_for(base_paths[bi], annotation_dir) is not None:
                        detect_rank = rank
                        detect_bi = bi
                        break
            else:
                if relevant_rank is not None and relevant_bi is not None:
                    detect_rank = relevant_rank
                    detect_bi = relevant_bi
            detect_caption = f"Text detection on retrieved #{detect_rank} ({base_labels[detect_bi]})"
            detect_tile = add_caption(
                make_detection_thumb(base_paths[detect_bi], (thumb_size[0] * 2, thumb_size[1] * 2), annotation_dir),
                detect_caption,
                "#e09f00",
            )

            width = max(query_tile.width + len(top_tiles) * thumb_size[0], detect_tile.width)
            height = query_tile.height + detect_tile.height + 18
            sheet = Image.new("RGB", (width, height), "white")
            sheet.paste(query_tile, (0, 0))
            for i, tile in enumerate(top_tiles):
                sheet.paste(tile, (query_tile.width + i * thumb_size[0], 0))
            sheet.paste(detect_tile, (0, query_tile.height + 18))
            safe_name = Path(query_paths[qi]).stem.replace("/", "_")
            sheet.save(out_dir / f"{label}_{seq}_{safe_name}.jpg", quality=92)


def make_clear_detection_examples(annotation_dir: Path, out_dir: Path, per_class: int = 2) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("*.jpg"):
        old_file.unlink()
    grouped: dict[str, list[Path]] = {label: [] for label in CLASS_NAMES}
    for json_path in sorted(annotation_dir.glob("*.json")):
        label = label_from_name(json_path)
        if label not in grouped:
            continue
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            image_path = json_path.with_suffix(ext)
            if image_path.exists():
                grouped[label].append(image_path)
                break

    for label, paths in grouped.items():
        for seq, image_path in enumerate(paths[:per_class], start=1):
            image = read_image(image_path)
            if image is None:
                continue
            drawn = draw_boxes(image, detect_text_boxes(image), load_labelme_boxes(str(image_path), annotation_dir))
            drawn = resize_max_side(drawn, 1600)
            write_image(out_dir / f"{label}_{seq}_{image_path.stem}_det_gt.jpg", drawn)


def main() -> None:
    parser = argparse.ArgumentParser(description="BJTU visual impression retrieval and text detection pipeline.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--topk", type=int, default=60)
    parser.add_argument("--shortlist", type=int, default=140)
    parser.add_argument("--orb-weight", type=float, default=0.18)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    base_items, query_items = scan_dataset(args.dataset)
    print(f"base images: {len(base_items)}")
    print(f"query images: {len(query_items)}")

    cache_dir = args.out / "cache"
    base_features, base_paths, base_labels = compute_or_load_features(base_items, cache_dir / "base_features.npz", args.rebuild_cache)
    query_features, query_paths, query_labels = compute_or_load_features(query_items, cache_dir / "query_features.npz", args.rebuild_cache)

    base_orb = compute_or_load_orb([ImageItem(Path(p), l, "base") for p, l in zip(base_paths, base_labels)], cache_dir / "base_orb.pkl", args.rebuild_cache)
    query_orb = compute_or_load_orb([ImageItem(Path(p), l, "query") for p, l in zip(query_paths, query_labels)], cache_dir / "query_orb.pkl", args.rebuild_cache)

    rankings = rank_with_optional_orb(
        query_features=query_features,
        base_features=base_features,
        query_paths=query_paths,
        base_paths=base_paths,
        query_orb=query_orb,
        base_orb=base_orb,
        shortlist=args.shortlist,
        orb_weight=args.orb_weight,
    )

    # Scores are the global cosine scores. The ORB rerank is reflected in the ordering.
    global_scores = query_features @ base_features.T
    retrieval_dir = args.out / "retrieval"
    write_topk_results(
        retrieval_dir / f"top{args.topk}_results.csv",
        rankings,
        query_paths,
        query_labels,
        base_paths,
        base_labels,
        global_scores,
        args.topk,
    )
    class_rows, overall_rows = compute_precision_rows(rankings, query_labels, base_labels)
    write_dict_csv(retrieval_dir / "precision_by_class.csv", class_rows)
    write_dict_csv(retrieval_dir / "precision_overall.csv", overall_rows)
    plot_precision(class_rows, retrieval_dir / "plots")
    make_retrieval_contact_sheets(retrieval_dir / "contact_sheets", rankings, query_paths, query_labels, base_paths, base_labels)
    make_text_visual_samples(
        args.out / "text_detection" / "visual_samples",
        rankings,
        query_paths,
        query_labels,
        base_paths,
        base_labels,
        annotation_dir=args.dataset / "object_detection" / "data",
    )
    make_clear_detection_examples(args.dataset / "object_detection" / "data", args.out / "text_detection" / "clear_examples")

    print("done")
    print(f"precision: {retrieval_dir / 'precision_by_class.csv'}")
    print(f"text samples: {args.out / 'text_detection' / 'visual_samples'}")


if __name__ == "__main__":
    main()
