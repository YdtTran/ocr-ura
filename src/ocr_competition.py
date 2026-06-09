from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlparse


def clean_text(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def ascii_fold(value) -> str:
    text = unicodedata.normalize("NFC", clean_text(value))
    text = text.replace("Đ", "D").replace("đ", "d")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def postprocess_ocr(text: str) -> str:
    """Normalize whitespace without deleting repeated words."""
    return clean_text(text)


def _bbox_stats(bbox: Sequence[Sequence[float]]) -> dict:
    xs = [float(point[0]) for point in bbox]
    ys = [float(point[1]) for point in bbox]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "center_x": (left + right) / 2,
        "center_y": (top + bottom) / 2,
        "width": max(right - left, 1.0),
        "height": max(bottom - top, 1.0),
    }


def normalize_detection(detection) -> dict:
    if isinstance(detection, Mapping):
        bbox = detection["bbox"]
        text = detection.get("text", "")
        confidence = detection.get("confidence", 0.0)
    else:
        bbox, text, confidence = detection[:3]
    item = {
        "bbox": [[float(x), float(y)] for x, y in bbox],
        "text": clean_text(text),
        "confidence": float(confidence),
    }
    item.update(_bbox_stats(item["bbox"]))
    return item


def order_detections(detections: Iterable, line_tolerance: float = 0.6) -> list[dict]:
    """Group boxes into visual lines, then order lines and boxes."""
    items = [normalize_detection(item) for item in detections]
    items = [item for item in items if item["text"]]
    items.sort(key=lambda item: (item["center_y"], item["left"]))

    lines: list[dict] = []
    for item in items:
        best = None
        best_distance = math.inf
        for line in lines:
            tolerance = line_tolerance * max(line["mean_height"], item["height"])
            distance = abs(item["center_y"] - line["center_y"])
            if distance <= tolerance and distance < best_distance:
                best = line
                best_distance = distance
        if best is None:
            lines.append(
                {
                    "items": [item],
                    "center_y": item["center_y"],
                    "mean_height": item["height"],
                }
            )
        else:
            best["items"].append(item)
            count = len(best["items"])
            best["center_y"] = sum(x["center_y"] for x in best["items"]) / count
            best["mean_height"] = sum(x["height"] for x in best["items"]) / count

    lines.sort(key=lambda line: min(item["top"] for item in line["items"]))
    ordered = []
    for line_index, line in enumerate(lines):
        line["items"].sort(key=lambda item: item["left"])
        for item in line["items"]:
            item["line_index"] = line_index
            ordered.append(item)
    return ordered


def select_detections(
    detections: Iterable,
    confidence_threshold: float = 0.25,
    line_tolerance: float = 0.6,
) -> list[dict]:
    selected = [
        normalize_detection(item)
        for item in detections
        if float(item.get("confidence", 0.0) if isinstance(item, Mapping) else item[2])
        >= confidence_threshold
    ]
    return order_detections(selected, line_tolerance=line_tolerance)


def detections_to_text(detections: Iterable[Mapping]) -> str:
    return postprocess_ocr(" ".join(clean_text(item.get("text", "")) for item in detections))


def detection_summary(detections: Sequence[Mapping]) -> dict:
    confidences = [float(item.get("confidence", 0.0)) for item in detections]
    return {
        "detection_count": len(detections),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
    }


def should_retry_ocr(
    text: str,
    detection_count: int,
    mean_confidence: float,
    min_chars: int = 6,
    min_mean_confidence: float = 0.40,
) -> bool:
    text = clean_text(text)
    return (
        not text
        or len(text) < min_chars
        or detection_count == 0
        or mean_confidence < min_mean_confidence
    )


def ocr_quality(text: str, detection_count: int, mean_confidence: float) -> float:
    text = clean_text(text)
    alpha_numeric = sum(char.isalnum() for char in text)
    length_score = min(alpha_numeric / 80.0, 1.0)
    box_score = min(detection_count / 8.0, 1.0)
    return 0.55 * mean_confidence + 0.30 * length_score + 0.15 * box_score


def choose_ocr_pass(primary: Mapping, retry: Mapping | None) -> Mapping:
    if retry is None:
        return primary
    primary_score = ocr_quality(
        primary.get("text", ""),
        int(primary.get("detection_count", 0)),
        float(primary.get("mean_confidence", 0.0)),
    )
    retry_score = ocr_quality(
        retry.get("text", ""),
        int(retry.get("detection_count", 0)),
        float(retry.get("mean_confidence", 0.0)),
    )
    return retry if retry_score > primary_score else primary


def resize_dimensions(
    width: int,
    height: int,
    min_long_side: int = 720,
    max_long_side: int = 1600,
) -> tuple[int, int]:
    long_side = max(width, height)
    if long_side < min_long_side:
        ratio = min_long_side / long_side
    elif long_side > max_long_side:
        ratio = max_long_side / long_side
    else:
        ratio = 1.0
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def product_semantic_key(value) -> str:
    text = unicodedata.normalize("NFC", clean_text(value))
    if not text:
        return ""
    folded = ascii_fold(text).lower()
    folded = re.sub(r"[^a-z0-9]+", " ", folded).strip()
    folded = re.sub(r"\bpat[e]?\b", "pate", folded)
    folded = re.sub(r"\bopti\s*pro\s*plus\b|\boptiproplus\b", "optipro plus", folded)
    folded = re.sub(r"\bopti\s*pro\b", "optipro", folded)
    folded = re.sub(
        r"\bi?nfini\s*pro\b|\bi?nfinipro\b|\bifinipro\b",
        "infinipro",
        folded,
    )
    folded = folded.replace("highland coffee", "highlands coffee")
    folded = folded.replace("halong canfoco", "ha long canfoco")
    return folded


def build_product_alias_map(
    values: Iterable,
    min_token_f1: float = 0.8,
) -> dict[str, str]:
    groups: dict[str, Counter] = defaultdict(Counter)
    for value in values:
        text = clean_text(value)
        if text:
            groups[product_semantic_key(text)][text] += 1

    aliases = {}
    for key, counts in groups.items():
        variants = list(counts)
        representative = max(
            variants,
            key=lambda candidate: (
                sum(token_f1(candidate, variant) * counts[variant] for variant in variants),
                counts[candidate],
                -len(candidate),
                candidate,
            ),
        )
        for variant in variants:
            aliases[variant] = (
                representative
                if token_f1(variant, representative) >= min_token_f1
                else variant
            )
    return aliases


def canonicalize_product(value, alias_map: Mapping[str, str] | None = None) -> str:
    text = clean_text(value)
    if not text or not alias_map:
        return text
    return alias_map.get(text, text)


_PRODUCT_RULES = [
    (r"\bnan\b.*\bi?nfini\s*pro\b|\bnan\b.*\bi?nfinipro\b|\bnan\b.*\bifinipro\b", "Nestlé NAN INFINIPRO A2", 100),
    (r"\bnan\b.*\bopti\s*pro\b.*\bplus\b|\bnan\b.*\boptipro\s*plus\b|\bnan\b.*\boptiproplus\b", "Nestlé NAN OPTIPRO PLUS", 100),
    (r"\bpate\b.*\bcot\b.*\bden\b|\bcot\b.*\bden\b.*\bpate\b", "Pate Cột Đèn Hải Phòng", 100),
    (r"\bvissan\b.*\bpate\s*heo\b", "Vissan Pate Heo", 95),
    (r"\bvinamilk\b.*\bflex\b", "Vinamilk Flex", 95),
    (r"\bdutch\s*lady\b.*\bgrow\b", "Dutch Lady Grow", 95),
    (r"\bdo\s*hop\s*ha\s*long\b", "Đồ Hộp Hạ Long", 70),
    (r"\bha\s*long\s*canfoco\b|\bhalong\s*canfoco\b", "Ha Long Canfoco", 70),
    (r"\bhighlands?\s*coffee\b", "Highlands Coffee", 70),
    (r"\bvinamilk\b", "Vinamilk", 70),
    (r"\bth\s*true\b|\bthtrue\b", "TH True Milk", 70),
    (r"\bdutch\s*lady\b", "Dutch Lady", 70),
    (r"\bnutifood\b|\bnuti\b", "Nutifood", 70),
    (r"\bpediasure\b", "Abbott PediaSure", 75),
    (r"\bsimilac\b", "Abbott Similac", 75),
    (r"\bglucerna\b", "Abbott Glucerna", 75),
    (r"\bensure\b", "Abbott Ensure", 70),
    (r"\bmilo\b", "Nestlé Milo", 75),
    (r"\baptamil\b", "Aptamil", 70),
    (r"\bfriso\b", "Friso", 70),
    (r"\bmeiji\b", "Meiji", 70),
    (r"\banlene\b", "Anlene", 70),
    (r"\byomost\b", "Yomost", 70),
    (r"\bfami\b", "Fami", 70),
    (r"\bvissan\b", "Vissan", 70),
    (r"\bcp\b", "CP", 65),
    (r"\bnan\b", "Nestlé NAN", 65),
    (r"\bnestle\b", "Nestlé", 25),
    (r"\bpate\b", "Pate", 20),
]


def product_candidates(text: str) -> list[dict]:
    normalized = ascii_fold(text).lower()
    tokens = normalized.split()
    news_context = len(tokens) >= 25 and any(
        marker in normalized
        for marker in ("news", "tin tuc", "thu hoi", "cong ty", "theo doi", "bao ")
    )
    candidates = []
    for pattern, product, specificity in _PRODUCT_RULES:
        matches = list(re.finditer(pattern, normalized, flags=re.IGNORECASE))
        if not matches:
            continue
        first = matches[0].start()
        support = len(matches)
        context_penalty = 20 if news_context and specificity < 90 else 0
        score = (
            specificity
            + min(support - 1, 3) * 5
            - min(first / 100, 8)
            - context_penalty
        )
        candidates.append(
            {
                "product_name": product,
                "score": score,
                "specificity": specificity,
                "support": support,
                "first_position": first,
            }
        )
    return sorted(candidates, key=lambda item: (-item["score"], item["first_position"]))


def predict_product_rule(text: str, min_score: float = 55) -> str:
    candidates = product_candidates(text)
    if not candidates or candidates[0]["score"] < min_score:
        return ""
    return candidates[0]["product_name"]


def url_group_id(url: str) -> str:
    parsed = urlparse(clean_text(url))
    parts = PurePosixPath(parsed.path).parts
    if len(parts) >= 2:
        return parts[-2]
    return clean_text(url)


def audit_label_links(
    labels: Iterable[Mapping],
    links: Iterable[Mapping],
) -> dict:
    labels_by_id = {
        clean_text(row.get("image_id")): (
            clean_text(row.get("ocr_text")),
            clean_text(row.get("product_name")),
        )
        for row in labels
    }
    url_by_id = {
        clean_text(row.get("image_id")): clean_text(row.get("ImageURL"))
        for row in links
    }
    ids_by_url: dict[str, list[str]] = defaultdict(list)
    for image_id, url in url_by_id.items():
        if image_id in labels_by_id and url:
            ids_by_url[url].append(image_id)

    conflicting_urls = {}
    excluded_ids = set()
    for url, image_ids in ids_by_url.items():
        label_values = {labels_by_id[image_id] for image_id in image_ids}
        if len(label_values) > 1:
            conflicting_urls[url] = sorted(image_ids)
            excluded_ids.update(image_ids)

    all_ids = set(labels_by_id) & set(url_by_id)
    return {
        "conflicting_url_count": len(conflicting_urls),
        "conflicting_urls": conflicting_urls,
        "excluded_ids": excluded_ids,
        "trusted_ids": all_ids - excluded_ids,
        "url_by_id": url_by_id,
        "group_by_id": {image_id: url_group_id(url) for image_id, url in url_by_id.items()},
    }


def grouped_dev_split(
    records: Iterable[Mapping],
    dev_fraction: float = 0.2,
    seed: int = 2026,
) -> tuple[set[str], set[str]]:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    groups: dict[str, list[str]] = defaultdict(list)
    for row in records:
        image_id = clean_text(row.get("image_id"))
        group_id = clean_text(row.get("group_id")) or image_id
        if image_id:
            groups[group_id].append(image_id)

    ranked = sorted(
        groups,
        key=lambda group_id: hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest(),
    )
    target = max(1, round(sum(len(ids) for ids in groups.values()) * dev_fraction))
    dev_ids = set()
    for group_id in ranked:
        if len(dev_ids) >= target:
            break
        dev_ids.update(groups[group_id])
    all_ids = {image_id for ids in groups.values() for image_id in ids}
    return all_ids - dev_ids, dev_ids


def token_f1(ground_truth, prediction) -> float:
    gt = clean_text(ground_truth)
    pred = clean_text(prediction)
    if not gt and not pred:
        return 1.0
    gt_tokens = set(gt.lower().split())
    pred_tokens = set(pred.lower().split())
    if not gt_tokens or not pred_tokens:
        return 0.0
    common = len(gt_tokens & pred_tokens)
    return 2 * common / (len(gt_tokens) + len(pred_tokens))


def character_error_rate(ground_truth, prediction) -> float:
    gt = clean_text(ground_truth)
    pred = clean_text(prediction)
    if not gt:
        return 0.0 if not pred else 1.0
    distance = list(range(len(pred) + 1))
    for row, gt_char in enumerate(gt, 1):
        previous, distance[0] = distance[0], row
        for column, pred_char in enumerate(pred, 1):
            old = distance[column]
            distance[column] = (
                previous
                if gt_char == pred_char
                else 1 + min(previous, distance[column], distance[column - 1])
            )
            previous = old
    return min(distance[-1] / len(gt), 1.0)


def composite_metrics(rows: Iterable[Mapping]) -> dict:
    rows = list(rows)
    if not rows:
        return {"rows": 0, "product_f1": 0.0, "ocr_cer": 0.0, "composite": 0.0}
    product = sum(
        token_f1(row.get("product_name_gt"), row.get("product_name_pred"))
        for row in rows
    ) / len(rows)
    cer = sum(
        character_error_rate(row.get("ocr_text_gt"), row.get("ocr_text_pred"))
        for row in rows
    ) / len(rows)
    return {
        "rows": len(rows),
        "product_f1": product,
        "ocr_cer": cer,
        "composite": 0.6 * product + 0.4 * (1.0 - cer),
    }
