from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocr_competition import (
    choose_ocr_pass,
    composite_metrics,
    detection_summary,
    detections_to_text,
    select_detections,
)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluation_rows(
    checkpoint: list[dict],
    labels_by_id: dict[str, dict],
    allowed_ids: set[str] | None = None,
) -> list[dict]:
    rows = []
    for prediction in checkpoint:
        image_id = prediction["image_id"]
        if allowed_ids is not None and image_id not in allowed_ids:
            continue
        label = labels_by_id.get(image_id, {})
        ocr_gt = prediction.get("label_ocr_text", label.get("ocr_text", ""))
        product_gt = prediction.get("label_product_name", label.get("product_name", ""))
        rows.append(
            {
                "ocr_text_gt": ocr_gt,
                "ocr_text_pred": prediction.get("pred_ocr_text", prediction.get("ocr_text", "")),
                "product_name_gt": product_gt,
                "product_name_pred": prediction.get(
                    "pred_product_name",
                    prediction.get("product_name", ""),
                ),
            }
        )
    return rows


def threshold_sweep(
    raw_path: Path,
    labels_by_id: dict[str, dict],
    thresholds: list[float],
    allowed_ids: set[str] | None,
) -> list[dict]:
    raw_rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output = []
    for threshold in thresholds:
        rows = []
        for raw in raw_rows:
            image_id = raw["image_id"]
            if image_id not in labels_by_id:
                continue
            if allowed_ids is not None and image_id not in allowed_ids:
                continue
            passes = []
            for raw_pass in raw.get("passes", []):
                selected = select_detections(
                    raw_pass.get("raw_detections", []),
                    confidence_threshold=threshold,
                )
                summary = detection_summary(selected)
                passes.append(
                    {
                        "text": detections_to_text(selected),
                        "detection_count": summary["detection_count"],
                        "mean_confidence": summary["mean_confidence"],
                        "pass_name": raw_pass.get("pass_name", ""),
                    }
                )
            if not passes:
                text = ""
            else:
                chosen = passes[0]
                for candidate in passes[1:]:
                    chosen = choose_ocr_pass(chosen, candidate)
                text = chosen["text"]
            rows.append(
                {
                    "ocr_text_gt": labels_by_id[image_id].get("ocr_text", ""),
                    "ocr_text_pred": text,
                    "product_name_gt": "",
                    "product_name_pred": "",
                }
            )
        metrics = composite_metrics(rows)
        output.append(
            {
                "threshold": threshold,
                "rows": metrics["rows"],
                "ocr_cer": metrics["ocr_cer"],
                "ocr_score": 1.0 - metrics["ocr_cer"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/artifacts/label_manifest.csv"))
    parser.add_argument("--raw-jsonl", type=Path)
    parser.add_argument(
        "--thresholds",
        default="0.20,0.25,0.30,0.35,0.40",
        help="Comma-separated thresholds for raw detection sweep.",
    )
    args = parser.parse_args()

    checkpoint = read_csv(args.checkpoint)
    labels = read_csv(args.labels)
    labels_by_id = {row["image_id"]: row for row in labels}
    manifest = read_csv(args.manifest) if args.manifest.exists() else []
    trusted_ids = {
        row["image_id"] for row in manifest if row.get("split") in {"train", "dev"}
    } or None
    dev_ids = {row["image_id"] for row in manifest if row.get("split") == "dev"} or None

    reports = {
        "all": composite_metrics(evaluation_rows(checkpoint, labels_by_id)),
        "trusted": composite_metrics(
            evaluation_rows(checkpoint, labels_by_id, trusted_ids)
        ),
        "dev": composite_metrics(evaluation_rows(checkpoint, labels_by_id, dev_ids)),
    }
    print(json.dumps(reports, ensure_ascii=False, indent=2))

    if args.raw_jsonl and args.raw_jsonl.exists():
        thresholds = [float(value) for value in args.thresholds.split(",")]
        sweep = threshold_sweep(args.raw_jsonl, labels_by_id, thresholds, dev_ids)
        print(json.dumps({"threshold_sweep_dev": sweep}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
