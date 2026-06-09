from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocr_competition import (  # noqa: E402
    audit_label_links,
    build_product_alias_map,
    canonicalize_product,
    clean_text,
    product_semantic_key,
    predict_product_rule,
)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_excluded_ids(labels_path: Path, pate_links_path: Path, milk_links_path: Path) -> set[str]:
    if not pate_links_path.exists() or not milk_links_path.exists():
        return set()
    labels = read_csv(labels_path)
    links = read_csv(pate_links_path) + read_csv(milk_links_path)
    return set(audit_label_links(labels, links)["excluded_ids"])


def normalize_product(value, alias_map: dict[str, str]) -> str:
    label = canonicalize_product(value, alias_map)
    return predict_product_rule(label, min_score=55) or label


def clean_rows(rows: list[dict], alias_map: dict[str, str]) -> list[dict]:
    cleaned = []
    for row in rows:
        cleaned.append(
            {
                "image_id": clean_text(row.get("image_id")),
                "ocr_text": clean_text(row.get("ocr_text")),
                "product_name": normalize_product(row.get("product_name"), alias_map),
            }
        )
    return cleaned


def label_summary(rows: list[dict]) -> dict:
    names = [clean_text(row.get("product_name")) for row in rows]
    nonempty = [name for name in names if name]
    counts = Counter(nonempty)
    return {
        "rows": len(rows),
        "ocr_text_empty": sum(1 for row in rows if not clean_text(row.get("ocr_text"))),
        "product_name_empty": len(rows) - len(nonempty),
        "product_name_nonempty": len(nonempty),
        "product_name_unique_exact": len(counts),
        "product_name_unique_semantic": len({product_semantic_key(name) for name in nonempty}),
        "product_name_singletons": sum(1 for value in counts.values() if value == 1),
        "product_name_long_gt_50": sum(1 for name in nonempty if len(name) > 50),
        "product_name_semicolon": sum(1 for name in nonempty if ";" in name),
        "product_name_comma": sum(1 for name in nonempty if "," in name),
        "top_product_names": counts.most_common(25),
    }


def submission_quality(rows: list[dict], train_rows: list[dict]) -> dict:
    train_keys = {
        product_semantic_key(row.get("product_name"))
        for row in train_rows
        if clean_text(row.get("product_name"))
    }
    names = [clean_text(row.get("product_name")) for row in rows]
    nonempty = [name for name in names if name]
    unknown = Counter(
        name for name in nonempty if product_semantic_key(name) not in train_keys
    )
    return {
        **label_summary(rows),
        "unknown_product_rows_vs_clean_train": sum(unknown.values()),
        "unknown_product_unique_vs_clean_train": len(unknown),
        "unknown_product_examples": unknown.most_common(25),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("version/v002_improved_ocr/submission.csv"),
    )
    parser.add_argument(
        "--pate-links",
        type=Path,
        default=Path("data/SMCE_Annotation_Form.xlsx - ImageLink_Pate.csv"),
    )
    parser.add_argument(
        "--milk-links",
        type=Path,
        default=Path("data/SMCE_Annotation_Form.xlsx - ImageLink_Milk.csv"),
    )
    parser.add_argument("--clean-train", type=Path, default=Path("data/train_labels_clean.csv"))
    parser.add_argument(
        "--model-ready",
        type=Path,
        default=Path("data/train_labels_model_ready.csv"),
    )
    parser.add_argument(
        "--clean-submission",
        type=Path,
        default=Path("version/v002_improved_ocr/submission_clean.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/artifacts/product_name_cleaning_report.json"),
    )
    args = parser.parse_args()

    raw_train = read_csv(args.train)
    alias_map = build_product_alias_map(row.get("product_name", "") for row in raw_train)
    clean_train = clean_rows(raw_train, alias_map)
    excluded_ids = load_excluded_ids(args.train, args.pate_links, args.milk_links)
    model_ready = [row for row in clean_train if row["image_id"] not in excluded_ids]

    fieldnames = ["image_id", "ocr_text", "product_name"]
    write_csv(args.clean_train, clean_train, fieldnames)
    write_csv(args.model_ready, model_ready, fieldnames)

    report = {
        "raw_train": label_summary(raw_train),
        "clean_train": label_summary(clean_train),
        "model_ready_train": {
            **label_summary(model_ready),
            "excluded_label_conflict_rows": len(excluded_ids),
        },
    }

    if args.submission.exists():
        raw_submission = read_csv(args.submission)
        clean_submission = clean_rows(raw_submission, alias_map)
        write_csv(args.clean_submission, clean_submission, fieldnames)
        report["raw_submission_v002"] = label_summary(raw_submission)
        report["clean_submission_v002"] = submission_quality(clean_submission, clean_train)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
