from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocr_competition import (
    audit_label_links,
    build_product_alias_map,
    canonicalize_product,
    grouped_dev_split,
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


def build_artifacts(
    labels_path: Path,
    pate_links_path: Path,
    milk_links_path: Path,
    output_dir: Path,
    dev_fraction: float = 0.2,
    seed: int = 2026,
) -> dict:
    labels = read_csv(labels_path)
    pate_links = read_csv(pate_links_path)
    milk_links = read_csv(milk_links_path)
    links = pate_links + milk_links
    category_by_id = {
        row["image_id"]: "pate" for row in pate_links
    } | {
        row["image_id"]: "milk" for row in milk_links
    }

    audit = audit_label_links(labels, links)
    product_aliases = build_product_alias_map(row.get("product_name", "") for row in labels)
    trusted_records = [
        {
            "image_id": image_id,
            "group_id": audit["group_by_id"].get(image_id, image_id),
            "category": category_by_id.get(image_id, ""),
        }
        for image_id in sorted(audit["trusted_ids"])
    ]
    train_ids, dev_ids = grouped_dev_split(
        trusted_records,
        dev_fraction=dev_fraction,
        seed=seed,
    )

    labels_by_id = {row["image_id"]: row for row in labels}
    manifest = []
    for image_id in sorted(labels_by_id):
        label = labels_by_id[image_id]
        if image_id in audit["excluded_ids"]:
            split = "excluded"
        elif image_id in dev_ids:
            split = "dev"
        elif image_id in train_ids:
            split = "train"
        else:
            split = "unlinked"
        manifest.append(
            {
                "image_id": image_id,
                "category": category_by_id.get(image_id, ""),
                "ImageURL": audit["url_by_id"].get(image_id, ""),
                "group_id": audit["group_by_id"].get(image_id, ""),
                "split": split,
                "label_conflict": image_id in audit["excluded_ids"],
                "ocr_text": label.get("ocr_text", ""),
                "product_name": label.get("product_name", ""),
                "canonical_product_name": canonicalize_product(
                    label.get("product_name", ""),
                    product_aliases,
                ),
            }
        )

    conflicts = []
    for url, image_ids in sorted(audit["conflicting_urls"].items()):
        for image_id in image_ids:
            label = labels_by_id[image_id]
            conflicts.append(
                {
                    "ImageURL": url,
                    "group_id": audit["group_by_id"].get(image_id, ""),
                    "image_id": image_id,
                    "ocr_text": label.get("ocr_text", ""),
                    "product_name": label.get("product_name", ""),
                }
            )

    write_csv(
        output_dir / "label_manifest.csv",
        manifest,
        [
            "image_id",
            "category",
            "ImageURL",
            "group_id",
            "split",
            "label_conflict",
            "ocr_text",
            "product_name",
            "canonical_product_name",
        ],
    )
    write_csv(
        output_dir / "label_conflicts.csv",
        conflicts,
        ["ImageURL", "group_id", "image_id", "ocr_text", "product_name"],
    )

    summary = {
        "labels": len(labels),
        "links": len(links),
        "unique_urls": len(set(audit["url_by_id"].values())),
        "conflicting_urls": audit["conflicting_url_count"],
        "excluded_rows": len(audit["excluded_ids"]),
        "train_rows": len(train_ids),
        "dev_rows": len(dev_ids),
        "unlinked_rows": sum(row["split"] == "unlinked" for row in manifest),
        "dev_fraction": dev_fraction,
        "seed": seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("data/train_labels.csv"))
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
    parser.add_argument("--output-dir", type=Path, default=Path("data/artifacts"))
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    summary = build_artifacts(
        args.labels,
        args.pate_links,
        args.milk_links,
        args.output_dir,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
