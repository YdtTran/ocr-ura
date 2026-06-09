import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocr_competition import (
    audit_label_links,
    build_product_alias_map,
    canonicalize_product,
    composite_metrics,
    grouped_dev_split,
    order_detections,
    postprocess_ocr,
    predict_product_rule,
    select_detections,
    should_retry_ocr,
)


def box(x, y, w=100, h=20):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


class OcrCompetitionTests(unittest.TestCase):
    def test_postprocess_preserves_repeated_words(self):
        self.assertEqual(
            postprocess_ocr("CP  vị hôi hôi\n bở bở không ngon"),
            "CP vị hôi hôi bở bở không ngon",
        )

    def test_detections_are_grouped_into_lines_before_sorting(self):
        detections = [
            (box(160, 13), "sau", 0.9),
            (box(10, 10), "Trước", 0.9),
            (box(15, 50), "Dòng hai", 0.9),
        ]
        ordered = order_detections(detections)
        self.assertEqual([item["text"] for item in ordered], ["Trước", "sau", "Dòng hai"])

    def test_selection_uses_inclusive_threshold_and_keeps_metadata(self):
        detections = [(box(0, 0), "NAN", 0.25), (box(0, 30), "noise", 0.24)]
        selected = select_detections(detections, confidence_threshold=0.25)
        self.assertEqual([item["text"] for item in selected], ["NAN"])
        self.assertEqual(selected[0]["confidence"], 0.25)

    def test_audit_excludes_conflicting_labels_for_same_url(self):
        labels = [
            {"image_id": "a", "ocr_text": "", "product_name": ""},
            {"image_id": "b", "ocr_text": "Nestle NAN", "product_name": "NAN"},
            {"image_id": "c", "ocr_text": "CP", "product_name": "CP"},
        ]
        links = [
            {"image_id": "a", "ImageURL": "https://host/post/music.jpg"},
            {"image_id": "b", "ImageURL": "https://host/post/music.jpg"},
            {"image_id": "c", "ImageURL": "https://host/other/cover.jpg"},
        ]
        audit = audit_label_links(labels, links)
        self.assertEqual(audit["conflicting_url_count"], 1)
        self.assertEqual(audit["excluded_ids"], {"a", "b"})
        self.assertEqual(audit["trusted_ids"], {"c"})

    def test_grouped_split_does_not_leak_url_groups(self):
        records = [
            {"image_id": "a", "group_id": "post-1", "category": "pate"},
            {"image_id": "b", "group_id": "post-1", "category": "pate"},
            {"image_id": "c", "group_id": "post-2", "category": "milk"},
            {"image_id": "d", "group_id": "post-3", "category": "milk"},
        ]
        train_ids, dev_ids = grouped_dev_split(records, dev_fraction=0.5, seed=7)
        train_groups = {r["group_id"] for r in records if r["image_id"] in train_ids}
        dev_groups = {r["group_id"] for r in records if r["image_id"] in dev_ids}
        self.assertFalse(train_groups & dev_groups)
        self.assertTrue(train_ids)
        self.assertTrue(dev_ids)

    def test_product_canonicalization_and_specific_rules(self):
        aliases = build_product_alias_map(["Nestle NAN", "NESTLE NAN", "Nestle NAN"])
        self.assertEqual(canonicalize_product("NESTLE NAN", aliases), "Nestle NAN")
        self.assertEqual(
            predict_product_rule("Nestle NAN OPTI pro PLUS digestion and immunity"),
            "Nestlé NAN OPTIPRO PLUS",
        )
        self.assertEqual(
            predict_product_rule("PATE CỘT ĐÈN HẢI PHÒNG chính hãng"),
            "Pate Cột Đèn Hải Phòng",
        )

    def test_retry_only_for_weak_primary_result(self):
        self.assertTrue(should_retry_ocr("", detection_count=0, mean_confidence=0.0))
        self.assertTrue(should_retry_ocr("NAN", detection_count=1, mean_confidence=0.3))
        self.assertFalse(
            should_retry_ocr(
                "Highlands Coffee ngừng bán trà sen vàng",
                detection_count=4,
                mean_confidence=0.72,
            )
        )

    def test_composite_metric_matches_competition_weights(self):
        metrics = composite_metrics(
            [
                {
                    "ocr_text_gt": "ABC",
                    "ocr_text_pred": "ABC",
                    "product_name_gt": "Nestlé NAN",
                    "product_name_pred": "Nestlé NAN",
                },
                {
                    "ocr_text_gt": "",
                    "ocr_text_pred": "noise",
                    "product_name_gt": "",
                    "product_name_pred": "",
                },
            ]
        )
        self.assertEqual(metrics["product_f1"], 1.0)
        self.assertEqual(metrics["ocr_cer"], 0.5)
        self.assertEqual(metrics["composite"], 0.8)


if __name__ == "__main__":
    unittest.main()
