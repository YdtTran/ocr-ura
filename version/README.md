# Version Archive

This folder is the comparison log for OCR brainstorming runs. Keep one folder per
Kaggle attempt so scores, notebooks, outputs, and follow-up ideas stay together.

## Index

| Version | Status | Kaggle score | Main change | Verdict |
|---|---|---:|---|---|
| `v001_baseline` | archived | 0.5264 | Reference baseline from the initial workflow | Useful baseline |
| `v002_improved_ocr` | awaiting score | TBD | Cached OCR detections, grouped label artifacts, lower OCR threshold, retry path, product canonicalization/rules | Need Kaggle score |

`score_log.csv` keeps the same comparison data in a quick-edit format.

## How to Add a Version

1. Copy `version/_TEMPLATE` to `version/vXXX_short_name`.
2. Put the Kaggle notebook for that run in `notebook.ipynb`.
3. Add `checkpoint.csv`, `submission.csv`, `raw_ocr_detections.jsonl`, and
   `results.zip` when available.
4. Fill in the score, observed output quality, what improved, what regressed,
   and recommended next experiment in that version's `README.md`.
5. Add one row to `score_log.csv`.
