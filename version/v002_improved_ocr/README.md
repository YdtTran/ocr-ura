# v002 Improved OCR

## Result

| Field | Value |
|---|---|
| Date | 2026-06-08 |
| Kaggle score | TBD |
| Local score | TBD |
| Runtime | TBD |
| Notebook | `notebook.ipynb` |
| Submission | `submission.csv` |
| Checkpoint | `checkpoint.csv` |
| Raw OCR cache | `raw_ocr_detections.jsonl` |
| Kaggle download | `results.zip` |

## What Changed

- Added grouped label manifest and conflict audit under `artifacts/`.
- Canonicalized product labels for product-head training.
- Lowered OCR confidence threshold and preserved repeated OCR words.
- Added visual-line ordering for OCR boxes.
- Added retry OCR path for empty, short, or low-confidence primary outputs.
- Cached raw OCR detections for later threshold sweeps.
- Added more specific product rules for common brands and variants.

## Output Quality

- Good:
  - Raw OCR detections are available, so threshold sweeps and postprocessing changes can be tested without rerunning OCR.
  - Checkpoint includes metadata such as OCR pass, confidence, detection count, runtime, and config hash.
- Unknown:
  - Kaggle score has not been recorded yet.
  - Need manual review of the new `submission.csv` against the baseline output.
- Possible risks:
  - Lower OCR threshold may add noisy text.
  - Product rules may increase false positives on news/social-media text.
  - Retry path may increase runtime if too many images are retried.

## Verdict

Inconclusive until the Kaggle score is added. This version is structurally better
for analysis because it preserves raw OCR detections and run metadata.

## Suggested Next Improvements

- Record Kaggle score and compare product fill rate with `v001_baseline`.
- Inspect false-positive product predictions from generic brand rules.
- Run `evaluate_ocr.py` against any labeled checkpoint with the raw JSONL cache.
- Try threshold sweeps before changing EasyOCR preprocessing again.

