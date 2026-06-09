# v001 Baseline

## Result

| Field | Value |
|---|---|
| Date | 2026-06-07 |
| Kaggle score | 0.5264 |
| Local score | 0.5284 on 500-image training sample |
| Runtime | About 3 hours end to end on CPU |
| Notebook | `notebook.ipynb` |
| Submission | `submission.csv` |
| Checkpoint | `checkpoint.csv` |
| Raw OCR cache | Not available |
| Kaggle download | Not available |

## What Changed

- Reference baseline from `lightweight-baseline-reference-starter.ipynb`.
- EasyOCR Vietnamese/English OCR with resize, contrast, and sharpen preprocessing.
- Product extraction used hand-written rules plus a TF-IDF/logistic-regression product head.

## Output Quality

- Good:
  - Reasonable OCR fill rate: root summary reports 1,585 non-empty OCR outputs from 2,006 predictions.
  - Competition score gives a stable baseline for later versions.
- Bad:
  - Product fill rate was lower than OCR fill rate.
  - Product labels were fragmented by casing and naming variants.
  - No raw OCR box cache, so OCR threshold/postprocessing experiments require rerunning OCR.

## Verdict

Useful baseline. Keep it as the comparison point, but do not treat it as a strong
solution because the workflow lacks leakage-aware offline validation and cached
OCR detections.

## Suggested Next Improvements

- Canonicalize product labels before training.
- Add grouped train/dev artifacts.
- Cache raw OCR detections so threshold and postprocessing changes can be tested cheaply.
- Tune product confidence threshold against token F1, not only classifier loss.

