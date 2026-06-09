# Kaggle OCR Improvement Workflow

## 1. Prepare audited labels

```powershell
python scripts/prepare_ocr_data.py
```

This writes:

- `data/artifacts/label_manifest.csv`: grouped train/dev/excluded split and canonical product target;
- `data/artifacts/label_conflicts.csv`: identical URLs with conflicting labels;
- `data/artifacts/audit_summary.json`: reproducible split statistics.

`train_labels.csv` is never modified. Rows marked `excluded` must not be used for
validation or product-model training.

## 2. Sync the self-contained notebook

```powershell
python scripts/sync_kaggle_notebook.py
```

The notebook embeds the tested helpers from `src/ocr_competition.py`. Run this
command after changing the module.

For Kaggle, upload:

- `notebooks/lightweight-baseline-reference-starter.ipynb`;
- the `data/artifacts` directory as a small private Kaggle Dataset if it is not already
  available in the competition input.

If `label_manifest.csv` is unavailable, the notebook remains runnable but prints a
warning and trains the product head on all labels.

## 3. OCR behavior

The notebook:

- preserves repeated words;
- groups detection boxes into visual lines;
- uses confidence threshold `0.25`;
- upscales small images and keeps the baseline 1280 px cap for runtime control;
- retries only empty, short, or low-confidence outputs using contrast and sharpen;
- saves raw boxes and confidence to `raw_ocr_detections.jsonl`;
- stores a configuration hash in `checkpoint.csv`.

A checkpoint is resumed only when its configuration hash matches the current OCR
configuration.

## 4. Evaluate

Evaluate an existing labeled checkpoint:

```powershell
python scripts/evaluate_ocr.py --checkpoint data/train_ocr_500_checkpoint.csv
```

After running OCR on labeled images with raw detection caching:

```powershell
python scripts/evaluate_ocr.py `
  --checkpoint train_checkpoint.csv `
  --raw-jsonl raw_ocr_detections.jsonl
```

The evaluator reports all/trusted/dev metrics and sweeps OCR thresholds
`0.20,0.25,0.30,0.35,0.40` on cached boxes.

## 5. Promotion gates

Promote a configuration to a full Kaggle test run only when:

- grouped dev OCR CER improves;
- grouped dev product token F1 does not regress;
- the 500-image composite exceeds `0.5284`;
- retry remains below roughly 25% of images;
- the full checkpoint has no unexpected errors or configuration mismatch.
