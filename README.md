# OCR Brainstorming Workspace

Lightweight workspace for brainstorming OCR competition improvements. The root is
kept intentionally small; code, scripts, notebooks, docs, data, tests, and
completed Kaggle attempts each have their own folder.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Workflow

Prepare audited training artifacts:

```powershell
python scripts/prepare_ocr_data.py
```

Run the test suite:

```powershell
python -m unittest discover -s tests
```

Sync helper code into the Kaggle notebook after changing `ocr_competition.py`:

```powershell
python scripts/sync_kaggle_notebook.py
```

Evaluate a labeled checkpoint:

```powershell
python scripts/evaluate_ocr.py --checkpoint data/train_ocr_500_checkpoint.csv
```

Archive each Kaggle run under `version/vXXX_short_name/` using the template in
`version/_TEMPLATE/README.md`.

See `docs/KAGGLE_OCR_WORKFLOW.md` and `docs/PROJECT_SUMMARY.md` for experiment
strategy.

## Folder Layout

| Folder | Purpose |
|---|---|
| `src/` | Shared OCR/product helper code |
| `scripts/` | Local prep, sync, and evaluation commands |
| `notebooks/` | Editable Kaggle notebook source |
| `docs/` | Analysis notes and workflow docs |
| `data/` | Local labels, link CSVs, sample images, and prepared artifacts |
| `tests/` | Unit tests for reusable helpers |
| `version/` | Archived Kaggle attempts and score notes |

## Version Archive

Each version folder should contain:

- `README.md`: score, change summary, output quality, and next ideas;
- `notebook.ipynb`: the notebook used for that Kaggle run;
- `checkpoint.csv`: raw OCR/product predictions when available;
- `submission.csv`: uploaded competition submission;
- `raw_ocr_detections.jsonl`: cached raw OCR boxes when available;
- `results.zip`: Kaggle output archive when downloaded.

## Local Data

Generated checkpoints, submissions, result archives, image samples, and raw OCR
detections at the workspace root are ignored by default. Move completed run
outputs into `version/` so they can be compared side by side.
