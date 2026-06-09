# v003 Brand Inference

## Result

| Field | Value |
|---|---|
| Date | 2026-06-08 |
| Kaggle score | TBD |
| Local score | TBD |
| Runtime | TBD |
| Kaggle paste file | `notebook-gencode.txt` |

## What Changed

- Started from `notebook-gencode-template/template.txt`.
- Kept OCR output separate from product inference: `ocr_text` is not padded with brand text.
- Reweighted product rules from `data/train_labels.csv` support:
  - `Đồ Hộp Hạ Long`: about 806 labeled rows.
  - `Nestlé/NAN`: about 678/603 labeled rows.
  - `Pate Cột Đèn`: about 492 labeled rows.
  - `Ha Long Canfoco`: about 169 labeled rows.
  - `Highlands Coffee`: about 109 labeled rows.
  - `Aptamil`: about 40 labeled rows.
- Added guarded brand inference for news/social OCR:
  - Specific product-family matches beat generic brand matches.
  - Long news-like OCR penalizes generic brand hits.
  - Multi-brand contexts avoid letting `Highlands Coffee` beat `Đồ Hộp Hạ Long` or `Pate` when those are the main issue context.
- Canonicalized training labels before fitting the sklearn product head, so the classifier does not emit long raw labels when a compact product family is available.
- Kept OCR, checkpointing, and sklearn product-head hyperparameters close to the template; the main change is product-rule priority and label canonicalization.

## Product Trade-off

The competition score gives product name more weight than OCR text:

```text
score = 0.6 * product_f1 + 0.4 * (1 - CER)
```

V003 therefore increases brand/product recall in `product_name`, while leaving `ocr_text` as OCR-only text to avoid hurting CER with invented text.

## Files

- `notebook-gencode.txt`: paste this into Kaggle as a notebook/script with `# %%` cells.

## Verification

- Cell 3 product rules were executed locally with smoke examples.
- Full Kaggle execution still needs the competition input mount and EasyOCR runtime.
