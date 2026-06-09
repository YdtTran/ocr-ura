# Analysis of `train_ocr_500_checkpoint.csv`

## Scope and Caveats

The checkpoint contains 500 unique training images, no runtime errors, and the expected label and prediction columns.

Two caveats matter:

1. OCR CER is a direct comparison between EasyOCR output and `train_labels.csv`.
2. Product metrics are optimistic because the product classifier was trained on all of `train_labels.csv`, including these sampled rows. They are useful for diagnosis and ablation, but not as an unbiased validation score.

## Main Results

| Metric | Result |
|---|---:|
| Images | 500 |
| Mean OCR CER | **0.5008** |
| OCR component, `1 - CER` | **0.4992** |
| Product token F1, full baseline | **0.5480** |
| Product token F1, rules only | **0.4517** |
| Sample composite, full baseline | **0.5284** |
| Sample composite, rules only | **0.4707** |
| Leaderboard score | **0.5264** |
| Mean runtime | 4.812 seconds/image |
| Total runtime | 40.1 minutes |

The sample composite is close to the leaderboard score, so the sample appears useful for rapid iteration. This does not remove the product-model leakage caveat.

At the observed speed, 2,006 test images take approximately 2.7 hours, consistent with the full baseline runtime.

## Does Training Help?

Yes. On the same OCR output:

- the trained product head raises product F1 from `0.4517` to `0.5480`;
- the corresponding composite estimate increases from `0.4707` to `0.5284`;
- the estimated composite gain is **0.0577**.

The trained model added 111 non-empty predictions where rules returned nothing:

| Outcome | Rows |
|---|---:|
| Better than rules | 79 |
| Same score as rules | 22 |
| Worse than rules | 10 |
| Exact product match | 35 |
| False positive | 10 |

The training step is therefore useful and should not be removed. It needs cleaner targets and honest held-out validation.

## OCR Findings

| OCR outcome | Rows |
|---|---:|
| Label OCR non-empty | 400 |
| Predicted OCR non-empty | 397 |
| Both empty | 70 |
| Label non-empty, prediction empty | 33 |
| Label empty, prediction non-empty | 30 |
| Exact trimmed match | 83 |

CER distribution:

| CER threshold | Rows at or below threshold |
|---|---:|
| `0.00` | 83 |
| `0.10` | 108 |
| `0.25` | 163 |
| `0.50` | 256 |
| `0.75` | 333 |
| `1.00` | 500 |

For rows with non-empty OCR labels, mean CER is `0.5510`. For rows where both label and prediction are non-empty, it is `0.5107`.

The problem is not only missed text. EasyOCR sometimes extracts extra contextual text where the label is empty or contains only selected text. Improving the score therefore requires text selection and noise filtering, not merely increasing OCR recall.

## Product Findings

| Product outcome | Rows |
|---|---:|
| Label product non-empty | 306 |
| Predicted product non-empty | 247 |
| Both empty | 165 |
| False negative | 88 |
| False positive | 29 |
| Both non-empty | 218 |
| Exact match, case-insensitive | 221 |

Product quality strongly depends on OCR quality:

| CER range | Rows | Product token F1 |
|---|---:|---:|
| `0.00`-`0.25` | 163 | **0.7798** |
| `0.25`-`0.50` | 89 | 0.4233 |
| `0.50`-`0.75` | 81 | 0.4822 |
| `0.75`-`1.00` | 167 | 0.4200 |

Of the 88 product false negatives:

- 20 have empty OCR output;
- 68 have non-empty OCR output;
- 60 have CER above `0.75`;
- 10 have CER at or below `0.25`.

Most false negatives are associated with poor OCR. The 10 low-CER false negatives are strong candidates for fixing rules, product-label normalization, or the binary gate threshold.

Frequent false-negative labels include:

- `Pate Cột Đèn Hải Phòng`;
- `Đồ hộp Hạ Long`;
- `NAN`;
- `Sữa Nestle`;
- `Nestlé`;
- `PATE GAN VISSAN`.

## Rule Findings

Rules produced 136 non-empty outputs:

- mean product F1 on those rows: `0.3737`;
- false positives: 19.

Rules remain useful for deterministic high-confidence matches, but broad first-match rules are weak on news and social-media images containing several company or product mentions.

The trained model produced 10 additional false positives, including outputs based on contextual mentions of NAN, Hạ Long, Highlands, and Nestlé products.

## Recommendations

### 1. Keep the Trained Product Head

The measured ablation shows a substantial gain over rules-only prediction. Removing training would likely lower the score.

### 2. Create an Honest Product Validation Split

Select the 500 IDs before training and exclude them from `ProductPredictor.fit()`. Then evaluate the product head on their OCR output.

Without this split, product metrics partly measure memorization of the same training rows.

### 3. Optimize OCR and Text Selection First

OCR is the largest measured bottleneck. Prioritize:

- reviewing the 33 non-empty labels with empty OCR;
- filtering watermarks, dates, usernames, URLs, and news-channel text;
- retaining bounding boxes and confidence values;
- retrying only empty or very short OCR outputs at another scale;
- comparing contrast-only, sharpen-only, and no-preprocessing variants on a smaller fixed subset.

Do not run another 500-image experiment for every preprocessing change. Use 50-100 difficult images first.

### 4. Tune the Product Gate Using Cached OCR

The baseline uses `prob_threshold=0.60`. Re-run the product head on `pred_ocr_text` with thresholds such as:

```text
0.35, 0.45, 0.50, 0.55, 0.60, 0.70
```

This requires no OCR rerun. Optimize token F1 while monitoring the false-positive count.

### 5. Canonicalize Product Labels Before Training

Merge casing, Unicode, accent, and spelling variants such as:

- `Nestle` and `Nestlé`;
- `pate` and `patê`;
- casing variants of `Đồ Hộp Hạ Long`;
- `Highland Coffee` and `Highlands Coffee`.

Keep genuinely different product lines such as NAN OPTIPRO and NAN INFINIPRO separate.

### 6. Make Rules More Conservative

Replace unconditional first-match behavior with candidate scoring. Bare terms such as `pate`, `nestle`, and company names should not automatically win when OCR resembles a news article.

Use:

- rule specificity;
- number of supporting tokens;
- repeated mentions;
- text position and bounding-box size;
- classifier confidence.

### 7. Add Targeted NAN and Pate Variants

The error list confirms recurring misses for NAN and Pate Cột Đèn variants. Add normalized patterns for OCR variants, but evaluate them against false positives before promoting them.

### 8. Preserve Raw OCR Metadata

The current checkpoint contains only final text. Future OCR checkpoints should also retain:

- text for each detected box;
- confidence;
- bounding box;
- accepted/rejected status.

That enables most OCR postprocessing experiments without another multi-hour EasyOCR run.

## Recommended Next Experiment

Use the existing 500 cached OCR strings and perform a product-only experiment:

1. remove these 500 IDs from product training;
2. canonicalize product labels;
3. train the product head on the remaining rows;
4. sweep the binary threshold;
5. compare rules-only, current hybrid, and canonicalized hybrid;
6. select the best configuration by held-out product token F1.

This experiment should take minutes and directly tests the product stage without paying the OCR cost again.
