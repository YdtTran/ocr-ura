# Baseline Output vs. Label Analysis

## Critical Limitation

A direct row-by-row comparison between `submission.csv` and `train_labels.csv` is not possible because they contain different image IDs.

| File | ID ranges |
|---|---|
| `train_labels.csv` | `img_0001`-`img_2933`, `img_4558`-`img_6516` |
| `submission.csv` | `img_2934`-`img_4557`, `img_6517`-`img_6610`, `img_6613`-`img_6900` |

There are **zero overlapping IDs**. The submission is output for the held-out/test partition, not output on the labeled training rows.

`submission.csv` contains 2,006 unique IDs, the required three columns, and no blank IDs or embedded newlines. It differs from `checkpoint.csv` only because empty prediction fields were converted to a single space by the notebook's export step. After trimming whitespace, their predictions are equivalent.

`img_6611` and `img_6612` do not appear, but this is not evidence of missing submission rows without the authoritative `test.csv` or `sample_submission.csv`. The competition data may intentionally omit those IDs.

Therefore:

- OCR CER and score components cannot be calculated from the provided files.
- The submitted file received an observed competition score of **0.5264**.
- Individual submission predictions cannot be classified as correct or incorrect without the hidden test labels.
- Recommendations below combine submission distribution analysis with an exact evaluation of the notebook's regex stage on labeled OCR text.

## Observed Submission Result

| Submission | Competition score |
|---|---:|
| Current lightweight baseline | **0.5264** |

This score measures the complete pipeline on hidden labels: EasyOCR, preprocessing, postprocessing, rules, and the learned product head. The leaderboard score does not expose OCR CER and product token F1 separately, so it cannot identify which stage contributes most to the remaining error.

Use **0.5264** as the end-to-end baseline that future full submissions must beat. The `0.6033` result below is a different metric slice: product token F1 from rules alone using ground-truth training OCR. It must not be compared directly with the leaderboard score.

## Observable Distribution Differences

| Measure | Training labels | Submission output |
|---|---:|---:|
| Rows | 4,892 | 2,006 |
| Non-empty OCR | 79.8% | 79.0% |
| Non-empty product | 59.3% | 44.3% |
| Mean non-empty OCR length | 124.3 | 78.5 |
| Median non-empty OCR length | 100 | 56 |
| OCR texts over 200 characters | 638 | 88 |
| OCR texts of 1-5 characters | 115 | 122 |

The OCR fill rate is similar, but submission OCR is much shorter. This may reflect test-set distribution shift, EasyOCR missing text, or differences between human labels and machine output. It cannot be resolved without matching labels.

Submission OCR also contains substantial contextual noise:

- `2026` occurs in 171 rows.
- Social/news terms occur in at least 159 rows.
- URLs occur in 13 rows.
- 122 non-empty OCR outputs contain at most five characters.

This matters because the current product extraction scans the entire OCR string and treats every mention as a product signal.

## Exact Regex-Stage Evaluation

The notebook's 34 current `BRAND_RULES` were run on the ground-truth `ocr_text` from all 4,892 training rows.

This is not a full baseline evaluation because it bypasses EasyOCR and excludes the learned classifier. It isolates the behavior of the first, highest-priority product stage.

| Result | Rows |
|---|---:|
| Rule produced a product | 2,431 |
| Correct empty prediction | 1,787 |
| False positive | 206 |
| False negative | 674 |
| Label and prediction both non-empty | 2,225 |
| Exact match, case-insensitive | 623 |
| Partial token overlap | 1,130 |
| No token overlap | 472 |
| Mean product token F1 | 0.6033 |

The rules recover useful product tokens, but exact product selection is weak. Only 623 of 2,225 non-empty pairs match exactly after ignoring case.

## Main Error Patterns

### 1. Context Mentions Cause False Positives

The rules cannot distinguish a depicted product from a company or product mentioned in news text.

Example:

```text
Label:      ""
Prediction: "Đồ Hộp Hạ Long"
OCR:        "Highlands Coffee ... thông tin liên quan Đồ hộp Hạ Long"
```

Largest false-positive outputs:

| Prediction | Count |
|---|---:|
| `Đồ Hộp Hạ Long` | 71 |
| `Nestlé` | 56 |
| `Ha Long Canfoco` | 27 |
| `Pate` | 14 |
| `Nestlé Nan` | 14 |
| `Pate Cột Đèn Hải Phòng` | 12 |

The generic `pate`, `nestle`, and Hạ Long rules are especially vulnerable because they match common article text.

### 2. First-Match Rule Ordering Selects the Wrong Entity

`extract_product()` returns immediately on the first regex match. When OCR contains several entities, rule order determines the answer.

Frequent examples:

| Label | Rule output | Count |
|---|---|---:|
| `Đồ hộp Hạ Long` variants | `Ha Long Canfoco` | 228 |
| `Highlands Coffee` variants | `Đồ Hộp Hạ Long` | 57 |
| `Coffee House` | `Pate` | 14 |

Example:

```text
Label:      "Highlands Coffee"
Prediction: "Đồ Hộp Hạ Long"
OCR:        "Highlands Coffee là đối tác ... Công ty Cổ phần Đồ hộp Hạ Long"
```

The current rule system does not score competing candidates or use prominence, frequency, position, or OCR geometry.

### 3. Broad Rules Collapse Specific Labels

The current rules often return a brand or family when the label is a specific product.

Examples:

- `Nestlé NAN OPTIPRO PLUS` becomes `Nestlé Nan`.
- `Nestlé NAN INFINIPRO A2` becomes `Nestlé Nan`.
- `Sữa Nestle` becomes `Nestlé`.
- `PATE CỘT ĐÈN` becomes `Pate Cột Đèn Hải Phòng`.

This can preserve some token F1 but loses exact product information and may add incorrect tokens.

### 4. Important NAN Variants Are Missed

NAN-related labels account for a large portion of rule false negatives. At least 254 rows whose labels contain `NAN` produce no rule result.

Examples include:

- `NAN`
- `sữa NAN`
- `Nestlé NAN`
- `NAN OPTI pro PLUS 2`
- `Nestlé NAN OPTIPRO PLUS`
- `Nestlé NAN INFINIPRO A2`

The current `nestle|nestlé` rule requires Nestlé text. OCR that contains only `NAN` is not recognized.

### 5. Training Targets Are Fragmented

The 2,899 non-empty product labels contain:

- 495 raw strings;
- 412 strings after Unicode, whitespace, and case normalization;
- approximately 379 strings after loose accent, punctuation, and common-alias normalization.

Examples of equivalent labels split into separate classes:

| Semantic label | Rows | Raw variants |
|---|---:|---:|
| `Đồ Hộp Hạ Long` | 707 | At least 8 |
| `Pate Cột Đèn Hải Phòng` | 278 | Many casing/Unicode variants |
| `Sữa Nestlé` | 105 | 4 common variants |
| `Sữa NAN` | 101 | 5 common variants |
| `Highlands Coffee` | 78 | 7 common variants |

The logistic-regression head trains directly on these raw strings. Equivalent labels therefore compete as separate classes and reduce examples per class.

### 6. Submission Product Vocabulary Is Inconsistent

The submission contains 123 distinct non-empty product strings. Most come from training labels or rules, but output specificity varies significantly:

- `Ha Long Canfoco`
- `Đồ Hộp Hạ Long`
- `Pate`
- `Pate Cột Đèn Hải Phòng`
- `CÔNG TY CỔ PHẦN ĐỒ HỘP HẠ LONG`
- `Highlands Coffee trà sen vàng, trà vải`
- `sản phẩm đồ hộp`

This confirms that the learned head can reproduce noisy, sentence-like training targets while rules emit shorter canonical names.

## Recommendations

### Priority 1: Canonicalize Product Labels

Create a canonical target column before fitting `ProductPredictor`.

Normalize:

- Unicode composition;
- case and whitespace;
- `Nestle`/`Nestlé`;
- `pate`/`patê`;
- known casing variants;
- stable aliases such as `Highland Coffee`/`Highlands Coffee`.

Keep product variants such as `NAN OPTIPRO PLUS` and `NAN INFINIPRO A2` separate when they represent genuinely different products.

Expected benefit: more examples per real class, consistent submission output, and better token F1.  
Cost: seconds; no OCR rerun.

### Priority 2: Replace First-Match Rules with Candidate Scoring

Collect every matching candidate instead of returning the first match. Score candidates using:

- specificity of the regex;
- number of supporting tokens;
- repeated mentions;
- occurrence near the start of OCR;
- penalties for generic words such as `pate`;
- classifier confidence;
- OCR confidence and text-box size when geometry becomes available.

Expected benefit: fewer Hạ Long/Highlands/Coffee House entity-selection errors.  
Cost: seconds for label-only testing.

### Priority 3: Add NAN Rules and Specific Product-Line Rules

Add standalone `NAN` recognition and detect:

- `OPTI PRO` / `OPTIPRO` / `OPTIPROPLUS`;
- `INFINIPRO A2`;
- product stage numbers;
- common OCR spacing and accent variants.

Specific rules must run before generic Nestlé/NAN rules.

Expected benefit: addresses one of the largest measured false-negative groups.  
Cost: seconds.

### Priority 4: Make Generic Rules Conservative

Do not allow weak rules such as bare `pate`, `nestle`, or company mentions to override stronger candidates.

Options:

- require a second product token;
- require repeated occurrence;
- use generic matches only as classifier features;
- suppress product output when the OCR resembles a news article and evidence is weak.

Expected benefit: fewer false positives and generic misclassifications.  
Cost: seconds to minutes.

### Priority 5: Train and Validate on Canonical Targets

Add held-out cross-validation for the product head using ground-truth OCR text. Compare:

1. rules only;
2. classifier only;
3. current rules-first pipeline;
4. scored candidates plus classifier;
5. direct classifier with confidence rejection.

Tune `prob_threshold` against product token F1, not logistic loss.

Near-duplicate OCR strings must stay in the same fold to avoid optimistic validation.

Expected benefit: identifies whether rules actually improve the learned model.  
Cost: minutes, no OCR rerun.

### Priority 6: Preserve Raw OCR Detections

Keep the checkpoint and create a sidecar cache containing:

- detected text;
- confidence;
- bounding box;
- preprocessing configuration.

This allows experiments with confidence thresholds, reading order, noise filtering, and product extraction without repeating the three-hour OCR pass.

Expected benefit: converts many future OCR experiments from hours to seconds.  
Cost: requires one future OCR run.

### Priority 7: Filter News and Social Overlays

The submission frequently includes dates, news-source names, TikTok text, and URLs. Use confidence and geometry to downweight:

- corner watermarks;
- timestamps;
- usernames and URLs;
- repeated channel names;
- long article captions.

Do not hard-delete these tokens until tested on a labeled image sample, because some may contain the target product.

Expected benefit: cleaner OCR and fewer contextual product false positives.  
Cost: minutes on cached detections; full benefit requires images or raw boxes.

### Priority 8: Diagnose Short OCR Outputs

The submission has 122 non-empty outputs of five characters or fewer, almost as many as the larger training set. Review a small sample and test:

- a lower EasyOCR confidence threshold;
- alternate preprocessing;
- retry at a larger image scale;
- fallback OCR only for empty or very short outputs.

Expected benefit: targeted OCR recall improvement without slowing every image.  
Cost: small sampled rerun first.

### Priority 9: Validate Against the Official Submission Template

When `test.csv` or `sample_submission.csv` is available, verify exact ID-set equality and row count before upload. Do not infer missing rows from gaps in numeric IDs.

Expected benefit: prevents format or coverage rejection.  
Cost: seconds.

## Recommended Next Experiment

Before another end-to-end OCR run:

1. Canonicalize the product labels.
2. Add NAN/product-line candidates.
3. Replace first-match selection with candidate scoring.
4. Run grouped cross-validation on labeled OCR text.
5. Compare product token F1 for the five pipeline variants.
6. Select one configuration for the next full checkpointed OCR run.

This targets the measured weaknesses of the current baseline while avoiding the three-hour OCR cost.

For the next full submission, require:

- a leaderboard score above **0.5264**;
- no regression in runtime or model size without a documented accuracy tradeoff;
- one coherent set of changes so any score movement remains interpretable.
