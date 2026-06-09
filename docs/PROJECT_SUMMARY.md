# OCR Hackathon Brainstorming Summary

## Purpose

This folder is a lightweight workspace for exploring ideas for the OCR hackathon. It is not yet a production project or a record of validated improvements.

The main constraint is iteration cost: one end-to-end CPU run takes about **3 hours**. Ideas should therefore be screened with cheap offline analysis, small samples, or cached OCR output before committing to a full run.

## Current Assets

| File | Role |
|---|---|
| `lightweight-baseline-reference-starter.ipynb` | End-to-end reference baseline |
| `train_labels.csv` | Training labels with `image_id`, `ocr_text`, and `product_name` |
| `checkpoint.csv` | Partial output from a baseline run |
| `submission.csv` | Final scoring artifact exported from the checkpoint |
| `train_ocr_500_checkpoint.csv` | Cached baseline OCR and product predictions for 500 random training images |
| `TRAIN_OCR_500_ANALYSIS.md` | Error analysis and recommendations from the 500-image sample |

No image files or competition `test.csv` are currently present in this folder.

## Baseline Architecture

The notebook uses a two-stage CPU pipeline:

1. **OCR**
   - EasyOCR with Vietnamese and English models
   - Images are resized to a maximum dimension of 1280 pixels
   - Contrast is increased and a sharpening filter is applied
   - Detections below confidence `0.35` are discarded
   - Remaining text is ordered approximately top-to-bottom and left-to-right
   - Whitespace and consecutive duplicate tokens are normalized

2. **Product extraction**
   - Hand-written regex rules run first
   - If no rule matches, a binary classifier predicts whether a product is present
   - A character n-gram TF-IDF plus logistic-regression classifier predicts the product name
   - Product classes occurring fewer than three times are excluded from the multiclass model
   - The binary gate uses a probability threshold of `0.60`

The product head is small and fast. EasyOCR is the main runtime and model-size cost.

## Evaluation Objective

The notebook defines the composite score as:

```text
0.6 * product token F1 + 0.4 * (1 - OCR character error rate)
```

This makes product-name quality more important than OCR quality, but OCR errors also affect the downstream product classifier. Improvements should be measured separately for:

- OCR character error rate
- Product token F1
- Composite score
- Runtime per image
- Model size and memory

The current exported `submission.csv` received a competition score of **0.5264**. This is the end-to-end reference score for future submissions.

The 500-image training sample produced a local composite estimate of **0.5284**, with OCR CER `0.5008` and product token F1 `0.5480`. Product metrics on this sample are optimistic because the product model was trained on these rows, but the sample is still useful for OCR diagnosis and rapid ablation.

## Dataset Snapshot

`train_labels.csv` contains **4,892 rows**:

- OCR text present: 3,905 rows, or **79.8%**
- Product name present: 2,899 rows, or **59.3%**
- Both OCR text and product name present: **2,871 rows**
- OCR text present but product empty: **1,034 rows**
- Product present but OCR text empty: **28 rows**
- Raw product-name variants: **495 non-empty values**
- Product classes with at least three examples: **259**
- Singleton product classes: **168**

The product labels contain substantial casing and naming variation. Examples such as `ĐỒ HỘP HẠ LONG`, `Đồ hộp Hạ Long`, and `đồ hộp Hạ Long` are currently treated as different classes by the classifier even though they appear semantically equivalent.

This label fragmentation is likely one of the cheapest high-impact areas to investigate.

## Checkpoint Status

`checkpoint.csv` and `submission.csv` contain the same **2,006 predictions**:

- OCR text present: 1,585 rows, or **79.0%**
- Product name present: 889 rows, or **44.3%**
- Mean non-empty OCR length: 78.5 characters

The submission and training labels have **zero overlapping `image_id` values**, despite sharing the same ID format. The submission therefore cannot be scored directly against `train_labels.csv`; its labels are held out by the competition.

The submission differs from the checkpoint only by converting empty fields to a single space, as required by the notebook export step.

Treat the checkpoint as:

- cached OCR output for qualitative analysis;
- a source for prediction distribution and failure-pattern inspection;
- a resumable partial run if its matching test dataset is restored.

Do not treat it as validation evidence for model quality.

## Highest-Priority Ideas

### 1. Canonicalize Product Labels Before Training

Normalize case, Unicode, punctuation, spacing, accents where appropriate, and known aliases into a canonical product vocabulary.

Why first:

- It requires no OCR rerun.
- It reduces artificial class fragmentation.
- It gives the classifier more examples per real product.
- It aligns training targets with the token-F1 metric.

Cheap test:

- Build a canonicalization report showing which labels collapse together.
- Run cross-validation using only `train_labels.csv`.
- Compare raw-label and canonical-label product token F1.

### 2. Establish Offline Product-Head Cross-Validation

The current notebook trains on all labels and does not provide a held-out product-head estimate. Add grouped or stratified folds over the labeled OCR text.

Questions to test cheaply:

- Rules only versus classifier only versus rules plus classifier
- `min_class_count`
- TF-IDF feature count and n-gram range
- Binary-gate threshold
- Whether the gate helps compared with direct classification plus confidence rejection
- Character analyzer versus `char_wb`

The split must avoid leaking near-duplicate OCR text across train and validation folds.

### 3. Tune for the Actual Metric

The product classifier optimizes logistic loss, while evaluation uses token F1. Confidence thresholds and canonical output strings should be selected against product token F1 and the composite formula.

Possible improvement:

- Return a shorter canonical brand/product phrase when uncertain instead of an over-specific wrong label.
- Measure whether token overlap rewards this behavior.

### 4. Expand Rules from Observed Errors

Rules are nearly free at inference time, but additions should come from measured false negatives and false positives rather than a speculative brand list.

Workflow:

1. Cross-validate the classifier on ground-truth OCR text.
2. Rank frequent product mistakes.
3. Add narrow rules for stable, high-frequency patterns.
4. Re-run only offline scoring.

### 5. Use the Checkpoint for Qualitative OCR Error Mining

Without matching ground truth, the checkpoint can still reveal:

- repeated EasyOCR confusions;
- noisy social-media overlays;
- empty-output patterns;
- product-rule false positives;
- common irrelevant tokens;
- unusually short or long outputs.

This analysis can generate hypotheses, but each hypothesis still needs validation on labeled data or a small manually reviewed image sample.

## OCR Ideas Requiring Images

These should be tested on a small, fixed development sample before any full run:

- Compare no preprocessing, contrast-only, sharpening-only, and grayscale variants.
- Tune the EasyOCR confidence threshold around `0.20` to `0.50`.
- Preserve detection confidence and geometry for downstream filtering.
- Filter likely watermark, timestamp, and social-media overlay text.
- Try multiple image scales only on low-confidence cases.
- Crop likely product regions before OCR.
- Compare EasyOCR with a genuinely lightweight backend such as Tesseract or a mobile OCR model.
- Cache raw OCR detections so postprocessing and product experiments never rerun OCR.

Any backend comparison should report accuracy, runtime, memory, and model size on the same sample.

## Efficient Experiment Strategy

Use three levels of validation:

1. **Seconds: label-only experiments**
   - Product canonicalization
   - Rules
   - Classifier cross-validation
   - Threshold tuning

2. **Minutes: fixed image sample**
   - Use a representative sample with empty/non-empty products, short/long text, and major product families
   - Cache raw OCR detections
   - Compare preprocessing and OCR parameters

3. **About 3 hours: full end-to-end run**
   - Run only after a hypothesis wins at a cheaper level
   - Save checkpoints and timing metadata
   - Change one coherent group of variables per run

## Suggested Experiment Log

For each idea, record:

| Field | Description |
|---|---|
| Hypothesis | Why the change may improve the score |
| Data | Labels, sample IDs, or full dataset used |
| Baseline | Exact configuration being compared |
| Change | One clearly scoped modification |
| Product F1 | Held-out token F1 |
| OCR CER | Held-out character error rate, if available |
| Composite | Competition formula |
| Runtime | Total and per-image time |
| Model size | Serialized weights and dependencies |
| Decision | Keep, reject, or investigate |

## Immediate Next Milestone

Before another end-to-end run:

1. Define a canonical product-name mapping.
2. Build leakage-aware offline cross-validation for the product head.
3. Score rules-only and rules-plus-classifier variants.
4. Create a small image development set when images become available.
5. Cache raw OCR detections for that set.
6. Promote only measured improvements to the next full run.

## Current Unknowns

- The competition image files and matching test metadata are not in this folder.
- The submission's matching ground-truth labels are unavailable.
- The aggregate submission score is known (`0.5264`), but OCR CER and product token F1 are not available separately.
- The exact source of missing labels is unclear: true negatives, annotation gaps, or weak-label limitations.
- Duplicate or near-duplicate images/text may affect naive random cross-validation.

These unknowns should remain explicit until evidence is available.
