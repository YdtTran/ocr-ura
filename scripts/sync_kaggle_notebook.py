from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "lightweight-baseline-reference-starter.ipynb"
MODULE = PROJECT_ROOT / "src" / "ocr_competition.py"


def lines(source: str) -> list[str]:
    parts = source.splitlines(keepends=True)
    if parts and not parts[-1].endswith("\n"):
        parts[-1] += "\n"
    return parts


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    module_source = MODULE.read_text(encoding="utf-8")

    notebook["cells"][6]["source"] = lines(
        module_source
        + """

# Compatibility name used by the product head and evaluation cells.
extract_product = predict_product_rule

rule_tests = [
    ("Nestle NAN OPTI pro PLUS digestion", "Nestlé NAN OPTIPRO PLUS"),
    ("PATE CỘT ĐÈN HẢI PHÒNG", "Pate Cột Đèn Hải Phòng"),
    ("CP vị hôi hôi bở bở", "CP"),
    ("No product in this text", ""),
]
for text, expected in rule_tests:
    got = extract_product(text)
    assert got == expected, (text, got, expected)
print(f"Scored product rules loaded: {len(_PRODUCT_RULES)}")
"""
    )

    notebook["cells"][8]["source"] = lines(
        """from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def find_manifest():
    candidates = [
        INPUT_DIR / "artifacts" / "label_manifest.csv",
        INPUT_DIR / "label_manifest.csv",
        Path("artifacts/label_manifest.csv"),
    ]
    if Path("/kaggle/input").exists():
        candidates.extend(sorted(Path("/kaggle/input").rglob("label_manifest.csv")))
    return next((path for path in candidates if path.exists()), None)


class ProductPredictor:
    def __init__(self, min_class_count=3, prob_threshold=0.55, max_features=3000):
        self.min_class_count = min_class_count
        self.prob_threshold = prob_threshold
        self.max_features = max_features
        self._has_clf = self._prod_clf = None

    def fit(self, train_labels, rule_fn):
        df = train_labels.copy()
        df["ocr_text"] = df["ocr_text"].astype(str).str.strip()
        self._alias_map = build_product_alias_map(df["product_name"])
        df["product_name"] = df["product_name"].map(
            lambda value: canonicalize_product(value, self._alias_map)
        )
        self._rule_fn = rule_fn
        self._has_clf = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                      max_features=self.max_features, min_df=2)),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
        ])
        self._has_clf.fit(df["ocr_text"], (df["product_name"] != "").astype(int))
        pos = df[(df["ocr_text"] != "") & (df["product_name"] != "")]
        keep = pos["product_name"].value_counts()
        pos = pos[pos["product_name"].isin(keep[keep >= self.min_class_count].index)]
        self._prod_clf = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                      max_features=self.max_features, min_df=2)),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
        ])
        if len(pos):
            self._prod_clf.fit(pos["ocr_text"], pos["product_name"])
        self._n_train = len(df)
        self._n_classes = pos["product_name"].nunique() if len(pos) else 0
        return self

    def predict(self, ocr_text):
        ocr_text = clean_text(ocr_text)
        if not ocr_text:
            return ""
        ruled = self._rule_fn(ocr_text)
        if ruled:
            return ruled
        if self._has_clf is None or self._prod_clf is None:
            return ""
        classes = list(self._has_clf.classes_)
        proba = self._has_clf.predict_proba([ocr_text])[0]
        if 1 not in classes or proba[classes.index(1)] < self.prob_threshold:
            return ""
        return str(self._prod_clf.predict([ocr_text])[0])


product_predictor = None
product_train_df = train_labels_df
manifest_path = find_manifest()
if train_labels_df is not None and manifest_path is not None:
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    train_ids = set(manifest.loc[manifest["split"] == "train", "image_id"])
    product_train_df = train_labels_df[train_labels_df["image_id"].isin(train_ids)].copy()
    print(f"Using trusted grouped train split: {len(product_train_df):,} rows")
elif train_labels_df is not None:
    print("Warning: label_manifest.csv not found; training on all labels.")

if product_train_df is not None:
    product_predictor = ProductPredictor(
        min_class_count=3,
        prob_threshold=0.55,
        max_features=3000,
    ).fit(product_train_df, extract_product)
    print(
        f"Product head: {product_predictor._n_train:,} rows, "
        f"{product_predictor._n_classes:,} canonical classes"
    )
    if manifest_path is not None:
        dev_ids = set(manifest.loc[manifest["split"] == "dev", "image_id"])
        product_dev_df = train_labels_df[train_labels_df["image_id"].isin(dev_ids)]
        threshold_scores = []
        for threshold in [0.35, 0.45, 0.55, 0.60, 0.70]:
            product_predictor.prob_threshold = threshold
            predictions = product_dev_df["ocr_text"].map(product_predictor.predict)
            score = sum(
                token_f1(gt, pred)
                for gt, pred in zip(product_dev_df["product_name"], predictions)
            ) / max(len(product_dev_df), 1)
            threshold_scores.append((score, threshold))
        best_score, best_threshold = max(threshold_scores)
        product_predictor.prob_threshold = best_threshold
        print(
            f"Grouped-dev product threshold={best_threshold:.2f}, "
            f"token F1={best_score:.4f}"
        )


def predict_product(ocr_text: str) -> str:
    if product_predictor is not None:
        return product_predictor.predict(ocr_text)
    return extract_product(ocr_text)
"""
    )

    notebook["cells"][10]["source"] = lines(
        """import easyocr
import json
import numpy as np

reader = easyocr.Reader(["vi", "en"], gpu=False, verbose=False)
OCR_CONFIG = {
    "primary_threshold": 0.25,
    "retry_threshold": 0.20,
    "min_long_side": 720,
    "max_long_side": 1280,
    "retry_min_chars": 6,
    "retry_min_confidence": 0.32,
}
CONFIG_HASH = hashlib.sha256(
    json.dumps(OCR_CONFIG, sort_keys=True).encode()
).hexdigest()[:12]
RAW_OCR_JSONL = WORK_DIR / "raw_ocr_detections.jsonl"
print(f"EasyOCR loaded; config={CONFIG_HASH}")


def load_image(image_id: str):
    path = IMAGES_DIR / f"{image_id}.jpg"
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def prepare_image(img, retry=False):
    size = resize_dimensions(
        *img.size,
        min_long_side=OCR_CONFIG["min_long_side"],
        max_long_side=OCR_CONFIG["max_long_side"],
    )
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    if retry:
        img = ImageEnhance.Contrast(img).enhance(1.35)
        img = img.filter(ImageFilter.SHARPEN)
    return img


def run_ocr_pass(img, pass_name, threshold):
    raw = reader.readtext(np.array(img), detail=1, paragraph=False)
    selected = select_detections(raw, confidence_threshold=threshold)
    summary = detection_summary(selected)
    return {
        "pass_name": pass_name,
        "text": detections_to_text(selected),
        "detection_count": summary["detection_count"],
        "mean_confidence": summary["mean_confidence"],
        "raw_detections": [normalize_detection(item) for item in raw],
    }


def run_ocr(image_id: str) -> dict:
    started = time.perf_counter()
    img = load_image(image_id)
    if img is None:
        return {
            "image_id": image_id, "ocr_text": "", "product_name": "",
            "runtime_seconds": time.perf_counter() - started,
            "ocr_pass": "missing_image", "detection_count": 0,
            "mean_confidence": 0.0, "error": "missing_or_invalid_image",
            "raw_passes": [],
        }
    try:
        primary = run_ocr_pass(
            prepare_image(img, retry=False),
            "primary",
            OCR_CONFIG["primary_threshold"],
        )
        retry = None
        if should_retry_ocr(
            primary["text"],
            primary["detection_count"],
            primary["mean_confidence"],
            min_chars=OCR_CONFIG["retry_min_chars"],
            min_mean_confidence=OCR_CONFIG["retry_min_confidence"],
        ):
            retry = run_ocr_pass(
                prepare_image(img, retry=True),
                "retry",
                OCR_CONFIG["retry_threshold"],
            )
        chosen = choose_ocr_pass(primary, retry)
        text = postprocess_ocr(chosen["text"])
        return {
            "image_id": image_id,
            "ocr_text": text,
            "product_name": predict_product(text),
            "runtime_seconds": time.perf_counter() - started,
            "ocr_pass": chosen["pass_name"],
            "detection_count": chosen["detection_count"],
            "mean_confidence": chosen["mean_confidence"],
            "error": "",
            "raw_passes": [item for item in (primary, retry) if item is not None],
        }
    except Exception as exc:
        return {
            "image_id": image_id, "ocr_text": "", "product_name": "",
            "runtime_seconds": time.perf_counter() - started,
            "ocr_pass": "error", "detection_count": 0,
            "mean_confidence": 0.0, "error": repr(exc), "raw_passes": [],
        }


print("\\nSmoke test on first image...")
smoke = run_ocr(test_df["image_id"].iloc[0])
print({key: value for key, value in smoke.items() if key != "raw_passes"})
"""
    )

    notebook["cells"][12]["source"] = lines(
        """SAVE_EVERY = 50
CHECKPOINT_COLUMNS = [
    "image_id", "ocr_text", "product_name", "runtime_seconds", "ocr_pass",
    "detection_count", "mean_confidence", "error", "config_hash",
]
done_ids = set()
results = []

if CHECKPOINT_CSV.exists():
    ckpt = pd.read_csv(CHECKPOINT_CSV, keep_default_na=False)
    hashes = set(ckpt.get("config_hash", pd.Series(dtype=str)).astype(str))
    if hashes == {CONFIG_HASH}:
        done_ids = set(ckpt["image_id"])
        results = ckpt.to_dict("records")
        print(f"Resuming matching checkpoint: {len(done_ids):,} images")
    else:
        print("Ignoring checkpoint from a different OCR configuration")

pending = [image_id for image_id in test_df["image_id"] if image_id not in done_ids]
print(f"Pending: {len(pending):,} | Done: {len(done_ids):,}")

raw_mode = "a" if done_ids and RAW_OCR_JSONL.exists() else "w"
with RAW_OCR_JSONL.open(raw_mode, encoding="utf-8") as raw_handle:
    for idx, image_id in enumerate(tqdm(pending, desc="OCR Progress")):
        result = run_ocr(image_id)
        raw_handle.write(json.dumps({
            "image_id": image_id,
            "config_hash": CONFIG_HASH,
            "passes": result.pop("raw_passes"),
        }, ensure_ascii=False) + "\\n")
        result["config_hash"] = CONFIG_HASH
        results.append(result)
        if (idx + 1) % SAVE_EVERY == 0:
            pd.DataFrame(results, columns=CHECKPOINT_COLUMNS).to_csv(
                CHECKPOINT_CSV, index=False, encoding="utf-8"
            )
            raw_handle.flush()

pd.DataFrame(results, columns=CHECKPOINT_COLUMNS).to_csv(
    CHECKPOINT_CSV, index=False, encoding="utf-8"
)
df_result = pd.DataFrame(results)
print(f"Processed     : {len(df_result):,}")
print(f"OCR fill rate : {(df_result['ocr_text'].str.strip() != '').mean():.1%}")
print(f"Product fill  : {(df_result['product_name'].str.strip() != '').mean():.1%}")
print(f"Retry selected: {(df_result['ocr_pass'] == 'retry').mean():.1%}")
print(f"Errors        : {(df_result['error'].str.strip() != '').sum():,}")
"""
    )

    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
