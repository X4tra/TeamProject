#!/usr/bin/env python3
"""
Train a binary classifier for relations between bounding boxes and
export error analysis (FP/FN) with image overlays.

- Split by image first to avoid image-level leakage.
- Only use boxes whose labels are in: schematic, icon, text, arrows.
- Drop tables and any other disallowed labels completely.
- Hard-negative downsampling with tunable parameters.
- Save:
    - pairs_dataset.csv
    - pair_classifier.joblib
    - pairs_with_predictions.csv (test set + predictions)
    - top-k FP/FN overlays in images/errors/{FP,FN}/
"""

import json
import re
import math
import itertools
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from PIL import Image, ImageDraw


with open('result.json', 'r') as f:
    tdata = json.load(f)
    data = tdata["images"]

res = {}

for e in data:
    eid = e["id"]
    ename = re.search(r"(?<=/)[^/]*(?=\.jpg)", e["file_name"]).group()
    res[f"{ename}"] = eid




# ── 1. Label & debug configuration ───────────────────────────────────────────

SCHEMATIC_LABEL = "schematic"      # or "large_symbol" if that's your label
ICON_LABEL = "images"

TEXT_LABELS = {
    "hn_text", "hdim_text", "mm_text", "text", "text_bloc",
    "vtext", "vdim_text", "vn_text",
}

ARROW_LABELS = {
    "ang_arrow", "cir_arrow", "dh_arrow",
    "dv_arrow", "sh_arrow", "sv_arrow",
}

TABLE_LABELS = {
    "table", "table_body", "table_header", "cell_table", "irregular_table",
}

# 'other' excluded
ALLOWED_LABELS = (
    {SCHEMATIC_LABEL, ICON_LABEL} |
    TEXT_LABELS |
    ARROW_LABELS
)

# Hard-negative sampling knobs
NEAR_DIST_THRESHOLD = 0.2
NEAR_IOU_THRESHOLD  = 0.0
FAR_NEG_KEEP_PROB   = 0.10

# Error overlay config
TOP_K_ERRORS = 200

# Label Studio local-files prefix
URL_PREFIX = "/data/local-files/?d="


# ── 2. Load data ─────────────────────────────────────────────────────────────

with open("result_v10.json", "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} annotated images.")


# ── 3. Feature extraction ────────────────────────────────────────────────────

def compute_features(bP, bQ, iw, ih):
    def to_abs(b):
        x = b["x"] / 100 * iw
        y = b["y"] / 100 * ih
        w = b["width"]  / 100 * iw
        h = b["height"] / 100 * ih
        return x + w/2, y + h/2, w, h, x, y

    cxP, cyP, wP, hP, x1P, y1P = to_abs(bP)
    cxQ, cyQ, wQ, hQ, x1Q, y1Q = to_abs(bQ)

    dist = math.sqrt((cxP - cxQ)**2 + (cyP - cyQ)**2)
    diag = math.sqrt(iw**2 + ih**2)

    ix1 = max(x1P, x1Q); iy1 = max(y1P, y1Q)
    ix2 = min(x1P + wP, x1Q + wQ); iy2 = min(y1P + hP, y1Q + hQ)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = wP*hP + wQ*hQ - inter
    iou = inter / union if union > 0 else 0.0

    theta = math.atan2(cyQ - cyP, cxQ - cxP)

    return dict(
        d_norm       = dist / diag,
        x_P_norm     = cxP / iw,  y_P_norm = cyP / ih,
        x_Q_norm     = cxQ / iw,  y_Q_norm = cyQ / ih,
        w_P_norm     = wP  / iw,  h_P_norm = hP  / ih,
        w_Q_norm     = wQ  / iw,  h_Q_norm = hQ  / ih,
        iou          = iou,
        theta        = theta,
        delta_x_norm = abs(cxP - cxQ) / iw,
        delta_y_norm = abs(cyP - cyQ) / ih,
        area_P_norm  = (wP * hP) / (iw * ih),
        area_Q_norm  = (wQ * hQ) / (iw * ih),
        area_ratio   = (wP * hP) / (wQ * hQ + 1e-6),
        aspect_P     = wP / (hP + 1e-6),
        aspect_Q     = wQ / (hQ + 1e-6),
    )


# ── 4. Split by image, then build pairs ──────────────────────────────────────

image_items = []
for sample in data:
    ann_src = sample.get("predictions") or sample.get("annotations") or []
    if not ann_src:
        continue
    results = ann_src[0].get("result", [])
    if not any(r["type"] == "rectanglelabels" for r in results):
        continue
    image_items.append(sample)

train_imgs, test_imgs = train_test_split(
    image_items, test_size=0.2, random_state=42
)

def build_pairs(samples, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    rows = []

    for sample in samples:
        ann_src = sample.get("predictions") or sample.get("annotations") or []
        if not ann_src:
            continue
        results = ann_src[0].get("result", [])

        boxes = {r["id"]: r for r in results if r["type"] == "rectanglelabels"}
        if not boxes:
            continue

        def box_label(r):
            return r["value"]["rectanglelabels"][0]

        allowed_ids = [
            bid for bid, r in boxes.items()
            if (box_label(r) in ALLOWED_LABELS) and (box_label(r) not in TABLE_LABELS)
        ]
        if not allowed_ids:
            continue

        pos_pairs = set()
        for r in results:
            if r["type"] != "relation":
                continue
            f, t = r["from_id"], r["to_id"]
            if f in allowed_ids and t in allowed_ids:
                pos_pairs.add((f, t))

        iw = next(iter(boxes.values()))["original_width"]
        ih = next(iter(boxes.values()))["original_height"]
        img = sample["data"].get("image", "")

        for idP, idQ in itertools.permutations(sorted(allowed_ids), 2):
            bP = boxes[idP]["value"]
            bQ = boxes[idQ]["value"]
            lP = bP["rectanglelabels"][0]
            lQ = bQ["rectanglelabels"][0]

            if lP in TABLE_LABELS or lQ in TABLE_LABELS:
                continue
            if lP not in ALLOWED_LABELS or lQ not in ALLOWED_LABELS:
                continue

            feats = compute_features(bP, bQ, iw, ih)
            is_pos = (idP, idQ) in pos_pairs

            if not is_pos:
                near = (feats["d_norm"] <= NEAR_DIST_THRESHOLD) or (feats["iou"] > NEAR_IOU_THRESHOLD)
                if (not near) and (rng.random() > FAR_NEG_KEEP_PROB):
                    continue

            rows.append({
                "image": img,
                "id_P": idP,
                "id_Q": idQ,
                "label_P": lP,
                "label_Q": lQ,
                **feats,
                "target": int(is_pos),
            })
    return pd.DataFrame(rows)

print("Building training pairs...")
df_train = build_pairs(train_imgs, rng_seed=42)
print("Building test pairs...")
df_test = build_pairs(test_imgs, rng_seed=123)

df = pd.concat([df_train, df_test], ignore_index=True)

n_pos = df.target.sum()
n_neg = (df.target == 0).sum()
print(f"Total pairs : {len(df):,}")
print(f"Positives   : {n_pos:,} ({100*n_pos/len(df):.2f}%)")
print(f"Negatives   : {n_neg:,} ({100*n_neg/len(df):.2f}%)")
print(f"Imbalance ratio (neg/pos): {n_neg/n_pos:.1f}x")

df.to_csv("pairs_dataset.csv", index=False)
print("Saved -> pairs_dataset.csv")


# ── 5. Prepare features ──────────────────────────────────────────────────────

df = pd.get_dummies(df, columns=["label_P", "label_Q"])

allowed_suffixes = (
    {SCHEMATIC_LABEL, ICON_LABEL} |
    TEXT_LABELS |
    ARROW_LABELS
)

def is_allowed_label_col(col):
    if not (col.startswith("label_P_") or col.startswith("label_Q_")):
        return True
    return any(col.endswith(suf) for suf in allowed_suffixes)

EXCLUDE = {"image", "id_P", "id_Q", "target"}
FEATURE_COLS = [
    c for c in df.columns
    if (c not in EXCLUDE) and is_allowed_label_col(c)
]

X = df[FEATURE_COLS].values.astype(np.float32)
y = df["target"].values

n_train = len(df_train)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]


# ── 6. Train/val split and class imbalance ───────────────────────────────────

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

train_neg = (y_tr == 0).sum()
train_pos = (y_tr == 1).sum()
scale_pos_weight = train_neg / train_pos
print(f"\nscale_pos_weight set to {scale_pos_weight:.2f}")


# ── 7. Scale & train ─────────────────────────────────────────────────────────

scaler = StandardScaler()
X_tr_scaled   = scaler.fit_transform(X_tr)
X_val_scaled  = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

clf = XGBClassifier(
    n_estimators          = 500,
    max_depth             = 6,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    scale_pos_weight      = scale_pos_weight,
    eval_metric           = "aucpr",
    early_stopping_rounds = 20,
    random_state          = 42,
    n_jobs                = -1,
)

clf.fit(
    X_tr_scaled, y_tr,
    eval_set=[(X_val_scaled, y_val)],
    verbose=50,
)

model = Pipeline([("scaler", scaler), ("clf", clf)])


# ── 8. Threshold search & evaluation ─────────────────────────────────────────

y_pred_prob = model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1_scores      = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx       = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\nDefault threshold  : 0.50")
print(f"Optimal threshold  : {best_threshold:.4f}")
print(f"Best F1 (positive) : {f1_scores[best_idx]:.4f}")

y_pred_default = (y_pred_prob >= 0.50).astype(int)
y_pred_optimal = (y_pred_prob >= best_threshold).astype(int)

print("\n── Classification Report (default threshold = 0.50) ──")
print(classification_report(y_test, y_pred_default,
                            target_names=["No Relation", "Related"]))

print("── Classification Report (optimal threshold) ──")
print(classification_report(y_test, y_pred_optimal,
                            target_names=["No Relation", "Related"]))

print(f"ROC-AUC : {roc_auc_score(y_test, y_pred_prob):.4f}")
print(f"PR-AUC  : {average_precision_score(y_test, y_pred_prob):.4f}")


# ── 9. Feature importances ───────────────────────────────────────────────────

importances = pd.Series(
    model.named_steps["clf"].feature_importances_,
    index=FEATURE_COLS
)

print("\n── Top 20 Features by Importance ──")
print(importances.nlargest(20).to_string())


# ── 10. Error analysis export ────────────────────────────────────────────────

df_test_pairs = df.iloc[n_train:].copy()
df_test_pairs["y_true"] = y_test
df_test_pairs["y_prob"] = y_pred_prob
df_test_pairs["y_pred"] = y_pred_optimal

df_test_pairs["error_type"] = "TN"
df_test_pairs.loc[(df_test_pairs.y_true == 1) & (df_test_pairs.y_pred == 0), "error_type"] = "FN"
df_test_pairs.loc[(df_test_pairs.y_true == 0) & (df_test_pairs.y_pred == 1), "error_type"] = "FP"
df_test_pairs.loc[(df_test_pairs.y_true == 1) & (df_test_pairs.y_pred == 1), "error_type"] = "TP"

df_test_pairs.to_csv("pairs_with_predictions.csv", index=False)
print("Saved -> pairs_with_predictions.csv")


# ── 11. Build rect index using basenames ─────────────────────────────────────

def extract_filename(raw_img: str) -> str:
    """
    From '/data/local-files/?d=/images/foo.jpg' -> 'foo.jpg'.
    """
    if raw_img.startswith(URL_PREFIX):
        raw_img = raw_img[len(URL_PREFIX):]  # '/images/foo.jpg'
    return Path(raw_img).name                # 'foo.jpg'

def build_rect_index(samples):
    image_to_results = {}
    for sample in samples:
        ann_src = sample.get("predictions") or sample.get("annotations") or []
        if not ann_src:
            continue
        results = ann_src[0].get("result", [])
        raw_img = sample["data"].get("image", "")
        if not raw_img:
            continue
        fname = extract_filename(raw_img)    # 'foo.jpg'
        rects = {r["id"]: r for r in results if r["type"] == "rectanglelabels"}
        if rects:
            image_to_results[fname] = rects
    return image_to_results

image_to_results = build_rect_index(data)

print("\n[DEBUG] image_to_results keys (first 5 basenames):")
print(list(image_to_results.keys())[:5])

base_img_dir = Path("images")
out_dir_fp = base_img_dir / "errors" / "FP"
out_dir_fn = base_img_dir / "errors" / "FN"
out_dir_fp.mkdir(parents=True, exist_ok=True)
out_dir_fn.mkdir(parents=True, exist_ok=True)

fps = df_test_pairs[df_test_pairs["error_type"] == "FP"].copy()
fns = df_test_pairs[df_test_pairs["error_type"] == "FN"].copy()

fps = fps.sort_values("y_prob", ascending=False).head(TOP_K_ERRORS)
fns = fns.sort_values("y_prob", ascending=True).head(TOP_K_ERRORS)

fp_fn = pd.concat([fps, fns])

print(f"[DEBUG] Using {len(fp_fn)} rows for overlays")
print("[DEBUG] Example image values among fp_fn:")
print(fp_fn["image"].head().to_string(index=False))

seen = []

def save_file(tname):
    iid = 0
    while f"{tname}_{iid}" in seen:
        iid += 1

    seen.append(f"{tname}_{iid}")

    if tname in res:
        if row["error_type"] == "FP":
            out_path =  out_dir_fp / f"{res[f"{tname}"]}_{iid}_{tname}_FP.png"
        else:
            out_path =  out_dir_fn / f"{res[f"{tname}"]}_{iid}_{tname}_FN.png"
    else:
        if row["error_type"] == "FP":
            out_path =  out_dir_fp / f"{tname}_{iid}_FP.png"
        else:
            out_path = out_dir_fn / f"{tname}_{iid}_FN.png"
    
    return out_path


def draw_pair_overlay(row, idx):
    raw_img = row["image"]
    fname = extract_filename(raw_img)        # 'foo.jpg'
    img_path = base_img_dir / fname         # images/foo.jpg

    if not img_path.exists():
        print(f"[DEBUG] Missing image file: {img_path}")
        return

    rects = image_to_results.get(fname)
    if rects is None:
        print(f"[DEBUG] No rects for filename key: {fname}")
        return

    idP = row["id_P"]
    idQ = row["id_Q"]
    if idP not in rects or idQ not in rects:
        print(f"[DEBUG] Missing rects for ids P={idP}, Q={idQ} in {fname}")
        return

    rP = rects[idP]["value"]
    rQ = rects[idQ]["value"]
    w = rects[idP]["original_width"]
    h = rects[idP]["original_height"]

    def to_abs_box(b):
        x = b["x"] / 100 * w
        y = b["y"] / 100 * h
        bw = b["width"]  / 100 * w
        bh = b["height"] / 100 * h
        return (x, y, x + bw, y + bh)

    boxP = to_abs_box(rP)
    boxQ = to_abs_box(rQ)

    try:
        im = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[DEBUG] PIL failed to open {img_path}: {e}")
        return

    draw = ImageDraw.Draw(im)

    draw.rectangle(boxP, outline="blue", width=3)
    draw.rectangle(boxQ, outline="orange", width=3)

    cxP = (boxP[0] + boxP[2]) / 2
    cyP = (boxP[1] + boxP[3]) / 2
    cxQ = (boxQ[0] + boxQ[2]) / 2
    cyQ = (boxQ[1] + boxQ[3]) / 2

    line_color = "red" if row["error_type"] == "FP" else "yellow"
    draw.line((cxP, cyP, cxQ, cyQ), fill=line_color, width=2)

    prob = row["y_prob"]
    lbl = f"{row['error_type']} p={prob:.2f}"
    draw.text(
        (min(boxP[0], boxQ[0]), max(0, min(boxP[1], boxQ[1]) - 15)),
        lbl,
        fill="white"
    )

    tname = fname[:-4]
    out_path = save_file(tname)

    im.save(out_path)
    print(f"[DEBUG] Saved overlay: {out_path}")


print("\nRendering overlays...")
for idx, row in fp_fn.iterrows():
    draw_pair_overlay(row, idx)
print("Done rendering overlays.")


# ── 12. Save model ───────────────────────────────────────────────────────────

joblib.dump(
    {"model": model, "threshold": best_threshold, "feature_cols": FEATURE_COLS},
    "pair_classifier.joblib",
)
print("\nModel saved -> pair_classifier.joblib")
