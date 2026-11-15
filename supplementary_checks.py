"""
supplementary_checks.py
สคริปต์ตรวจสอบและคำนวณค่าสถิติเพิ่มเติม (ที่มีเครื่องหมาย † ใน Model Card)
สำหรับกลุ่ม OK67 — รายวิชา HCD-302 Creating Explainable AI
วิธีรัน: python supplementary_checks.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

RANDOM_STATE = 42
CHOSEN_THRESHOLD = 0.35

def bootstrap_metric_ci(y_true, y_score_or_pred, metric_fn, n_boot=2000, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt_b = y_true[idx]
        yp_b = y_score_or_pred[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        val = metric_fn(yt_b, yp_b)
        if not np.isnan(val):
            stats.append(val)
    return np.percentile(stats, 2.5), np.percentile(stats, 97.5)

def fpr_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else np.nan

def fnr_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fn / (fn + tp) if (fn + tp) > 0 else np.nan

def main():
    print("=" * 72)
    print("SUPPLEMENTARY CHECKS FOR MODEL CARD (GROUP OK67)")
    print("=" * 72)

    pipe = joblib.load("best_lgbm_model.joblib")
    train_df = pd.read_csv("train_split_seed42.csv")
    test_df = pd.read_csv("test_split_seed42.csv")

    X_test = test_df.drop(columns=["Target"])
    y_test = test_df["Target"].values

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= CHOSEN_THRESHOLD).astype(int)

    overall_auc = roc_auc_score(y_test, y_prob)
    auc_low, auc_high = bootstrap_metric_ci(y_test, y_prob, roc_auc_score)
    print("\n[1] Overall Performance & 95% CI† (Test Set, n=885, Threshold=0.35)")
    print(f"    - ROC-AUC  : {overall_auc:.4f} (95% CI†: {auc_low:.4f} - {auc_high:.4f})")
    print(f"    - Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"    - Precision: {precision_score(y_test, y_pred):.4f}")

    print("\n[2] Disaggregated Subgroup Evaluation with 95% CI† & AUC† (Threshold=0.35)")
    subgroups = [
        ("เพศ: หญิง (0)", X_test["Gender"] == 0),
        ("เพศ: ชาย (1)", X_test["Gender"] == 1),
        ("อายุ <= 20 ปี", X_test["Age at enrollment"] <= 20),
        ("อายุ > 20 ปี", X_test["Age at enrollment"] > 20),
        ("ทุน: ไม่ได้รับ (0)", X_test["Scholarship holder"] == 0),
        ("ทุน: ได้รับ (1)", X_test["Scholarship holder"] == 1),
    ]

    for label, mask in subgroups:
        yt = y_test[mask.values]
        pr = y_prob[mask.values]
        yp = y_pred[mask.values]

        sub_auc = roc_auc_score(yt, pr)
        fpr = fpr_score(yt, yp)
        fnr = fnr_score(yt, yp)
        fpr_l, fpr_h = bootstrap_metric_ci(yt, yp, fpr_score)
        fnr_l, fnr_h = bootstrap_metric_ci(yt, yp, fnr_score)

        print(
            f"    - {label:20s} | n={len(yt):3d} | AUC†={sub_auc:.3f} | "
            f"FPR={fpr*100:5.2f}% [{fpr_l*100:.1f}-{fpr_h*100:.1f}]† | "
            f"FNR={fnr*100:5.2f}% [{fnr_l*100:.1f}-{fnr_h*100:.1f}]†"
        )

    print("\n[3] Permutation Feature Importance (PFI) for Sensitive Attributes†")
    pfi = permutation_importance(
        pipe, X_test, y_test, scoring="roc_auc", n_repeats=15, random_state=RANDOM_STATE, n_jobs=-1
    )
    pfi_df = (
        pd.DataFrame({"Feature": X_test.columns, "Mean": pfi.importances_mean, "Std": pfi.importances_std})
        .sort_values("Mean", ascending=False)
        .reset_index(drop=True)
    )
    pfi_df["Rank"] = pfi_df.index + 1
    for feat in ["Scholarship holder", "Gender", "Age at enrollment"]:
        row = pfi_df[pfi_df["Feature"] == feat].iloc[0]
        print(f"    - Rank {int(row['Rank']):2d}/36 : {feat:20s} | PFI = {row['Mean']:.4f} ± {row['Std']:.4f}")

    sch_train = train_df[train_df["Scholarship holder"] == 1]
    print("\n[4] Scholarship Holder Exploration† (Train Set & Subgroup Threshold Sweep)")
    print(
        f"    - Train Set Scholarship (1): n={len(sch_train)}, "
        f"Actual Dropout={int(sch_train['Target'].sum())} ({sch_train['Target'].mean()*100:.2f}%†)"
    )

    sch_mask = (X_test["Scholarship holder"] == 1).values
    yt_sch = y_test[sch_mask]
    pr_sch = y_prob[sch_mask]
    print("    - Test Set Threshold Sweep (Scholarship = 1, n=231, Actual Dropout=29):")
    for th in [0.35, 0.25, 0.20, 0.15]:
        yp_s = (pr_sch >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt_sch, yp_s).ravel()
        print(
            f"      * Threshold = {th:.2f} -> FN = {fn:2d}/29 (FNR = {fn/29*100:5.2f}%) | "
            f"FP = {fp:2d}/{fp+tn} (FPR = {fp/(fp+tn)*100:5.2f}%)"
        )
    print("=" * 72)

if __name__ == "__main__":
    main()
