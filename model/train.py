"""
Feature engineering + model training for RISK//RING.

Features are computed causally (only from transaction history available
*before* the current transaction's timestamp) to avoid leaking the future
into training -- the same discipline a real streaming fraud system needs.
Train/test split is time-based (first 70% of simulated days -> train,
last 30% -> test), not a random shuffle, because a random split would let
the model see future transactions from the same accounts during training.
"""
import json
import os
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

REPORTING_THRESHOLD = 10_000.0
DATA_PATH = "data/transactions.csv"
OUT_DIR = "data"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("step").reset_index(drop=True)

    df["hour_of_day"] = df["step"] % 24
    df["near_threshold"] = ((df["amount"] >= 0.75 * REPORTING_THRESHOLD) &
                             (df["amount"] < REPORTING_THRESHOLD)).astype(int)
    for t in ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]:
        df[f"type_{t}"] = (df["tx_type"] == t).astype(int)

    # causal (history-only) velocity / fan-out features
    orig_seen = {}
    dest_seen = {}
    orig_last_time = {}
    orig_unique_dest = {}
    dest_unique_orig = {}
    pair_seen = {}

    orig_tx_count, dest_tx_count = [], []
    time_since_orig_last, orig_fanout, dest_fanin, pair_prior_count = [], [], [], []

    for _, row in df.iterrows():
        o, d, t = row["orig"], row["dest"], row["step"]

        orig_tx_count.append(orig_seen.get(o, 0))
        dest_tx_count.append(dest_seen.get(d, 0))
        time_since_orig_last.append(t - orig_last_time.get(o, t))
        orig_fanout.append(len(orig_unique_dest.get(o, set())))
        dest_fanin.append(len(dest_unique_orig.get(d, set())))
        pair_prior_count.append(pair_seen.get((o, d), 0))

        orig_seen[o] = orig_seen.get(o, 0) + 1
        dest_seen[d] = dest_seen.get(d, 0) + 1
        orig_last_time[o] = t
        orig_unique_dest.setdefault(o, set()).add(d)
        dest_unique_orig.setdefault(d, set()).add(o)
        pair_seen[(o, d)] = pair_seen.get((o, d), 0) + 1

    df["orig_tx_count_prior"] = orig_tx_count
    df["dest_tx_count_prior"] = dest_tx_count
    df["time_since_orig_last"] = time_since_orig_last
    df["orig_fanout_prior"] = orig_fanout
    df["dest_fanin_prior"] = dest_fanin
    df["pair_prior_count"] = pair_prior_count
    df["is_first_time_pair"] = (df["pair_prior_count"] == 0).astype(int)

    return df


FEATURE_COLS = [
    "amount", "hour_of_day", "near_threshold",
    "type_PAYMENT", "type_TRANSFER", "type_CASH_OUT", "type_DEBIT",
    "orig_tx_count_prior", "dest_tx_count_prior", "time_since_orig_last",
    "orig_fanout_prior", "dest_fanin_prior", "pair_prior_count", "is_first_time_pair",
]


def time_split(df, train_frac=0.7):
    cutoff = df["step"].quantile(train_frac)
    train = df[df["step"] < cutoff]
    test = df[df["step"] >= cutoff]
    return train, test


def evaluate(y_true, y_score, y_pred):
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    train, test = time_split(df)
    X_train, y_train = train[FEATURE_COLS], train["is_fraud"]
    X_test, y_test = test[FEATURE_COLS], test["is_fraud"]

    print(f"Train: {len(train)} rows ({y_train.mean()*100:.2f}% fraud)")
    print(f"Test:  {len(test)} rows ({y_test.mean()*100:.2f}% fraud)")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    logreg = LogisticRegression(class_weight="balanced", max_iter=1000)
    logreg.fit(X_train_sc, y_train)
    logreg_score = logreg.predict_proba(X_test_sc)[:, 1]
    logreg_pred = (logreg_score >= 0.5).astype(int)
    logreg_metrics = evaluate(y_test, logreg_score, logreg_pred)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        scale_pos_weight=neg / pos, eval_metric="aucpr", random_state=42,
    )
    model.fit(X_train, y_train)
    xgb_score = model.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_score >= 0.5).astype(int)
    xgb_metrics = evaluate(y_test, xgb_score, xgb_pred)

    print("Logistic Regression:", logreg_metrics)
    print("XGBoost:            ", xgb_metrics)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    test = test.copy()
    test["risk_score"] = xgb_score
    top_n = 30
    top_idx = np.argsort(-xgb_score)[:top_n]

    alerts = []
    for rank, i in enumerate(top_idx):
        row = test.iloc[i]
        sv = shap_values[i]
        contribs = sorted(
            zip(FEATURE_COLS, sv.tolist()), key=lambda x: -abs(x[1])
        )[:4]
        alerts.append({
            "tx_id": row["tx_id"],
            "orig": row["orig"],
            "dest": row["dest"],
            "amount": round(float(row["amount"]), 2),
            "tx_type": row["tx_type"],
            "risk_score": round(float(row["risk_score"]), 4),
            "is_fraud_ground_truth": int(row["is_fraud"]),
            "ring_id": row["ring_id"] if isinstance(row["ring_id"], str) else "",
            "typology": row["typology"] if isinstance(row["typology"], str) else "",
            "top_factors": [
                {"feature": f, "shap_value": round(float(v), 4)} for f, v in contribs
            ],
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump({
            "dataset": {
                "total_transactions": int(len(df)),
                "fraud_transactions": int(df["is_fraud"].sum()),
                "fraud_rate_pct": round(float(df["is_fraud"].mean() * 100), 3),
                "rings": int(df[df.is_fraud == 1]["ring_id"].nunique()),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
            },
            "logistic_regression": logreg_metrics,
            "xgboost": xgb_metrics,
        }, f, indent=2)

    with open(f"{OUT_DIR}/alerts.json", "w") as f:
        json.dump(alerts, f, indent=2)

    test.to_csv(f"{OUT_DIR}/test_scored.csv", index=False)

    df["risk_score"] = model.predict_proba(df[FEATURE_COLS])[:, 1]
    df.to_csv(f"{OUT_DIR}/all_scored.csv", index=False)

    print(f"\nWrote {OUT_DIR}/metrics.json, {OUT_DIR}/alerts.json, "
          f"{OUT_DIR}/test_scored.csv, {OUT_DIR}/all_scored.csv")


if __name__ == "__main__":
    main()
