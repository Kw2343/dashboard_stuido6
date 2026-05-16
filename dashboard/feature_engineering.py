import pandas as pd
import numpy as np

POPULARITY_M = 50


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["review_year"].astype(str) + "-" + df["review_month"].astype(str) + "-01",
        errors="coerce",
    )

    feats = df.groupby("parent_asin").agg(
        purchase_count=("user_id",   "count"),
        unique_users=  ("user_id",   "nunique"),
        avg_rating=    ("rating",    "mean"),
        last_purchase= ("timestamp", "max"),
    ).reset_index()

    today = pd.Timestamp.today()
    feats["days_since_last_purchase"] = (today - feats["last_purchase"]).dt.days

    freq = (
        df.groupby(["parent_asin", "user_id", "review_year_month"])
        .size()
        .reset_index(name="freq")
        .groupby("parent_asin")["freq"]
        .sum()
        .reset_index(name="purchase_frequency")
    )
    feats = feats.merge(freq, on="parent_asin", how="left")

    C = feats["avg_rating"].mean()
    m = POPULARITY_M
    v = feats["purchase_count"]
    R = feats["avg_rating"]
    feats["score"] = (v / (v + m)) * R + (m / (v + m)) * C

    return feats


def product_feature_importance(products: pd.DataFrame) -> pd.DataFrame:
    candidate_cols = {
        "title":           "Title",
        "store_clean":     "Store",
        "has_price":       "Price",
        "has_description": "Description",
        "has_features":    "Features",
        "has_categories":  "Categories",
    }
    rows = []
    for col, label in candidate_cols.items():
        if col in products.columns:
            val = products[col].mean() * 100 if col.startswith("has_") else products[col].notna().mean() * 100
        else:
            val = 0.0
        rows.append({"feature": label, "value": round(val, 1)})
    return pd.DataFrame(rows)