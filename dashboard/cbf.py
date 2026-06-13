#!/usr/bin/env python3
"""
Build Content-Based Filtering (CBF) recommendations.
 
Mirrors the structure and conventions of the train/test split script:
- Reads the same TRAIN split (latest 80%) produced by that script
- Robust column resolution (strips whitespace, handles name variants)
- Prints the same style of diagnostics before/after processing
- Writes a dashboard-ready CSV: user_id, rank, parent_asin, predicted_score,
  model, title, store
 
How CBF works here
-------------------
1. Build a TF-IDF matrix over product text (title + features + description + categories)
2. For each user, build a "profile vector" = average TF-IDF vector of all
   products they interacted with in TRAIN
3. Compute cosine similarity between the user's profile vector and every
   product's TF-IDF vector
4. Exclude products the user already interacted with (in TRAIN)
5. Recommend the top-N most similar products
"""
 
from __future__ import annotations
 
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_DIR  = r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard\data"
OUTPUT_DIR = r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard\dashboard_data"
 
REVIEWS_FILE       = "reviews_clean_no_exact_duplicates.csv"  # source interactions
PRODUCTS_FILE      = "products_clean.csv"                     # product catalogue with text fields
 
CBF_OUTPUT_FILE    = "cbf_dashboard.csv"
 
TOP_N              = 10          # recommendations per user
MIN_REVIEWS_FOR_REC = 1          # min train interactions to generate recs for a user
MAX_FEATURES       = 20_000      # TF-IDF vocabulary cap
 
# ── Train/test split rules (mirrors the split script) ─────────────────────────
MIN_TRAIN_SIZE       = 4
MIN_REVIEWS_FOR_TEST = 5
TEST_FRACTION        = 0.20
 
 
# ── Column resolution helpers (same pattern as the split script) ──────────────
 
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df
 
 
def resolve_user_id_column(df: pd.DataFrame) -> str:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    candidates = [
        "user_id", "userid", "user id",
        "reviewerid", "reviewer_id",
        "customer_id", "customerid",
    ]
    for key in candidates:
        if key in normalized:
            return normalized[key]
    raise ValueError(
        "Could not find the user_id column. "
        f"Available columns: {list(df.columns)}"
    )
 
 
def resolve_asin_column(df: pd.DataFrame) -> str:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    candidates = [
        "parent_asin", "asin", "product_id", "productid",
        "item_id", "itemid",
    ]
    for key in candidates:
        if key in normalized:
            return normalized[key]
    raise ValueError(
        "Could not find the parent_asin/product id column. "
        f"Available columns: {list(df.columns)}"
    )
 
 
# ── Chronological 80/20 split (same rules as the split script) ────────────────
 
from math import floor
 
 
def get_holdout_size(n_interactions: int) -> int:
    """Latest 20% go to test for users with >= MIN_REVIEWS_FOR_TEST interactions."""
    if n_interactions < MIN_REVIEWS_FOR_TEST:
        return 0
    holdout = floor(n_interactions * TEST_FRACTION + 0.5)
    holdout = max(1, holdout)
    holdout = min(holdout, n_interactions - MIN_TRAIN_SIZE)
    return max(0, holdout)
 
 
def split_train_test(
    reviews: pd.DataFrame, user_col: str
) -> pd.DataFrame:
    """
    Sort each user's reviews by timestamp_unix and keep only the earliest
    80% (train portion). Users with < MIN_REVIEWS_FOR_TEST interactions
    keep everything (all-train).
 
    Returns the TRAIN dataframe only — CBF is trained on training data.
    """
    ts_col = "timestamp_unix" if "timestamp_unix" in reviews.columns else None
    if ts_col is None:
        print("  ⚠ 'timestamp_unix' not found — using row order as chronological order")
 
    train_parts = []
    eligible, ineligible = 0, 0
 
    for user_id, user_df in reviews.groupby(user_col, sort=False):
        if ts_col:
            user_df = user_df.sort_values(ts_col, ascending=True)
        n = len(user_df)
        holdout = get_holdout_size(n)
 
        if holdout == 0:
            train_parts.append(user_df)
            ineligible += 1
            continue
 
        split_idx = n - holdout
        train_parts.append(user_df.iloc[:split_idx])
        eligible += 1
 
    train_df = pd.concat(train_parts, axis=0, ignore_index=True) if train_parts \
        else reviews.iloc[0:0].copy()
 
    print(f"  Users with >= {MIN_REVIEWS_FOR_TEST} reviews (80/20 split): {eligible:,}")
    print(f"  Users with <  {MIN_REVIEWS_FOR_TEST} reviews (all-train)  : {ineligible:,}")
    print(f"  Train rows: {len(train_df):,} / {len(reviews):,} total")
 
    return train_df
 
 
# ── Text building for TF-IDF ───────────────────────────────────────────────────
 
def _build_text_field(products: pd.DataFrame) -> pd.Series:
    """Combine title + features + description + categories into one text field."""
    parts = []
    for col in ["title", "features", "description", "categories", "store"]:
        if col in products.columns:
            parts.append(products[col].fillna("").astype(str))
        else:
            parts.append(pd.Series([""] * len(products), index=products.index))
 
    combined = parts[0]
    for p in parts[1:]:
        combined = combined + " " + p
 
    # Collapse whitespace
    return combined.str.replace(r"\s+", " ", regex=True).str.strip()
 
 
# ── Main ────────────────────────────────────────────────────────────────────
 
def main() -> None:
    input_dir  = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    reviews_path  = input_dir / REVIEWS_FILE
    products_path = input_dir / PRODUCTS_FILE
 
    # ── Load ────────────────────────────────────────────────────────────────
    print("Reading input files...")
 
    if not reviews_path.exists():
        raise FileNotFoundError(f"Reviews file not found at {reviews_path}.")
    if not products_path.exists():
        raise FileNotFoundError(f"Products file not found at {products_path}.")
 
    reviews = pd.read_csv(reviews_path, low_memory=False)
    reviews = normalize_columns(reviews)
 
    if products_path.suffix.lower() == ".xlsx":
        products = pd.read_excel(products_path, engine="openpyxl")
    else:
        products = pd.read_csv(products_path, low_memory=False)
    products = normalize_columns(products)
 
    print("Reviews columns :", list(reviews.columns))
    print("Products columns:", list(products.columns))
 
    user_col      = resolve_user_id_column(reviews)
    asin_col_train = resolve_asin_column(reviews)
    asin_col_prod  = resolve_asin_column(products)
 
    reviews[user_col]       = reviews[user_col].astype(str).str.strip()
    reviews[asin_col_train] = reviews[asin_col_train].astype(str).str.strip()
    products[asin_col_prod] = products[asin_col_prod].astype(str).str.strip()
 
    print(f"\nTotal reviews: {len(reviews):,} | Unique users: {reviews[user_col].nunique():,}")
    print("\nSplitting into train (earliest 80%) / test (latest 20%) per user...")
    train_df = split_train_test(reviews, user_col)
 
    # Deduplicate products by ASIN — keep first occurrence
    products = products.drop_duplicates(subset=[asin_col_prod]).reset_index(drop=True)
 
    print(f"\nTrain interactions : {len(train_df):,}")
    print(f"Unique users       : {train_df[user_col].nunique():,}")
    print(f"Unique products    : {products[asin_col_prod].nunique():,}")
 
    # ── TF-IDF over product text ───────────────────────────────────────────
    print("\nBuilding TF-IDF matrix over product text (title + features + "
          "description + categories + store)...")
 
    products["_text"] = _build_text_field(products)
 
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(products["_text"])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
 
    # Map ASIN -> row index in tfidf_matrix / products
    asin_to_idx = {asin: i for i, asin in enumerate(products[asin_col_prod])}
 
    # ── Build per-user profile vectors ─────────────────────────────────────
    print("\nBuilding user profile vectors from TRAIN interactions...")
 
    user_groups = train_df.groupby(user_col)[asin_col_train].apply(list)
 
    eligible_users   = 0
    ineligible_users = 0
    no_product_match = 0
 
    rec_rows = []
 
    for user_id, asins in user_groups.items():
        # Map to TF-IDF row indices, dropping ASINs not found in the catalogue
        idxs = [asin_to_idx[a] for a in asins if a in asin_to_idx]
 
        if len(idxs) < MIN_REVIEWS_FOR_REC:
            if not idxs:
                no_product_match += 1
            else:
                ineligible_users += 1
            continue
 
        # User profile = mean of their interacted products' TF-IDF vectors
        user_vector = tfidf_matrix[idxs].mean(axis=0)
        user_vector = np.asarray(user_vector)  # sparse mean -> dense (1, n_features)
 
        # Cosine similarity against ALL products
        sims = cosine_similarity(user_vector, tfidf_matrix).flatten()
 
        # Exclude products already interacted with
        seen = set(idxs)
        sims_masked = sims.copy()
        for i in seen:
            sims_masked[i] = -1.0
 
        # Top-N most similar unseen products
        top_idx = np.argpartition(sims_masked, -TOP_N)[-TOP_N:]
        top_idx = top_idx[np.argsort(sims_masked[top_idx])[::-1]]
 
        for rank, prod_idx in enumerate(top_idx, start=1):
            score = float(sims_masked[prod_idx])
            if score <= 0:
                continue
            row = {
                "user_id":         user_id,
                "rank":            rank,
                "parent_asin":     products.iloc[prod_idx][asin_col_prod],
                "predicted_score": round(score, 6),
                "model":           "CBF",
                "title":           products.iloc[prod_idx].get("title", ""),
                "store":           products.iloc[prod_idx].get("store",
                                       products.iloc[prod_idx].get("store_clean", "")),
            }
            rec_rows.append(row)
 
        eligible_users += 1
 
        if eligible_users % 100 == 0:
            print(f"  ... {eligible_users:,} users processed")
 
    cbf_df = pd.DataFrame(rec_rows)
 
    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = output_dir / CBF_OUTPUT_FILE
    cbf_df.to_csv(out_path, index=False)
 
    # ── Diagnostics (mirrors the split script's style) ─────────────────────
    print("\nDiagnostics")
    print("-" * 40)
    print(f"Total users in train split        : {train_df[user_col].nunique():,}")
    print(f"Users with CBF recs generated      : {eligible_users:,}")
    print(f"Users skipped — < {MIN_REVIEWS_FOR_REC} matched products : {ineligible_users:,}")
    print(f"Users skipped — no products matched: {no_product_match:,}")
    print(f"Total recommendation rows written  : {len(cbf_df):,}")
    if not cbf_df.empty:
        print(f"Avg recs per user                  : {len(cbf_df) / eligible_users:.2f}")
        print(f"Score range                        : {cbf_df['predicted_score'].min():.4f} "
              f"– {cbf_df['predicted_score'].max():.4f}")
 
    print(f"\nWrote: {out_path}")
    print("Done.")
 
 
if __name__ == "__main__":
    main()
  