from __future__ import annotations
 
from pathlib import Path
from typing import Optional, Union
 
import pandas as pd
import streamlit as st
 
from config import (
    DEFAULT_REVIEWS,
    DEFAULT_PRODUCTS,
    DEFAULT_USERS,
    DEFAULT_ASIN_ITEM,
    BOUGHT_TOGETHER_FILE,
    SCATTER_FILE,
)
 
 
# ── Internal helpers ──────────────────────────────────────────────────────────
 
def _reset(obj):
    """Seek to 0 for UploadedFile; no-op for Path / str."""
    try:
        obj.seek(0)
    except Exception:
        pass
    return obj
 
 
def _resolve(default: Path) -> Optional[Path]:
    """Return default path when it exists, else None."""
    return default if default.exists() else None
 
 
def maybe_source(upload, default: Path):
    """Prefer an uploaded file; fall back to the default path."""
    return upload if upload is not None else _resolve(default)
 
 
# ── Loaders ───────────────────────────────────────────────────────────────────
 
@st.cache_data(show_spinner=False)
def load_reviews(source) -> pd.DataFrame:
    usecols = [
        "rating", "parent_asin", "user_id",
        "review_year", "review_month", "review_year_month",
        "verified_purchase", "helpful_vote",
        "has_review_text", "review_length_words",
    ]
    df = pd.read_csv(_reset(source), usecols=usecols)
    df["verified_purchase"]    = df["verified_purchase"].fillna(False).astype(bool)
    df["has_review_text"]      = df["has_review_text"].fillna(False).astype(bool)
    df["helpful_vote"]         = pd.to_numeric(df["helpful_vote"],         errors="coerce").fillna(0).astype(int)
    df["review_length_words"]  = pd.to_numeric(df["review_length_words"],  errors="coerce").fillna(0)
    df["review_year"]          = pd.to_numeric(df["review_year"],          errors="coerce").fillna(-1).astype(int)
    df["review_month"]         = pd.to_numeric(df["review_month"],         errors="coerce").fillna(-1).astype(int)
    df["rating"]               = pd.to_numeric(df["rating"],               errors="coerce")
    return df
 
 
@st.cache_data(show_spinner=False)
def load_products(source) -> pd.DataFrame:
    usecols = [
        "parent_asin", "title", "average_rating", "rating_number",
        "price", "store_clean", "year_first_available",
        "has_price", "has_description", "has_features",
        "has_store", "has_categories",
    ]
    df = pd.read_csv(_reset(source), usecols=usecols)
    df["title"]       = df["title"].fillna("(missing title)")
    df["store_clean"] = df["store_clean"].fillna("(missing store)")
    return df.drop_duplicates(subset=["parent_asin"])
 
 
@st.cache_data(show_spinner=False)
def load_users(source) -> pd.DataFrame:
    usecols = [
        "user_id", "num_reviews", "unique_products_reviewed",
        "mean_rating_given", "median_rating_given",
        "verified_purchase_ratio", "mean_helpful_vote_received",
        "avg_review_length_words", "reviewing_time_span_days",
    ]
    return pd.read_csv(_reset(source), usecols=usecols).drop_duplicates(subset=["user_id"])
 
 
@st.cache_data(show_spinner=False)
def load_asin_item(source) -> pd.DataFrame:
    usecols = ["parent_asin", "Item", "title"]
    return pd.read_csv(_reset(source), usecols=usecols).drop_duplicates(subset=["parent_asin"])
 
 
@st.cache_data(show_spinner=False)
def load_bought_together() -> Optional[pd.DataFrame]:
    if not BOUGHT_TOGETHER_FILE.exists():
        return None
    df = pd.read_excel(BOUGHT_TOGETHER_FILE)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={
        "parent_asin_a": "parent_asin_1",
        "parent_asin_b": "parent_asin_2",
        "frequency":     "count",
        "pair_count":    "count",
    })
    required = {"parent_asin_1", "parent_asin_2", "count"}
    if not required.issubset(df.columns):
        return None
    return df
 
 
@st.cache_data(show_spinner=False)
def load_scatter() -> Optional[pd.DataFrame]:
    if not SCATTER_FILE.exists():
        return None
    df = pd.read_excel(SCATTER_FILE)
    df = df.rename(columns={
        "X_MaxCosSim":  "MaxCosine",
        "Y_PredRating": "Predicted_Rating",
    })
    return df.dropna()
 
 
# ── Products lookup (merged helper) ──────────────────────────────────────────
 
def build_products_lookup(products: pd.DataFrame, asin_item: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge products with asin_item titles for display."""
    lookup = products[
        ["parent_asin", "title", "store_clean", "average_rating", "rating_number", "price"]
    ].copy()
 
    if asin_item is not None:
        lookup = lookup.merge(
            asin_item[["parent_asin", "Item", "title"]].rename(columns={"title": "asin_item_title"}),
            on="parent_asin",
            how="left",
        )
        lookup["display_title"]      = lookup["Item"].fillna(lookup["title"])
        lookup["full_title_tooltip"] = lookup["asin_item_title"].fillna(lookup["title"])
    else:
        lookup["display_title"]      = lookup["title"]
        lookup["full_title_tooltip"] = lookup["title"]
 
    return lookup