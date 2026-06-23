#!/usr/bin/env python3
"""
Offline ranking evaluation for recommendation models.
 
This script reuses the CBF split rules from `dashboard/cbf.py`, then computes
ranking metrics for a model at top-K:
  - recall@K
  - hit_rate@K
  - MAP@K
  - NDCG@K
  - coverage@K
 
Current built-in model:
  - popularity
 
Future models can be added by registering another recommendation builder or by
passing a CSV with per-user recommendations.
"""
 
from __future__ import annotations
 
import argparse
import importlib.util
import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable
 
import numpy as np
import pandas as pd
 
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm should exist, but keep a fallback
    tqdm = None
 
 
ROOT = Path(__file__).resolve().parent
DEFAULT_REVIEWS = ROOT / "data" / "reviews_clean_no_exact_duplicates.csv"
DEFAULT_PRODUCTS = ROOT / "data" / "products_clean.csv"
DEFAULT_OUTPUT_DIR = ROOT / "evaluation_results"
DEFAULT_BLENDED_CSV = ROOT / "dashboard_data" / "blended_products_top100.csv"
CBF_HELPER_PATH = ROOT / "cbf.py"
POPULARITY_HELPER_PATH = ROOT / "tabs" / "popularity.py"
CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache"
SPLIT_CACHE_VERSION = 2
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
 
 
def _load_cbf_helpers():
    spec = importlib.util.spec_from_file_location("dashboard_cbf_helpers", CBF_HELPER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module from {CBF_HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
 
 
def _load_popularity_helpers():
    spec = importlib.util.spec_from_file_location(
        "dashboard_popularity_helpers", POPULARITY_HELPER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load popularity model from {POPULARITY_HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
 
 
def _log(message: str) -> None:
    print(f"[model-eval] {message}", flush=True)
 
 
def _cache_key(reviews_path: Path, helper_path: Path) -> str:
    payload = {
        "cache_version": SPLIT_CACHE_VERSION,
        "reviews": str(reviews_path.resolve()),
        "reviews_mtime": reviews_path.stat().st_mtime if reviews_path.exists() else None,
        "reviews_size": reviews_path.stat().st_size if reviews_path.exists() else None,
        "helper": str(helper_path.resolve()),
        "helper_mtime": helper_path.stat().st_mtime if helper_path.exists() else None,
        "helper_size": helper_path.stat().st_size if helper_path.exists() else None,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return digest
 
 
def _split_cache_paths(key: str) -> tuple[Path, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"split_{key}.pkl", CACHE_DIR / f"split_{key}.json"
 
 
def _load_split_cache(cache_file: Path) -> dict | None:
    if not cache_file.exists():
        return None
    try:
        with cache_file.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None
 
 
def _save_split_cache(cache_file: Path, manifest_file: Path, payload: dict) -> None:
    temp_cache_file = cache_file.with_suffix(".tmp")
    with temp_cache_file.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    temp_cache_file.replace(cache_file)
    manifest_file.write_text(json.dumps(payload["manifest"], indent=2, sort_keys=True))
 
 
def _progress_iter(iterable, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="user", ncols=90, leave=False)
 
 
def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")
 
 
def _load_reviews(reviews_path: Path) -> pd.DataFrame:
    _log(f"Loading reviews from {reviews_path}")
    wanted_columns = {
        "user_id", "userid", "user id", "reviewerid", "reviewer_id",
        "customer_id", "customerid", "parent_asin", "asin", "product_id",
        "productid", "item_id", "itemid", "rating", "timestamp_unix",
    }
    reviews = pd.read_csv(
        reviews_path,
        usecols=lambda column: str(column).strip().lower() in wanted_columns,
        low_memory=False,
    )
    cbf = _load_cbf_helpers()
    _log("Normalizing review columns")
    reviews = cbf.normalize_columns(reviews)
 
    user_col = cbf.resolve_user_id_column(reviews)
    asin_col = cbf.resolve_asin_column(reviews)
 
    reviews[user_col] = reviews[user_col].astype(str).str.strip()
    reviews[asin_col] = reviews[asin_col].astype(str).str.strip()
    reviews["rating"] = _coerce_numeric(reviews["rating"]) if "rating" in reviews.columns else np.nan
    if "timestamp_unix" in reviews.columns:
        reviews["timestamp_unix"] = _coerce_numeric(reviews["timestamp_unix"])
 
    return reviews, user_col, asin_col, cbf
 
 
def _load_products(products_path: Path | None) -> pd.DataFrame | None:
    if products_path is None or not products_path.exists():
        _log("No products file found, skipping metadata-based metrics")
        return None
    _log(f"Loading products from {products_path}")
    products = pd.read_csv(products_path, low_memory=False)
    products = products.copy()
    products.columns = [str(c).strip() for c in products.columns]
    for candidate in ["parent_asin", "asin", "product_id", "item_id"]:
        if candidate in products.columns:
            products[candidate] = products[candidate].astype(str).str.strip()
            return products
    return products
 
 
def _split_like_cbf(reviews: pd.DataFrame, user_col: str, cbf) -> tuple[pd.DataFrame, pd.DataFrame]:
    _log("Building train/test split with CBF rules")
    ts_col = "timestamp_unix" if "timestamp_unix" in reviews.columns else None
    total_users = reviews[user_col].nunique(dropna=True)
    _log(f"Vectorized split for {len(reviews):,} rows across {total_users:,} users")
 
    work = reviews.copy()
    work["_source_order"] = np.arange(len(work))
 
    _log("Split 1/3: sorting interactions")
    sort_columns = [user_col, ts_col] if ts_col else [user_col, "_source_order"]
    work = work.sort_values(sort_columns, kind="mergesort", na_position="last")
 
    _log("Split 2/3: calculating each user's holdout size")
    group_sizes = work.groupby(user_col, sort=False)[user_col].transform("size").to_numpy()
    positions = work.groupby(user_col, sort=False).cumcount().to_numpy()
 
    holdout = np.zeros(len(work), dtype=np.int64)
    eligible = group_sizes >= cbf.MIN_REVIEWS_FOR_TEST
    rounded = np.floor(group_sizes[eligible] * cbf.TEST_FRACTION + 0.5).astype(np.int64)
    rounded = np.maximum(1, rounded)
    rounded = np.minimum(rounded, group_sizes[eligible] - cbf.MIN_TRAIN_SIZE)
    holdout[eligible] = np.maximum(0, rounded)
 
    _log("Split 3/3: materializing train and test rows")
    test_mask = (holdout > 0) & (positions >= (group_sizes - holdout))
    train_df = work.loc[~test_mask].drop(columns="_source_order").reset_index(drop=True)
    test_df = work.loc[test_mask].drop(columns="_source_order").reset_index(drop=True)
    _log(f"Split complete: train={len(train_df):,}, test={len(test_df):,}")
    return train_df, test_df


def _filter_test_to_training_items(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_col: str,
    item_col: str,
) -> pd.DataFrame:
    """Remove held-out products that were never present in the training catalog."""
    train_items = set(train_df[item_col].dropna().astype(str))
    test_items = test_df[item_col].astype(str)
    keep_mask = test_items.isin(train_items)
    filtered = test_df.loc[keep_mask].copy()

    removed_rows = int((~keep_mask).sum())
    users_before = int(test_df[user_col].nunique())
    users_after = int(filtered[user_col].nunique())
    _log(
        "Training-catalog filter: "
        f"removed {removed_rows:,} held-out rows and "
        f"{users_before - users_after:,} users with no remaining test item"
    )
    return filtered


def _build_ground_truth(test_df: pd.DataFrame, user_col: str, item_col: str) -> dict[str, set[str]]:
    if test_df.empty:
        return {}
    _log("Building ground truth from all held-out interactions")
    _log(f"Ground truth users: {test_df[user_col].nunique():,}")
    return test_df.groupby(user_col)[item_col].apply(lambda s: set(map(str, s.tolist()))).to_dict()
 
 
def _embedding_cache_path(products_path: Path) -> Path:
    payload = {
        "model": EMBEDDING_MODEL,
        "products": str(products_path.resolve()),
        "products_mtime": products_path.stat().st_mtime,
        "products_size": products_path.stat().st_size,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"product_embeddings_{digest}.pkl"
 
 
def _build_item_embeddings(
    products: pd.DataFrame | None,
    item_ids: list[str],
    products_path: Path,
) -> dict[str, np.ndarray]:
    if products is None:
        return {}
 
    item_col = None
    for candidate in ["parent_asin", "asin", "product_id", "item_id"]:
        if candidate in products.columns:
            item_col = candidate
            break
    if item_col is None:
        return {}
 
    text_cols = [c for c in ["title", "categories", "description", "features", "store"] if c in products.columns]
    if not text_cols:
        return {}
 
    cache_path = _embedding_cache_path(products_path)
    vectors: dict[str, np.ndarray] = {}
    if cache_path.exists():
        try:
            with cache_path.open("rb") as fh:
                vectors = pickle.load(fh)
            _log(f"Loaded {len(vectors):,} cached product embeddings")
        except Exception:
            vectors = {}
 
    requested = set(map(str, item_ids))
    missing = sorted(requested - set(vectors))
    if not missing:
        return {item: vectors[item] for item in requested if item in vectors}
 
    rows = (
        products[products[item_col].astype(str).isin(missing)]
        .drop_duplicates(subset=[item_col])
        [[item_col] + text_cols]
        .fillna("")
    )
    if rows.empty:
        return {item: vectors[item] for item in requested if item in vectors}
 
    texts = (
        rows[text_cols]
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .tolist()
    )
    row_ids = rows[item_col].astype(str).tolist()
 
    _log(f"Encoding {len(texts):,} product texts with {EMBEDDING_MODEL}")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Embedding diversity requires `sentence-transformers`. "
            "Install dashboard/requirements_dashboard.txt."
        ) from exc
 
    try:
        model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load cached embedding model `{EMBEDDING_MODEL}`. "
            "Run once with network access so Sentence Transformers can download it."
        ) from exc
    encoded = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 32,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    vectors.update(
        {item_id: np.asarray(vector, dtype=np.float32) for item_id, vector in zip(row_ids, encoded)}
    )
 
    temp_cache = cache_path.with_suffix(".tmp")
    with temp_cache.open("wb") as fh:
        pickle.dump(vectors, fh, protocol=pickle.HIGHEST_PROTOCOL)
    temp_cache.replace(cache_path)
    _log(f"Cached product embeddings to {cache_path}")
 
    return {item: vectors[item] for item in requested if item in vectors}
 
 
def _build_popularity_product_stats(
    train_df: pd.DataFrame,
    item_col: str,
) -> pd.DataFrame:
    stats = (
        train_df.groupby(item_col, sort=False)
        .agg(
            purchase_frequency=(item_col, "size"),
            average_rating=("rating", "mean"),
            rating_number=("rating", "count"),
        )
        .reset_index()
        .rename(columns={item_col: "parent_asin"})
    )
    stats["parent_asin"] = stats["parent_asin"].astype(str)
    return stats
 
 
def _build_popularity_lookup(
    train_df: pd.DataFrame,
    item_col: str,
    catalog_items: list[str],
) -> dict[str, float]:
    counts = train_df[item_col].astype(str).value_counts()
    if counts.empty:
        return {item: 0.0 for item in catalog_items}
 
    ranked_items = counts.index.astype(str).tolist()
    max_rank = max(1, len(ranked_items) - 1)
    lookup = {item: 1.0 - (idx / max_rank) for idx, item in enumerate(ranked_items)}
    for item in catalog_items:
        lookup.setdefault(item, 0.0)
    return lookup
 
 
def _recommend_popularity_from_blended_csv(
    blended_csv: Path,
    user_ids: list[str],
    top_k: int,
) -> dict[str, list[str]]:
    """
    Read the pre-built blended_products_top100.csv produced by
    build_popularity_lists.py (80 popular + 20 discovery slots) and return
    the top-K items as a global recommendation list for every user.
 
    The CSV must contain a `parent_asin` column and optionally a `blend_rank`
    column (used for ordering). If `blend_rank` is absent the file order is used.
    """
    _log(f"Loading blended popularity list from {blended_csv}")
    if not blended_csv.exists():
        raise FileNotFoundError(
            f"Blended CSV not found: {blended_csv}\n"
            "Run build_popularity_lists.py first to generate it."
        )
    df = pd.read_csv(blended_csv, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
 
    asin_col = next(
        (c for c in df.columns if c.lower() in ("parent_asin", "asin")), None
    )
    if asin_col is None:
        raise ValueError(
            f"blended_products_top100.csv must contain a 'parent_asin' column. "
            f"Found: {list(df.columns)}"
        )
 
    if "blend_rank" in df.columns:
        df = df.sort_values("blend_rank", ascending=True)
 
    recs = df[asin_col].astype(str).str.strip().tolist()[:top_k]
 
    n_popular   = int((df["tier"] == "popular").sum())   if "tier" in df.columns else "?"
    n_discovery = int((df["tier"] == "discovery").sum()) if "tier" in df.columns else "?"
    _log(
        f"Blended list: {len(df)} rows total "
        f"({n_popular} popular + {n_discovery} discovery), using top-{top_k}"
    )
    _log(f"Serving the same global Top-{top_k} list to all {len(user_ids):,} users")
    return {user_id: list(recs) for user_id in user_ids}
 
 
def _recommend_popularity(
    train_df: pd.DataFrame,
    user_ids: list[str],
    item_col: str,
    top_k: int,
    discovery_share: float,
    min_discovery_rating: float,
    min_discovery_reviews: int,
    blended_csv: Path | None = None,
) -> dict[str, list[str]]:
    # ── Fast path: use pre-built blended CSV ────────────────────────────────
    if blended_csv is not None:
        return _recommend_popularity_from_blended_csv(blended_csv, user_ids, top_k)
 
    # ── Fallback: recompute scores live from train_df ────────────────────────
    _log(f"No blended CSV supplied — recomputing scores live from {POPULARITY_HELPER_PATH}")
    popularity = _load_popularity_helpers()
    product_stats = _build_popularity_product_stats(train_df, item_col)
 
    popular_scored = popularity._compute_popular_score(product_stats)
    discovery_scored = popularity._compute_discovery_score(product_stats)
 
    popular_ranked = (
        popular_scored.sort_values(
            ["popular_score", "parent_asin"], ascending=[False, True]
        )["parent_asin"]
        .astype(str)
        .tolist()
    )
    discovery_ranked = (
        discovery_scored[
            (discovery_scored["average_rating"] >= min_discovery_rating)
            & (discovery_scored["rating_number"] >= min_discovery_reviews)
        ]
        .sort_values(["discovery_score", "parent_asin"], ascending=[False, True])[
            "parent_asin"
        ]
        .astype(str)
        .tolist()
    )
 
    popular_slots = max(1, round(top_k * (1 - discovery_share)))
    discovery_slots = top_k - popular_slots
    _log(
        f"Popularity.py blend: {popular_slots} popular + "
        f"{discovery_slots} discovery items"
    )
    popular_recs = popular_ranked[:popular_slots]
    discovery_recs = [
        item for item in discovery_ranked if item not in set(popular_recs)
    ][:discovery_slots]
    recs = popular_recs + discovery_recs
 
    if len(recs) < top_k:
        recs.extend(
            item
            for item in popular_ranked
            if item not in set(recs)
        )
        recs = recs[:top_k]
 
    _log(f"Using the same global Top-{top_k} list for all {len(user_ids):,} users")
    return {user_id: list(recs) for user_id in user_ids}
 
 
def _intra_list_diversity(recs: list[str], item_vectors: dict[str, np.ndarray]) -> float:
    recs = [r for r in recs if r in item_vectors]
    if len(recs) < 2:
        return 0.0
 
    sims = []
    for i in range(len(recs)):
        a = item_vectors[recs[i]]
        for j in range(i + 1, len(recs)):
            b = item_vectors[recs[j]]
            sims.append(float(np.clip(np.dot(a, b), 0.0, 1.0)))
 
    return float(np.clip(1.0 - mean(sims), 0.0, 1.0)) if sims else 0.0
 
 
def _load_recommendations_csv(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(column).strip() for column in df.columns]
    if "user_id" not in df.columns:
        raise ValueError("Recommendations CSV is missing column: user_id")

    item_col = next(
        (
            candidate
            for candidate in ["recommended_asin", "parent_asin", "neighbour_item"]
            if candidate in df.columns
        ),
        None,
    )
    if item_col is None:
        raise ValueError(
            "Recommendations CSV must contain one of: "
            "recommended_asin, parent_asin, neighbour_item"
        )

    df["user_id"] = df["user_id"].astype(str).str.strip()
    df[item_col] = df[item_col].astype(str).str.strip()

    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df = df.sort_values(["user_id", "rank"], ascending=[True, True])
    elif "predicted_score" in df.columns:
        df = df.sort_values(["user_id", "predicted_score"], ascending=[True, False])

    df = df.drop_duplicates(subset=["user_id", item_col], keep="first")
    return df.groupby("user_id")[item_col].apply(lambda s: list(s.astype(str))).to_dict()
 
 
def _average_precision_at_k(rels: list[int], top_k: int, n_relevant: int) -> float:
    if n_relevant == 0:
        return 0.0
    denom = min(n_relevant, top_k)
    score = 0.0
    hits = 0
    for rank, rel in enumerate(rels[:top_k], start=1):
        if rel:
            hits += 1
            score += hits / rank
    return score / denom
 
 
def _ndcg_at_k(rels: list[int], top_k: int, n_relevant: int) -> float:
    def _dcg(values: list[int]) -> float:
        return sum((rel / np.log2(rank + 1)) for rank, rel in enumerate(values[:top_k], start=1))
 
    dcg = _dcg(rels)
    ideal = [1] * min(n_relevant, top_k)
    idcg = _dcg(ideal)
    return (dcg / idcg) if idcg > 0 else 0.0
 
 
@dataclass
class EvalResult:
    model: str
    top_k: int
    users_evaluated: int
    recall_at_k: float
    precision_at_k: float
    hit_rate_at_k: float
    mrr_at_k: float
    map_at_k: float
    ndcg_at_k: float
    intra_list_diversity_at_k: float
    popularity_bias_at_k: float
    coverage_at_k: float
 
 
def evaluate_rankings(
    recs_by_user: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    catalog_items: list[str],
    top_k: int,
    model_name: str,
    item_vectors: dict[str, np.ndarray] | None = None,
    popularity_lookup: dict[str, float] | None = None,
) -> tuple[EvalResult, pd.DataFrame]:
    _log(f"Evaluating {model_name} at top-{top_k}")
    user_rows = []
    unique_recommended_items: set[str] = set()
    item_vectors = item_vectors or {}
    popularity_lookup = popularity_lookup or {}
 
    for user_id, true_items in ground_truth.items():
        recs = [str(x) for x in recs_by_user.get(user_id, [])[:top_k]]
        rels = [1 if item in true_items else 0 for item in recs]
        n_hits = sum(rels)
        unique_recommended_items.update(recs)
 
        user_rows.append({
            "user_id": user_id,
            "hit_rate_at_k": int(n_hits > 0),
            "recall_at_k": round(n_hits / len(true_items), 6) if true_items else 0.0,
            "precision_at_k": round(n_hits / top_k, 6) if top_k else 0.0,
            "mrr_at_k": round(next((1.0 / rank for rank, rel in enumerate(rels, start=1) if rel), 0.0), 6),
            "map_at_k": round(_average_precision_at_k(rels, top_k, len(true_items)), 6),
            "ndcg_at_k": round(_ndcg_at_k(rels, top_k, len(true_items)), 6),
            "intra_list_diversity_at_k": round(_intra_list_diversity(recs, item_vectors), 6),
            "popularity_bias_at_k": round(float(mean([popularity_lookup.get(item, 0.0) for item in recs])) if recs else 0.0, 6),
            "n_true_items": len(true_items),
            "n_hits": n_hits,
            "recommendations": "|".join(recs),
        })
 
    detail_df = pd.DataFrame(user_rows)
    _log(f"Evaluation complete for {len(detail_df):,} users")
    if detail_df.empty:
        result = EvalResult(
            model=model_name,
            top_k=top_k,
            users_evaluated=0,
            recall_at_k=0.0,
            precision_at_k=0.0,
            hit_rate_at_k=0.0,
            mrr_at_k=0.0,
            map_at_k=0.0,
            ndcg_at_k=0.0,
            intra_list_diversity_at_k=0.0,
            popularity_bias_at_k=0.0,
            coverage_at_k=0.0,
        )
        return result, detail_df
 
    result = EvalResult(
        model=model_name,
        top_k=top_k,
        users_evaluated=len(detail_df),
        recall_at_k=float(detail_df["recall_at_k"].mean()),
        precision_at_k=float(detail_df["precision_at_k"].mean()),
        hit_rate_at_k=float(detail_df["hit_rate_at_k"].mean()),
        mrr_at_k=float(detail_df["mrr_at_k"].mean()),
        map_at_k=float(detail_df["map_at_k"].mean()),
        ndcg_at_k=float(detail_df["ndcg_at_k"].mean()),
        intra_list_diversity_at_k=float(detail_df["intra_list_diversity_at_k"].mean()),
        popularity_bias_at_k=float(detail_df["popularity_bias_at_k"].mean()),
        coverage_at_k=(len(unique_recommended_items) / len(catalog_items)) if catalog_items else 0.0,
    )
    return result, detail_df
 
 
def _build_catalog_items(
    reviews: pd.DataFrame | None,
    item_col: str,
    products_path: Path | None,
    fallback_items: list[str] | None = None,
) -> list[str]:
    if products_path and products_path.exists():
        products = pd.read_csv(products_path, low_memory=False)
        products = products.copy()
        products.columns = [str(c).strip() for c in products.columns]
        for candidate in ["parent_asin", "asin", "product_id", "item_id"]:
            if candidate in products.columns:
                return sorted(products[candidate].astype(str).str.strip().dropna().unique().tolist())
 
    if reviews is not None:
        return sorted(reviews[item_col].astype(str).str.strip().dropna().unique().tolist())
    return sorted(fallback_items or [])
 
 
def _result_row(result: EvalResult) -> dict[str, object]:
    return {
        "model": result.model,
        "top_k": result.top_k,
        "users_evaluated": result.users_evaluated,
        "recall_at_k": round(result.recall_at_k, 6),
        "precision_at_k": round(result.precision_at_k, 6),
        "hit_rate_at_k": round(result.hit_rate_at_k, 6),
        "mrr_at_k": round(result.mrr_at_k, 6),
        "map_at_k": round(result.map_at_k, 6),
        "ndcg_at_k": round(result.ndcg_at_k, 6),
        "intra_list_diversity_at_k": round(result.intra_list_diversity_at_k, 6),
        "popularity_bias_at_k": round(result.popularity_bias_at_k, 6),
        "coverage_at_k": round(result.coverage_at_k, 6),
    }


def _safe_model_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_").replace("-", "_")


def _save_outputs(result: EvalResult, detail_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([_result_row(result)])

    safe_name = _safe_model_name(result.model)
    summary_path = output_dir / f"{safe_name}_metrics.csv"
    detail_path = output_dir / f"{safe_name}_user_details.csv"
 
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)
 
    print(f"[SAVE] Summary -> {summary_path}")
    print(f"[SAVE] Detail   -> {detail_path}")


def _save_comparison_outputs(
    results: list[EvalResult],
    details: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "model_comparison_metrics.csv"
    pd.DataFrame([_result_row(result) for result in results]).to_csv(
        metrics_path, index=False
    )
    print(f"[SAVE] Combined metrics -> {metrics_path}")

    for model_name, detail_df in details.items():
        detail_path = output_dir / f"{_safe_model_name(model_name)}_user_details.csv"
        detail_df.to_csv(detail_path, index=False)
        print(f"[SAVE] Detail           -> {detail_path}")
 
 
def _print_result(result: EvalResult) -> None:
    print("\nFINAL RESULTS")
    print("=" * 72)
    print(f"Model          : {result.model}")
    print(f"Users evaluated : {result.users_evaluated}")
    print(f"Recall@{result.top_k:<2}      : {result.recall_at_k:.4f}")
    print(f"Precision@{result.top_k:<2}   : {result.precision_at_k:.4f}")
    print(f"HitRate@{result.top_k:<2}     : {result.hit_rate_at_k:.4f}")
    print(f"MRR@{result.top_k:<2}         : {result.mrr_at_k:.4f}")
    print(f"MAP@{result.top_k:<2}         : {result.map_at_k:.4f}")
    print(f"NDCG@{result.top_k:<2}        : {result.ndcg_at_k:.4f}")
    print(f"Diversity@{result.top_k:<2}   : {result.intra_list_diversity_at_k:.4f}")
    print(f"PopBias@{result.top_k:<2}     : {result.popularity_bias_at_k:.4f}")
    print(f"Coverage@{result.top_k:<2}    : {result.coverage_at_k:.4f}")
 
 
def build_popularity_recommendations(
    train_df: pd.DataFrame,
    user_col: str,
    item_col: str,
    catalog_items: list[str],
    top_k: int,
    user_ids: list[str],
    discovery_share: float,
    min_discovery_rating: float,
    min_discovery_reviews: int,
    blended_csv: "Path | None" = None,
) -> dict[str, list[str]]:
    return _recommend_popularity(
        train_df=train_df,
        user_ids=user_ids,
        item_col=item_col,
        top_k=top_k,
        discovery_share=discovery_share,
        min_discovery_rating=min_discovery_rating,
        min_discovery_reviews=min_discovery_reviews,
        blended_csv=blended_csv,
    )
 
 
MODEL_REGISTRY: dict[str, Callable[..., dict[str, list[str]]]] = {
    "popularity": build_popularity_recommendations,
}
 
 
def main() -> int:
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Evaluate ranking models with offline metrics.")
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS, help="Reviews CSV file.")
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS, help="Products CSV file used for coverage denominator when available.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for evaluation outputs.")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY.keys()), default="popularity", help="Built-in model to evaluate.")
    parser.add_argument(
        "--recommendations-csv",
        type=Path,
        default=None,
        help=(
            "Optional content-based recommendations CSV. It must contain user_id, "
            "an item column such as recommended_asin, and optionally rank. When "
            "provided, the evaluator compares it with the popularity baseline."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10, help="Cutoff for ranking metrics.")
    parser.add_argument("--discovery-share", type=float, default=0.20, help="Share of popularity.py recommendations reserved for discovery products.")
    parser.add_argument("--min-discovery-rating", type=float, default=3.5, help="Minimum average rating used by popularity.py's discovery gate.")
    parser.add_argument("--min-discovery-reviews", type=int, default=5, help="Minimum review count used by popularity.py's discovery gate.")
    parser.add_argument(
        "--blended-csv",
        type=Path,
        default=None,
        help=(
            "Optional pre-built blended popularity CSV from "
            "build_popularity_lists.py. By default, the evaluator computes the "
            "Top-K popularity/discovery blend from the training split."
        ),
    )
    args = parser.parse_args()
    # Allow --blended-csv none to disable the pre-built list
    if hasattr(args, "blended_csv") and str(args.blended_csv).lower() == "none":
        args.blended_csv = None
 
    _log("Starting evaluation run")
    split_key = _cache_key(args.reviews, CBF_HELPER_PATH)
    split_cache_file, split_manifest_file = _split_cache_paths(split_key)
    cached_split = _load_split_cache(split_cache_file)
 
    if cached_split is not None:
        _log(f"Loading cached split from {split_cache_file}")
        train_df = cached_split["train_df"]
        test_df = cached_split["test_df"]
        user_col = cached_split["user_col"]
        item_col = cached_split["item_col"]
        cached_catalog = cached_split.get("catalog_items")
        if cached_catalog is None:
            cached_catalog = sorted(
                pd.unique(pd.concat([train_df[item_col], test_df[item_col]], ignore_index=True).astype(str))
            )
            cached_split["catalog_items"] = cached_catalog
            _save_split_cache(split_cache_file, split_manifest_file, cached_split)
            _log("Upgraded split cache with catalog metadata")
        reviews = None
    else:
        reviews, user_col, item_col, cbf = _load_reviews(args.reviews)
        train_df, test_df = _split_like_cbf(reviews, user_col, cbf)
        cached_catalog = _build_catalog_items(reviews, item_col, products_path=None)
        _log(f"Saving split cache to {split_cache_file}")
        _save_split_cache(
            split_cache_file,
            split_manifest_file,
            {
                "manifest": {
                    "cache_version": SPLIT_CACHE_VERSION,
                    "reviews": str(args.reviews.resolve()),
                    "reviews_mtime": args.reviews.stat().st_mtime,
                    "reviews_size": args.reviews.stat().st_size,
                    "helper": str(CBF_HELPER_PATH.resolve()),
                    "helper_mtime": CBF_HELPER_PATH.stat().st_mtime,
                    "helper_size": CBF_HELPER_PATH.stat().st_size,
                },
                "train_df": train_df,
                "test_df": test_df,
                "user_col": user_col,
                "item_col": item_col,
                "catalog_items": cached_catalog,
            },
        )
        _log(f"Cached split to {split_cache_file}")
 
    catalog_items = _build_catalog_items(
        reviews,
        item_col,
        args.products,
        fallback_items=cached_catalog,
    )
    products = _load_products(args.products)
    filtered_test_df = _filter_test_to_training_items(
        train_df, test_df, user_col, item_col
    )
    ground_truth = _build_ground_truth(filtered_test_df, user_col, item_col)

    builder = MODEL_REGISTRY[args.model]
    _log(f"Building recommendations with model={args.model}")
    popularity_recs = builder(
        train_df=train_df,
        user_col=user_col,
        item_col=item_col,
        catalog_items=catalog_items,
        top_k=args.top_k,
        user_ids=sorted(ground_truth),
        discovery_share=args.discovery_share,
        min_discovery_rating=args.min_discovery_rating,
        min_discovery_reviews=args.min_discovery_reviews,
        blended_csv=getattr(args, "blended_csv", None),
    )
    recommendations_by_model = {"popularity": popularity_recs}

    if args.recommendations_csv is not None:
        _log(f"Loading content-based recommendations from {args.recommendations_csv}")
        recommendations_by_model["content-based"] = _load_recommendations_csv(
            args.recommendations_csv
        )

    # Embedding diversity only needs vectors for products recommended by either model.
    recommended_items = sorted({
        item
        for recs_by_user in recommendations_by_model.values()
        for user_id in ground_truth
        for user_recs in [recs_by_user.get(user_id, [])]
        for item in user_recs[:args.top_k]
    })
    item_vectors = _build_item_embeddings(products, recommended_items, args.products)
    popularity_lookup = _build_popularity_lookup(train_df, item_col, catalog_items)

    results = []
    details = {}
    for model_name, recs_by_user in recommendations_by_model.items():
        result, detail_df = evaluate_rankings(
            recs_by_user=recs_by_user,
            ground_truth=ground_truth,
            catalog_items=catalog_items,
            top_k=args.top_k,
            model_name=model_name,
            item_vectors=item_vectors,
            popularity_lookup=popularity_lookup,
        )
        results.append(result)
        details[model_name] = detail_df
        _print_result(result)

    if len(results) > 1:
        _save_comparison_outputs(results, details, args.output_dir)
    else:
        _save_outputs(results[0], details[results[0].model], args.output_dir)
    _log(f"Done in {time.perf_counter() - start:.1f}s")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
