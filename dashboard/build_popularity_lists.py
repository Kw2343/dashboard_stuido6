#!/usr/bin/env python3
"""
build_popularity_lists.py
=========================
Builds a single blended_products_top100.csv from the pre-filtered
warm_hot_users_interactions_only file.
 
SOURCE OF TRUTH
---------------
Reads directly from warm_hot_users_interactions_only.xlsx (or .csv).
This file already contains ONLY warm and hot users (>= 2 interactions),
so cold users and single-interaction products are already excluded.
 
BLEND DESIGN
------------
  • Slots  1– 80 → top 80 by popular_score   ("Popular" tier)
  • Slots 81–100 → top 20 by discovery_score  ("Discovery" tier)
                   products NOT already in the popular tier,
                   with avg_rating >= 3.5 and >= 2 interactions
 
TRAIN/TEST SPLIT (mirrors evaluation split exactly)
---------------------------------------------------
  - Users with < 5 interactions → all training
  - Users with >= 5 interactions → 80/20 chronological split
  - Scoring is done on TRAIN interactions only
 
Output (saved to OUTPUT_DIR):
  blended_products_top100.csv      — 80 popular + 20 discovery
  all_train_products_scored.csv    — all scored products ranked by popular_score
"""
 
from __future__ import annotations
import sys
from math import floor
from pathlib import Path
import pandas as pd
 
# ── Paths ─────────────────────────────────────────────────────────────────────
WARM_HOT_FILE = r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard_data\warm_hot_users_interactions_only.xlsx"
OUTPUT_DIR    = r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard\dashboard_data"
 
# ── Split rules ───────────────────────────────────────────────────────────────
MIN_REVIEWS_FOR_TEST = 5
MIN_TRAIN_SIZE       = 4
TEST_FRACTION        = 0.20
 
# ── Blend parameters ──────────────────────────────────────────────────────────
POPULAR_SLOTS    = 80
DISCOVERY_SLOTS  = 20
TOTAL_N          = POPULAR_SLOTS + DISCOVERY_SLOTS
 
# ── Discovery quality gate ────────────────────────────────────────────────────
MIN_DISC_RATING  = 3.5
MIN_DISC_REVIEWS = 2
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _norm(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)
 
 
def _holdout_size(n: int) -> int:
    if n < MIN_REVIEWS_FOR_TEST:
        return 0
    h = floor(n * TEST_FRACTION + 0.5)
    h = max(1, h)
    h = min(h, n - MIN_TRAIN_SIZE)
    return max(0, h)
 
 
def _split_train(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows, test_rows = [], []
    for _, grp in df.groupby("user_id", sort=False):
        grp = grp.sort_values("timestamp_unix", ascending=True)
        h = _holdout_size(len(grp))
        if h == 0:
            train_rows.append(grp)
        else:
            train_rows.append(grp.iloc[:-h])
            test_rows.append(grp.iloc[-h:])
    train_df = pd.concat(train_rows, ignore_index=True)
    test_df  = pd.concat(test_rows,  ignore_index=True) if test_rows else pd.DataFrame()
    return train_df, test_df
 
 
def _score(train_df: pd.DataFrame) -> pd.DataFrame:
    stats = train_df.groupby("parent_asin").agg(
        interaction_count=("parent_asin", "count"),
        avg_rating=("rating", "mean"),
        unique_users=("user_id", "nunique"),
    ).reset_index()
    stats["avg_rating"] = stats["avg_rating"].round(2)
 
    pop_n   = _norm(stats["interaction_count"])
    sat_n   = _norm(stats["avg_rating"])
    trend_n = _norm(stats["unique_users"])
    low_exp = 1 - pop_n
 
    stats["popular_score"]   = ((0.50 * pop_n + 0.30 * sat_n + 0.20 * trend_n) * 5).round(3)
    stats["discovery_score"] = ((0.40 * sat_n + 0.30 * trend_n + 0.30 * low_exp) * 5).round(3)
    stats["low_exposure"]    = low_exp.round(4)
    return stats
 
 
def _safe_save(df: pd.DataFrame, path: Path, label: str) -> None:
    try:
        df.to_csv(path, index=False)
        print(f"  ✅ {label:55s} {len(df):>6,} rows  →  {path.name}")
    except PermissionError:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.parent / f"{path.stem}_{stamp}{path.suffix}"
        df.to_csv(alt, index=False)
        print(f"  ⚠  Locked — saved as {alt.name}")
 
 
# ── Main ─────────────────────────────────────────────────────────────────────
 
def main() -> None:
    SEP = "=" * 72
    print(SEP)
    print("  Build Blended Popularity+Discovery List")
    print(f"  Source : warm_hot_users_interactions_only (pre-filtered)")
    print(f"  Blend  : {POPULAR_SLOTS} popular + {DISCOVERY_SLOTS} discovery = {TOTAL_N} total")
    print(SEP)
 
    warm_hot_path = Path(WARM_HOT_FILE)
    out_dir       = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    # ── Load ────────────────────────────────────────────────────────────────
    print("\n[LOAD]  warm_hot_users_interactions_only ...")
    if not warm_hot_path.exists():
        print(f"  ERROR: {warm_hot_path} not found"); sys.exit(1)
 
    suffix = warm_hot_path.suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        df = pd.read_excel(warm_hot_path, engine="openpyxl")
    elif suffix == ".xls":
        df = pd.read_excel(warm_hot_path, engine="xlrd")
    else:
        df = pd.read_csv(warm_hot_path, low_memory=False)
 
    df.columns = [str(c).strip() for c in df.columns]
    print(f"  {len(df):,} rows | {df['user_id'].nunique():,} unique users "
          f"| {df['parent_asin'].nunique():,} unique products")
 
    # ── Step 1: Split ────────────────────────────────────────────────────────
    print("\n[SPLIT] Applying 80/20 chronological split ...")
    train_df, test_df = _split_train(df)
 
    eligible   = sum(1 for _, g in df.groupby("user_id") if _holdout_size(len(g)) > 0)
    ineligible = df["user_id"].nunique() - eligible
    print(f"  Users >= {MIN_REVIEWS_FOR_TEST} reviews (80/20 split): {eligible:,}")
    print(f"  Users <  {MIN_REVIEWS_FOR_TEST} reviews (all-train)  : {ineligible:,}")
    print(f"  Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")
    print(f"  Train unique products: {train_df['parent_asin'].nunique():,}")
 
    # ── Step 2: Score ────────────────────────────────────────────────────────
    print("\n[SCORE] Computing popularity & discovery scores from TRAIN ...")
    stats = _score(train_df)
    print(f"  Products scored       : {len(stats):,}")
    print(f"  Popular score range   : {stats['popular_score'].min():.3f} – {stats['popular_score'].max():.3f}")
    print(f"  Discovery score range : {stats['discovery_score'].min():.3f} – {stats['discovery_score'].max():.3f}")
 
    # Interaction count breakdown
    print(f"\n  Interaction count breakdown (train):")
    print(f"    1 interaction   : {(stats['interaction_count']==1).sum():,} products")
    print(f"    2 interactions  : {(stats['interaction_count']==2).sum():,} products")
    print(f"    3–5             : {stats['interaction_count'].between(3,5).sum():,} products")
    print(f"    6–10            : {stats['interaction_count'].between(6,10).sum():,} products")
    print(f"    > 10            : {(stats['interaction_count']>10).sum():,} products")
 
    # ── Step 3: Build blended 80/20 list ─────────────────────────────────────
    print("\n[BLEND] Building 80/20 blended list ...")
    popular_tier = stats.sort_values("popular_score", ascending=False).head(POPULAR_SLOTS).copy()
    popular_tier["tier"]       = "popular"
    popular_tier["blend_rank"] = range(1, len(popular_tier) + 1)
    pop_asin_set = set(popular_tier["parent_asin"])
 
    disc_pool = stats[
        (~stats["parent_asin"].isin(pop_asin_set)) &
        (stats["avg_rating"] >= MIN_DISC_RATING) &
        (stats["interaction_count"] >= MIN_DISC_REVIEWS)
    ].sort_values("discovery_score", ascending=False)
 
    discovery_tier = disc_pool.head(DISCOVERY_SLOTS).copy()
    discovery_tier["tier"]       = "discovery"
    discovery_tier["blend_rank"] = range(POPULAR_SLOTS + 1, POPULAR_SLOTS + len(discovery_tier) + 1)
 
    blended = pd.concat([popular_tier, discovery_tier], ignore_index=True)
    print(f"  Popular tier   : {len(popular_tier)} / {POPULAR_SLOTS} slots filled")
    print(f"  Discovery pool : {len(disc_pool):,} candidates")
    print(f"  Discovery tier : {len(discovery_tier)} / {DISCOVERY_SLOTS} slots filled")
 
    # ── Step 4: Print ranked tables ───────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  TOP {POPULAR_SLOTS} MOST POPULAR PRODUCTS  (Popular Score = 50% popularity + 30% rating + 20% trending)")
    print(f"{SEP}")
    print(f"  {'#':<4} {'parent_asin':<14} {'interactions':>13} {'avg_rating':>10} {'unique_users':>13} {'popular_score':>13}")
    print(f"  {'-'*72}")
    for _, row in popular_tier.iterrows():
        print(f"  {int(row['blend_rank']):<4} {row['parent_asin']:<14} "
              f"{int(row['interaction_count']):>13,} {row['avg_rating']:>10.2f} "
              f"{int(row['unique_users']):>13,} {row['popular_score']:>13.3f}")
 
    print(f"\n{SEP}")
    print(f"  TOP {DISCOVERY_SLOTS} LEAST POPULAR / MOST DISCOVERABLE PRODUCTS")
    print(f"  (Discovery Score = 40% rating + 30% trending + 30% low exposure | gate: rating >= {MIN_DISC_RATING}, interactions >= {MIN_DISC_REVIEWS})")
    print(f"{SEP}")
    print(f"  {'#':<4} {'parent_asin':<14} {'interactions':>13} {'avg_rating':>10} {'unique_users':>13} {'disc_score':>10} {'low_exp':>9}")
    print(f"  {'-'*80}")
    for _, row in discovery_tier.iterrows():
        print(f"  {int(row['blend_rank']):<4} {row['parent_asin']:<14} "
              f"{int(row['interaction_count']):>13,} {row['avg_rating']:>10.2f} "
              f"{int(row['unique_users']):>13,} {row['discovery_score']:>10.3f} {row['low_exposure']:>9.4f}")
 
    # ── Step 5: Theoretical hit rate ceiling ──────────────────────────────────
    reachable = len(stats)
    print(f"\n{SEP}")
    print("  THEORETICAL MAX HIT RATE  (exact-match ceiling)")
    print(f"{SEP}")
    print(f"  Warm/hot train products : {reachable:,}")
    for top_n in [10, 20, 50, 100]:
        ceiling = min(top_n, reachable) / reachable * 100 if reachable else 0
        marker = " ← this list" if top_n == TOTAL_N else ""
        print(f"  Top-{top_n:<4} → theoretical max hit rate ~ {ceiling:5.1f}%{marker}")
 
    # ── Step 6: Save ──────────────────────────────────────────────────────────
    export_cols = [c for c in [
        "blend_rank", "tier", "parent_asin",
        "interaction_count", "avg_rating", "unique_users",
        "popular_score", "discovery_score", "low_exposure",
    ] if c in blended.columns]
 
    all_cols = [c for c in [
        "parent_asin", "interaction_count", "avg_rating", "unique_users",
        "popular_score", "discovery_score", "low_exposure",
    ] if c in stats.columns]
 
    print(f"\n[SAVE]  Writing to {out_dir} ...")
    _safe_save(blended[export_cols], out_dir / "blended_products_top100.csv",
               "blended_products_top100.csv  (80 popular + 20 discovery)")
    _safe_save(stats[all_cols].sort_values("popular_score", ascending=False),
               out_dir / "all_train_products_scored.csv",
               "all_train_products_scored.csv (all warm/hot train products)")
 
    print(f"\n[DONE]\n{SEP}")
 
 
if __name__ == "__main__":
    main()