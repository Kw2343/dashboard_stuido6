"""
prepare_dashboard_data.py
=========================
Run this ONCE after training to prepare all datasets for the dashboard.
 
What it produces (in dashboard_data/):
  popular_products.csv     — ranked product catalogue with Bayesian popularity score
  ncf_dashboard.csv        — NCF top-N recommendations enriched with product metadata
  cbf_dashboard.csv        — CBF top-N recommendations enriched with product metadata
  model_comparison.csv     — NCF vs CBF vs Popularity side-by-side evaluation metrics
  user_profiles.csv        — per-user behavioural summary for the Users tab
  product_catalog.csv      — full enriched product catalogue (title, store, rating, etc.)
  copurchase_pairs.csv     — bought-together pairs with product names resolved
 
Run from your dashboard/ folder:
  python prepare_dashboard_data.py
"""
 
from __future__ import annotations
 
import sys
from pathlib import Path
 
import numpy  as np
import pandas as pd
 
# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit paths if your layout differs
# ════════════════════════════════════════════════════════════════════════════
 
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
NCF_DIR   = BASE_DIR / "ncf_results"
CBF_DIR   = BASE_DIR / "cbf_results"
OUT_DIR   = BASE_DIR / "dashboard_data"
 
# Source files
REVIEWS_FILE    = DATA_DIR / "reviews_clean_no_exact_duplicates.csv"
PRODUCTS_FILE   = DATA_DIR / "products_clean.csv"
USERS_FILE      = DATA_DIR / "user_summary.csv"
ASIN_ITEM_FILE  = DATA_DIR / "asin_item.csv"
COPURCHASE_FILE = DATA_DIR / "products_bought_together_pair_counts.xlsx"
 
# Model output files
NCF_RECS_FILE   = NCF_DIR / "ncf_user_recommendations.csv"
NCF_EVAL_FILE   = NCF_DIR / "ncf_evaluation_detail.csv"
NCF_SUMMARY_FILE= NCF_DIR / "ncf_summary.csv"
CBF_RECS_FILE   = CBF_DIR / "cbf_user_recommendations.csv"
CBF_EVAL_FILE   = CBF_DIR / "cbf_evaluation_detail.csv"
SIG_FILE        = CBF_DIR / "significance_tests.csv"
 
# Bayesian popularity prior (same value as your popularity tab)
BAYESIAN_M = 50
 
 
# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════
 
def _log(msg: str) -> None:
    print(f"  {msg}")
 
 
def _safe_read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    """Return DataFrame or None (with a warning) if file missing."""
    if not path.exists():
        print(f"  ⚠️  Not found (skipped): {path}")
        return None
    df = pd.read_csv(path, **kwargs)
    _log(f"✅  Loaded  {path.name:50s}  →  {len(df):,} rows")
    return df
 
 
def _save(df: pd.DataFrame, name: str) -> None:
    out = OUT_DIR / name
    df.to_csv(out, index=False)
    _log(f"💾  Saved   {name:50s}  →  {len(df):,} rows")
 
 
# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD SOURCE DATA
# ════════════════════════════════════════════════════════════════════════════
 
def load_sources() -> dict:
    print("\n📂  Loading source data …")
 
    reviews  = _safe_read_csv(
        REVIEWS_FILE,
        usecols=["user_id","parent_asin","rating",
                 "timestamp_unix","verified_purchase","helpful_vote",
                 "review_year","review_month","review_length_words"],
        low_memory=False,
    )
 
    products = _safe_read_csv(
        PRODUCTS_FILE,
        usecols=["parent_asin","title","average_rating","rating_number",
                 "price","store_clean","has_description","has_features",
                 "has_price","has_categories"],
        low_memory=False,
    )
    if products is not None:
        products = products.drop_duplicates(subset=["parent_asin"])
        products["parent_asin"] = products["parent_asin"].astype(str)
 
    users = _safe_read_csv(USERS_FILE)
    if users is not None:
        users["user_id"] = users["user_id"].astype(str)
 
    asin_item = _safe_read_csv(ASIN_ITEM_FILE)
    if asin_item is not None:
        asin_item["parent_asin"] = asin_item["parent_asin"].astype(str)
 
    # Co-purchase pairs (Excel)
    copurchase = None
    if COPURCHASE_FILE.exists():
        try:
            copurchase = pd.read_excel(COPURCHASE_FILE)
            _log(f"✅  Loaded  {COPURCHASE_FILE.name:50s}  →  {len(copurchase):,} rows")
        except Exception as e:
            print(f"  ⚠️  Could not read {COPURCHASE_FILE.name}: {e}")
 
    # Model outputs
    ncf_recs    = _safe_read_csv(NCF_RECS_FILE)
    ncf_eval    = _safe_read_csv(NCF_EVAL_FILE)
    ncf_summary = _safe_read_csv(NCF_SUMMARY_FILE)
    cbf_recs    = _safe_read_csv(CBF_RECS_FILE)
    cbf_eval    = _safe_read_csv(CBF_EVAL_FILE)
    sig_tests   = _safe_read_csv(SIG_FILE)
 
    return dict(
        reviews=reviews, products=products, users=users,
        asin_item=asin_item, copurchase=copurchase,
        ncf_recs=ncf_recs, ncf_eval=ncf_eval, ncf_summary=ncf_summary,
        cbf_recs=cbf_recs, cbf_eval=cbf_eval, sig_tests=sig_tests,
    )
 
 
# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 — ENRICHED PRODUCT CATALOGUE
# ════════════════════════════════════════════════════════════════════════════
 
def build_product_catalog(products: pd.DataFrame,
                          asin_item: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge product metadata with human-readable item names.
    Adds a 'display_name' column that prioritises the asin_item short name.
    """
    print("\n📦  Building enriched product catalogue …")
 
    cat = products.copy()
 
    if asin_item is not None:
        cat = cat.merge(
            asin_item[["parent_asin","Item"]].rename(columns={"Item":"item_name"}),
            on="parent_asin", how="left",
        )
        cat["display_name"] = cat["item_name"].fillna(cat["title"])
    else:
        cat["display_name"] = cat["title"]
 
    # Clean store name
    cat["store"] = cat.get("store_clean", "Unknown").fillna("Unknown")
    cat["store"] = cat["store"].replace("(missing store)", "Unknown")
 
    # Truncate long titles for display
    cat["short_title"] = cat["title"].str[:60]
 
    # Metadata completeness score (0–100%)
    has_cols = [c for c in ["has_description","has_features","has_price","has_categories"]
                if c in cat.columns]
    if has_cols:
        cat["metadata_completeness_pct"] = cat[has_cols].mean(axis=1) * 100
 
    _log(f"Product catalogue: {len(cat):,} products")
    return cat
 
 
# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 — POPULARITY SCORE  (Bayesian, same as dashboard tab)
# ════════════════════════════════════════════════════════════════════════════
 
def build_popular_products(reviews: pd.DataFrame,
                           products: pd.DataFrame) -> pd.DataFrame:
    """
    WHY BAYESIAN INSTEAD OF MinMaxScaler:
      MinMaxScaler normalises each column independently to [0,5].
      A product with 1 review at 5★ gets the same rating_norm as one
      with 10,000 reviews at 5★ — that's misleading.
 
      The Bayesian formula shrinks low-volume products toward the
      catalogue average, so a product needs BOTH high ratings AND
      sufficient volume to rank near the top.
 
    Formula:
      score = 0.7 × [(v/(v+m))·R + (m/(v+m))·C]
            + 0.3 × log-normalised purchase frequency
      where v = review count, m = 50, R = product avg, C = catalogue avg
    """
    print("\n🏆  Computing popularity scores (Bayesian) …")
 
    # Aggregate from reviews (ground truth)
    agg = (
        reviews.groupby("parent_asin")
        .agg(
            review_count    =("rating",         "count"),
            avg_rating_raw  =("rating",         "mean"),
            purchase_freq   =("user_id",        "nunique"),
            helpful_votes   =("helpful_vote",   "sum"),
            verified_pct    =("verified_purchase","mean"),
        )
        .reset_index()
    )
 
    # Bayesian rating
    C     = agg["avg_rating_raw"].mean()
    m     = BAYESIAN_M
    v     = agg["review_count"]
    R     = agg["avg_rating_raw"]
    f     = agg["purchase_freq"]
    f_max = f.max()
    f_norm = np.log1p(f) / (np.log1p(f_max + 1) or 1)
 
    agg["bayesian_rating"]  = (v / (v + m)) * R + (m / (v + m)) * C
    agg["popularity_score"] = (0.7 * agg["bayesian_rating"]) + (0.3 * f_norm * 5)
    agg["popularity_rank"]  = agg["popularity_score"].rank(ascending=False).astype(int)
 
    # Merge with product metadata
    meta_cols = [c for c in ["parent_asin","title","display_name","short_title",
                              "store","price","metadata_completeness_pct"]
                 if c in products.columns]
    pop = agg.merge(products[meta_cols], on="parent_asin", how="left")
 
    # Rating tier for chart colouring
    def _tier(r):
        if pd.isna(r): return "Unknown"
        if r >= 4.5:   return "Excellent (4.5+)"
        if r >= 4.0:   return "Good (4.0–4.4)"
        if r >= 3.5:   return "Average (3.5–3.9)"
        return "Below average (<3.5)"
 
    pop["rating_tier"] = pop["avg_rating_raw"].apply(_tier)
    pop = pop.sort_values("popularity_score", ascending=False)
 
    _log(f"Popular products: {len(pop):,} products scored")
    return pop
 
 
# ════════════════════════════════════════════════════════════════════════════
#  STEP 4 — ENRICH RECOMMENDATION FILES WITH PRODUCT METADATA
# ════════════════════════════════════════════════════════════════════════════
 
def enrich_recommendations(recs: pd.DataFrame,
                            products: pd.DataFrame,
                            model_label: str) -> pd.DataFrame:
    """
    Join recommendation output with product metadata so the dashboard
    can display product names, ratings, prices, etc.
    """
    if recs is None:
        return pd.DataFrame()
 
    # Normalise ASIN column name (may be 'parent_asin' or 'asin')
    if "asin" in recs.columns and "parent_asin" not in recs.columns:
        recs = recs.rename(columns={"asin": "parent_asin"})
    recs = recs.copy()
    recs["parent_asin"] = recs["parent_asin"].astype(str)
 
    meta_cols = [c for c in ["parent_asin","title","display_name","short_title",
                              "store","price","average_rating","rating_number",
                              "metadata_completeness_pct"]
                 if c in products.columns]
 
    enriched = recs.merge(products[meta_cols], on="parent_asin", how="left")
    enriched["model"] = model_label
 
    # Normalise score column name for the dashboard
    for col in ["predicted_score","score","predicted_rating","cosine_similarity"]:
        if col in enriched.columns and "display_score" not in enriched.columns:
            enriched["display_score"] = enriched[col]
 
    _log(f"{model_label} recommendations: {len(enriched):,} rows")
    return enriched
 
 
# ════════════════════════════════════════════════════════════════════════════
#  STEP 5 — MODEL COMPARISON TABLE
# ════════════════════════════════════════════════════════════════════════════
 
def build_model_comparison(ncf_eval:    pd.DataFrame | None,
                            cbf_eval:    pd.DataFrame | None,
                            pop_df:      pd.DataFrame,
                            ncf_summary: pd.DataFrame | None) -> pd.DataFrame:
    """
    One row per model with Hit Rate, Precision@10, Recall@10.
    Includes Popularity baseline computed from the reviews data.
    """
    print("\n📊  Building model comparison table …")
 
    rows = []
 
    def _metrics(df: pd.DataFrame, label: str):
        if df is None or df.empty:
            return
        row = {"Model": label}
        for col, metric in [
            ("hit",            "Hit Rate@10"),
            ("precision_at_k", "Precision@10"),
            ("recall_at_k",    "Recall@10"),
        ]:
            row[metric] = round(df[col].mean(), 6) if col in df.columns else None
        rows.append(row)
 
    _metrics(ncf_eval, "NCF — Neural Collaborative Filtering")
    _metrics(cbf_eval, "CBF — Content-Based Filtering")
 
    # Popularity baseline: HR = % of users whose most-purchased product
    # is in the global top-10 (simple proxy)
    if ncf_summary is not None:
        pop_row = ncf_summary[ncf_summary["Model"].str.contains("Popularity", case=False, na=False)]
        if not pop_row.empty:
            r = pop_row.iloc[0]
            row = {"Model": "Popularity Baseline"}
            for c in r.index:
                if "HitRate"   in c: row["Hit Rate@10"]  = float(r[c])
                if "Precision" in c: row["Precision@10"] = float(r[c])
                if "Recall"    in c: row["Recall@10"]    = float(r[c])
            rows.append(row)
 
    if not rows:
        _log("⚠️  No evaluation data found — model_comparison.csv will be empty")
        return pd.DataFrame()
 
    comp = pd.DataFrame(rows)
 
    # Add lift vs popularity baseline
    baseline = comp[comp["Model"].str.contains("Popularity", case=False, na=False)]["Hit Rate@10"]
    baseline = float(baseline.iloc[0]) if len(baseline) else None
    if baseline:
        comp["HR Lift vs Baseline (pp)"] = (comp["Hit Rate@10"] - baseline).round(6)
        comp["HR Lift vs Baseline (%)"]  = ((comp["Hit Rate@10"] / baseline - 1) * 100).round(2)
 
    _log(f"Model comparison: {len(comp)} models")
    return comp
 
 
# ════════════════════════════════════════════════════════════════════════════
#  STEP 6 — USER PROFILES  (for the Users tab)
# ════════════════════════════════════════════════════════════════════════════
 
def build_user_profiles(reviews:  pd.DataFrame,
                         users:    pd.DataFrame | None,
                         ncf_eval: pd.DataFrame | None,
                         cbf_eval: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge user_summary with per-user model evaluation results.
    Useful for the Users tab to show who benefited most from recommendations.
    """
    print("\n👤  Building user profiles …")
 
    # Compute engagement metrics from reviews
    eng = reviews.groupby("user_id").agg(
        review_count       =("rating",          "count"),
        avg_rating_given   =("rating",          "mean"),
        pct_5star          =("rating",          lambda x: (x==5).mean()),
        pct_1star          =("rating",          lambda x: (x==1).mean()),
        verified_pct       =("verified_purchase","mean"),
        avg_helpful_votes  =("helpful_vote",    "mean"),
        avg_review_words   =("review_length_words","mean"),
    ).reset_index()
 
    eng["user_segment"] = pd.cut(
        eng["review_count"],
        bins   = [0, 1, 5, 20, 50, float("inf")],
        labels = ["One-time", "Casual (2–5)", "Active (6–20)", "Power (21–50)", "Super (51+)"],
    ).astype(str)
 
    # Merge with user_summary if available
    if users is not None:
        users["user_id"] = users["user_id"].astype(str)
        eng = eng.merge(users, on="user_id", how="left", suffixes=("","_summary"))
 
    # Merge NCF evaluation per user
    if ncf_eval is not None and "user_id" in ncf_eval.columns:
        ncf_u = ncf_eval[["user_id","hit","precision_at_k","recall_at_k"]].rename(
            columns={"hit":"ncf_hit","precision_at_k":"ncf_precision","recall_at_k":"ncf_recall"}
        )
        eng = eng.merge(ncf_u, on="user_id", how="left")
 
    # Merge CBF evaluation per user
    if cbf_eval is not None and "user_id" in cbf_eval.columns:
        cbf_u = cbf_eval[["user_id","hit","precision_at_k","recall_at_k"]].rename(
            columns={"hit":"cbf_hit","precision_at_k":"cbf_precision","recall_at_k":"cbf_recall"}
        )
        eng = eng.merge(cbf_u, on="user_id", how="left")
 
    _log(f"User profiles: {len(eng):,} users")
    return eng
 
 
# ════════════════════════════════════════════════════════════════════════════
#  STEP 7 — CO-PURCHASE PAIRS WITH PRODUCT NAMES
# ════════════════════════════════════════════════════════════════════════════
 
def build_copurchase_dashboard(copurchase: pd.DataFrame | None,
                                products:   pd.DataFrame) -> pd.DataFrame | None:
    if copurchase is None:
        return None
 
    print("\n🛒  Enriching co-purchase pairs …")
 
    df = copurchase.copy()
    df.columns = [c.strip().lower() for c in df.columns]
 
    # Normalise column names
    df = df.rename(columns={
        "parent_asin_a": "parent_asin_1",
        "parent_asin_b": "parent_asin_2",
        "frequency":     "pair_count",
    })
 
    titles = products[["parent_asin","display_name","store","price"]].copy()
 
    df = df.merge(titles.rename(columns={
        "parent_asin":"parent_asin_1",
        "display_name":"product_1_name",
        "store":"product_1_store",
        "price":"product_1_price",
    }), on="parent_asin_1", how="left")
 
    df = df.merge(titles.rename(columns={
        "parent_asin":"parent_asin_2",
        "display_name":"product_2_name",
        "store":"product_2_store",
        "price":"product_2_price",
    }), on="parent_asin_2", how="left")
 
    df["pair_label"] = (
        df["product_1_name"].fillna(df["parent_asin_1"]).str[:40] + " + " +
        df["product_2_name"].fillna(df["parent_asin_2"]).str[:40]
    )
 
    df = df.sort_values("pair_count", ascending=False)
    _log(f"Co-purchase pairs: {len(df):,} pairs enriched")
    return df
 
 
# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
 
def main():
    print("=" * 60)
    print("  Dashboard Data Preparation")
    print("=" * 60)
 
    OUT_DIR.mkdir(exist_ok=True)
 
    # ── 1. Load everything ────────────────────────────────────────────────
    src = load_sources()
 
    if src["reviews"] is None or src["products"] is None:
        print("\n❌  reviews or products file missing — cannot continue.")
        sys.exit(1)
 
    # ── 2. Enriched product catalogue ─────────────────────────────────────
    catalog = build_product_catalog(src["products"], src["asin_item"])
    _save(catalog, "product_catalog.csv")
 
    # ── 3. Popularity scores ──────────────────────────────────────────────
    pop = build_popular_products(src["reviews"], catalog)
    _save(pop, "popular_products.csv")
 
    # ── 4. Enrich recommendations with product metadata ───────────────────
    if src["ncf_recs"] is not None:
        ncf_dash = enrich_recommendations(src["ncf_recs"], catalog, "NCF")
        _save(ncf_dash, "ncf_dashboard.csv")
 
    if src["cbf_recs"] is not None:
        cbf_dash = enrich_recommendations(src["cbf_recs"], catalog, "CBF")
        _save(cbf_dash, "cbf_dashboard.csv")
 
    # ── 5. Model comparison ────────────────────────────────────────────────
    comp = build_model_comparison(
        src["ncf_eval"], src["cbf_eval"], pop, src["ncf_summary"]
    )
    if not comp.empty:
        _save(comp, "model_comparison.csv")
 
    # ── 6. User profiles ───────────────────────────────────────────────────
    profiles = build_user_profiles(
        src["reviews"], src["users"], src["ncf_eval"], src["cbf_eval"]
    )
    _save(profiles, "user_profiles.csv")
 
    # ── 7. Co-purchase pairs ───────────────────────────────────────────────
    pairs = build_copurchase_dashboard(src["copurchase"], catalog)
    if pairs is not None:
        _save(pairs, "copurchase_dashboard.csv")
 
    # ── 8. Significance tests (copy through) ──────────────────────────────
    if src["sig_tests"] is not None:
        _save(src["sig_tests"], "significance_tests.csv")
 
    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅  Dashboard data ready in:  dashboard_data/")
    print("=" * 60)
    print()
    created = list(OUT_DIR.glob("*.csv"))
    for f in sorted(created):
        try:
            rows = sum(1 for _ in open(f, encoding="utf-8", errors="ignore")) - 1
        except Exception:
            rows = -1
        print(f"  {f.name:45s}  {rows:>8,} rows")
    print()
    print("  Next step: run  python -m streamlit run app.py")
 
 
if __name__ == "__main__":
    main()