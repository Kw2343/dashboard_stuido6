"""
enrich_reviews_cbf.py
=====================
Takes reviews_clean_no_exact_duplicates.csv (all original columns intact)
and adds CBF-derived columns for each (user, product) row.
 
OUTPUT: dashboard_data/reviews_cbf_enriched.csv
  — identical structure to the original reviews file
  — with these NEW columns appended:
 
  item_content_richness    float  0–1   How much usable text the product has
                                        (title + description + features + categories)
                                        0 = no metadata  1 = fully described product
 
  user_item_similarity     float  0–1   Cosine similarity between this user's
                                        taste profile (avg TF-IDF of liked items)
                                        and this product's TF-IDF vector.
                                        High = product matches user's typical preferences.
 
  cbf_predicted_rating     float  0–5   Blended score:
                                        0.5 × actual catalogue rating
                                      + 0.5 × (user_item_similarity × 5)
                                        Matches the Y_PredRating in your scatter plot.
 
  cbf_rank_for_user        int         Rank of this product among all products
                                        for this user (1 = best CBF match).
                                        NaN if user has no liked items in training.
 
  is_cbf_top10             bool        True if this product is in the user's
                                        top-10 CBF recommendations.
 
Run from your dashboard/ folder:
  python enrich_reviews_cbf.py
 
Requires:  pip install scikit-learn tqdm
"""
 
from __future__ import annotations
 
import warnings
warnings.filterwarnings("ignore")
 
from pathlib import Path
import numpy  as np
import pandas as pd
from tqdm   import tqdm
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.preprocessing           import normalize
 
 
# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
 
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "dashboard_data"
 
REVIEWS_FILE  = DATA_DIR / "reviews_clean_no_exact_duplicates.csv"
PRODUCTS_FILE = DATA_DIR / "products_clean.csv"
ASIN_FILE     = DATA_DIR / "asin_item.csv"
OUTPUT_FILE   = OUT_DIR  / "reviews_cbf_enriched.csv"
 
LIKED_THRESHOLD    = 4.0     # ratings >= this count as "liked" for profile building
TFIDF_MAX_FEATURES = 5_000   # vocabulary size
TOP_N              = 10      # for is_cbf_top10 flag
BATCH_SIZE         = 500     # users processed per batch (memory management)
 
 
# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
 
def load_data():
    print("\n📂  Loading data …")
 
    reviews = pd.read_csv(REVIEWS_FILE, low_memory=False)
    print(f"   Reviews:  {len(reviews):,} rows  ×  {len(reviews.columns)} columns")
 
    products = pd.read_csv(
        PRODUCTS_FILE,
        usecols=["parent_asin","title","average_rating",
                 "description","features","categories","store_clean"],
        low_memory=False,
    ).drop_duplicates(subset=["parent_asin"])
    products["parent_asin"] = products["parent_asin"].astype(str)
    print(f"   Products: {len(products):,} unique products")
 
    asin_item = None
    if ASIN_FILE.exists():
        asin_item = pd.read_csv(ASIN_FILE)[["parent_asin","Item"]]
        asin_item["parent_asin"] = asin_item["parent_asin"].astype(str)
 
    # Ensure consistent dtypes in reviews
    reviews["parent_asin"] = reviews["parent_asin"].astype(str)
    reviews["user_id"]     = reviews["user_id"].astype(str)
    reviews["rating"]      = pd.to_numeric(reviews["rating"], errors="coerce")
 
    return reviews, products, asin_item
 
 
# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 — BUILD TF-IDF PRODUCT FINGERPRINTS
# ════════════════════════════════════════════════════════════════════════════
 
def build_product_text(row: pd.Series) -> str:
    """Combine all text fields into one string — same logic as CBF model."""
    parts = []
    title = str(row.get("title","")).strip()
    if title and title.lower() not in ("nan","none",""):
        parts.extend([title] * 3)   # weight title 3×
    for col in ["categories","features","description"]:
        val = str(row.get(col,"")).strip()
        if val and val.lower() not in ("nan","none",""):
            parts.append(val[:400])
    store = str(row.get("store_clean","")).strip()
    if store and store not in ("nan","none","(missing store)",""):
        parts.append(store)
    return " ".join(parts) if parts else "unknown product"
 
 
def fit_tfidf(products: pd.DataFrame):
    print("\n⚙️   Fitting TF-IDF on product catalogue …")
 
    products = products.copy()
    products["_text"] = products.apply(build_product_text, axis=1)
    word_counts = products["_text"].str.split().str.len()
 
    # Content richness: normalised word count (0–1, capped at 200 words = 1.0)
    products["item_content_richness"] = (word_counts.clip(0, 200) / 200).round(4)
 
    vectorizer = TfidfVectorizer(
        max_features = TFIDF_MAX_FEATURES,
        ngram_range  = (1, 2),
        stop_words   = "english",
        strip_accents = "unicode",
        min_df       = 2,
        sublinear_tf = True,
    )
    tfidf_matrix = vectorizer.fit_transform(products["_text"])
 
    # L2-normalise rows so cosine_similarity = dot product (faster later)
    tfidf_normed = normalize(tfidf_matrix, norm="l2")
 
    # Build ASIN → matrix row index
    asin_index = {str(a): i for i, a in enumerate(products["parent_asin"])}
 
    print(f"   Vocabulary: {tfidf_matrix.shape[1]:,} terms")
    print(f"   Products with text: {len(products):,}")
 
    return vectorizer, tfidf_normed, asin_index, products[["parent_asin","item_content_richness"]]
 
 
# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 — COMPUTE PER-ROW CBF SCORES
# ════════════════════════════════════════════════════════════════════════════
 
def compute_cbf_scores(
    reviews:         pd.DataFrame,
    products:        pd.DataFrame,
    tfidf_normed,
    asin_index:      dict,
    richness_df:     pd.DataFrame,
) -> pd.DataFrame:
    """
    For every (user_id, parent_asin) pair in reviews, compute:
      user_item_similarity  — cosine sim between user taste profile and product
      cbf_predicted_rating  — blended score
      cbf_rank_for_user     — rank of this product among all products for user
      is_cbf_top10          — True if in top-10 for this user
    """
    print(f"\n🔢  Computing CBF scores for {reviews['user_id'].nunique():,} users …")
    print(f"    Processing in batches of {BATCH_SIZE} users …")
 
    # Merge catalogue rating onto reviews (for blended predicted rating)
    cat_rating = products[["parent_asin","average_rating"]].copy()
    cat_rating["parent_asin"] = cat_rating["parent_asin"].astype(str)
    cat_rating["average_rating"] = pd.to_numeric(cat_rating["average_rating"], errors="coerce")
 
    reviews = reviews.merge(cat_rating, on="parent_asin", how="left", suffixes=("","_cat"))
    avg_global_rating = cat_rating["average_rating"].mean()
 
    # Output columns — initialise as NaN/False
    reviews["user_item_similarity"]  = np.nan
    reviews["cbf_predicted_rating"]  = np.nan
    reviews["cbf_rank_for_user"]     = np.nan
    reviews["is_cbf_top10"]          = False
 
    all_users   = reviews["user_id"].unique().tolist()
    n_users     = len(all_users)
    n_processed = 0
    n_skipped   = 0
 
    # ── Batch loop ──────────────────────────────────────────────────────────
    for batch_start in tqdm(range(0, n_users, BATCH_SIZE),
                            desc="  Users", ncols=70):
        batch_users = all_users[batch_start : batch_start + BATCH_SIZE]
 
        for user_id in batch_users:
            user_mask    = reviews["user_id"] == user_id
            user_reviews = reviews.loc[user_mask]
 
            # ── Build user taste profile ──────────────────────────────────
            liked_asins = user_reviews.loc[
                user_reviews["rating"] >= LIKED_THRESHOLD, "parent_asin"
            ].unique()
 
            if len(liked_asins) == 0:
                liked_asins = user_reviews["parent_asin"].unique()   # fallback: all
 
            liked_indices = [asin_index[a] for a in liked_asins if a in asin_index]
 
            if not liked_indices:
                n_skipped += 1
                continue
 
            # User profile = mean of liked item vectors (already L2-normalised rows)
            user_profile = np.asarray(
                tfidf_normed[liked_indices].mean(axis=0)
            )   # shape (1, n_features)
 
            # ── Compute similarity for every product in this user's review rows ──
            user_asins   = user_reviews["parent_asin"].unique()
            user_asin_idx = [asin_index[a] for a in user_asins if a in asin_index]
            valid_asins  = [a for a in user_asins if a in asin_index]
 
            if not user_asin_idx:
                n_skipped += 1
                continue
 
            item_matrix = tfidf_normed[user_asin_idx]   # (n_items, n_features)
            sims        = cosine_similarity(user_profile, item_matrix).flatten()
            sim_map     = dict(zip(valid_asins, sims))
 
            # ── Assign scores back to review rows ─────────────────────────
            for asin, sim in sim_map.items():
                asin_mask = user_mask & (reviews["parent_asin"] == asin)
                cat_rat   = reviews.loc[asin_mask, "average_rating"].iloc[0] \
                            if asin_mask.any() else np.nan
                cat_rat   = cat_rat if pd.notna(cat_rat) else avg_global_rating
 
                pred_rating = round(0.5 * float(cat_rat) + 0.5 * (sim * 5), 4)
 
                reviews.loc[asin_mask, "user_item_similarity"] = round(float(sim), 4)
                reviews.loc[asin_mask, "cbf_predicted_rating"] = pred_rating
 
            # ── Rank ALL products for this user to get cbf_rank_for_user ──
            # (among products this user actually reviewed)
            sorted_asins = sorted(sim_map, key=sim_map.get, reverse=True)
            rank_map     = {a: r+1 for r, a in enumerate(sorted_asins)}
 
            # is_cbf_top10 among all known products, not just reviewed ones
            # Full similarity over all products (for is_cbf_top10 flag)
            all_sims     = cosine_similarity(user_profile, tfidf_normed).flatten()
            top10_indices = set(all_sims.argsort()[::-1][:TOP_N])
            all_asins_list= list(asin_index.keys())
            top10_asins   = {all_asins_list[i] for i in top10_indices
                             if i < len(all_asins_list)}
 
            for asin in valid_asins:
                asin_mask = user_mask & (reviews["parent_asin"] == asin)
                reviews.loc[asin_mask, "cbf_rank_for_user"] = rank_map.get(asin, np.nan)
                if asin in top10_asins:
                    reviews.loc[asin_mask, "is_cbf_top10"] = True
 
            n_processed += 1
 
    print(f"\n   Users processed: {n_processed:,}")
    print(f"   Users skipped (no liked items): {n_skipped:,}")
 
    # Drop the temporary catalogue rating column we added
    if "average_rating_cat" in reviews.columns:
        reviews = reviews.drop(columns=["average_rating_cat"])
    if "average_rating" in reviews.columns and "average_rating" not in pd.read_csv(
        REVIEWS_FILE, nrows=1
    ).columns:
        reviews = reviews.drop(columns=["average_rating"])
 
    return reviews
 
 
# ════════════════════════════════════════════════════════════════════════════
#  STEP 4 — MERGE CONTENT RICHNESS AND SAVE
# ════════════════════════════════════════════════════════════════════════════
 
def save_enriched(reviews: pd.DataFrame, richness_df: pd.DataFrame) -> None:
    print("\n💾  Merging content richness and saving …")
 
    enriched = reviews.merge(
        richness_df.rename(columns={"item_content_richness":"item_content_richness"}),
        on="parent_asin", how="left",
    )
 
    OUT_DIR.mkdir(exist_ok=True)
    enriched.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
 
    print(f"\n{'='*60}")
    print(f"  ✅  Saved: {OUTPUT_FILE}")
    print(f"  Rows : {len(enriched):,}")
    print(f"  Cols : {len(enriched.columns)}")
    print(f"\n  New columns added:")
    new_cols = ["item_content_richness","user_item_similarity",
                "cbf_predicted_rating","cbf_rank_for_user","is_cbf_top10"]
    for col in new_cols:
        if col in enriched.columns:
            non_null = enriched[col].notna().sum()
            sample   = enriched[col].dropna().iloc[0] if non_null > 0 else "—"
            print(f"    {col:30s}  {non_null:>8,} non-null  sample={sample}")
    print(f"{'='*60}\n")
 
 
# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
 
def main():
    print("=" * 60)
    print("  CBF Review Enrichment")
    print("=" * 60)
 
    reviews, products, asin_item = load_data()
 
    vectorizer, tfidf_normed, asin_index, richness_df = fit_tfidf(products)
 
    enriched_reviews = compute_cbf_scores(
        reviews, products, tfidf_normed, asin_index, richness_df
    )
 
    save_enriched(enriched_reviews, richness_df)
 
 
if __name__ == "__main__":
    main()