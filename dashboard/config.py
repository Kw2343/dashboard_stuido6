from pathlib import Path
 
# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "data"
 
DEFAULT_REVIEWS      = DATA_DIR / "reviews_clean_no_exact_duplicates.csv"
DEFAULT_PRODUCTS     = DATA_DIR / "products_clean.csv"
DEFAULT_USERS        = DATA_DIR / "user_summary.csv"
DEFAULT_ASIN_ITEM    = DATA_DIR / "asin_item.csv"
BOUGHT_TOGETHER_FILE = DATA_DIR / "products_bought_together_pair_counts.xlsx"
SCATTER_FILE         = DATA_DIR / "EShop_Product_Recommendations_Scatterplot_Inputs.xlsx"
 
# ── Recommendation defaults ───────────────────────────────────────────────────
POPULARITY_M = 50          # Bayesian prior (minimum votes)
 
# ── Scatter groups ────────────────────────────────────────────────────────────
TOP_ORDER = ["Top1", "Top2", "Top3", "Top4", "Top5"]