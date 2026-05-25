import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
 
import streamlit as st
 
st.set_page_config(
    page_title="Health & Household Reviews Dashboard",
    page_icon="📊",
    layout="wide",
)
 
import pandas as pd
 
from config import DEFAULT_REVIEWS, DEFAULT_PRODUCTS, DEFAULT_USERS, DEFAULT_ASIN_ITEM
from data_loader import (
    load_reviews, load_products, load_users, load_asin_item,
    build_products_lookup, maybe_source,
)
from feature_engineering import feature_engineering
from utils import human_int, pct
 
from tabs.overview            import show_overview_tab
from tabs.products            import show_products_tab
from tabs.users               import show_users_tab
from tabs.scatter             import show_scatter_tab
from tabs.bought_together     import show_bought_together_tab
from tabs.popularity          import show_popularity_tab
from tabs.feature_engineering import show_feature_tab
 
 
def _load_css(path: str) -> None:
    """Load CSS using absolute path resolved relative to this file."""
    css_path = Path(__file__).resolve().parent / path
    try:
        css_content = css_path.read_text()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {css_path}")
 
_load_css("styles.css")
st.markdown("<style>.block-container{padding-top:1rem}</style>", unsafe_allow_html=True)
 
st.title("Interactive Dashboard — E-Shop Recommendation System")
st.write("Explore reviews, products, users, and recommendation outputs with real-time filters.")
 
with st.sidebar:
    st.header("Filters")
    _reviews_src = maybe_source(None, DEFAULT_REVIEWS)
    if _reviews_src is None:
        st.warning("No reviews CSV found. Upload one below.")
        _year_range = (2000, 2024)
    else:
        _all_years = sorted([
            int(y) for y in
            pd.read_csv(_reviews_src, usecols=["review_year"])["review_year"]
            .dropna().unique() if int(y) > 0
        ])
        _yr_min, _yr_max = min(_all_years), max(_all_years)
        _year_range = st.slider("Review year range", _yr_min, _yr_max, (_yr_min, _yr_max))
 
    ratings          = st.multiselect("Ratings", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    verified_filter  = st.selectbox("Verified purchase", ["All", "Verified only", "Non-verified only"])
    min_helpful      = st.slider("Minimum helpful votes", 0, 500, 0)
    review_text_only = st.checkbox("Only reviews with text", value=False)
 
    st.subheader("Recommendation filters")
    min_user_reviews    = st.slider("Min reviews / user",         1,  50, 1)
    min_product_reviews = st.slider("Min reviews / product",      1, 100, 1)
    min_rating_count    = st.slider("Min rating count / product", 1, 500, 1)
 
    st.divider()
    st.header("Data sources")
    reviews_upload   = st.file_uploader("Reviews CSV",         type="csv", key="reviews")
    products_upload  = st.file_uploader("Products CSV",        type="csv", key="products")
    users_upload     = st.file_uploader("Users CSV",           type="csv", key="users")
    asin_item_upload = st.file_uploader("ASIN Item CSV (opt)", type="csv", key="asin_item")
 
reviews_source   = maybe_source(reviews_upload,   DEFAULT_REVIEWS)
products_source  = maybe_source(products_upload,  DEFAULT_PRODUCTS)
users_source     = maybe_source(users_upload,     DEFAULT_USERS)
asin_item_source = maybe_source(asin_item_upload, DEFAULT_ASIN_ITEM)
 
missing = [str(p) for p, s in [
    (DEFAULT_REVIEWS,  reviews_source),
    (DEFAULT_PRODUCTS, products_source),
    (DEFAULT_USERS,    users_source),
] if s is None]
 
if missing:
    st.warning("Missing required files: " + ", ".join(missing))
    st.stop()
 
with st.spinner("Loading data…"):
    reviews  = load_reviews(reviews_source)
    products = load_products(products_source)
    users    = load_users(users_source)
    asin_item = None
    if asin_item_source is not None:
        try:
            asin_item = load_asin_item(asin_item_source)
        except Exception:
            st.warning("Could not load asin_item.csv — product titles will use defaults.")
 
@st.cache_data(show_spinner=False)
def _cached_features(df):
    return feature_engineering(df)
 
features_df = _cached_features(reviews)
 
products = products.merge(
    features_df[[
        "parent_asin", "purchase_count", "unique_users",
        "avg_rating", "days_since_last_purchase",
        "purchase_frequency", "score",
    ]],
    on="parent_asin", how="left",
)
 
products_lookup = build_products_lookup(products, asin_item)
 
filtered = reviews[
    reviews["review_year"].between(_year_range[0], _year_range[1])
    & reviews["rating"].isin(ratings)
    & (reviews["helpful_vote"] >= min_helpful)
].copy()
 
if verified_filter == "Verified only":
    filtered = filtered[filtered["verified_purchase"]]
elif verified_filter == "Non-verified only":
    filtered = filtered[~filtered["verified_purchase"]]
 
if review_text_only:
    filtered = filtered[filtered["has_review_text"]]
 
active_users     = reviews["user_id"].value_counts()
active_users     = active_users[active_users >= min_user_reviews].index
filtered         = filtered[filtered["user_id"].isin(active_users)]
 
popular_products = reviews["parent_asin"].value_counts()
popular_products = popular_products[popular_products >= min_product_reviews].index
filtered         = filtered[filtered["parent_asin"].isin(popular_products)]
 
rated_products   = products[products["rating_number"] >= min_rating_count]["parent_asin"]
filtered         = filtered[filtered["parent_asin"].isin(rated_products)]
 
if filtered.empty:
    st.error("No reviews match the current filters. Loosen the sidebar controls.")
    st.stop()
 
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Filtered reviews",  human_int(len(filtered)))
c2.metric("Unique users",      human_int(filtered["user_id"].nunique()))
c3.metric("Unique products",   human_int(filtered["parent_asin"].nunique()))
c4.metric("Avg rating",        f"{filtered['rating'].mean():.2f}")
c5.metric("Verified",          pct(filtered["verified_purchase"].mean()))
c6.metric("Helpful vote > 0",  pct((filtered["helpful_vote"] > 0).mean()))
 
(
    tab_overview, tab_products, tab_users,
    tab_scatter, tab_bought, tab_popularity, tab_features,
) = st.tabs([
    "Overview", "Products", "Users",
    "Scatter Plot", "Bought Together", "Popularity", "Feature Engineering",
])
 
with tab_overview:
    show_overview_tab(reviews, products, users, filtered)
with tab_products:
    show_products_tab(filtered, products_lookup, products)
with tab_users:
    show_users_tab(filtered, users)
with tab_scatter:
    show_scatter_tab()
with tab_bought:
    show_bought_together_tab(products_lookup)
with tab_popularity:
    show_popularity_tab(products)
with tab_features:
    show_feature_tab(reviews, products, features_df)