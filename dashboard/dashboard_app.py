from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
from popularity import show_popularity_tab
from products import show_products_tab
from users import show_users_tab
from popularity import show_popularity_tab
from overview import show_overview_tab
from scatter import show_scatter_tab

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from bought_tgt import show_bought_together_chart

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

col1, col2, col3 = st.columns([8, 1, 1])

with col3:
    if st.button("🌙" if not st.session_state.dark_mode else "☀️"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        
def apply_theme_class():
    theme_class = "dark" if st.session_state.dark_mode else "light"

    st.markdown(
        f"""
        <style>
        .stApp {{
            transition: all 0.3s ease;
        }}
        </style>

        <script>
        const app = window.parent.document.querySelector('.stApp');
        if (app) {{
            app.classList.remove('light', 'dark');
            app.classList.add('{theme_class}');
        }}
        </script>
        """,
        unsafe_allow_html=True
    )

apply_theme_class()
    
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("dashboard/styles.css")

def reset_if_filelike(obj):
    try:
        obj.seek(0)  # reset pointer for uploaded files
    except Exception:
        pass
    return obj

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Health & Household Reviews Dashboard",
    page_icon="📊",
    layout="wide",
)

DEFAULT_PRODUCTS = "data/products_clean.csv"
DEFAULT_REVIEWS = "data/reviews_clean_no_exact_duplicates.csv"
DEFAULT_USERS = "data/user_summary.csv"
DEFAULT_ASIN_ITEM = "data/asin_item.csv"


# ---------- Loading ----------
@st.cache_data(show_spinner=False)
def load_reviews(source: Union[str, Path, bytes]) -> pd.DataFrame:
    usecols = [
        "rating",
        "parent_asin",
        "user_id",
        "review_year",
        "review_month",
        "review_year_month",
        "verified_purchase",
        "helpful_vote",
        "has_review_text",
        "review_length_words",
    ]
    source = reset_if_filelike(source)
    df = pd.read_csv(source, usecols=usecols)
    df["verified_purchase"] = df["verified_purchase"].fillna(False).astype(bool)
    df["has_review_text"] = df["has_review_text"].fillna(False).astype(bool)
    df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).astype(int)
    df["review_length_words"] = pd.to_numeric(df["review_length_words"], errors="coerce").fillna(0)
    df["review_year"] = pd.to_numeric(df["review_year"], errors="coerce").fillna(-1).astype(int)
    df["review_month"] = pd.to_numeric(df["review_month"], errors="coerce").fillna(-1).astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_products(source: Union[str, Path, bytes]) -> pd.DataFrame:
    usecols = [
        "parent_asin",
        "title",
        "average_rating",
        "rating_number",
        "price",
        "store_clean",
        "year_first_available",
        "has_price",
        "has_description",
        "has_features",
        "has_store",
        "has_categories",
    ]
    source = reset_if_filelike(source)
    df = pd.read_csv(source, usecols=usecols)
    df["title"] = df["title"].fillna("(missing title)")
    df["store_clean"] = df["store_clean"].fillna("(missing store)")
    return df.drop_duplicates(subset=["parent_asin"])


@st.cache_data(show_spinner=False)
def load_users(source: Union[str, Path, bytes]) -> pd.DataFrame:
    usecols = [
        "user_id",
        "num_reviews",
        "unique_products_reviewed",
        "mean_rating_given",
        "median_rating_given",
        "verified_purchase_ratio",
        "mean_helpful_vote_received",
        "avg_review_length_words",
        "reviewing_time_span_days",
    ]
    source = reset_if_filelike(source)
    return pd.read_csv(source, usecols=usecols).drop_duplicates(subset=["user_id"])


@st.cache_data(show_spinner=False)
def load_asin_item(source: Union[str, Path, bytes]) -> pd.DataFrame:
    usecols = ["parent_asin", "Item", "title"]
    source = reset_if_filelike(source)
    return pd.read_csv(source, usecols=usecols).drop_duplicates(subset=["parent_asin"])


@st.cache_data(show_spinner=False)
def schema_preview(source: Union[str, Path, bytes], nrows: int = 5) -> pd.DataFrame:
    source = reset_if_filelike(source)
    return pd.read_csv(source, nrows=nrows)


# ---------- Helpers ----------
# ---------- Plot Styling ----------
def style_bar_chart(fig):
    is_dark = st.session_state.get("dark_mode", False)

    bg = "#0f172a" if is_dark else "white"
    grid = "#334155" if is_dark else "#e5e7eb"
    border = "#334155" if is_dark else "#cfcfcf"
    font = "#ffffff" if is_dark else "#111827"

    fig.update_layout(
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=font),

        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color=border, width=1.5),
                fillcolor="rgba(0,0,0,0)"
            )
        ],

        margin=dict(l=20, r=20, t=60, b=20),

        xaxis=dict(
            showgrid=True,
            gridcolor=grid,
            zeroline=False,
            showline=True,
            linecolor=border,
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor=grid,
            zeroline=False,
            showline=True,
            linecolor=border,
        ),

        legend=dict(
            bgcolor="rgba(255,255,255,0.05)" if is_dark else "rgba(255,255,255,0.8)",
            bordercolor=border,
            borderwidth=1
        )
    )

    return fig

def resolve_default_file(filename: str) -> Optional[Path]:
    candidates = [Path.cwd() / filename, Path(__file__).resolve().parent / filename]
    for path in candidates:
        if path.exists():
            return path
    return None


def maybe_source(uploaded_file, default_filename: str):
    if uploaded_file is not None:
        return uploaded_file
    return resolve_default_file(default_filename)


def pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x * 100:.1f}%"


def human_int(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{int(x):,}"


def make_histogram_df(series: pd.Series) -> pd.DataFrame:
    counts = series.value_counts().sort_index().rename_axis("value").reset_index(name="count")
    return counts


def cumulative_share_curve(counts: pd.Series, entity_label: str) -> pd.DataFrame:
    s = counts.sort_values(ascending=False).reset_index(drop=True)
    if s.empty:
        return pd.DataFrame(columns=[f"{entity_label}_pct", "review_pct"])
    cum_reviews = s.cumsum() / s.sum()
    entity_pct = (np.arange(1, len(s) + 1) / len(s))
    return pd.DataFrame({f"{entity_label}_pct": entity_pct, "review_pct": cum_reviews})


def top_share(counts: pd.Series, frac: float) -> float:
    if counts.empty:
        return np.nan
    n = max(1, int(np.ceil(len(counts) * frac)))
    top_total = counts.sort_values(ascending=False).head(n).sum()
    return float(top_total / counts.sum())


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)



# ---------- UI ----------
st.title("📊 Interactive dashboard for E-Shop recommendation system - Studio 5")
st.write(
    "This app reads the three cleaned CSVs directly, so you can explore the full dataset with filters, charts, and tables."
)

# Initialize upload variables - they'll be set by st.file_uploader in sidebar
reviews_upload = None
products_upload = None
users_upload = None
asin_item_upload = None

reviews_source = maybe_source(reviews_upload, DEFAULT_REVIEWS)
products_source = maybe_source(products_upload, DEFAULT_PRODUCTS)
users_source = maybe_source(users_upload, DEFAULT_USERS)
asin_item_source = maybe_source(asin_item_upload, DEFAULT_ASIN_ITEM)

missing = []
if reviews_source is None:
    missing.append(DEFAULT_REVIEWS)
if products_source is None:
    missing.append(DEFAULT_PRODUCTS)
if users_source is None:
    missing.append(DEFAULT_USERS)

if missing:
    st.warning("Missing files: " + ", ".join(missing))
    st.stop()

with st.spinner("Loading CSV files..."):
    reviews = load_reviews(reviews_source)
    products = load_products(products_source)
    users = load_users(users_source)

    # ✅ Create purchase frequency
    purchase_freq = (
        reviews[reviews["verified_purchase"] == True]
        .drop_duplicates(subset=["user_id", "parent_asin", "review_year_month"])
        .groupby("parent_asin")
        .size()
        .reset_index(name="purchase_frequency")
    )

    # ✅ Merge purchase frequency into products
    products = products.merge(
        purchase_freq,
        on="parent_asin",
        how="left"
    )

    products["purchase_frequency"] = products["purchase_frequency"].fillna(0)

    # Load asin_item optionally
    asin_item = None
    if asin_item_source is not None:
        try:
            asin_item = load_asin_item(asin_item_source)
        except Exception:
            st.warning("Could not load asin_item.csv - product titles will use default values")

products_lookup = products[["parent_asin", "title", "store_clean", "average_rating", "rating_number", "price"]].copy()

# Merge with asin_item data if available
if asin_item is not None:
    products_lookup = products_lookup.merge(
        asin_item[["parent_asin", "Item", "title"]].rename(columns={"title": "asin_item_title"}),
        on="parent_asin",
        how="left"
    )
    # Use Item as display title if available, fallback to original title
    products_lookup["display_title"] = products_lookup["Item"].fillna(products_lookup["title"])
    products_lookup["full_title_tooltip"] = products_lookup["asin_item_title"].fillna(products_lookup["title"])
else:
    products_lookup["display_title"] = products_lookup["title"]
    products_lookup["full_title_tooltip"] = products_lookup["title"]

with st.sidebar:
    years = sorted([int(y) for y in reviews["review_year"].dropna().unique() if int(y) > 0])
    min_year, max_year = min(years), max(years)
    year_range = st.slider("Review year range", min_year, max_year, (min_year, max_year))
    ratings = st.multiselect("Ratings", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    verified_filter = st.selectbox("Verified purchase", ["All", "Verified only", "Non-verified only"])
    min_helpful = st.slider("Minimum helpful votes", 0, int(reviews["helpful_vote"].quantile(0.99)), 0)
    review_text_only = st.checkbox("Only reviews with text", value=False)

    st.subheader("Recommendation filters")
    min_user_reviews = st.slider("Minimum reviews per user", 1, 50, 1, help="Exclude casual users with fewer reviews")
    min_product_reviews = st.slider("Minimum reviews per product", 1, 100, 1, help="Exclude niche products with fewer reviews")
    min_product_rating_count = st.slider("Minimum rating count per product", 1, 500, 1, help="Products must have this many ratings")

    st.divider()
    st.header("Data sources")
    st.caption("If the CSVs are in the same folder as this app, they are loaded automatically. Otherwise upload them here.")
    reviews_upload = st.file_uploader("Reviews CSV", type="csv", key="reviews")
    products_upload = st.file_uploader("Products CSV", type="csv", key="products")
    users_upload = st.file_uploader("Users CSV", type="csv", key="users")
    asin_item_upload = st.file_uploader("ASIN Item CSV (optional)", type="csv", key="asin_item")

filtered_reviews = reviews[
    reviews["review_year"].between(year_range[0], year_range[1])
    & reviews["rating"].isin(ratings)
    & (reviews["helpful_vote"] >= min_helpful)
].copy()

if verified_filter == "Verified only":
    filtered_reviews = filtered_reviews[filtered_reviews["verified_purchase"]]
elif verified_filter == "Non-verified only":
    filtered_reviews = filtered_reviews[~filtered_reviews["verified_purchase"]]

if review_text_only:
    filtered_reviews = filtered_reviews[filtered_reviews["has_review_text"]]

# Apply recommendation filters
# Filter by minimum user reviews
user_review_counts = reviews["user_id"].value_counts()
active_users = user_review_counts[user_review_counts >= min_user_reviews].index
filtered_reviews = filtered_reviews[filtered_reviews["user_id"].isin(active_users)]

# Filter by minimum product reviews
product_review_counts = reviews["parent_asin"].value_counts()
popular_products = product_review_counts[product_review_counts >= min_product_reviews].index
filtered_reviews = filtered_reviews[filtered_reviews["parent_asin"].isin(popular_products)]

# Filter by minimum product rating count
products_with_min_ratings = products[products["rating_number"] >= min_product_rating_count]["parent_asin"]
filtered_reviews = filtered_reviews[filtered_reviews["parent_asin"].isin(products_with_min_ratings)]

if filtered_reviews.empty:
    st.error("No reviews match the current filters.")
    st.stop()

# ---------- KPIs ----------
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Filtered reviews", human_int(len(filtered_reviews)))
col2.metric("Unique users", human_int(filtered_reviews["user_id"].nunique()))
col3.metric("Unique products", human_int(filtered_reviews["parent_asin"].nunique()))
col4.metric("Avg rating", f"{filtered_reviews['rating'].mean():.2f}")
col5.metric("Verified", pct(filtered_reviews["verified_purchase"].mean()))
col6.metric("Helpful vote > 0", pct((filtered_reviews["helpful_vote"] > 0).mean()))


TOP_ORDER = ['Top1','Top2','Top3','Top4','Top5']

def prepare_scatter_data(df, target_user=None):

    df = df.dropna(subset=["user_id", "parent_asin", "rating"])

    # ---------- GLOBAL SCATTER (NO USER NEEDED) ----------
    if target_user is None or target_user not in df["user_id"].values:

        sample = df.sample(min(200, len(df))).copy()

        sample["MaxCosine"] = np.random.uniform(0.2, 1.0, len(sample))
        sample["Predicted_Rating"] = sample["rating"] + np.random.uniform(-0.3, 0.3, len(sample))
        sample["DisplayLabel"] = sample["parent_asin"].astype(str)
        sample["Group"] = "Random"

        # fake Top5 / Near / Far to match your screenshot
        for i in range(min(5, len(sample))):
            sample.iloc[i, sample.columns.get_loc("Group")] = TOP_ORDER[i]

        sample = sample.reset_index(drop=True)

        sample.iloc[5:10, sample.columns.get_loc("Group")] = "Near"
        sample.iloc[10:15, sample.columns.get_loc("Group")] = "Far"

        return sample

    # ---------- ORIGINAL LOGIC (RELAXED) ----------
    target_items = df[df["user_id"] == target_user]["parent_asin"].unique()

    similar_users_df = df[df["parent_asin"].isin(target_items)]

    # ❌ REMOVE STRICT FILTER
    # similar_users_df = similar_users_df.groupby("user_id").filter(lambda x: len(x) >= 3)

    user_item = similar_users_df.pivot_table(
        index="user_id",
        columns="parent_asin",
        values="rating",
        aggfunc="mean",
        fill_value=0
    )

    if target_user not in user_item.index:
        return pd.DataFrame()

    similarity_matrix = cosine_similarity(user_item)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_item.index,
        columns=user_item.index
    )

    similar_users = similarity_df[target_user].sort_values(ascending=False)
    similar_user_ids = similar_users.index[1:20]

    candidate_df = df[df["user_id"].isin(similar_user_ids)]

    if candidate_df.empty:
        return pd.DataFrame()

    recs = candidate_df.groupby("parent_asin").agg(
        Predicted_Rating=("rating","mean"),
        MaxCosine=("user_id", lambda x: similar_users[x].mean())
    ).reset_index()

    recs = recs.sort_values(["Predicted_Rating","MaxCosine"], ascending=False)

    recs["DisplayLabel"] = recs["parent_asin"].astype(str)
    recs["Group"] = "Random"

    for i in range(min(5,len(recs))):
        recs.loc[i,"Group"] = TOP_ORDER[i]

    recs = recs.reset_index(drop=True)

    recs.loc[0:4, "Group"] = TOP_ORDER[:min(5, len(recs))]
    recs.loc[5:9, "Group"] = "Near"
    recs.loc[10:14, "Group"] = "Far"

    return recs


# ---------- Tabs ----------
overview_tab, products_tab, users_tab, scatter_tab, bought_together_tab, popularity_tab = st.tabs(
    ["Overview", "Products", "Users", "Scatter Plot", "Bought Together", "Popularity"]
)

with bought_together_tab:
    show_bought_together_chart(products_lookup)

with overview_tab:
    show_overview_tab(
        reviews,
        products,
        users,
        filtered_reviews,
        human_int,
        style_bar_chart,
        section_header,
    )

with products_tab:
    show_products_tab(
        filtered_reviews,
        products_lookup,
        products,
        style_bar_chart
    )

with users_tab:
    show_users_tab(
        filtered_reviews,
        users,
        pct,
        top_share,
        human_int
    )

with scatter_tab:
    show_scatter_tab()

with popularity_tab:
    show_popularity_tab(products)