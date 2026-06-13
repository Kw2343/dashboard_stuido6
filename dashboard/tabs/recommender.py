"""
recommender.py
------------------
"Compare Recommendations" tab for the E-Shop dashboard.
 
Shows three recommendation approaches side-by-side for any selected user:
  1.  Popular Products      — same for everyone (popularity baseline)
  2.  Collaborative Filtering (NCF) — personalised per user
  3.  Content-Based Filtering (CBF) — personalised per user
 
Data paths (relative to this file's parent directory):
  Popular  →  dashboard_data/popular_products.csv
  NCF      →  data/ncf_dashboard.csv
  CBF      →  dashboard_data/cbf_dashboard.csv
 
How to wire into app.py
------------------------
  from tabs.recommender_tab import show_recommender_tab
 
  with tab_recommender:
      show_recommender_tab()
"""
 
from __future__ import annotations
 
from pathlib import Path
from typing import Optional
 
import pandas as pd
import streamlit as st
 
# ── Paths ─────────────────────────────────────────────────────────────────────
_DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard_data"
_POPULAR_CSV   = _DASHBOARD_DIR / "popular_products.csv"
_NCF_CSV       = _DASHBOARD_DIR / "ncf_dashboard.csv"
_CBF_CSV       = _DASHBOARD_DIR / "cbf_dashboard.csv"
 
_TOP_N = 5
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
 
@st.cache_data(show_spinner=False)
def _load_popular() -> Optional[pd.DataFrame]:
    if not _POPULAR_CSV.exists():
        return None
    df = pd.read_csv(_POPULAR_CSV)
    for c in ["popularity_score", "bayesian_rating", "avg_rating_raw",
              "tfidf_popularity_score", "dl_content_score",
              "average_rating", "review_count", "rating_number"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
 
 
@st.cache_data(show_spinner=False)
def _load_ncf() -> Optional[pd.DataFrame]:
    if not _NCF_CSV.exists():
        return None
    df = pd.read_csv(_NCF_CSV)
    for c in ["display_score", "predicted_score", "score", "average_rating", "rank"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
 
 
@st.cache_data(show_spinner=False)
def _load_cbf() -> Optional[pd.DataFrame]:
    if not _CBF_CSV.exists():
        return None
    df = pd.read_csv(_CBF_CSV)
    for c in ["display_score", "predicted_score", "score",
              "cbf_predicted_rating", "cosine_similarity",
              "average_rating", "rank"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
 
def _name_col(df: pd.DataFrame) -> str:
    for c in ["display_name", "short_title", "title"]:
        if c in df.columns:
            return c
    return df.columns[0]
 
 
def _score_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["display_score", "predicted_score", "cbf_predicted_rating",
              "score", "cosine_similarity", "tfidf_popularity_score",
              "dl_content_score", "popularity_score",
              "bayesian_rating", "avg_rating_raw", "average_rating"]:
        if c in df.columns:
            return c
    return None
 
 
def _trunc(val: str, n: int = 52) -> str:
    s = str(val)
    return s if len(s) <= n else s[:n - 1] + "…"
 
 
def _fmt_score(val) -> str:
    try:
        if val is None or pd.isna(val):
            return "—"
        return f"{float(val):.3f}"
    except (TypeError, ValueError):
        return "—"
 
 
def _rank_badge(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  CARD RENDERER
# ══════════════════════════════════════════════════════════════════════════════
 
def _render_cards(rows: list[dict], score_label: str) -> None:
    for row in rows:
        badge = _rank_badge(row["rank"])
        name  = _trunc(row["name"])
        score = _fmt_score(row.get("score"))
        st.markdown(
            f"""
<div style="
    display:flex; align-items:center; gap:14px;
    padding:10px 14px; margin-bottom:8px;
    background:var(--card-bg, #ffffff);
    border:1px solid var(--card-border, #dce6f0);
    border-radius:10px;
    box-shadow:var(--card-shadow, 0 1px 4px rgba(0,0,0,0.07));
    ">
  <span style="font-size:1.5rem; min-width:2rem; text-align:center;">{badge}</span>
  <div style="flex:1; overflow:hidden;">
    <div style="
        font-weight:600; font-size:0.88rem;
        color:var(--text-primary, #1e3a5f);
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
        ">{name}</div>
    <div style="font-size:0.78rem; color:var(--text-muted, #5a7a9a); margin-top:2px;">
        {score_label}: <strong style="color:var(--accent, #1a6fc4);">{score}</strong>
    </div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  DATA EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════
 
def _popular_rows(df: pd.DataFrame) -> list[dict]:
    sc  = _score_col(df)
    nc  = _name_col(df)
    top = df.sort_values(sc, ascending=False).head(_TOP_N) if sc else df.head(_TOP_N)
    return [
        {"rank": i + 1, "name": row[nc], "score": row.get(sc)}
        for i, (_, row) in enumerate(top.iterrows())
    ]
 
 
def _ncf_rows(df: pd.DataFrame, user_id: str) -> tuple[list[dict], bool]:
    user_df = df[df["user_id"].astype(str) == user_id]
    if user_df.empty:
        return [], False
    sc  = _score_col(user_df)
    nc  = _name_col(user_df)
    top = (
        user_df.sort_values("rank").head(_TOP_N)
        if "rank" in user_df.columns
        else user_df.sort_values(sc, ascending=False).head(_TOP_N) if sc
        else user_df.head(_TOP_N)
    )
    return [
        {"rank": i + 1, "name": row[nc], "score": row.get(sc)}
        for i, (_, row) in enumerate(top.iterrows())
    ], True
 
 
def _cbf_rows(df: pd.DataFrame, user_id: str) -> tuple[list[dict], bool]:
    user_df = df[df["user_id"].astype(str) == user_id]
    if user_df.empty:
        return [], False
    sc  = _score_col(user_df)
    nc  = _name_col(user_df)
    top = (
        user_df.sort_values("rank").head(_TOP_N)
        if "rank" in user_df.columns
        else user_df.sort_values(sc, ascending=False).head(_TOP_N) if sc
        else user_df.head(_TOP_N)
    )
    return [
        {"rank": i + 1, "name": row[nc], "score": row.get(sc)}
        for i, (_, row) in enumerate(top.iterrows())
    ], True
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION RENDERERS
# ══════════════════════════════════════════════════════════════════════════════
 
def _section_popular(popular_df: Optional[pd.DataFrame]) -> list[dict]:
    st.markdown("###  What's Trending")
    st.caption(
        "These are the most popular products across all customers — "
        "ranked by ratings, purchase volume, and customer reviews. "
        "A good starting point for promotions and homepage placements."
    )
    if popular_df is None:
        st.warning(f"popular_products.csv not found at `{_POPULAR_CSV}`.")
        return []
    rows = _popular_rows(popular_df)
    _render_cards(rows, score_label="Popularity score")
    return rows
 
 
def _section_ncf(ncf_df: Optional[pd.DataFrame], user_id: str) -> list[dict]:
    st.markdown("###  Customers Like You Also Bought")
    st.caption(
        "These products are recommended based on what other customers "
        "with similar buying habits purchased. The more purchases and "
        "ratings in common, the stronger the recommendation."
    )
    if ncf_df is None:
        st.warning(f"ncf_dashboard.csv not found at `{_NCF_CSV}`.")
        return []
    rows, found = _ncf_rows(ncf_df, user_id)
    if not found:
        sample = ncf_df["user_id"].dropna().astype(str).unique()[:5].tolist()
        st.warning(
            f"No recommendations found for user **{user_id}**.  \n"
            f"Try one of these sample User IDs:  \n"
            + "  \n".join(f"• `{u}`" for u in sample)
        )
        return []
    _render_cards(rows, score_label="Match score")
    return rows
 
 
def _section_cbf(cbf_df: Optional[pd.DataFrame], user_id: str) -> list[dict]:
    st.markdown("###  Picked for You")
    st.caption(
        "These products match this customer's personal taste — "
        "selected by analysing the categories, descriptions, and features "
        "of products they've previously rated highly."
    )
    if cbf_df is None:
        st.warning(f"cbf_dashboard.csv not found at `{_CBF_CSV}`.")
        return []
    rows, found = _cbf_rows(cbf_df, user_id)
    if not found:
        sample = cbf_df["user_id"].dropna().astype(str).unique()[:5].tolist()
        st.warning(
            f"No recommendations found for user **{user_id}**.  \n"
            f"Try one of these sample User IDs:  \n"
            + "  \n".join(f"• `{u}`" for u in sample)
        )
        return []
    _render_cards(rows, score_label="Match score")
    return rows
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
 
def _comparison_table(
    pop_rows: list[dict],
    ncf_rows: list[dict],
    cbf_rows: list[dict],
) -> None:
    st.markdown("### How Do the Three Approaches Compare?")
    st.caption(
        "Each column shows the top 5 products recommended by a different approach. "
        "Comparing them reveals how much personalisation changes what a customer sees."
    )
 
    def _col(rows: list[dict]) -> list[str]:
        return [f"{_rank_badge(r['rank'])} {_trunc(r['name'], 38)}" for r in rows]
 
    max_len = max(len(pop_rows), len(ncf_rows), len(cbf_rows), 1)
 
    def _pad(lst: list[str], n: int) -> list[str]:
        return lst + ["—"] * (n - len(lst))
 
    comp = pd.DataFrame({
        "Rank":                        [_rank_badge(i + 1) for i in range(max_len)],
        " What's Trending":          _pad(_col(pop_rows), max_len),
        " Customers Like You Bought": _pad(_col(ncf_rows), max_len),
        " Picked for You":           _pad(_col(cbf_rows), max_len),
    })
 
    st.dataframe(
        comp,
        use_container_width=True,
        hide_index=True,
        height=(_TOP_N + 1) * 38,
    )
 
    # ── Business-friendly overlap insight ────────────────────────────────────
    pop_names = {r["name"] for r in pop_rows}
    ncf_names = {r["name"] for r in ncf_rows}
    cbf_names = {r["name"] for r in cbf_rows}
    all_agree = pop_names & ncf_names & cbf_names
    ncf_cbf   = (ncf_names & cbf_names) - pop_names
    only_pop  = pop_names - ncf_names - cbf_names
    only_pers = (ncf_names | cbf_names) - pop_names
 
    st.markdown("####  What This Tells Us")
 
    if all_agree:
        names = ", ".join(_trunc(n, 35) for n in list(all_agree)[:3])
        st.success(
            f"**{len(all_agree)} product(s) appear in all three lists** — "
            f"these are universally strong picks that are both trending *and* "
            f"personally relevant: {names}. "
            f"These are safe bets for any promotion."
        )
 
    if ncf_cbf:
        names = ", ".join(_trunc(n, 35) for n in list(ncf_cbf)[:3])
        st.info(
            f"**{len(ncf_cbf)} product(s) are recommended by both personalisation models "
            f"but aren't in the trending list** — {names}. "
            f"These are hidden gems that this customer is likely to buy, "
            f"even though they aren't bestsellers site-wide."
        )
 
    if only_pop:
        st.warning(
            f"**{len(only_pop)} trending product(s) don't appear in the personalised lists.** "
            f"These are popular site-wide but may not suit this particular customer's taste. "
            f"Pushing them too hard risks irrelevant promotions."
        )
 
    if only_pers and not ncf_cbf and not all_agree:
        st.info(
            f"**The personalised models surface completely different products from the trending list.** "
            f"This customer has distinct preferences — personalised recommendations are "
            f"likely to drive higher engagement than a one-size-fits-all approach."
        )
 
    if not all_agree and not ncf_cbf:
        st.info(
            "**All three models recommend different products.** "
            "This customer's taste is quite specific — "
            "personalised recommendations are especially valuable here, "
            "since trending products alone would miss what they actually want."
        )
 
    # ── Summary scorecard ─────────────────────────────────────────────────────
    st.markdown("####  Recommendation Strategy Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Products in common (all 3)",
        len(all_agree),
        help="Products recommended by all three approaches — highest confidence picks"
    )
    c2.metric(
        "Personalised-only picks",
        len(only_pers),
        help="Products the personalised models find for this customer that aren't trending site-wide"
    )
    c3.metric(
        "Trending but not personalised",
        len(only_pop),
        help="Trending products that don't match this customer's specific taste"
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  USER SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
 
def _user_selector(ncf_df: Optional[pd.DataFrame],
                   cbf_df: Optional[pd.DataFrame]) -> Optional[str]:
    st.markdown("#### Select a Customer")
 
    known_ids: list[str] = []
    for df in [ncf_df, cbf_df]:
        if df is not None and "user_id" in df.columns:
            known_ids.extend(df["user_id"].dropna().astype(str).unique().tolist())
    known_ids = sorted(set(known_ids))
 
    col_input, col_drop = st.columns([3, 2])
 
    with col_input:
        typed = st.text_input(
            "Enter Customer ID",
            placeholder="e.g. AFCQ2QKSQD7G3",
            key="rec_uid_input",
        )
 
    with col_drop:
        if known_ids:
            chosen = st.selectbox(
                "Or choose from list",
                options=["— select a customer —"] + known_ids[:500],
                key="rec_uid_select",
            )
        else:
            chosen = "— select a customer —"
            st.caption("(no customer list available)")
 
    uid = typed.strip() if typed.strip() else (
        chosen if chosen != "— select a customer —" else None
    )
    return uid
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  STYLES
# ══════════════════════════════════════════════════════════════════════════════
 
_CSS = """
<style>
[data-testid="stAppViewContainer"] {
    --card-bg:      #ffffff;
    --card-border:  #dce6f0;
    --card-shadow:  0 1px 4px rgba(0,0,0,0.07);
    --text-primary: #1e3a5f;
    --text-muted:   #5a7a9a;
    --accent:       #1a6fc4;
}
[data-testid="column"] {
    border-right: 1px solid rgba(100,140,180,0.12);
    padding-right: 1rem !important;
}
[data-testid="column"]:last-child {
    border-right: none;
}
</style>
"""
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  TAB ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
 
def show_recommender_tab() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
 
    st.header("Product Recommendation Comparison")
    st.write(
        "See what each recommendation approach suggests for any customer. "
        "Use this to understand how personalised recommendations differ "
        "from simply showing what's trending — and where the biggest "
        "opportunities for targeted promotions lie."
    )
 
    # ── How it works expander (for stakeholders) ──────────────────────────────
    with st.expander("ℹ️ How does each approach work?", expanded=False):
        st.markdown("""
| Approach | How it works | Best used for |
|---|---|---|
|  **What's Trending** | Shows the most popular products across all customers based on ratings and purchase volume | Homepage banners, mass promotions, new customer landing pages |
|  **Customers Like You Bought** | Finds customers with similar purchase history and recommends what they bought | Cross-sell emails, "you might also like" widgets |
|  **Picked for You** | Analyses the product features this customer prefers (category, description, brand type) | Personalised emails, loyalty programme offers, returning customer pages |
        """)
 
    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading recommendation data…"):
        popular_df = _load_popular()
        ncf_df     = _load_ncf()
        cbf_df     = _load_cbf()
 
    # ── User selection ────────────────────────────────────────────────────────
    user_id = _user_selector(ncf_df, cbf_df)
 
    if not user_id:
        st.info(" Enter or select a Customer ID above to see their personalised recommendations.")
        st.divider()
        st.markdown("###  What's Trending Right Now")
        st.caption("Showing trending products — select a customer above to see personalised picks alongside these.")
        if popular_df is not None:
            _render_cards(_popular_rows(popular_df), score_label="Popularity score")
        else:
            st.warning(f"popular_products.csv not found at `{_POPULAR_CSV}`.")
        return
 
    st.success(f"Showing recommendations for customer **{user_id}**")
    st.divider()
 
    # ── Three-column layout ───────────────────────────────────────────────────
    col_pop, col_ncf, col_cbf = st.columns(3, gap="medium")
 
    with col_pop:
        pop_rows = _section_popular(popular_df)
 
    with col_ncf:
        ncf_rows = _section_ncf(ncf_df, user_id)
 
    with col_cbf:
        cbf_rows = _section_cbf(cbf_df, user_id)
 
    st.divider()
 
    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_compare, tab_pop_detail, tab_ncf_detail, tab_cbf_detail = st.tabs([
        " Comparison & Insights",
        " Trending — Full Details",
        " Collaborative — Full Details",
        " Content-Based — Full Details",
    ])
 
    with tab_compare:
        _comparison_table(pop_rows, ncf_rows, cbf_rows)
 
    with tab_pop_detail:
        st.markdown("#### Most Popular Products — Full Details")
        st.caption("These products perform best across all customers based on ratings and purchase counts.")
        if popular_df is not None and pop_rows:
            sc = _score_col(popular_df)
            nc = _name_col(popular_df)
            top = (
                popular_df.sort_values(sc, ascending=False).head(_TOP_N)
                if sc else popular_df.head(_TOP_N)
            )
            show_cols = [c for c in [nc, sc, "review_count", "rating_number",
                                      "average_rating", "store"]
                         if c and c in top.columns]
            st.dataframe(top[show_cols].reset_index(drop=True),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No trending products data available.")
 
    with tab_ncf_detail:
        st.markdown("#### Collaborative Recommendations — Full Details")
        st.caption(
            "Products recommended because similar customers bought them. "
            "The match score reflects how closely this customer's history "
            "aligns with others who bought these products."
        )
        if ncf_df is not None:
            user_ncf = ncf_df[ncf_df["user_id"].astype(str) == user_id]
            if not user_ncf.empty:
                sc = _score_col(user_ncf)
                nc = _name_col(user_ncf)
                top = (
                    user_ncf.sort_values("rank").head(_TOP_N)
                    if "rank" in user_ncf.columns
                    else user_ncf.sort_values(sc, ascending=False).head(_TOP_N)
                    if sc else user_ncf.head(_TOP_N)
                )
                show_cols = [c for c in ["rank", nc, sc, "average_rating", "store", "price"]
                             if c and c in top.columns]
                st.dataframe(top[show_cols].reset_index(drop=True),
                             use_container_width=True, hide_index=True)
            else:
                st.info(f"No collaborative recommendations found for customer {user_id}.")
        else:
            st.info("Collaborative filtering data not loaded.")
 
    with tab_cbf_detail:
        st.markdown("#### Content-Based Recommendations — Full Details")
        st.caption(
            "Products matched to this customer's personal taste — "
            "selected based on the categories, features, and descriptions "
            "of products they've previously rated highly."
        )
        if cbf_df is not None:
            user_cbf = cbf_df[cbf_df["user_id"].astype(str) == user_id]
            if not user_cbf.empty:
                sc = _score_col(user_cbf)
                nc = _name_col(user_cbf)
                top = (
                    user_cbf.sort_values("rank").head(_TOP_N)
                    if "rank" in user_cbf.columns
                    else user_cbf.sort_values(sc, ascending=False).head(_TOP_N)
                    if sc else user_cbf.head(_TOP_N)
                )
                show_cols = [c for c in ["rank", nc, sc, "average_rating", "store", "price"]
                             if c and c in top.columns]
                st.dataframe(top[show_cols].reset_index(drop=True),
                             use_container_width=True, hide_index=True)
            else:
                st.info(f"No content-based recommendations found for customer {user_id}.")
        else:
            st.info("Content-based filtering data not loaded.")