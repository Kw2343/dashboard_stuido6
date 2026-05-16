from __future__ import annotations
 
import pandas as pd
import streamlit as st
 
from utils import pct, top_share, human_int
 
 
def show_users_tab(
    filtered_reviews: pd.DataFrame,
    users: pd.DataFrame,
) -> None:
    """
    Render the Users tab.
 
    Signature simplified: helpers (pct, top_share, human_int) are
    imported from utils so callers don't need to pass them.
    """
    st.markdown("### User analysis")
 
    # ── Summary KPIs ──────────────────────────────────────────────────────────
    u1, u2, u3 = st.columns(3)
 
    reviews_per_user = filtered_reviews.groupby("user_id").size()
 
    u1.metric("Median reviews / user",  f"{reviews_per_user.median():.0f}")
    u2.metric("Top 10 % users share",   pct(top_share(reviews_per_user, 0.1)))
    u3.metric("Single-review users",    pct((reviews_per_user == 1).mean()))
 
    # ── User stats table (merged) ─────────────────────────────────────────────
    st.subheader("User statistics")
 
    active_ids = filtered_reviews["user_id"].unique()
    user_subset = users[users["user_id"].isin(active_ids)].copy()
 
    st.dataframe(user_subset.head(500), use_container_width=True)
 