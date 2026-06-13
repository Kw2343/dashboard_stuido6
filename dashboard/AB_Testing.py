from __future__ import annotations
 
"""
ab_testing.py — Offline A/B Test: Global Popularity Baseline vs CBF
=====================================================================
 
Two evaluation modes run simultaneously:
 
  MODE 1 — Fair Comparison (same N)
    Model A: Top-N most popular products the user hasn't seen (per-user filtered)
    Model B: Top-N personalised CBF recommendations
    → Answers: Does personalisation beat popularity at equal list size?
 
  MODE 2 — Business Reality (different N)
    Model A: Fixed global Top-50 popularity list (same for all users, like a homepage)
    Model B: Top-10 personalised CBF recommendations (like a personalised email)
    → Answers: Which drives more hits in real deployment scenarios?
 
Evaluation strategy: Leave-One-Out
  Latest interaction per user is withheld as answer key.
  All earlier interactions are training history.
 
Significance tests: chi-squared (hit rate), t-test (avg rating), p < 0.05
"""
 
from math import floor
from pathlib import Path
 
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
 
# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA_DIR      = Path(r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard_data")
_CBF_CSV       = _DATA_DIR / "cbf_dashboard.csv"
_POPULAR_CSV   = _DATA_DIR / "popular_products_top100.csv"
 
# ── Constants ─────────────────────────────────────────────────────────────────
MIN_REVIEWS_FOR_TEST = 5
MIN_TRAIN_SIZE       = 4
TEST_FRACTION        = 0.20
 
# Mode 1 — Fair comparison (same list size for both models)
DEFAULT_TOP_N_FAIR   = 10
 
# Mode 2 — Business reality (different list sizes)
DEFAULT_TOP_N_A_BIZ  = 50   # homepage / mass promotion
DEFAULT_TOP_N_B_BIZ  = 10   # personalised email / widget
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
 
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
 
 
def _chronological_split(user_reviews: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts_col = "timestamp_unix" if "timestamp_unix" in user_reviews.columns else None
    if ts_col:
        ur = user_reviews.sort_values(ts_col, ascending=True).reset_index(drop=True)
    else:
        sort_cols = [c for c in ["review_year", "review_month"] if c in user_reviews.columns]
        ur = (user_reviews.sort_values(sort_cols, ascending=True).reset_index(drop=True)
              if sort_cols else user_reviews.reset_index(drop=True))
    n = len(ur)
    h = _holdout_size(n)
    if h == 0:
        return ur, ur.iloc[0:0]
    return ur.iloc[:n - h].copy(), ur.iloc[n - h:].copy()
 
 
def _popular_scored(products: pd.DataFrame) -> pd.DataFrame:
    """Fallback scorer when popular_products_top100.csv is missing."""
    df = products.copy()
    for col in ["rating_number", "average_rating", "purchase_frequency"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    df["popular_score"] = (
        0.50 * _norm(df["purchase_frequency"])
      + 0.30 * _norm(df["average_rating"])
      + 0.20 * _norm(df["rating_number"])
    ).round(4)
    return df.sort_values("popular_score", ascending=False)
 
 
def _get_score_col(df: pd.DataFrame) -> str | None:
    for c in ["popular_score", "popularity_score",
              "tfidf_popularity_score", "average_rating"]:
        if c in df.columns:
            return c
    return None
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION GENERATORS
# ══════════════════════════════════════════════════════════════════════════════
 
def _recs_popularity_personfiltered(
    popular_file: pd.DataFrame | None,
    products_df: pd.DataFrame,
    train_asins: set,
    top_n: int,
) -> list[str]:
    """
    MODE 1 — Fair: Top-N popular products the user hasn't already seen.
    Filters out training history per user → different list per user.
    """
    if popular_file is not None and "parent_asin" in popular_file.columns:
        sc  = _get_score_col(popular_file)
        src = popular_file.sort_values(sc, ascending=False) if sc else popular_file
        candidates = src["parent_asin"].tolist()
    else:
        scored     = _popular_scored(products_df)
        candidates = scored["parent_asin"].tolist()
 
    # Remove items already seen in training
    unseen = [a for a in candidates if a not in train_asins]
    return unseen[:top_n]
 
 
def _recs_popularity_global(
    popular_file: pd.DataFrame | None,
    products_df: pd.DataFrame,
    top_n: int,
) -> list[str]:
    """
    MODE 2 — Business: Same fixed global Top-N for EVERY user.
    No personalisation, no history filtering.
    """
    if popular_file is not None and "parent_asin" in popular_file.columns:
        sc  = _get_score_col(popular_file)
        src = popular_file.sort_values(sc, ascending=False) if sc else popular_file
        return src["parent_asin"].head(top_n).tolist()
    scored = _popular_scored(products_df)
    return scored["parent_asin"].head(top_n).tolist()
 
 
def _recs_cbf(cbf_user: pd.DataFrame, top_n: int) -> list[str]:
    """Both modes — personalised CBF recs for one user."""
    if cbf_user.empty:
        return []
    if "rank" in cbf_user.columns:
        cbf_user = cbf_user.sort_values("rank", ascending=True)
    elif "predicted_score" in cbf_user.columns:
        cbf_user = cbf_user.sort_values("predicted_score", ascending=False)
    return cbf_user["parent_asin"].head(top_n).tolist()
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  PER-USER METRICS
# ══════════════════════════════════════════════════════════════════════════════
 
def _user_metrics(recs: list[str], held_out: pd.DataFrame,
                  train_reviews: pd.DataFrame) -> dict:
    held_asins   = set(held_out["parent_asin"].tolist())
    train_asins  = set(train_reviews["parent_asin"].tolist())
    full_history = held_asins | train_asins
 
    # Exact — withheld item only
    exact_hits = [r for r in recs if r in held_asins]
    hit_exact  = int(len(exact_hits) > 0)
    prec_exact = len(exact_hits) / len(recs) if recs else 0.0
    avg_rating_exact = (
        held_out[held_out["parent_asin"].isin(exact_hits)]["rating"].mean()
        if exact_hits else float("nan")
    )
 
    # History — full purchase history
    hist_hits    = [r for r in recs if r in full_history]
    hit_history  = int(len(hist_hits) > 0)
    prec_history = len(hist_hits) / len(recs) if recs else 0.0
    all_reviews  = pd.concat([held_out, train_reviews])
    avg_rating_history = (
        all_reviews[all_reviews["parent_asin"].isin(hist_hits)]["rating"].mean()
        if hist_hits else float("nan")
    )
 
    return {
        "hit": hit_exact, "precision": prec_exact, "avg_rating": avg_rating_exact,
        "hit_history": hit_history, "precision_history": prec_history,
        "avg_rating_history": avg_rating_history,
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION + SIGNIFICANCE
# ══════════════════════════════════════════════════════════════════════════════
 
def _agg(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"hit_rate": 0.0, "precision": 0.0, "avg_rating": float("nan"),
                "hit_rate_history": 0.0, "precision_history": 0.0,
                "avg_rating_history": float("nan"), "n_users": 0,
                "hits_arr": [], "ratings_arr": [], "hits_history_arr": []}
    return {
        "hit_rate":           df["hit"].mean() * 100,
        "precision":          df["precision"].mean() * 100,
        "avg_rating":         df["avg_rating"].dropna().mean(),
        "hit_rate_history":   df["hit_history"].mean() * 100,
        "precision_history":  df["precision_history"].mean() * 100,
        "avg_rating_history": df["avg_rating_history"].dropna().mean(),
        "n_users":            len(df),
        "hits_arr":           df["hit"].tolist(),
        "ratings_arr":        df["avg_rating"].dropna().tolist(),
        "hits_history_arr":   df["hit_history"].tolist(),
    }
 
 
def _significance(agg_a: dict, agg_b: dict) -> tuple[dict, dict]:
    sig_hit = sig_rating = {}
    if agg_a["n_users"] > 0 and agg_b["n_users"] > 0:
        ha = sum(agg_a["hits_arr"]); ma = agg_a["n_users"] - ha
        hb = sum(agg_b["hits_arr"]); mb = agg_b["n_users"] - hb
        if min(ha, ma, hb, mb) > 0:
            chi2, p = stats.chi2_contingency([[ha, ma], [hb, mb]])[:2]
            sig_hit = {"chi2": chi2, "p": p, "significant": p < 0.05}
    if len(agg_a["ratings_arr"]) >= 2 and len(agg_b["ratings_arr"]) >= 2:
        t, p = stats.ttest_ind(agg_a["ratings_arr"], agg_b["ratings_arr"])
        sig_rating = {"t": t, "p": p, "significant": p < 0.05}
    return sig_hit, sig_rating
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════
 
@st.cache_data(show_spinner=False)
def run_ab_experiment(
    reviews_df:   pd.DataFrame,
    products_df:  pd.DataFrame,
    cbf_df:       pd.DataFrame,
    popular_file: pd.DataFrame | None = None,
    top_n_fair:   int = DEFAULT_TOP_N_FAIR,
    top_n_a_biz:  int = DEFAULT_TOP_N_A_BIZ,
    top_n_b_biz:  int = DEFAULT_TOP_N_B_BIZ,
    random_seed:  int = 42,
) -> dict:
    rng = np.random.default_rng(random_seed)
 
    cbf_grouped = {uid: grp for uid, grp in cbf_df.groupby("user_id")}
    cbf_users   = set(cbf_grouped.keys())
 
    user_counts  = reviews_df["user_id"].value_counts()
    ineligible   = user_counts[user_counts < MIN_REVIEWS_FOR_TEST].index.tolist()
    all_eligible = user_counts[user_counts >= MIN_REVIEWS_FOR_TEST].index.tolist()
    eligible     = [u for u in all_eligible if u in cbf_users]
    no_cbf       = [u for u in all_eligible if u not in cbf_users]
 
    order = np.array(eligible, dtype=object)
    rng.shuffle(order)
    mid     = len(order) // 2
    group_a = set(order[:mid])
    group_b = set(order[mid:])
 
    # Mode 2: precompute the fixed global list once
    global_top_a_biz = _recs_popularity_global(popular_file, products_df, top_n_a_biz)
 
    grouped  = {uid: grp for uid, grp in reviews_df.groupby("user_id")}
    # Four result buckets: mode1_a, mode1_b, mode2_a, mode2_b
    res_m1a, res_m1b, res_m2a, res_m2b = [], [], [], []
    split_log = []
 
    for user_id, user_reviews in grouped.items():
        if user_id not in group_a and user_id not in group_b:
            continue
 
        train_reviews, test_reviews = _chronological_split(user_reviews)
        train_asins = set(train_reviews["parent_asin"].tolist())
 
        split_log.append({
            "user_id":       user_id,
            "group":         "A — Popularity" if user_id in group_a else "B — CBF",
            "n_total":       len(user_reviews),
            "n_train":       len(train_reviews),
            "n_test":        len(test_reviews),
            "withheld_item": test_reviews.iloc[0]["parent_asin"] if not test_reviews.empty else "",
            "has_cbf":       user_id in cbf_users,
        })
 
        if test_reviews.empty:
            continue
 
        if user_id in group_a:
            # Mode 1 — fair: per-user filtered popularity
            recs_m1 = _recs_popularity_personfiltered(
                popular_file, products_df, train_asins, top_n_fair)
            # Mode 2 — business: same fixed global list
            recs_m2 = global_top_a_biz
            if recs_m1:
                res_m1a.append({"user_id": user_id,
                                **_user_metrics(recs_m1, test_reviews, train_reviews)})
            if recs_m2:
                res_m2a.append({"user_id": user_id,
                                **_user_metrics(recs_m2, test_reviews, train_reviews)})
        else:
            cbf_user = cbf_grouped.get(user_id, pd.DataFrame())
            # Mode 1 — fair: top_n_fair CBF recs
            recs_m1 = _recs_cbf(cbf_user, top_n_fair)
            # Mode 2 — business: top_n_b_biz CBF recs
            recs_m2 = _recs_cbf(cbf_user, top_n_b_biz)
            if recs_m1:
                res_m1b.append({"user_id": user_id,
                                **_user_metrics(recs_m1, test_reviews, train_reviews)})
            if recs_m2:
                res_m2b.append({"user_id": user_id,
                                **_user_metrics(recs_m2, test_reviews, train_reviews)})
 
    df_m1a = pd.DataFrame(res_m1a); df_m1b = pd.DataFrame(res_m1b)
    df_m2a = pd.DataFrame(res_m2a); df_m2b = pd.DataFrame(res_m2b)
 
    agg_m1a = _agg(df_m1a); agg_m1b = _agg(df_m1b)
    agg_m2a = _agg(df_m2a); agg_m2b = _agg(df_m2b)
 
    sig_hit_m1, sig_rat_m1 = _significance(agg_m1a, agg_m1b)
    sig_hit_m2, sig_rat_m2 = _significance(agg_m2a, agg_m2b)
 
    return {
        # Mode 1 — fair
        "m1_a": agg_m1a, "m1_b": agg_m1b,
        "df_m1a": df_m1a, "df_m1b": df_m1b,
        "sig_hit_m1": sig_hit_m1, "sig_rat_m1": sig_rat_m1,
        # Mode 2 — business
        "m2_a": agg_m2a, "m2_b": agg_m2b,
        "df_m2a": df_m2a, "df_m2b": df_m2b,
        "sig_hit_m2": sig_hit_m2, "sig_rat_m2": sig_rat_m2,
        # Shared
        "split_df":      pd.DataFrame(split_log),
        "global_top_a":  global_top_a_biz,
        "n_eligible":    len(eligible),
        "n_ineligible":  len(ineligible),
        "no_cbf_users":  len(no_cbf),
        "params": {
            "top_n_fair":  top_n_fair,
            "top_n_a_biz": top_n_a_biz,
            "top_n_b_biz": top_n_b_biz,
        },
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
 
def _delta(vb, va):
    if va is None or pd.isna(va): return None
    return f"{vb - va:+.2f}"
 
 
def _sig_table(sig_hit: dict, sig_rating: dict) -> None:
    rows = []
    if sig_hit:
        rows.append({
            "What we tested": "Hit Rate (exact match)",
            "Test": "Chi-squared",
            "p-value": f"{sig_hit['p']:.4f}",
            "Result": "✅ Real difference" if sig_hit["significant"] else "❌ Could be chance",
            "Plain English": ("One model genuinely hits more often — not luck."
                              if sig_hit["significant"] else
                              "Not enough evidence to declare a winner yet."),
        })
    if sig_rating:
        rows.append({
            "What we tested": "Avg rating of matched products",
            "Test": "t-test",
            "p-value": f"{sig_rating['p']:.4f}",
            "Result": "✅ Real difference" if sig_rating["significant"] else "❌ Could be chance",
            "Plain English": ("One model genuinely recommends higher-rated products."
                              if sig_rating["significant"] else
                              "Rating difference may be random — more data needed."),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data to run significance tests.")
 
 
def _metrics_columns(a: dict, b: dict, label_a: str, label_b: str) -> None:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("**Hit Rate — Exact**")
        st.caption("Did the withheld item appear in the rec list?")
        st.metric(label_a, f"{a['hit_rate']:.1f}%")
        st.metric(label_b, f"{b['hit_rate']:.1f}%",
                  delta=_delta(b["hit_rate"], a["hit_rate"]))
    with m2:
        st.markdown("**Hit Rate — History**")
        st.caption("Did any rec appear in the user's full purchase history?")
        st.metric(label_a, f"{a['hit_rate_history']:.1f}%")
        st.metric(label_b, f"{b['hit_rate_history']:.1f}%",
                  delta=_delta(b["hit_rate_history"], a["hit_rate_history"]))
    with m3:
        st.markdown("**Precision (Exact)**")
        st.caption("% of rec slots that were correct")
        st.metric(label_a, f"{a['precision']:.2f}%")
        st.metric(label_b, f"{b['precision']:.2f}%",
                  delta=_delta(b["precision"], a["precision"]))
    with m4:
        st.markdown("**Avg Rating of Hits**")
        st.caption("How well users rated the matched products")
        a_r = a["avg_rating"]; b_r = b["avg_rating"]
        st.metric(label_a, f"{a_r:.2f}★" if pd.notna(a_r) else "N/A")
        st.metric(label_b, f"{b_r:.2f}★" if pd.notna(b_r) else "N/A",
                  delta=_delta(b_r, a_r) if (pd.notna(a_r) and pd.notna(b_r)) else None)
 
 
def _bar_chart(a: dict, b: dict, label_a: str, label_b: str, title: str) -> None:
    rows = []
    for metric, va, vb in [
        ("Hit Rate — Exact (%)",   a["hit_rate"],   b["hit_rate"]),
        ("Hit Rate — History (%)", a["hit_rate_history"], b["hit_rate_history"]),
        ("Precision — Exact (%)",  a["precision"],  b["precision"]),
        ("Avg Rating",
         a["avg_rating"] if pd.notna(a["avg_rating"]) else 0,
         b["avg_rating"] if pd.notna(b["avg_rating"]) else 0),
    ]:
        rows += [{"Model": label_a, "Metric": metric, "Value": round(va, 2)},
                 {"Model": label_b, "Metric": metric, "Value": round(vb, 2)}]
    fig = px.bar(pd.DataFrame(rows), x="Metric", y="Value", color="Model",
                 barmode="group", text="Value",
                 color_discrete_map={label_a: "#94a3b8", label_b: "#0d9488"},
                 title=title)
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=420, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)
 
 
def _plain_summary(a: dict, b: dict, label_a: str, label_b: str,
                   sig_hit: dict, sig_rat: dict, list_size_a: int, list_size_b: int) -> None:
    winner = label_b if b["hit_rate_history"] >= a["hit_rate_history"] else label_a
    sig_note = ("The difference **is statistically significant** (p < 0.05)."
                if (sig_hit.get("significant") or sig_rat.get("significant"))
                else "The difference is **not yet statistically significant** — more data needed.")
    hr_diff = abs(b["hit_rate_history"] - a["hit_rate_history"])
    lift    = (f"{label_b}'s history hit rate is **{hr_diff:.1f} percentage points higher**."
               if b["hit_rate_history"] > a["hit_rate_history"]
               else f"{label_a}'s history hit rate is **{hr_diff:.1f} percentage points higher**.")
    a_r  = a["avg_rating"];  b_r  = b["avg_rating"]
    a_rs = f"{a_r:.2f}★" if pd.notna(a_r) else "N/A"
    b_rs = f"{b_r:.2f}★" if pd.notna(b_r) else "N/A"
    st.markdown(f"""
- **{label_a}** (list size {list_size_a}) hit at least one correct product for
  **{a['hit_rate']:.0f}%** of customers (exact) / **{a['hit_rate_history']:.0f}%** (history).
- **{label_b}** (list size {list_size_b}) hit at least one correct product for
  **{b['hit_rate']:.0f}%** of customers (exact) / **{b['hit_rate_history']:.0f}%** (history).
- {lift}
- Matched products were rated {a_rs} ({label_a}) and {b_rs} ({label_b}) on average.
- **Winner: {winner}** (by history hit rate).
- {sig_note}
    """)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT TAB
# ══════════════════════════════════════════════════════════════════════════════
 
def show_ab_testing_tab(
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    cbf_df: pd.DataFrame | None = None,
    popular_file: pd.DataFrame | None = None,
) -> None:
    st.header("🧪 A/B Test — Popularity Baseline vs Content-Based Filtering")
 
    if cbf_df is None or cbf_df.empty:
        st.error("❌ No CBF dataset loaded. Pass `cbf_df` into `show_ab_testing_tab()`.")
        return
 
    with st.expander("ℹ️ Two evaluation modes — what's the difference?", expanded=False):
        st.markdown("""
| | Mode 1 — Fair Comparison | Mode 2 — Business Reality |
|---|---|---|
| **Model A list size** | Same N as CBF (e.g. Top-10) | Larger (e.g. Top-50, like a homepage) |
| **Model A filtering** | Removes items user already bought | Same fixed list for every user |
| **Model B list size** | Same N as Model A | Smaller (e.g. Top-10, like a personalised email) |
| **Answers** | Is personalisation better at equal list size? | Which drives more hits in real deployment? |
| **Why it matters** | Fairest scientific comparison | Reflects how you'd actually use each model |
 
**Hit Rate — Exact:** Did the user's withheld (next) purchase appear in the rec list?
 
**Hit Rate — History:** Did any rec appear anywhere in the user's full purchase history?
*(Fairer for CBF — it recommends things similar to what the user already likes)*
        """)
 
    st.subheader("⚙️ Settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        top_n_fair  = st.slider("Mode 1 — List size (both models)", 5, 30, DEFAULT_TOP_N_FAIR,
                                key="ab_fair_n")
    with c2:
        top_n_a_biz = st.slider("Mode 2 — Popularity list size (Model A)", 10, 100, DEFAULT_TOP_N_A_BIZ,
                                key="ab_biz_na")
    with c3:
        top_n_b_biz = st.slider("Mode 2 — CBF list size (Model B)", 5, 30, DEFAULT_TOP_N_B_BIZ,
                                key="ab_biz_nb")
 
    if st.button("▶ Run Both A/B Tests", type="primary", key="ab_run"):
        with st.spinner("Running both evaluation modes…"):
            results = run_ab_experiment(
                reviews_df=reviews, products_df=products, cbf_df=cbf_df,
                popular_file=popular_file,
                top_n_fair=top_n_fair,
                top_n_a_biz=top_n_a_biz,
                top_n_b_biz=top_n_b_biz,
            )
        st.session_state["ab_results"] = results
        st.success("✅ Both tests complete!")
 
    results = st.session_state.get("ab_results")
    if not results:
        st.info("Configure settings above and click **▶ Run Both A/B Tests** to start.")
        return
 
    p = results["params"]
 
    # ── Split summary ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("👥 Customer Split")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("In test",                 f"{results['n_eligible']:,}")
    d2.metric("Excluded (< 5 reviews)",  f"{results['n_ineligible']:,}")
    d3.metric("Excluded (no CBF data)",  f"{results['no_cbf_users']:,}")
    d4.metric("Group A — Popularity",    f"{results['m1_a']['n_users']:,}")
    d5.metric("Group B — CBF",           f"{results['m1_b']['n_users']:,}")
 
    with st.expander("🔍 Per-customer split log", expanded=False):
        st.dataframe(results["split_df"], use_container_width=True, hide_index=True)
        st.download_button("⬇ Download split log",
                           data=results["split_df"].to_csv(index=False).encode("utf-8"),
                           file_name="ab_split_log.csv", mime="text/csv", key="dl_split")
 
    # ── Mode 1 — Fair ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"📊 Mode 1 — Fair Comparison  (Top-{p['top_n_fair']} vs Top-{p['top_n_fair']})")
    st.caption(
        f"Both models recommend the same number of products ({p['top_n_fair']}).  "
        f"Model A filters out each user's purchase history first.  "
        f"**This is the scientifically fair comparison.**"
    )
    label_m1a = f"🔥 Popularity Top-{p['top_n_fair']} (filtered)"
    label_m1b = f"🧠 CBF Top-{p['top_n_fair']}"
    _metrics_columns(results["m1_a"], results["m1_b"], label_m1a, label_m1b)
    _bar_chart(results["m1_a"], results["m1_b"], label_m1a, label_m1b,
               f"Mode 1 — Fair Comparison (Top-{p['top_n_fair']} each)")
 
    st.markdown("#### Statistical Significance — Mode 1")
    _sig_table(results["sig_hit_m1"], results["sig_rat_m1"])
 
    st.markdown("#### What This Tells Us — Mode 1")
    _plain_summary(results["m1_a"], results["m1_b"], label_m1a, label_m1b,
                   results["sig_hit_m1"], results["sig_rat_m1"],
                   p["top_n_fair"], p["top_n_fair"])
 
    # ── Mode 2 — Business ─────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"📊 Mode 2 — Business Reality  "
                 f"(Popularity Top-{p['top_n_a_biz']} vs CBF Top-{p['top_n_b_biz']})")
    st.caption(
        f"Model A = fixed global Top-{p['top_n_a_biz']} list (homepage / mass promotion — same for everyone).  "
        f"Model B = personalised Top-{p['top_n_b_biz']} CBF list (email / widget — unique per user).  "
        f"**This reflects how you'd actually deploy each model.**"
    )
    label_m2a = f"🔥 Popularity Top-{p['top_n_a_biz']} (global)"
    label_m2b = f"🧠 CBF Top-{p['top_n_b_biz']}"
    _metrics_columns(results["m2_a"], results["m2_b"], label_m2a, label_m2b)
    _bar_chart(results["m2_a"], results["m2_b"], label_m2a, label_m2b,
               f"Mode 2 — Business Reality (Popularity Top-{p['top_n_a_biz']} vs CBF Top-{p['top_n_b_biz']})")
 
    st.markdown("#### Statistical Significance — Mode 2")
    _sig_table(results["sig_hit_m2"], results["sig_rat_m2"])
 
    st.markdown("#### What This Tells Us — Mode 2")
    _plain_summary(results["m2_a"], results["m2_b"], label_m2a, label_m2b,
                   results["sig_hit_m2"], results["sig_rat_m2"],
                   p["top_n_a_biz"], p["top_n_b_biz"])
 
    # ── Side-by-side mode comparison ──────────────────────────────────────────
    st.divider()
    st.subheader("🔁 Mode 1 vs Mode 2 — Does the evaluation method change the conclusion?")
    comp = pd.DataFrame([
        {"Evaluation Mode":    f"Mode 1 — Fair (Top-{p['top_n_fair']} each)",
         "Model A Hit Rate %": f"{results['m1_a']['hit_rate']:.1f}%",
         "Model B Hit Rate %": f"{results['m1_b']['hit_rate']:.1f}%",
         "Winner":             ("🧠 CBF" if results["m1_b"]["hit_rate"] >= results["m1_a"]["hit_rate"]
                                else "🔥 Popularity"),
         "Significant?":       "✅ Yes" if results["sig_hit_m1"].get("significant") else "❌ No"},
        {"Evaluation Mode":    f"Mode 2 — Business (Pop Top-{p['top_n_a_biz']} vs CBF Top-{p['top_n_b_biz']})",
         "Model A Hit Rate %": f"{results['m2_a']['hit_rate']:.1f}%",
         "Model B Hit Rate %": f"{results['m2_b']['hit_rate']:.1f}%",
         "Winner":             ("🧠 CBF" if results["m2_b"]["hit_rate"] >= results["m2_a"]["hit_rate"]
                                else "🔥 Popularity"),
         "Significant?":       "✅ Yes" if results["sig_hit_m2"].get("significant") else "❌ No"},
    ])
    st.dataframe(comp, use_container_width=True, hide_index=True)
 
    m1_winner = "CBF" if results["m1_b"]["hit_rate"] >= results["m1_a"]["hit_rate"] else "Popularity"
    m2_winner = "CBF" if results["m2_b"]["hit_rate"] >= results["m2_a"]["hit_rate"] else "Popularity"
    if m1_winner == m2_winner:
        st.success(f"✅ Both evaluation methods agree — **{m1_winner}** wins regardless of how you measure it. "
                   f"This is a robust result.")
    else:
        st.warning(f"⚠️ The two methods disagree — Mode 1 favours **{m1_winner}** "
                   f"but Mode 2 favours **{m2_winner}**. "
                   f"This likely means the larger list size in Mode 2 is driving the difference, "
                   f"not the quality of recommendations. Use Mode 1 for scientific comparison.")
 
    # ── Downloads ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Download Results")
    t1, t2, t3, t4 = st.tabs(["Mode 1 Group A", "Mode 1 Group B",
                               "Mode 2 Group A", "Mode 2 Group B"])
    for tab, df, fname, label in [
        (t1, results["df_m1a"], "ab_mode1_group_a.csv", "Mode 1 — Popularity"),
        (t2, results["df_m1b"], "ab_mode1_group_b.csv", "Mode 1 — CBF"),
        (t3, results["df_m2a"], "ab_mode2_group_a.csv", "Mode 2 — Popularity"),
        (t4, results["df_m2b"], "ab_mode2_group_b.csv", "Mode 2 — CBF"),
    ]:
        with tab:
            if df.empty:
                st.info("No results.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(f"⬇ Download {label}",
                                   data=df.to_csv(index=False).encode("utf-8"),
                                   file_name=fname, mime="text/csv", key=f"dl_{fname}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    import sys
 
    _ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_ROOT))
 
    def _noop_cache(*args, **kwargs):
        def decorator(fn): return fn
        if args and callable(args[0]): return args[0]
        return decorator
    st.cache_data = _noop_cache
 
    _OUT_DIR      = _DATA_DIR
    _REVIEWS_CSV  = Path(r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard\data\reviews_clean_no_exact_duplicates.csv")
    _PRODUCTS_CSV = Path(r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard\data\products_clean.csv")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
 
    SEP = "=" * 65
    print(SEP)
    print("  A/B Test — Standalone Mode")
    print("  Mode 1: Fair Comparison (same list size)")
    print("  Mode 2: Business Reality (different list sizes)")
    print(SEP)
 
    print("\n[LOAD]  Reviews ...")
    if not _REVIEWS_CSV.exists(): print(f"  ERROR: {_REVIEWS_CSV}"); sys.exit(1)
    reviews_df = pd.read_csv(_REVIEWS_CSV, low_memory=False)
    print(f"  {len(reviews_df):,} rows")
 
    print("[LOAD]  Products ...")
    if not _PRODUCTS_CSV.exists(): print(f"  ERROR: {_PRODUCTS_CSV}"); sys.exit(1)
    products_df = pd.read_csv(_PRODUCTS_CSV, low_memory=False)
    print(f"  {len(products_df):,} rows")
 
    print("[LOAD]  CBF recommendations ...")
    if not _CBF_CSV.exists(): print(f"  ERROR: {_CBF_CSV}"); sys.exit(1)
    cbf_df = pd.read_csv(_CBF_CSV)
    for c in ["predicted_score", "rank"]:
        if c in cbf_df.columns: cbf_df[c] = pd.to_numeric(cbf_df[c], errors="coerce")
    print(f"  {len(cbf_df):,} rows | {cbf_df['user_id'].nunique():,} users")
 
    popular_file = pd.read_csv(_POPULAR_CSV) if _POPULAR_CSV.exists() else None
    if popular_file is not None: print(f"[LOAD]  popular_products_top100.csv ({len(popular_file):,} rows)")
    else:                        print("[WARN]  popular_products_top100.csv not found — scoring from catalogue")
 
    print(f"\n[RUN]   Mode 1: Top-{DEFAULT_TOP_N_FAIR} vs Top-{DEFAULT_TOP_N_FAIR}")
    print(f"[RUN]   Mode 2: Popularity Top-{DEFAULT_TOP_N_A_BIZ} vs CBF Top-{DEFAULT_TOP_N_B_BIZ}")
 
    results = run_ab_experiment(
        reviews_df=reviews_df, products_df=products_df, cbf_df=cbf_df,
        popular_file=popular_file,
        top_n_fair=DEFAULT_TOP_N_FAIR,
        top_n_a_biz=DEFAULT_TOP_N_A_BIZ,
        top_n_b_biz=DEFAULT_TOP_N_B_BIZ,
    )
 
    p = results["params"]
 
    def _print_mode(tag, a, b, sh, sr, la, lb):
        print(f"\n  {tag}")
        print(f"  {'Metric':<35} {la:>18} {lb:>18} {'Delta':>10}")
        print(f"  {'-'*83}")
        for label, va, vb in [
            ("Hit Rate — Exact (%)",    a["hit_rate"],          b["hit_rate"]),
            ("Hit Rate — History (%)",  a["hit_rate_history"],  b["hit_rate_history"]),
            ("Precision — Exact (%)",   a["precision"],         b["precision"]),
        ]:
            print(f"  {label:<35} {va:>17.2f}% {vb:>17.2f}% {vb-va:>+9.2f}%")
        a_r = a["avg_rating"]; b_r = b["avg_rating"]
        a_rs = f"{a_r:.3f}" if pd.notna(a_r) else "N/A"
        b_rs = f"{b_r:.3f}" if pd.notna(b_r) else "N/A"
        d_rs = f"{b_r-a_r:+.3f}" if (pd.notna(a_r) and pd.notna(b_r)) else "N/A"
        print(f"  {'Avg Rating (Exact Hits)':<35} {a_rs:>18} {b_rs:>18} {d_rs:>10}")
        if sh: print(f"  Hit Rate p={sh['p']:.4f}  {'✓ SIGNIFICANT' if sh['significant'] else '✗ not significant'}")
        if sr: print(f"  Avg Rating p={sr['p']:.4f}  {'✓ SIGNIFICANT' if sr['significant'] else '✗ not significant'}")
        winner = lb if b["hit_rate_history"] >= a["hit_rate_history"] else la
        print(f"  Winner (history hit rate): {winner}")
 
    print("\n" + SEP + "\n  RESULTS\n" + SEP)
    print(f"  Eligible: {results['n_eligible']:,}  |  Group A: {results['m1_a']['n_users']:,}  |  Group B: {results['m1_b']['n_users']:,}")
 
    _print_mode(
        f"MODE 1 — Fair Comparison (Top-{p['top_n_fair']} each)",
        results["m1_a"], results["m1_b"],
        results["sig_hit_m1"], results["sig_rat_m1"],
        f"Pop Top-{p['top_n_fair']}", f"CBF Top-{p['top_n_fair']}",
    )
    _print_mode(
        f"MODE 2 — Business Reality (Pop Top-{p['top_n_a_biz']} vs CBF Top-{p['top_n_b_biz']})",
        results["m2_a"], results["m2_b"],
        results["sig_hit_m2"], results["sig_rat_m2"],
        f"Pop Top-{p['top_n_a_biz']}", f"CBF Top-{p['top_n_b_biz']}",
    )
 
    m1w = "CBF" if results["m1_b"]["hit_rate"] >= results["m1_a"]["hit_rate"] else "Popularity"
    m2w = "CBF" if results["m2_b"]["hit_rate"] >= results["m2_a"]["hit_rate"] else "Popularity"
    print(f"\n  Both modes agree: {'YES — ' + m1w + ' wins both' if m1w == m2w else 'NO — Mode 1=' + m1w + ', Mode 2=' + m2w}")
 
    # Save
    print(f"\n[SAVE]  Writing to {_OUT_DIR} ...")
    for df, fname in [
        (results["df_m1a"], "ab_mode1_group_a.csv"),
        (results["df_m1b"], "ab_mode1_group_b.csv"),
        (results["df_m2a"], "ab_mode2_group_a.csv"),
        (results["df_m2b"], "ab_mode2_group_b.csv"),
        (results["split_df"], "ab_split_log.csv"),
    ]:
        df.to_csv(_OUT_DIR / fname, index=False)
        print(f"  → {fname}  ({len(df):,} rows)")
 
    # Summary
    rows = []
    for mode, a, b, sh, la, lb in [
        (f"Mode 1 — Fair Top-{p['top_n_fair']}",
         results["m1_a"], results["m1_b"], results["sig_hit_m1"],
         f"Pop Top-{p['top_n_fair']}", f"CBF Top-{p['top_n_fair']}"),
        (f"Mode 2 — Business Pop-{p['top_n_a_biz']} vs CBF-{p['top_n_b_biz']}",
         results["m2_a"], results["m2_b"], results["sig_hit_m2"],
         f"Pop Top-{p['top_n_a_biz']}", f"CBF Top-{p['top_n_b_biz']}"),
    ]:
        for grp, agg, lbl in [("A", a, la), ("B", b, lb)]:
            rows.append({"mode": mode, "group": grp, "model": lbl,
                         "n_users": agg["n_users"],
                         "hit_rate_exact": round(agg["hit_rate"], 4),
                         "hit_rate_history": round(agg["hit_rate_history"], 4),
                         "precision_exact": round(agg["precision"], 4),
                         "avg_rating": round(agg["avg_rating"], 4) if pd.notna(agg["avg_rating"]) else None,
                         "sig_hit_p": round(sh["p"], 6) if sh else None,
                         "sig_hit": sh.get("significant") if sh else None})
    pd.DataFrame(rows).to_csv(_OUT_DIR / "ab_results_summary.csv", index=False)
    print("  → ab_results_summary.csv")
    print(f"\n[DONE]\n{SEP}")