from __future__ import annotations
 
"""
ab_testing.py — Offline Evaluation: Global Popularity Baseline vs CBF
======================================================================
 
No user splitting — every eligible customer is evaluated against BOTH
models simultaneously. Results are compared directly.
 
Two evaluation modes run simultaneously:
 
  MODE 1 — Fair Comparison (same N)
    Model A: Top-N popular products the user hasn't seen (per-user filtered)
    Model B: Top-N personalised CBF recommendations
    → Answers: Does personalisation beat popularity at equal list size?
 
  MODE 2 — Business Reality (different N)
    Model A: Fixed global Top-50 popularity list (same for all, like a homepage)
    Model B: Top-10 personalised CBF recommendations (like a personalised email)
    → Answers: Which drives more hits in real deployment scenarios?
 
Evaluation strategy: Leave-One-Out
  Latest interaction per user is withheld as answer key.
  All earlier interactions are training history.
 
Significance tests: Wilcoxon signed-rank (paired, same users), p < 0.05
"""
 
from math import floor
from pathlib import Path
 
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
 
# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA_DIR      = Path(r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard\dashboard_data")
_CBF_CSV       = _DATA_DIR / "cbf_dashboard.csv"
_POPULAR_CSV   = _DATA_DIR / "popular_products_top100.csv"
_DISCOVERY_CSV = _DATA_DIR / "discovery_products_top100.csv"
 
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
        sort_cols = [c for c in ["review_year", "review_month"]
                     if c in user_reviews.columns]
        ur = (user_reviews.sort_values(sort_cols, ascending=True).reset_index(drop=True)
              if sort_cols else user_reviews.reset_index(drop=True))
    n = len(ur)
    h = _holdout_size(n)
    if h == 0:
        return ur, ur.iloc[0:0]
    return ur.iloc[:n - h].copy(), ur.iloc[n - h:].copy()
 
 
def _popular_scored(products: pd.DataFrame) -> pd.DataFrame:
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
    popular_file:   pd.DataFrame | None,
    discovery_file: pd.DataFrame | None,
    products_df:    pd.DataFrame,
    train_asins:    set,
    top_n:          int,
) -> list[str]:
    """Top-N popular products the user hasn't already seen (per-user filtered)."""
    if popular_file is not None and "parent_asin" in popular_file.columns:
        sc         = _get_score_col(popular_file)
        src        = popular_file.sort_values(sc, ascending=False) if sc else popular_file
        pop_list   = src["parent_asin"].tolist()
    else:
        pop_list   = _popular_scored(products_df)["parent_asin"].tolist()
 
    if discovery_file is not None and "parent_asin" in discovery_file.columns:
        disc_sc    = _get_score_col(discovery_file)
        disc_src   = discovery_file.sort_values(disc_sc, ascending=False) if disc_sc else discovery_file
        disc_list  = [a for a in disc_src["parent_asin"].tolist() if a not in set(pop_list)]
        n_disc     = max(1, round(top_n * 0.20))
        candidates = pop_list + disc_list[:n_disc]
    else:
        candidates = pop_list
 
    return [a for a in candidates if a not in train_asins][:top_n]
 
 
def _recs_popularity_global(
    popular_file:   pd.DataFrame | None,
    discovery_file: pd.DataFrame | None,
    products_df:    pd.DataFrame,
    top_n:          int,
) -> list[str]:
    """Same fixed global Top-N for every user (no history filtering)."""
    if popular_file is not None and "parent_asin" in popular_file.columns:
        sc       = _get_score_col(popular_file)
        src      = popular_file.sort_values(sc, ascending=False) if sc else popular_file
        pop_list = src["parent_asin"].tolist()
    else:
        pop_list = _popular_scored(products_df)["parent_asin"].tolist()
 
    if discovery_file is not None and "parent_asin" in discovery_file.columns:
        disc_sc   = _get_score_col(discovery_file)
        disc_src  = discovery_file.sort_values(disc_sc, ascending=False) if disc_sc else discovery_file
        pop_set   = set(pop_list)
        disc_list = [a for a in disc_src["parent_asin"].tolist() if a not in pop_set]
        n_disc    = max(1, round(top_n * 0.20))
        n_pop     = top_n - n_disc
        return pop_list[:n_pop] + disc_list[:n_disc]
 
    return pop_list[:top_n]
 
 
def _recs_cbf(cbf_user: pd.DataFrame, top_n: int) -> list[str]:
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
 
    exact_hits = [r for r in recs if r in held_asins]
    hist_hits  = [r for r in recs if r in full_history]
 
    all_reviews = pd.concat([held_out, train_reviews])
 
    return {
        "hit":               int(len(exact_hits) > 0),
        "precision":         len(exact_hits) / len(recs) if recs else 0.0,
        "avg_rating":        (held_out[held_out["parent_asin"].isin(exact_hits)]["rating"].mean()
                              if exact_hits else float("nan")),
        "hit_history":       int(len(hist_hits) > 0),
        "precision_history": len(hist_hits) / len(recs) if recs else 0.0,
        "avg_rating_history":(all_reviews[all_reviews["parent_asin"].isin(hist_hits)]["rating"].mean()
                              if hist_hits else float("nan")),
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════
 
def _agg(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"hit_rate": 0.0, "precision": 0.0, "avg_rating": float("nan"),
                "hit_rate_history": 0.0, "precision_history": 0.0,
                "avg_rating_history": float("nan"), "n_users": 0,
                "hits_arr": [], "ratings_arr": [], "prec_arr": []}
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
        "prec_arr":           df["precision"].tolist(),
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  SIGNIFICANCE  (paired Wilcoxon — same users evaluated on both models)
# ══════════════════════════════════════════════════════════════════════════════
 
def _significance(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """
    Paired Wilcoxon signed-rank test on per-user hit and precision.
    Both DataFrames must be aligned on user_id (same users, same order).
    """
    if df_a.empty or df_b.empty:
        return {}
 
    # Align on user_id
    merged = df_a[["user_id", "hit", "precision", "avg_rating"]].merge(
        df_b[["user_id", "hit", "precision", "avg_rating"]],
        on="user_id", suffixes=("_a", "_b"),
    )
    if len(merged) < 10:
        return {}
 
    results = {}
    for metric in ["hit", "precision"]:
        xa = merged[f"{metric}_a"].values.astype(float)
        xb = merged[f"{metric}_b"].values.astype(float)
        diff = xa - xb
        if np.all(diff == 0):
            results[metric] = {"p": 1.0, "significant": False, "direction": "no difference"}
            continue
        try:
            _, p = stats.wilcoxon(xa, xb, alternative="two-sided", zero_method="wilcox")
            results[metric] = {
                "p":           p,
                "significant": p < 0.05,
                "direction":   "Model B higher" if xb.mean() > xa.mean() else "Model A higher",
            }
        except Exception:
            results[metric] = {"p": float("nan"), "significant": False, "direction": "error"}
 
    # t-test on avg_rating (unpaired — not every user has a rating hit)
    ra = df_a["avg_rating"].dropna().tolist()
    rb = df_b["avg_rating"].dropna().tolist()
    if len(ra) >= 2 and len(rb) >= 2:
        _, p = stats.ttest_ind(ra, rb)
        results["avg_rating"] = {
            "p":           p,
            "significant": p < 0.05,
            "direction":   "Model B higher" if np.mean(rb) > np.mean(ra) else "Model A higher",
        }
 
    return results
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════
 
@st.cache_data(show_spinner=False)
def run_ab_experiment(
    reviews_df:      pd.DataFrame,
    products_df:     pd.DataFrame,
    cbf_df:          pd.DataFrame,
    popular_file:    pd.DataFrame | None = None,
    discovery_file:  pd.DataFrame | None = None,
    top_n_fair:      int = DEFAULT_TOP_N_FAIR,
    top_n_a_biz:     int = DEFAULT_TOP_N_A_BIZ,
    top_n_b_biz:     int = DEFAULT_TOP_N_B_BIZ,
    random_seed:     int = 42,
) -> dict:
    """
    Every eligible customer is evaluated against BOTH models.
    No group splitting — direct paired comparison.
    """
    cbf_grouped = {uid: grp for uid, grp in cbf_df.groupby("user_id")}
    cbf_users   = set(cbf_grouped.keys())
 
    user_counts  = reviews_df["user_id"].value_counts()
    ineligible   = user_counts[user_counts < MIN_REVIEWS_FOR_TEST].index.tolist()
    all_eligible = user_counts[user_counts >= MIN_REVIEWS_FOR_TEST].index.tolist()
    eligible     = [u for u in all_eligible if u in cbf_users]
    no_cbf       = [u for u in all_eligible if u not in cbf_users]
 
    # Precompute fixed global list for Mode 2 Model A (same for all users)
    global_pop_biz = _recs_popularity_global(
        popular_file, discovery_file, products_df, top_n_a_biz)
 
    grouped   = {uid: grp for uid, grp in reviews_df.groupby("user_id")}
    # Each list stores one row per user for each model/mode combination
    res_m1a, res_m1b = [], []   # Mode 1 — fair
    res_m2a, res_m2b = [], []   # Mode 2 — business
    eval_log = []
 
    for user_id in eligible:
        user_reviews = grouped.get(user_id)
        if user_reviews is None:
            continue
 
        train_reviews, test_reviews = _chronological_split(user_reviews)
        if test_reviews.empty:
            continue
 
        train_asins = set(train_reviews["parent_asin"].tolist())
        cbf_user    = cbf_grouped.get(user_id, pd.DataFrame())
 
        # ── Mode 1 — Fair (same N, history-filtered popularity) ───────────────
        recs_m1a = _recs_popularity_personfiltered(
            popular_file, discovery_file, products_df, train_asins, top_n_fair)
        recs_m1b = _recs_cbf(cbf_user, top_n_fair)
 
        # ── Mode 2 — Business (different N, global fixed list) ────────────────
        recs_m2a = global_pop_biz
        recs_m2b = _recs_cbf(cbf_user, top_n_b_biz)
 
        if recs_m1a:
            res_m1a.append({"user_id": user_id,
                            **_user_metrics(recs_m1a, test_reviews, train_reviews)})
        if recs_m1b:
            res_m1b.append({"user_id": user_id,
                            **_user_metrics(recs_m1b, test_reviews, train_reviews)})
        if recs_m2a:
            res_m2a.append({"user_id": user_id,
                            **_user_metrics(recs_m2a, test_reviews, train_reviews)})
        if recs_m2b:
            res_m2b.append({"user_id": user_id,
                            **_user_metrics(recs_m2b, test_reviews, train_reviews)})
 
        eval_log.append({
            "user_id":       user_id,
            "n_total":       len(user_reviews),
            "n_train":       len(train_reviews),
            "n_test":        len(test_reviews),
            "withheld_item": test_reviews.iloc[0]["parent_asin"],
            "has_cbf":       user_id in cbf_users,
        })
 
    df_m1a = pd.DataFrame(res_m1a); df_m1b = pd.DataFrame(res_m1b)
    df_m2a = pd.DataFrame(res_m2a); df_m2b = pd.DataFrame(res_m2b)
 
    return {
        # Mode 1
        "m1_a":   _agg(df_m1a), "m1_b":   _agg(df_m1b),
        "df_m1a": df_m1a,       "df_m1b": df_m1b,
        "sig_m1": _significance(df_m1a, df_m1b),
        # Mode 2
        "m2_a":   _agg(df_m2a), "m2_b":   _agg(df_m2b),
        "df_m2a": df_m2a,       "df_m2b": df_m2b,
        "sig_m2": _significance(df_m2a, df_m2b),
        # Shared
        "eval_log":      pd.DataFrame(eval_log),
        "global_pop_biz": global_pop_biz,
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
#  STREAMLIT DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
 
def _delta(vb, va):
    if va is None or pd.isna(va): return None
    return f"{vb - va:+.2f}"
 
 
def _sig_table(sig: dict) -> None:
    rows = []
    labels = {"hit": "Hit Rate (exact)", "precision": "Precision (exact)",
              "avg_rating": "Avg Rating of Hits"}
    for key, label in labels.items():
        s = sig.get(key)
        if not s:
            continue
        rows.append({
            "Metric":      label,
            "p-value":     f"{s['p']:.4f}" if not pd.isna(s["p"]) else "N/A",
            "Result":      "✅ Real difference" if s["significant"] else "❌ Could be chance",
            "Direction":   s.get("direction", ""),
            "Plain English": ("One model is genuinely better here — not luck."
                              if s["significant"] else
                              "Difference may be random — more data needed."),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Not enough paired data to run significance tests.")
 
 
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
        st.markdown("**Precision**")
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
        ("Hit Rate — Exact (%)",   a["hit_rate"],          b["hit_rate"]),
        ("Hit Rate — History (%)", a["hit_rate_history"],  b["hit_rate_history"]),
        ("Precision (%)",          a["precision"],         b["precision"]),
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
                   sig: dict, size_a: int, size_b: int) -> None:
    winner   = label_b if b["hit_rate_history"] >= a["hit_rate_history"] else label_a
    sig_hit  = sig.get("hit", {})
    sig_rat  = sig.get("avg_rating", {})
    sig_note = ("The difference **is statistically significant** (p < 0.05)."
                if (sig_hit.get("significant") or sig_rat.get("significant"))
                else "The difference is **not yet statistically significant** — more data needed.")
    hr_diff  = abs(b["hit_rate_history"] - a["hit_rate_history"])
    lift     = (f"{label_b}'s history hit rate is **{hr_diff:.1f} percentage points higher**."
                if b["hit_rate_history"] > a["hit_rate_history"]
                else f"{label_a}'s history hit rate is **{hr_diff:.1f} percentage points higher**.")
    a_rs = f"{a['avg_rating']:.2f}★" if pd.notna(a["avg_rating"]) else "N/A"
    b_rs = f"{b['avg_rating']:.2f}★" if pd.notna(b["avg_rating"]) else "N/A"
    st.markdown(f"""
- **{label_a}** (list size {size_a}) correctly predicted the next purchase for
  **{a['hit_rate']:.1f}%** of customers (exact) / **{a['hit_rate_history']:.1f}%** (history match).
- **{label_b}** (list size {size_b}) correctly predicted the next purchase for
  **{b['hit_rate']:.1f}%** of customers (exact) / **{b['hit_rate_history']:.1f}%** (history match).
- {lift}
- Matched products rated {a_rs} ({label_a}) vs {b_rs} ({label_b}) on average.
- **Winner: {winner}** (by history hit rate).
- {sig_note}
    """)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT TAB
# ══════════════════════════════════════════════════════════════════════════════
 
def show_ab_testing_tab(
    reviews:        pd.DataFrame,
    products:       pd.DataFrame,
    cbf_df:         pd.DataFrame | None = None,
    popular_file:   pd.DataFrame | None = None,
    discovery_file: pd.DataFrame | None = None,
) -> None:
    st.header("🧪 Model Evaluation — Popularity Baseline vs Content-Based Filtering")
    st.caption(
        "Every eligible customer is evaluated against both models simultaneously — "
        "no group splitting. Results are a direct paired comparison."
    )
 
    if cbf_df is None or cbf_df.empty:
        st.error("❌ No CBF dataset loaded.")
        return
 
    with st.expander("ℹ️ How this evaluation works", expanded=False):
        st.markdown(f"""
**Every customer is tested on both models at the same time.**
 
1. Each customer's most recent purchase is hidden as the answer key
2. Both models generate recommendations using only their earlier purchase history
3. We check whether each model's recommendations contained the hidden item
 
| | Mode 1 — Fair | Mode 2 — Business Reality |
|---|---|---|
| **Popularity list** | Top-N unseen items per customer | Fixed global Top-{DEFAULT_TOP_N_A_BIZ} (same for everyone) |
| **CBF list** | Top-N personalised recs | Top-{DEFAULT_TOP_N_B_BIZ} personalised recs |
| **Answers** | Which model is scientifically better? | Which performs better in real deployment? |
 
**Hit Rate — Exact:** Did the model predict exactly what the customer bought next?
 
**Hit Rate — History:** Did any recommendation match anything in the customer's full purchase history?
        """)
 
    st.subheader("⚙️ Settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        top_n_fair  = st.slider("Mode 1 — List size (both models)",
                                5, 30, DEFAULT_TOP_N_FAIR, key="ab_fair_n")
    with c2:
        top_n_a_biz = st.slider("Mode 2 — Popularity list size",
                                10, 100, DEFAULT_TOP_N_A_BIZ, key="ab_biz_na")
    with c3:
        top_n_b_biz = st.slider("Mode 2 — CBF list size",
                                5, 30, DEFAULT_TOP_N_B_BIZ, key="ab_biz_nb")
 
    if st.button("▶ Run Evaluation", type="primary", key="ab_run"):
        with st.spinner("Evaluating both models on all eligible customers…"):
            results = run_ab_experiment(
                reviews_df=reviews, products_df=products, cbf_df=cbf_df,
                popular_file=popular_file, discovery_file=discovery_file,
                top_n_fair=top_n_fair, top_n_a_biz=top_n_a_biz, top_n_b_biz=top_n_b_biz,
            )
        st.session_state["ab_results"] = results
        st.success("✅ Evaluation complete!")
 
    results = st.session_state.get("ab_results")
    if not results:
        st.info("Configure settings above and click **▶ Run Evaluation** to start.")
        return
 
    p = results["params"]
 
    # ── Coverage summary ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Customer Coverage")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Customers evaluated",         f"{results['n_eligible']:,}")
    d2.metric("Excluded (< 5 purchases)",    f"{results['n_ineligible']:,}")
    d3.metric("Excluded (no CBF data)",      f"{results['no_cbf_users']:,}")
    d4.metric("Mode 1 paired comparisons",   f"{len(results['df_m1a']):,}")
 
    with st.expander("🔍 Per-customer evaluation log", expanded=False):
        st.dataframe(results["eval_log"], use_container_width=True, hide_index=True)
        st.download_button("⬇ Download evaluation log",
                           data=results["eval_log"].to_csv(index=False).encode("utf-8"),
                           file_name="eval_log.csv", mime="text/csv", key="dl_log")
 
    # ── Mode 1 ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"📊 Mode 1 — Fair Comparison  (Top-{p['top_n_fair']} each)")
    st.caption(
        f"Both models recommend {p['top_n_fair']} products per customer.  "
        "Popularity list is filtered to remove each customer's purchase history.  "
        "**This is the scientifically fair comparison.**"
    )
    label_m1a = f"🔥 Popularity Top-{p['top_n_fair']}"
    label_m1b = f"🧠 CBF Top-{p['top_n_fair']}"
    _metrics_columns(results["m1_a"], results["m1_b"], label_m1a, label_m1b)
    _bar_chart(results["m1_a"], results["m1_b"], label_m1a, label_m1b,
               f"Mode 1 — Fair Comparison (Top-{p['top_n_fair']} each)")
    st.markdown("#### Statistical Significance — Mode 1")
    _sig_table(results["sig_m1"])
    st.markdown("#### Summary — Mode 1")
    _plain_summary(results["m1_a"], results["m1_b"], label_m1a, label_m1b,
                   results["sig_m1"], p["top_n_fair"], p["top_n_fair"])
 
    # ── Mode 2 ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"📊 Mode 2 — Business Reality  "
                 f"(Popularity Top-{p['top_n_a_biz']} vs CBF Top-{p['top_n_b_biz']})")
    st.caption(
        f"Popularity = fixed global Top-{p['top_n_a_biz']} (same for all customers, like a homepage).  "
        f"CBF = personalised Top-{p['top_n_b_biz']} per customer (like a targeted email).  "
        "**This reflects real deployment conditions.**"
    )
    label_m2a = f"🔥 Popularity Top-{p['top_n_a_biz']} (global)"
    label_m2b = f"🧠 CBF Top-{p['top_n_b_biz']}"
    _metrics_columns(results["m2_a"], results["m2_b"], label_m2a, label_m2b)
    _bar_chart(results["m2_a"], results["m2_b"], label_m2a, label_m2b,
               f"Mode 2 — Business Reality")
    st.markdown("#### Statistical Significance — Mode 2")
    _sig_table(results["sig_m2"])
    st.markdown("#### Summary — Mode 2")
    _plain_summary(results["m2_a"], results["m2_b"], label_m2a, label_m2b,
                   results["sig_m2"], p["top_n_a_biz"], p["top_n_b_biz"])
 
    # ── Agreement check ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔁 Do Both Modes Agree?")
    m1w = "CBF" if results["m1_b"]["hit_rate"] >= results["m1_a"]["hit_rate"] else "Popularity"
    m2w = "CBF" if results["m2_b"]["hit_rate"] >= results["m2_a"]["hit_rate"] else "Popularity"
 
    comp = pd.DataFrame([
        {"Evaluation Mode":   f"Mode 1 — Fair (Top-{p['top_n_fair']} each)",
         "Popularity Hit %":  f"{results['m1_a']['hit_rate']:.1f}%",
         "CBF Hit %":         f"{results['m1_b']['hit_rate']:.1f}%",
         "Winner":            "🧠 CBF" if m1w == "CBF" else "🔥 Popularity",
         "Significant?":      "✅ Yes" if results["sig_m1"].get("hit", {}).get("significant") else "❌ No"},
        {"Evaluation Mode":   f"Mode 2 — Business (Pop {p['top_n_a_biz']} vs CBF {p['top_n_b_biz']})",
         "Popularity Hit %":  f"{results['m2_a']['hit_rate']:.1f}%",
         "CBF Hit %":         f"{results['m2_b']['hit_rate']:.1f}%",
         "Winner":            "🧠 CBF" if m2w == "CBF" else "🔥 Popularity",
         "Significant?":      "✅ Yes" if results["sig_m2"].get("hit", {}).get("significant") else "❌ No"},
    ])
    st.dataframe(comp, use_container_width=True, hide_index=True)
 
    if m1w == m2w:
        st.success(f"✅ Both modes agree — **{m1w}** wins regardless of evaluation method. "
                   "This is a robust result you can present with confidence.")
    else:
        st.warning(f"⚠️ The modes disagree — Mode 1 favours **{m1w}** but Mode 2 favours **{m2w}**. "
                   f"This is likely because Mode 2 gives Popularity {p['top_n_a_biz']} slots vs "
                   f"CBF's {p['top_n_b_biz']} — more slots = more chances to hit. "
                   "Trust Mode 1 for the scientific conclusion.")
 
    # ── Downloads ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Download Full Results")
    t1, t2, t3, t4 = st.tabs(["Mode 1 — Popularity", "Mode 1 — CBF",
                               "Mode 2 — Popularity", "Mode 2 — CBF"])
    for tab, df, fname, label in [
        (t1, results["df_m1a"], "eval_mode1_popularity.csv", "Mode 1 Popularity"),
        (t2, results["df_m1b"], "eval_mode1_cbf.csv",        "Mode 1 CBF"),
        (t3, results["df_m2a"], "eval_mode2_popularity.csv", "Mode 2 Popularity"),
        (t4, results["df_m2b"], "eval_mode2_cbf.csv",        "Mode 2 CBF"),
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
 
    _REVIEWS_CSV  = Path(r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard\data\reviews_clean_no_exact_duplicates.csv")
    _PRODUCTS_CSV = Path(r"C:\Users\kelvi\Downloads\dashboard - Copy (2) - Copy\dashboard\data\products_clean.csv")
    _OUT_DIR      = _DATA_DIR
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
 
    SEP = "=" * 65
    print(SEP)
    print("  Model Evaluation — Standalone Mode")
    print("  All eligible customers evaluated on BOTH models (no splitting)")
    print(SEP)
 
    for path, label in [(_REVIEWS_CSV, "Reviews"), (_PRODUCTS_CSV, "Products"), (_CBF_CSV, "CBF")]:
        if not path.exists():
            print(f"  ERROR: {label} file not found at {path}"); sys.exit(1)
 
    print("\n[LOAD]  Reviews ..."); reviews_df = pd.read_csv(_REVIEWS_CSV, low_memory=False)
    print(f"  {len(reviews_df):,} rows")
    print("[LOAD]  Products ..."); products_df = pd.read_csv(_PRODUCTS_CSV, low_memory=False)
    print(f"  {len(products_df):,} rows")
    print("[LOAD]  CBF recommendations ..."); cbf_df = pd.read_csv(_CBF_CSV)
    for c in ["predicted_score", "rank"]:
        if c in cbf_df.columns: cbf_df[c] = pd.to_numeric(cbf_df[c], errors="coerce")
    print(f"  {len(cbf_df):,} rows | {cbf_df['user_id'].nunique():,} users")
 
    popular_file   = pd.read_csv(_POPULAR_CSV)   if _POPULAR_CSV.exists()   else None
    discovery_file = pd.read_csv(_DISCOVERY_CSV) if _DISCOVERY_CSV.exists() else None
    if popular_file   is not None: print(f"[LOAD]  popular_products_top100.csv   ({len(popular_file):,} rows)")
    else:                          print("[WARN]  popular_products_top100.csv not found")
    if discovery_file is not None: print(f"[LOAD]  discovery_products_top100.csv ({len(discovery_file):,} rows)")
    else:                          print("[WARN]  discovery_products_top100.csv not found")
 
    print(f"\n[RUN]   Mode 1: Top-{DEFAULT_TOP_N_FAIR} vs Top-{DEFAULT_TOP_N_FAIR} (fair)")
    print(f"[RUN]   Mode 2: Popularity Top-{DEFAULT_TOP_N_A_BIZ} vs CBF Top-{DEFAULT_TOP_N_B_BIZ} (business)")
 
    results = run_ab_experiment(
        reviews_df=reviews_df, products_df=products_df, cbf_df=cbf_df,
        popular_file=popular_file, discovery_file=discovery_file,
        top_n_fair=DEFAULT_TOP_N_FAIR, top_n_a_biz=DEFAULT_TOP_N_A_BIZ,
        top_n_b_biz=DEFAULT_TOP_N_B_BIZ,
    )
 
    p = results["params"]
    print(f"\n{SEP}\n  RESULTS\n{SEP}")
    print(f"  Customers evaluated : {results['n_eligible']:,}")
    print(f"  Excluded (< 5 purchases)  : {results['n_ineligible']:,}")
    print(f"  Excluded (no CBF data)    : {results['no_cbf_users']:,}")
 
    def _print_mode(tag, a, b, sig, la, lb):
        print(f"\n  {tag}")
        print(f"  {'Metric':<30} {la:>20} {lb:>20} {'Delta':>10}")
        print(f"  {'-'*82}")
        for lbl, va, vb in [
            ("Hit Rate — Exact (%)",   a["hit_rate"],         b["hit_rate"]),
            ("Hit Rate — History (%)", a["hit_rate_history"], b["hit_rate_history"]),
            ("Precision (%)",          a["precision"],        b["precision"]),
        ]:
            print(f"  {lbl:<30} {va:>19.2f}% {vb:>19.2f}% {vb-va:>+9.2f}%")
        a_r = a["avg_rating"]; b_r = b["avg_rating"]
        a_rs = f"{a_r:.3f}" if pd.notna(a_r) else "N/A"
        b_rs = f"{b_r:.3f}" if pd.notna(b_r) else "N/A"
        d_rs = f"{b_r-a_r:+.3f}" if (pd.notna(a_r) and pd.notna(b_r)) else "N/A"
        print(f"  {'Avg Rating (Exact Hits)':<30} {a_rs:>20} {b_rs:>20} {d_rs:>10}")
        for metric in ["hit", "precision", "avg_rating"]:
            s = sig.get(metric)
            if s:
                sig_str = "✓ SIGNIFICANT" if s["significant"] else "✗ not significant"
                print(f"  {metric} p={s['p']:.4f}  {sig_str}  ({s.get('direction','')})")
        winner = lb if b["hit_rate_history"] >= a["hit_rate_history"] else la
        print(f"  → Winner (history hit rate): {winner}")
 
    _print_mode(f"MODE 1 — Fair (Top-{p['top_n_fair']} each)",
                results["m1_a"], results["m1_b"], results["sig_m1"],
                f"Popularity Top-{p['top_n_fair']}", f"CBF Top-{p['top_n_fair']}")
    _print_mode(f"MODE 2 — Business (Pop Top-{p['top_n_a_biz']} vs CBF Top-{p['top_n_b_biz']})",
                results["m2_a"], results["m2_b"], results["sig_m2"],
                f"Popularity Top-{p['top_n_a_biz']}", f"CBF Top-{p['top_n_b_biz']}")
 
    m1w = "CBF" if results["m1_b"]["hit_rate"] >= results["m1_a"]["hit_rate"] else "Popularity"
    m2w = "CBF" if results["m2_b"]["hit_rate"] >= results["m2_a"]["hit_rate"] else "Popularity"
    print(f"\n  Both modes agree: {'YES — ' + m1w + ' wins both' if m1w == m2w else 'NO — Mode 1=' + m1w + ', Mode 2=' + m2w}")
 
    print(f"\n[SAVE]  Writing to {_OUT_DIR} ...")
    summary_rows = []
    for mode_label, a, b, sig, la, lb in [
        (f"Mode1_Fair_Top{p['top_n_fair']}",
         results["m1_a"], results["m1_b"], results["sig_m1"],
         f"Popularity_Top{p['top_n_fair']}", f"CBF_Top{p['top_n_fair']}"),
        (f"Mode2_Business_Pop{p['top_n_a_biz']}_CBF{p['top_n_b_biz']}",
         results["m2_a"], results["m2_b"], results["sig_m2"],
         f"Popularity_Top{p['top_n_a_biz']}", f"CBF_Top{p['top_n_b_biz']}"),
    ]:
        for grp, agg, lbl in [("Popularity", a, la), ("CBF", b, lb)]:
            summary_rows.append({
                "mode": mode_label, "model": lbl,
                "n_users": agg["n_users"],
                "hit_rate_exact":   round(agg["hit_rate"], 4),
                "hit_rate_history": round(agg["hit_rate_history"], 4),
                "precision":        round(agg["precision"], 4),
                "avg_rating":       round(agg["avg_rating"], 4) if pd.notna(agg["avg_rating"]) else None,
                "sig_hit_p":   round(sig.get("hit", {}).get("p", float("nan")), 6),
                "sig_hit":     sig.get("hit", {}).get("significant"),
            })
 
    pd.DataFrame(summary_rows).to_csv(_OUT_DIR / "eval_summary.csv", index=False)
    print("  → eval_summary.csv")
 
    for df, fname in [
        (results["df_m1a"], "eval_mode1_popularity.csv"),
        (results["df_m1b"], "eval_mode1_cbf.csv"),
        (results["df_m2a"], "eval_mode2_popularity.csv"),
        (results["df_m2b"], "eval_mode2_cbf.csv"),
        (results["eval_log"], "eval_log.csv"),
    ]:
        df.to_csv(_OUT_DIR / fname, index=False)
        print(f"  → {fname}  ({len(df):,} rows)")
 
    print(f"\n[DONE]\n{SEP}")