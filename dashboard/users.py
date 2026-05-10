import streamlit as st


def show_users_tab(
    filtered_reviews,
    users,
    pct,
    top_share,
    human_int
):

    st.markdown("### User concentration and behaviour")

    user_counts = (
        filtered_reviews.groupby("user_id", as_index=False)
        .size()
        .rename(columns={"size": "filtered_review_count"})
        .merge(users, on="user_id", how="left")
        .sort_values(
            "filtered_review_count",
            ascending=False
        )
    )

    u1, u2, u3, u4 = st.columns(4)

    counts_series = user_counts[
        "filtered_review_count"
    ]

    u1.metric(
        "Top 1% user share",
        pct(top_share(counts_series, 0.01))
    )

    u2.metric(
        "Top 5% user share",
        pct(top_share(counts_series, 0.05))
    )

    u3.metric(
        "Median reviews per active user",
        f"{counts_series.median():.0f}"
    )

    u4.metric(
        "Most active user",
        human_int(counts_series.max())
    )

    st.markdown(
        "#### Most active users under current filters"
    )

    st.data_editor(
        user_counts.head(250),
        use_container_width=True,
        hide_index=True,
        num_rows="fixed"
    )
    
    
    
    
