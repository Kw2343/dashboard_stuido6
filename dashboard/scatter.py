from pathlib import Path
import streamlit as st

from scatter_plot import (
    load_scatter_data,
    create_scatter_plot
)

TOP_ORDER = ['Top1','Top2','Top3','Top4','Top5']


def show_scatter_tab():

    st.header("📊 Product Recommendation Scatter Plot")

    SCATTER_FILE = (
        Path(__file__).parent
        / "data"
        / "EShop_Product_Recommendations_Scatterplot_Inputs.xlsx"
    )

    df = load_scatter_data(SCATTER_FILE)

    user_input = st.text_input(
        "Search by User ID",
        placeholder="Enter User ID..."
    )

    if user_input.strip() == "":
        st.info("Enter a User ID to view recommendations.")
        return

    plot_df = df[
        df["User_ID"].astype(str) == user_input.strip()
    ].copy()

    if plot_df.empty:
        st.warning("No data found for this user.")
        return

    top = plot_df[
        plot_df["Group"].isin(TOP_ORDER)
    ].copy()

    if not top.empty:

        st.subheader("Top 5 Product Recommendations")

        top["order"] = top["Group"].map({
            "Top1": 1,
            "Top2": 2,
            "Top3": 3,
            "Top4": 4,
            "Top5": 5
        })

        top = top.sort_values("order")

        table_df = top[[
            "DisplayLabel",
            "MaxCosine",
            "Predicted_Rating"
        ]].rename(columns={
            "DisplayLabel": "Product",
            "MaxCosine": "Cosine Similarity",
            "Predicted_Rating": "Predicted Rating"
        })

        table_df["Cosine Similarity"] = (
            table_df["Cosine Similarity"].round(3)
        )

        table_df["Predicted Rating"] = (
            table_df["Predicted Rating"].round(2)
        )

        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            height=260
        )

    st.markdown("<br>", unsafe_allow_html=True)

    fig = create_scatter_plot(plot_df)

    st.plotly_chart(
        fig,
        use_container_width=True
    )