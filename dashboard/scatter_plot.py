import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

TOP_ORDER = ["Top1", "Top2", "Top3", "Top4", "Top5"]


# ---------- LOAD ----------
def load_scatter_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path)

    df = df.rename(columns={
        "X_MaxCosSim": "MaxCosine",
        "Y_PredRating": "Predicted_Rating"
    })

    return df.dropna()


# ---------- PLOT ----------
def create_scatter_plot(df: pd.DataFrame):

    top = df[df["Group"].isin(TOP_ORDER)]
    near = df[df["Group"] == "Near"]
    far = df[df["Group"] == "Far"]
    random_pts = df[df["Group"] == "Random"]

    top["order"] = top["Group"].map({g: i for i, g in enumerate(TOP_ORDER)})
    top = top.sort_values("order")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=random_pts["MaxCosine"],
        y=random_pts["Predicted_Rating"],
        mode="markers",
        name="Random",
        marker=dict(size=6, color="rgba(120,120,120,0.25)"),
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=near["MaxCosine"],
        y=near["Predicted_Rating"],
        mode="markers",
        name="Near",
        marker=dict(size=10, color="green")
    ))

    fig.add_trace(go.Scatter(
        x=far["MaxCosine"],
        y=far["Predicted_Rating"],
        mode="markers",
        name="Far",
        marker=dict(size=10, color="red")
    ))

    # Top 5 glow
    fig.add_trace(go.Scatter(
        x=top["MaxCosine"],
        y=top["Predicted_Rating"],
        mode="markers",
        name="Top Glow",
        marker=dict(size=26, color="rgba(59,130,246,0.25)"),
        hoverinfo="skip",
        showlegend=False
    ))

    # Top 5 line (IMPORTANT FIXED)
    fig.add_trace(go.Scatter(
        x=top["MaxCosine"],
        y=top["Predicted_Rating"],
        mode="lines+markers+text",
        text=top["DisplayLabel"],
        textposition="top center",
        name="Top 5",
        line=dict(color="#3b82f6", width=3),
        marker=dict(size=14, color="#3b82f6")
    ))

    fig.update_layout(
        title="Recommendation Scatter Plot",
        height=650
    )

    return fig


# ---------- TAB (LIKE USERS TAB STYLE) ----------
def show_scatter_tab():

    st.header("📊 Scatter Plot & Recommendations")

    file_path = Path(__file__).parent / "data" / "EShop_Product_Recommendations_Scatterplot_Inputs.xlsx"

    df = load_scatter_data(file_path)

    user_id = st.text_input("Search by User ID")

    # SAME STYLE AS USERS TAB → SIMPLE FLOW
    if not user_id:
        st.info("Enter a User ID to view recommendations")
        return

    plot_df = df[df["User_ID"] == user_id]

    if plot_df.empty:
        st.warning("No data found for this user")
        return

    # ---------- TABLE (FIXED - NOW ALWAYS SHOWS) ----------
    st.subheader("Top 5 Product Recommendations")

    top = plot_df[plot_df["Group"].isin(TOP_ORDER)].copy()

    if not top.empty:
        top["order"] = top["Group"].map({g: i for i, g in enumerate(TOP_ORDER)})
        top = top.sort_values("order")

        st.dataframe(
            top[["DisplayLabel", "MaxCosine", "Predicted_Rating"]],
            use_container_width=True,
            hide_index=True
        )

    # ---------- SCATTER ----------
    fig = create_scatter_plot(plot_df)
    st.plotly_chart(fig, use_container_width=True)