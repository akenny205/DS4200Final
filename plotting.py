"""
Visualization functions for Spotify Popularity Explorer
--------------------------------------------------------
1. make_popularity_by_group()  - box plot: popularity by genre group    (Plotly)
2. make_feature_importance()   - bar chart: RF feature importance       (Altair)
3. make_feature_heatmap()      - heatmap: avg features per genre group  (Altair)
4. make_scatter()              - scatter: any two features              (Plotly)
5. make_histogram()            - histogram: popularity distribution     (Plotly)

Table helper:
- make_table()
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import panel as pn

# Top features from analysis (drop energy + liveness — not significant)
TOP_FEATURES = [
    "acousticness", "valence", "danceability",
    "duration_ms", "speechiness", "instrumentalness",
    "loudness", "tempo"
]

GROUP_COLOR_MAP = {
    "Rock / Electronic":       "#e63946",
    "Pop / Folk / World":      "#f4a261",
    "Dance / Hip-Hop / Latin": "#2a9d8f",
    "Acoustic / Jazz / Chill": "#457b9d",
    "Metal / Hard Electronic": "#6a0572",
    "Classical / Ambient":     "#a8dadc",
    "Techno / IDM":            "#e9c46a",
    "Comedy":                  "#8d99ae",
}


# ── 1. Popularity by Genre Group (Plotly box plot) ───────────────────────────

def make_popularity_by_group(df: pd.DataFrame):
    if "genre_group" not in df.columns or df.empty:
        return go.Figure()

    order = (
        df.groupby("genre_group")["popularity"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig = px.box(
        df,
        x="genre_group",
        y="popularity",
        color="genre_group",
        category_orders={"genre_group": order},
        color_discrete_map=GROUP_COLOR_MAP,
        title="Popularity Distribution by Genre Group",
        labels={"genre_group": "Genre Group", "popularity": "Popularity"},
        points=False
    )
    fig.update_layout(
        height=450,
        showlegend=False,
        xaxis_tickangle=-25,
        margin=dict(l=20, r=20, t=50, b=80),
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font_color="white"
    )
    return fig


# ── 2. RF Feature Importance (Altair bar chart) ──────────────────────────────

def make_feature_importance(filepath="feature_importance.csv"):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return pn.pane.Str("feature_importance.csv not found. Run feature_analysis.py first.")

    # drop insignificant features
    df = df[~df["feature"].isin(["energy", "liveness"])]
    df = df.sort_values("rf_importance", ascending=False)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("rf_importance:Q",
                    title="Importance Score",
                    axis=alt.Axis(format=".2f")),
            y=alt.Y("feature:N",
                    sort="-x",
                    title="Feature"),
            color=alt.Color("rf_importance:Q",
                            scale=alt.Scale(scheme="greens"),
                            legend=None),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("rf_importance:Q", title="RF Importance", format=".4f"),
                alt.Tooltip("pearson_r:Q",    title="Correlation",   format=".4f"),
            ]
        )
        .properties(
            title="Which Audio Features Best Predict Popularity?",
            width=500,
            height=300
        )
    )
    return chart


# ── 3. Feature Heatmap per Genre Group (Altair) ──────────────────────────────

def make_feature_heatmap(df: pd.DataFrame):
    if "genre_group" not in df.columns or df.empty:
        return pn.pane.Str("No genre_group column found.")

    # normalize each feature to 0-1 so loudness doesn't dominate
    avail = [f for f in TOP_FEATURES if f in df.columns]
    melted = df.groupby("genre_group")[avail].mean().reset_index()

    for feat in avail:
        col_min, col_max = melted[feat].min(), melted[feat].max()
        if col_max > col_min:
            melted[feat] = (melted[feat] - col_min) / (col_max - col_min)

    melted = melted.melt(
        id_vars="genre_group",
        value_vars=avail,
        var_name="feature",
        value_name="normalized_mean"
    )

    chart = (
        alt.Chart(melted)
        .mark_rect()
        .encode(
            x=alt.X("feature:N", title="Audio Feature"),
            y=alt.Y("genre_group:N", title="Genre Group"),
            color=alt.Color(
                "normalized_mean:Q",
                scale=alt.Scale(scheme="viridis"),
                title="Normalized Mean"
            ),
            tooltip=[
                alt.Tooltip("genre_group:N",      title="Genre Group"),
                alt.Tooltip("feature:N",          title="Feature"),
                alt.Tooltip("normalized_mean:Q",  title="Value", format=".3f"),
            ]
        )
        .properties(
            title="Average Audio Features by Genre Group (Normalized)",
            width=500,
            height=280
        )
    )
    return chart


# ── 4. Scatter Plot (Plotly) ─────────────────────────────────────────────────

def make_scatter(df: pd.DataFrame, x_feature: str, y_feature: str):
    if df.empty:
        return go.Figure()

    color_col = "genre_group" if "genre_group" in df.columns else "track_genre"

    fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color=color_col,
        color_discrete_map=GROUP_COLOR_MAP,
        hover_data=["track_name", "artists"],
        opacity=0.5,
        title=f"{x_feature.title()} vs {y_feature.title()}",
        labels={x_feature: x_feature.title(), y_feature: y_feature.title()}
    )
    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font_color="white"
    )
    return fig


# ── 5. Popularity Histogram (Plotly) ─────────────────────────────────────────

def make_histogram(df: pd.DataFrame):
    if df.empty:
        return go.Figure()

    fig = px.histogram(
        df,
        x="popularity",
        nbins=40,
        title="Popularity Distribution",
        labels={"popularity": "Popularity Score", "count": "Track Count"},
        color_discrete_sequence=["#2a9d8f"]
    )
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font_color="white"
    )
    return fig


# ── Table helper ─────────────────────────────────────────────────────────────

def make_table(df: pd.DataFrame):
    if df.empty:
        return pn.pane.Str("No tracks match the current filters.")

    top = df.sort_values("popularity", ascending=False).head(10)
    cols = ["track_name", "artists", "genre_group", "popularity"]
    cols = [c for c in cols if c in top.columns]

    return pn.pane.DataFrame(top[cols], height=300)