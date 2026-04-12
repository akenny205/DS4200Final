import pandas as pd
import panel as pn
import plotly.express as px

pn.extension("plotly")

# load data
df = pd.read_csv("dataset.csv")

# Optional: reduce size if very large
df = df.sample(10000, random_state=42)

# API Layer

class SpotifyAPI:
    def __init__(self, df):
        self.df = df

    def filter_data(self, genres, popularity, energy,
                    danceability, valence, explicit):

        data = self.df.copy()

        if genres:
            data = data[data["track_genre"].isin(genres)]

        data = data[
            (data["popularity"] >= popularity[0]) &
            (data["popularity"] <= popularity[1]) &
            (data["energy"] >= energy[0]) &
            (data["energy"] <= energy[1]) &
            (data["danceability"] >= danceability[0]) &
            (data["danceability"] <= danceability[1]) &
            (data["valence"] >= valence[0]) &
            (data["valence"] <= valence[1])
        ]

        if explicit is not None:
            data = data[data["explicit"] == explicit]

        return data


api = SpotifyAPI(df)


# Widgets

genre_options = sorted(df["track_genre"].unique())

genre_select = pn.widgets.MultiSelect(
    name="Genre",
    options=genre_options,
    size=8
)

popularity_slider = pn.widgets.RangeSlider(
    name="Popularity",
    start=0, end=100, value=(0, 100)
)

energy_slider = pn.widgets.RangeSlider(
    name="Energy",
    start=0.0, end=1.0, value=(0.0, 1.0), step=0.01
)

dance_slider = pn.widgets.RangeSlider(
    name="Danceability",
    start=0.0, end=1.0, value=(0.0, 1.0), step=0.01
)

valence_slider = pn.widgets.RangeSlider(
    name="Valence",
    start=0.0, end=1.0, value=(0.0, 1.0), step=0.01
)

explicit_toggle = pn.widgets.Select(
    name="Explicit",
    options=[None, True, False],
    value=None
)

feature_options = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "speechiness",
    "loudness"
]

x_axis_select = pn.widgets.Select(
    name="X-Axis Feature",
    options=feature_options,
    value="danceability"
)

y_axis_select = pn.widgets.Select(
    name="Y-Axis Feature",
    options=feature_options + ["popularity"],
    value="popularity"
)

# -----------------------------
# Plot Functions
# -----------------------------
@pn.depends(
    genre_select,
    popularity_slider,
    energy_slider,
    dance_slider,
    valence_slider,
    explicit_toggle,
    x_axis_select,
    y_axis_select
)
def scatter_plot(genres, popularity, energy,
                 danceability, valence, explicit,
                 x_feature, y_feature):

    filtered = api.filter_data(
        genres, popularity, energy,
        danceability, valence, explicit
    )

    fig = px.scatter(
        filtered,
        x=x_feature,
        y=y_feature,
        color="track_genre",
        hover_data=["track_name", "artists"],
        opacity=0.6
    )

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


@pn.depends(
    genre_select,
    popularity_slider,
    energy_slider,
    dance_slider,
    valence_slider,
    explicit_toggle
)
def popularity_histogram(genres, popularity, energy,
                         danceability, valence, explicit):

    filtered = api.filter_data(
        genres, popularity, energy,
        danceability, valence, explicit
    )

    fig = px.histogram(
        filtered,
        x="popularity",
        nbins=30
    )

    fig.update_layout(height=300)

    return fig


@pn.depends(
    genre_select,
    popularity_slider,
    energy_slider,
    dance_slider,
    valence_slider,
    explicit_toggle
)
def top_tracks_table(genres, popularity, energy,
                     danceability, valence, explicit):

    filtered = api.filter_data(
        genres, popularity, energy,
        danceability, valence, explicit
    )

    top = filtered.sort_values(
        by="popularity", ascending=False
    ).head(10)

    return pn.pane.DataFrame(
        top[["track_name", "artists", "popularity"]],
        height=250
    )

# -----------------------------
# Layout
# -----------------------------
controls = pn.Column(
    "## Controls",
    genre_select,
    popularity_slider,
    energy_slider,
    dance_slider,
    valence_slider,
    explicit_toggle,
    x_axis_select,
    y_axis_select,
    width=300
)

main_panel = pn.Column(
    "## What Makes a Song Popular?",
    scatter_plot,
    popularity_histogram,
    top_tracks_table
)

dashboard = pn.Row(controls, main_panel)

dashboard.servable()
