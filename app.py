"""
UI Layer - Spotify Popularity Explorer

Tabs:
  1. Introduction  — project overview and story setup
  2. Initial Insights — popularity by genre group + feature importance
  3. Feature Deep Dive — heatmap + scatter
  4. Distribution  — popularity histogram
  5. Top Tracks    — filtered table
  6. Conclusions   — findings and takeaways
"""

import panel as pn
import spotify_api
import plotting as plt

CARD_WIDTH = 320

api = spotify_api.SpotifyAPI(spotify_api.DATA_FILE_PATH)

# ── Reusable style helpers ────────────────────────────────────────────────────

def section_text(body: str) -> pn.pane.Markdown:
    """Styled body text between visualizations."""
    return pn.pane.Markdown(
        body,
        styles={
            "background": "#1a1a1a",
            "border-left": "4px solid #2a9d8f",
            "padding": "12px 16px",
            "border-radius": "4px",
            "color": "#cccccc",
            "font-size": "14px",
            "line-height": "1.7"
        }
    )

def section_header(title: str) -> pn.pane.Markdown:
    """Consistent section headers."""
    return pn.pane.Markdown(
        f"### {title}",
        styles={"color": "#2a9d8f", "margin-top": "20px"}
    )

# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_filtered_data(genre_groups, genres, popularity, energy,
                      danceability, valence, explicit, sample_size):
    return api.filter_data(
        genres=genres,
        genre_groups=genre_groups,
        popularity=popularity,
        energy=energy,
        danceability=danceability,
        valence=valence,
        explicit=explicit,
        sample_size=sample_size
    )

def get_box(genre_groups, genres, popularity, energy,
            danceability, valence, explicit, sample_size):
    df = get_filtered_data(genre_groups, genres, popularity, energy,
                           danceability, valence, explicit, sample_size)
    return plt.make_popularity_by_group(df)

def get_heatmap(genre_groups, genres, popularity, energy,
                danceability, valence, explicit, sample_size):
    df = get_filtered_data(genre_groups, genres, popularity, energy,
                           danceability, valence, explicit, sample_size)
    return plt.make_feature_heatmap(df)

def get_scatter(genre_groups, genres, popularity, energy, danceability,
                valence, explicit, sample_size, x_feature, y_feature):
    df = get_filtered_data(genre_groups, genres, popularity, energy,
                           danceability, valence, explicit, sample_size)
    return plt.make_scatter(df, x_feature, y_feature)

def get_histogram(genre_groups, genres, popularity, energy,
                  danceability, valence, explicit, sample_size):
    df = get_filtered_data(genre_groups, genres, popularity, energy,
                           danceability, valence, explicit, sample_size)
    return plt.make_histogram(df)

def get_table(genre_groups, genres, popularity, energy,
              danceability, valence, explicit, sample_size):
    df = get_filtered_data(genre_groups, genres, popularity, energy,
                           danceability, valence, explicit, sample_size)
    return plt.make_table(df)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    pn.extension("plotly", "vega")
    api.process_data()

    # ── Widgets ───────────────────────────────────────────────────────────────

    genre_group_select = pn.widgets.MultiChoice(
        name="Genre Group", options=api.get_genre_groups()
    )
    genre_select = pn.widgets.MultiChoice(
        name="Specific Genre", options=api.get_genres()
    )
    popularity_slider = pn.widgets.RangeSlider(
        name="Popularity", start=0, end=100, value=(0, 100)
    )
    energy_slider = pn.widgets.RangeSlider(
        name="Energy", start=0.0, end=1.0, value=(0.0, 1.0)
    )
    dance_slider = pn.widgets.RangeSlider(
        name="Danceability", start=0.0, end=1.0, value=(0.0, 1.0)
    )
    valence_slider = pn.widgets.RangeSlider(
        name="Valence", start=0.0, end=1.0, value=(0.0, 1.0)
    )
    explicit_toggle = pn.widgets.RadioButtonGroup(
        name="Explicit Content",
        options={"All Songs": None, "Explicit Only": True, "Clean Only": False},
        value=None
    )
    sample_slider = pn.widgets.IntSlider(
        name="Sample Size", start=500, end=20000, step=500, value=5000
    )
    feature_options = [
        "acousticness", "valence", "danceability",
        "speechiness", "instrumentalness", "loudness", "tempo"
    ]
    x_axis = pn.widgets.Select(
        name="X Feature", options=feature_options, value="acousticness"
    )
    y_axis = pn.widgets.Select(
        name="Y Feature", options=feature_options + ["popularity"], value="popularity"
    )

    # ── Bindings ──────────────────────────────────────────────────────────────

    shared = [
        genre_group_select, genre_select, popularity_slider,
        energy_slider, dance_slider, valence_slider,
        explicit_toggle, sample_slider
    ]

    box             = pn.bind(get_box,       *shared)
    heatmap         = pn.bind(get_heatmap,   *shared)
    scatter         = pn.bind(get_scatter,   *shared, x_axis, y_axis)
    histogram       = pn.bind(get_histogram, *shared)
    table           = pn.bind(get_table,     *shared)
    importance_chart = plt.make_feature_importance()

    # ── Tab Content ───────────────────────────────────────────────────────────

    # Tab 1 — Introduction
    intro_tab = pn.Column(
        pn.pane.Markdown(
            "# What Makes a Song Popular on Spotify?",
            styles={"color": "#2a9d8f", "font-size": "22px"}
        ),
        section_text(
            """
            Every day, millions of songs compete for listener attention on Spotify.
            Some tracks go viral overnight while others with similar sounds never break through.
            **Is there a formula to popularity — or is it largely unpredictable?**

            This project explores over **113,000 Spotify tracks** spanning 114 genres to answer
            that question using audio features like acousticness, danceability, valence, and more.
            """
        ),
        pn.pane.Markdown("### Our Approach", styles={"color": "white"}),
        section_text(
            """
            **Step 1: Genre Grouping:** With 114 genres in the dataset, direct comparison is noisy.
            We used KMeans clustering on average audio feature profiles to group similar genres into
            8 broader categories (e.g. *Dance / Hip-Hop / Latin*, *Classical / Ambient*).

            **Step 2: Feature Analysis:** We ran Pearson correlations, linear regression, and a
            Random Forest regressor to rank which audio features best predict popularity.

            **Step 3: Visualization:** We built this interactive dashboard to explore the results
            and let you filter by genre group, audio characteristics, and more.

            Use the **sidebar filters** to narrow down the data at any point, then work through
            the tabs from left to right to follow the full story.
            """
        ),
        pn.pane.Markdown("### Dataset", styles={"color": "white"}),
        section_text(
            """
            - **Source:** Spotify Tracks Dataset (Kaggle)
            - **Size:** ~114,000 tracks across 114 genres
            - **Features:** danceability, energy, acousticness, valence, tempo, speechiness,
              instrumentalness, loudness, liveness, duration, explicit, popularity
            - **Note:** Popularity scores reflect a point-in-time snapshot and do not account
              for artist fame, release timing, or playlist placement — all of which likely
              matter more than audio features alone.
            """
        )
    )

    # Tab 2 — Initial Insights
    insights_tab = pn.Column(
        section_header("Step 1 — Does Genre Group Affect Popularity?"),
        section_text(
            """
            Before diving into individual features, we first ask: does the type of music matter?
            The box plot below shows the spread of popularity scores across our 8 genre groups.
            Look for differences in median popularity and the width of each box — a wider box
            means more variability within that group.
            """
        ),
        box,
        section_text(
            """
            **What we found:** Dance / Hip-Hop / Latin and Pop / Folk / World tend to have the
            highest median popularity, while Classical / Ambient and Techno / IDM skew lower.
            This suggests genre context matters — but the wide spreads in every group tell us
            genre alone doesn't determine a song's fate. Something else is at play.
            That leads us to our next question: which specific audio features are driving popularity?
            """
        ),
        section_header("Step 2 — Which Audio Features Predict Popularity?"),
        section_text(
            """
            We tested every audio feature against popularity using both linear regression and a
            Random Forest regressor. The chart below ranks features by their Random Forest
            importance score. The higher the bar, the more that feature contributed to the model's
            predictions. Hover over each bar to see the correlation and p-value as well.
            """
        ),
        importance_chart,
        section_text(
            """
            **What we found:** Acousticness, valence, and danceability are the top three predictors.
            Notably, acousticness and valence are *negatively* correlated with popularity, meaning
            louder, less acoustic, and more emotionally neutral songs tend to score higher.
            Energy and liveness were statistically insignificant (p > 0.05) and were excluded.
            Despite these signals, our models only explained about 16% of variance in popularity (R² = 0.16),
            suggesting audio features alone are far from the whole story.
            """
        )
    )

    # Tab 3 — Feature Deep Dive
    deep_tab = pn.Column(
        section_header("Step 3 — How Do Audio Profiles Differ Across Genre Groups?"),
        section_text(
            """
            Now that we know which features matter most, we can ask: *how do those features
            vary by genre group?* The heatmap below shows the normalized average of each audio
            feature per genre group. Brighter cells mean that group scores relatively high on
            that feature compared to others.
            """
        ),
        heatmap,
        section_text(
            """
            **What we found:** Classical / Ambient scores extremely high on acousticness and
            instrumentalness, consistent with its lower popularity. Dance / Hip-Hop / Latin
            leads on danceability and energy. These distinct profiles confirm our genre clusters
            are meaningful and reflect real sonic differences between groups.
            """
        ),
        section_header("Step 4 — Explore Any Two Features Directly"),
        section_text(
            """
            Use the **Scatter Settings** panel in the sidebar to choose any two features and
            see how they relate across the filtered dataset. Each point is a song, colored by
            genre group. Try plotting **acousticness vs popularity** or **danceability vs valence**
            to see the patterns the models identified, or explore combinations of your own.
            """
        ),
        scatter,
        section_text(
            """
            **Tip:** Narrow the Genre Group filter in the sidebar to isolate a single group
            and see how its feature relationships differ from the overall trend.
            """
        )
    )

    # Tab 4 — Distribution
    dist_tab = pn.Column(
        section_header("How is Popularity Distributed Across the Dataset?"),
        section_text(
            """
            Before drawing conclusions it is worth understanding the shape of our target variable.
            The histogram below shows how popularity scores are distributed across all filtered tracks.
            Use the sidebar to see how the shape changes when you filter by genre group or audio features.
            """
        ),
        histogram,
        section_text(
            """
            **What we found:** Popularity is heavily right-skewed — the vast majority of songs
            score below 40, with very few reaching the 80–100 range. This means truly popular
            songs are rare outliers, not the norm. It also means our models are predicting against
            a dataset dominated by obscure tracks, which partly explains the low R² scores.
            """
        )
    )

    # Tab 5 — Top Tracks
    table_tab = pn.Column(
        section_header("Top Tracks Under Your Current Filters"),
        section_text(
            """
            The table below shows the 10 highest-popularity tracks matching your current sidebar
            filters. Use the Genre Group or Specific Genre filters to drill into a particular
            corner of the dataset and see which songs rise to the top.
            """
        ),
        table
    )

    # Tab 6 — Conclusions
    conclusions_tab = pn.Column(
        pn.pane.Markdown(
            "# What Did We Learn?",
            styles={"color": "#2a9d8f", "font-size": "22px"}
        ),
        section_text(
            """
            **1. Genre context shapes popularity ceilings.**
            Dance, Hip-Hop, and Pop genre groups consistently produce the most popular tracks,
            while Classical and Ambient genres cluster at the lower end. If you are making music
            for reach, genre positioning matters.
            """
        ),
        section_text(
            """
            **2. Less acoustic, more danceable songs tend to be more popular — but the effect is modest.**
            Acousticness and instrumentalness are negatively associated with popularity, while
            danceability and loudness are positively associated. However, with R² = 0.16, audio
            features only explain a small fraction of what makes a song popular. External factors
            like artist fame, playlist placement, and release timing likely dominate.
            """
        ),
        section_text(
            """
            **3. Popularity is rare by design.**
            The heavily skewed popularity distribution reveals that most songs on Spotify go largely
            unheard. The dataset is dominated by mid-tier and obscure tracks, meaning the few
            breakout hits are genuine statistical outliers — not just songs that happened to sound
            a certain way.
            """
        ),
        pn.pane.Markdown("### What Could Be Done Next?", styles={"color": "#2a9d8f"}),
        section_text(
            """
            - **Add temporal data:** Knowing when a song was released and how its popularity
              changed over time would reveal whether certain audio trends rise and fall with eras.
            - **Include artist-level features:** Follower counts, prior hit history, and label
              affiliation would likely dramatically improve predictive power.
            - **Model by genre group separately:** A random forest trained only on Dance / Hip-Hop
              tracks might find very different feature importance patterns than one trained on all genres.
            """
        )
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────

    filter_card = pn.Card(
        pn.Column(
            genre_group_select, genre_select, popularity_slider,
            energy_slider, dance_slider, valence_slider,
            explicit_toggle, sample_slider
        ),
        title="Filters",
        width=CARD_WIDTH,
        collapsed=False
    )

    axis_card = pn.Card(
        pn.Column(x_axis, y_axis),
        title="Scatter Settings",
        width=CARD_WIDTH,
        collapsed=True
    )

    # ── Template ──────────────────────────────────────────────────────────────

    pn.template.FastListTemplate(
        title="Spotify Popularity Explorer",
        sidebar=[filter_card, axis_card],
        theme_toggle=False,
        header_background="black",
        main=[
            pn.Tabs(
                ("Introduction",    intro_tab),
                ("Initial Insights", insights_tab),
                ("Feature Deep Dive", deep_tab),
                ("Distribution",    dist_tab),
                ("Top Tracks",      table_tab),
                ("Conclusions",     conclusions_tab),
                active=0
            )
        ]
    ).servable()


main()