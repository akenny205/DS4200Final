"""
Genre Grouper
-------------
Step 1: Auto-clusters genres by their mean audio features using KMeans.
Step 2: Prints cluster contents so you can rename them.
Step 3: Adds a 'genre_group' column to the dataframe.

Usage:
    from genre_grouper import add_genre_groups
    df = add_genre_groups(df)
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Features used to cluster genres - we used audio features only
CLUSTER_FEATURES = [
    "danceability", "energy", "acousticness",
    "valence", "tempo", "speechiness", "instrumentalness", "loudness"
]

GENRE_GROUP_NAMES = {
    0: "Techno / IDM",
    1: "Pop / Folk / World",
    2: "Metal / Hard Electronic",
    3: "Classical / Ambient",
    4: "Comedy",
    5: "Rock / Electronic",
    6: "Dance / Hip-Hop / Latin",
    7: "Acoustic / Jazz / Chill",
}

N_CLUSTERS = 8


def build_genre_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the mean audio feature vector for each genre."""
    return df.groupby("track_genre")[CLUSTER_FEATURES].mean()


def cluster_genres(genre_profiles: pd.DataFrame, n_clusters: int) -> pd.Series:
    """Fit KMeans on genre profiles, return a Series: genre -> cluster_id."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(genre_profiles)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(scaled)

    return pd.Series(km.labels_, index=genre_profiles.index, name="cluster_id")


def print_clusters(cluster_series: pd.Series) -> None:
    """Print each cluster and the genres inside it."""
    print("\n" + "="*55)
    print("  GENRE CLUSTERS  —  fill in GENRE_GROUP_NAMES above")
    print("="*55)
    for cid in sorted(cluster_series.unique()):
        genres = sorted(cluster_series[cluster_series == cid].index.tolist())
        print(f"\nCluster {cid}:")
        for g in genres:
            print(f"    {g}")
    print("\n" + "="*55)
    print("Re-run after filling in GENRE_GROUP_NAMES to add the column.")
    print("="*55 + "\n")


def add_genre_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point.
    - First run (GENRE_GROUP_NAMES empty): prints clusters and returns df unchanged.
    - After you fill GENRE_GROUP_NAMES: adds 'genre_group' column and returns df.
    """
    profiles = build_genre_profiles(df)
    cluster_series = cluster_genres(profiles, N_CLUSTERS)

    if not GENRE_GROUP_NAMES:
        print_clusters(cluster_series)
        return df  # nothing added yet — go fill in the names first

    # Validate that all cluster IDs are covered
    found = set(cluster_series.unique())
    named = set(GENRE_GROUP_NAMES.keys())
    missing = found - named
    if missing:
        raise ValueError(
            f"GENRE_GROUP_NAMES is missing cluster IDs: {missing}\n"
            f"Run with empty GENRE_GROUP_NAMES first to see all clusters."
        )

    # Build genre -> group_name mapping
    genre_to_group = (
        cluster_series
        .map(GENRE_GROUP_NAMES)
        .rename("genre_group")
    )

    df = df.copy()
    df["genre_group"] = df["track_genre"].map(genre_to_group)

    # Summary
    print("\nGenre groups added. Distribution:")
    print(df["genre_group"].value_counts().to_string())
    print()

    return df


if __name__ == "__main__":
    df = pd.read_csv("dataset.csv").dropna()
    df = add_genre_groups(df)