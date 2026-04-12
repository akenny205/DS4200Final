"""
Spotify Data Handler
Handles loading, cleaning, genre grouping, and filtering.
"""

import pandas as pd
from genre_grouper import add_genre_groups

DATA_FILE_PATH = "dataset.csv"


class SpotifyAPI:

    def __init__(self, filename=DATA_FILE_PATH):
        self.df = pd.read_csv(filename)

    def process_data(self):
        self.df = self.df.dropna()
        self.df = add_genre_groups(self.df)  # adds 'genre_group' column

    def get_genres(self):
        return sorted(self.df["track_genre"].unique())

    def get_genre_groups(self):
        if "genre_group" not in self.df.columns:
            return []
        return sorted(self.df["genre_group"].unique())

    def filter_data(self,
                    genres=None,
                    genre_groups=None,
                    popularity=(0, 100),
                    energy=(0, 1),
                    danceability=(0, 1),
                    valence=(0, 1),
                    explicit=None,
                    sample_size=None):

        data = self.df.copy()

        # filter by broad genre group first, then narrow genre
        if genre_groups:
            data = data[data["genre_group"].isin(genre_groups)]
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

        if sample_size and sample_size < len(data):
            data = data.sample(sample_size, random_state=42)

        return data