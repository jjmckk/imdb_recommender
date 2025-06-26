"""
recommender.py

A modular content-based recommender system for IMDb-style movie datasets.
Uses SBERT similarities and metadata such as genres, weighted ratings, and language
to generate personalized movie recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine

class Recommender:
    """
    A content-based recommender system for IMDb-style datasets using SBERT similarities and metadata.
    """

    def __init__(self, df, target_urn, system_col='sbert_scores', col_weights=None, language_col='original_language'):
        """
        Initializes the recommender with a target movie and prepares the DataFrame for scoring and filtering.

        Args:
            df (pd.DataFrame): DataFrame containing movie data.
            target_urn (int): URN of the target movie to generate recommendations for.
            system_col (str): Column containing precomputed similarity scores.
            col_weights (dict): Weights for combining individual feature scores.
            language_col (str): Column for original language metadata.
        """
        print("""
****************************************************
Initialising Recommender System
----------------------------------------------------
Target URN: {}
System Similarity Column: {}
Language Column: {}
****************************************************
""".format(target_urn, system_col, language_col))

        self.df = df.copy()
        self.target_urn = target_urn
        self.system_col = system_col
        self.language_col = language_col

        if not col_weights:
            col_weights = {
                'overview_similarity': 5,
                'genre_similarity': 3,
                'vote_weighted_scaled': 2,
                'language_match': 1
            }
        self.col_weights = col_weights

        all_genres = [
            "action", "adventure", "animation", "anime", "biography", "comedy", "crime",
            "documentary", "drama", "family", "fantasy", "film_noir", "game_show", "history",
            "horror", "lifestyle", "music", "musical", "mystery", "reality_tv", "romance",
            "sci_fi", "seasonal", "short", "sport", "thriller", "war", "western"
        ]
        self.genre_cols = [x for x in self.df.columns if x in all_genres]

        self.target_language = self.df[self.df['urn'] == self.target_urn][self.language_col].values[0]
        self.genre_base_vector = self.df[self.df['urn'] == self.target_urn][self.genre_cols].values[0]

        print("Filtering DataFrame by similarity...")
        self._filter_by_similarity()

    def _filter_by_similarity(self):
        """
        Filters the DataFrame to only include movies that have similarity scores with the target,
        and adds the 'overview_similarity' column.
        """
        recommendations = self.df[self.df['urn'] == self.target_urn][self.system_col].values[0]
        urns = [x[0] for x in recommendations]
        scores = [x[1] for x in recommendations]

        urns.insert(0, self.target_urn)
        scores.insert(0, 1.0)

        score_lookup = dict(zip(urns, scores))

        self.df = self.df[self.df['urn'].isin(urns)].copy()
        self.df['overview_similarity'] = self.df['urn'].map(score_lookup)
        self.df.drop(columns=[self.system_col], inplace=True)

    def _apply_language_match(self):
        """
        Adds a binary 'language_match' column indicating whether the movie matches the target language.
        """
        print("\n* Applying Language Match...")
        self.df['language_match'] = self.df[self.language_col].apply(lambda x: float(x == self.target_language))
        self.df.drop(columns=[self.language_col], inplace=True)

    def _apply_genre_similarity(self):
        """
        Computes cosine similarity between genre vectors and adds a 'genre_similarity' column.
        """
        print("\n* Computing Genre Similarity...")
        self.df['genre_similarity'] = self.df[self.genre_cols].apply(
            lambda row: 1 - cosine(row.values, self.genre_base_vector), axis=1
        )
        self.df.drop(columns=self.genre_cols, inplace=True)

    def _apply_vote_scaling(self):
        """
        Scales the 'vote_weighted' column to [0, 1] and stores it as 'vote_weighted_scaled'.
        """
        print("\n* Scaling Vote Weight...")
        scaler = MinMaxScaler()
        self.df['vote_weighted_scaled'] = scaler.fit_transform(self.df[['vote_weighted']])
        self.df.drop(columns=['vote_weighted'], inplace=True)

    def _apply_weighted_score(self):
        """
        Combines all prioritisation features into a single 'rec_score' using the assigned weights.
        """
        print("\n* Calculating Weighted Recommendation Score...")
        self.df['rec_score'] = (
            self.df['overview_similarity'] * self.col_weights['overview_similarity'] +
            self.df['genre_similarity'] * self.col_weights['genre_similarity'] +
            self.df['vote_weighted_scaled'] * self.col_weights['vote_weighted_scaled'] +
            self.df['language_match'] * self.col_weights['language_match']
        )

    def show_recommendations(self, top_n=None):
        """
        Returns a table of recommended movies ranked by final weighted score.

        Args:
            top_n (int or None): Number of top recommendations to return. Returns all if None.

        Returns:
            pd.DataFrame: Sorted recommendations table.
        """
        print("""
****************************************************
Running Recommender Steps
----------------------------------------------------
1. Language Match
2. Genre Similarity
3. Vote Scaling
4. Weighted Score Calculation
****************************************************
""")
        self._apply_language_match()
        self._apply_genre_similarity()
        self._apply_vote_scaling()
        self._apply_weighted_score()

        col_order = [
            'urn', 'title', 'overview', 'overview_similarity',
            'genre_similarity', 'vote_weighted_scaled', 'language_match', 'rec_score'
        ]
        self.df = self.df[col_order].sort_values(by='rec_score', ascending=False)

        print("\nRecommendation process complete.")
        return self.df.head(top_n).reset_index(drop=True) if top_n else self.df.reset_index(drop=True)

# Example test run
if __name__ == "__main__":
    # Dummy test data
    import pickle

    # Assume processed_df is a previously saved DataFrame containing all required columns
    with open("test_data/processed_df.pkl", "rb") as f:
        processed_df = pickle.load(f)

    target_urn = 1
    top_n = 3
    recommender = Recommender(processed_df, target_urn=target_urn, system_col='sbert_scores')
    recommendations = recommender.show_recommendations(top_n=top_n)

    print("\nTop Recommendations:")
    print(recommendations)
