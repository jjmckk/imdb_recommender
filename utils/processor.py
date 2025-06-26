
"""
rec_processor.py

A preprocessing utility for movie-based recommendation systems using embeddings and zero-shot classification.

This module defines the RecProcessor class, which prepares a movie dataset by:
- Assigning proper data types
- Generating unique resource numbers (URNs)
- Applying weighted vote scoring
- Embedding movie overviews with Sentence-BERT
- Computing cosine similarity between overviews
- Identifying top-N most similar movies for each entry
- Using a zero-shot classifier to score genre relevance

It is intended for use in content-based movie recommender pipelines.

Author: [Your Name]
Date: [Date]
"""

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple, Union
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers.utils import logging

#turn logging off
logging.set_verbosity_error()

#enable tqdm.pandas
tqdm.pandas()

class RecProcessor:
    """
    A processing class for preparing movie data for a content-based recommender system using SBERT and zero-shot classification.
    """

    def __init__(
        self, 
        df: pd.DataFrame, 
        embedding_model_name: str = None,
        classification_model_name: str = None
    ):
        """
        Initialize the RecProcessor with the provided DataFrame and optional model names.

        Args:
            df (pd.DataFrame): Input DataFrame containing movie data.
            embedding_model_name (str, optional): Name of the SBERT model. Defaults to 'all-MiniLM-L6-v2'.
            classification_model_name (str, optional): Name of the classification model. Defaults to 'facebook/bart-large-mnli'.

        Output:
            None
        """
        self.df = df.copy()
        self.device = 0 if torch.cuda.is_available() else -1
        print("*" * 60)
        if self.device == 0:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Device Setup] CUDA detected → using GPU: {gpu_name}")
        else:
            print("[Device Setup] CUDA not available → using CPU")
        print("*" * 60 + "\n")

        self.embedding_model_name = embedding_model_name or 'all-MiniLM-L6-v2'
        self.classification_model_name = classification_model_name or 'facebook/bart-large-mnli'

    def assign_data_types(self, type_dic: Dict[str, str] = None):
        """
        Assigns data types to key columns.

        Args:
            type_dic (dict, optional): Dictionary mapping column names to data types.
                Defaults to standard schema for title, overview, language, vote counts and averages.

        Output:
            None
        """
        print("-" * 50)
        print("[Stage] Assigning data types...")
        print("-" * 50)
        if type_dic is None:
            type_dic = {
                'title': 'string',
                'overview': 'string',
                'original_language': 'string',
                'vote_count': 'float64',
                'vote_average': 'float64'
            }
        self.df = self.df.astype(type_dic)

    def create_urn(self, drop_index: bool = True):
        """
        Creates a unique row identifier for each movie.

        Args:
            drop_index (bool): Whether to reset the index before creating the URN. Defaults to True.

        Output:
            None
        """
        print("\n" + "-" * 50)
        print("[Stage] Creating URNs for each row...")
        print("-" * 50)
        if drop_index:
            self.df = self.df.reset_index(drop=True)
        self.df['urn'] = self.df.index

    def apply_weighted_rating(self, average: str = 'mean', percentile: float = 0.5):
        """
        Applies IMDB-style weighted rating to each row.

        Args:
            average (str): Either 'mean' or 'median' to define central tendency of vote_average.
            percentile (float): Percentile threshold for vote_count cutoff (e.g., 0.5 = median).

        Output:
            None (adds 'vote_weighted' column to df)
        """
        print("\n" + "-" * 50)
        print("[Stage] Calculating weighted vote scores...")
        print("-" * 50)
        C = self.df['vote_average'].mean() if average == 'mean' else self.df['vote_average'].median()
        m = self.df['vote_count'].quantile(percentile)
        self.df['vote_weighted'] = self.df.apply(
            lambda row: (row['vote_count'] / (row['vote_count'] + m)) * row['vote_average'] +
                        (m / (row['vote_count'] + m)) * C, axis=1)
        self.df.drop(columns=['vote_average', 'vote_count'], inplace=True)

    def clean_overviews_for_transformer(self):
        """
        Strips whitespace from the 'overview' column and creates a new column for model input.

        Output:
            None (adds 'overview_transformer' column)
        """
        print("\n" + "-" * 50)
        print("[Stage] Cleaning overview text...")
        print("-" * 50)
        self.df['overview_transformer'] = self.df['overview'].str.strip()

    def extract_sbert_vectors(self) -> np.ndarray:
        """
        Encodes the movie overviews using SBERT.

        Output:
            np.ndarray: Matrix of embedding vectors.
        """
        print("\n" + "-" * 50)
        print(f"[Stage] Encoding overviews with SBERT → {self.embedding_model_name}")
        print("-" * 50)
        model = SentenceTransformer(self.embedding_model_name, device=self.device)
        assert 'overview_transformer' in self.df.columns, "'overview_transformer' column missing"
        self.sbert_vectors = model.encode(self.df['overview_transformer'].tolist(), show_progress_bar=True)
        return self.sbert_vectors

    def compute_cosine_similarity_matrix(self) -> np.ndarray:
        """
        Computes cosine similarity between SBERT vectors.

        Output:
            np.ndarray: Cosine similarity matrix.
        """
        print("\n" + "-" * 50)
        print("[Stage] Computing cosine similarity matrix...")
        print("-" * 50)
        self.similarity_matrix = cosine_similarity(self.sbert_vectors, self.sbert_vectors)
        return self.similarity_matrix

    def get_top_n_sim(self, sim_scores: np.ndarray, top_n: int = 10, return_scores: bool = True) -> Union[List[int], List[Tuple[int, float]]]:
        """
        Retrieves top-N most similar movies.

        Args:
            sim_scores (np.ndarray): Similarity scores for a single item.
            top_n (int): Number of top similar items to return. Defaults to 10.
            return_scores (bool): Whether to return similarity scores along with indices.

        Output:
            List[int] or List[Tuple[int, float]]: List of top similar indices or (index, score) pairs.
        """
        urn_sim_scores = list(enumerate(sim_scores))
        sorted_sim_scores = sorted(urn_sim_scores, key=lambda x: x[1], reverse=True)
        top_n_sim_scores = sorted_sim_scores[1:top_n+1]
        non_zero_sim_scores = [x for x in top_n_sim_scores if x[1] != 0]
        return non_zero_sim_scores if return_scores else [x[0] for x in non_zero_sim_scores]

    def add_closest_sim(self, destination_col_name: str = 'sbert_scores', top_n: int = 10):
        """
        Adds a column with top-N most similar movies for each row.

        Args:
            destination_col_name (str): Name of the column to store similarity results.
            top_n (int): Number of similar items to store. Defaults to 10.

        Output:
            None (modifies dataframe in-place)
        """
        print("\n" + "-" * 50)
        print(f"[Stage] Finding top-{top_n} most similar overviews per movie...")
        print("-" * 50)
        assert self.similarity_matrix.shape[0] == self.df.shape[0], "Matrix and DataFrame mismatch"
        self.df[destination_col_name] = None
        for idx in tqdm(self.df.index, desc=f"[Progress] Generating '{destination_col_name}'"):
            sim_scores = self.similarity_matrix[idx]
            self.df.at[idx, destination_col_name] = self.get_top_n_sim(sim_scores=sim_scores, top_n=top_n)

    def _genre_scores_urn_row(self, row, classifier, genres):
        """
        Internal helper to score genres for a single row using zero-shot classification.

        Args:
            row (pd.Series): Row from the DataFrame.
            classifier (Pipeline): Hugging Face zero-shot classifier.
            genres (List[str]): List of candidate genres.

        Output:
            dict: Mapping of genre to score with row URN.
        """
        result = classifier(row['overview'], candidate_labels=genres, multi_label=True)
        genre_dict = {label: score for label, score in zip(result['labels'], result['scores'])}
        genre_dict['urn'] = row['urn']
        return genre_dict

    def extract_genre_scores(self, genres: List[str]):
        """
        Adds genre scores to the DataFrame using zero-shot classification.

        Args:
            genres (List[str]): List of genre labels to score.

        Output:
            None (adds genre score columns to df)
        """
        print("\n" + "-" * 50)
        print(f"[Stage] Extracting genre scores using zero-shot classification → {self.classification_model_name}")
        print("-" * 50)
        assert 'urn' in self.df.columns and 'overview' in self.df.columns, "df must contain 'urn' and 'overview'"
        assert isinstance(genres, list) and genres, "genres must be a non-empty list"

        classifier = pipeline("zero-shot-classification", model=self.classification_model_name, device=self.device)

        print("[Progress] Running classification across all rows...")
        genre_score_dicts = self.df.progress_apply(
            lambda row: self._genre_scores_urn_row(row, classifier, genres), axis=1
        )

        scores_df = pd.DataFrame(genre_score_dicts.tolist())
        scores_df.columns = (
            scores_df.columns
            .str.lower()
            .str.replace(' ', '_')
            .str.replace('-', '_')
        )

        self.df = self.df.merge(scores_df, on='urn', how='left')

    def run_pipeline(
        self, 
        genres: List[str]=None, 
        rating_average: str = 'mean',
        rating_percentile: float = 0.5,
        sim_top_n: int = 10,
        return_df: bool = True
    ) -> Union[pd.DataFrame, None]:
        """
        Runs the full processing pipeline, applying all transformations and model predictions.

        Args:
            genres (List[str]): List of genres to classify.
            rating_average (str): 'mean' or 'median' for central vote average. Defaults to 'mean'.
            rating_percentile (float): Percentile for vote count threshold. Defaults to 0.5.
            sim_top_n (int): Number of top similar movies to record. Defaults to 10.
            return_df (bool): Whether to return the transformed DataFrame. Defaults to True.

        Output:
            pd.DataFrame or None: Final processed DataFrame or None if return_df is False.
        """
        if not genres:
            genres = [
                "action", "adventure", "animation", "anime", "biography", "comedy", "crime",
                "documentary", "drama", "family", "fantasy", "film_noir", "game_show", "history",
                "horror", "lifestyle", "music", "musical", "mystery", "reality_tv", "romance",
                "sci_fi", "seasonal", "short", "sport", "thriller", "war", "western"
            ]

        print("\n" + "*" * 60)
        print("        Running RecProcessor Full Pipeline")
        print("*" * 60)

        self.assign_data_types()
        self.create_urn()
        self.apply_weighted_rating(average=rating_average, percentile=rating_percentile)
        self.clean_overviews_for_transformer()
        self.extract_sbert_vectors()
        self.compute_cosine_similarity_matrix()
        self.add_closest_sim(top_n=sim_top_n)
        self.extract_genre_scores(genres=genres)

        print("\n" + "*" * 60)
        print("        ✅ Pipeline complete — DataFrame ready")
        print("*" * 60 + "\n")

        return self.df if return_df else None


# Example usage
if __name__ == "__main__":
    print("Testing RecProcessor...")

    mock_data = pd.DataFrame({
        'title': ['Movie A', 'Movie B', 'Movie C'],
        'overview': ['A space saga.', 'A romantic drama.', 'A sci-fi action thriller.'],
        'original_language': ['en', 'en', 'fr'],
        'vote_count': [120, 85, 300],
        'vote_average': [7.8, 6.5, 8.2]
    })

    genres = ['sci_fi', 'romance', 'drama', 'action']
    processor = RecProcessor(mock_data)
    processed_df = processor.run_pipeline(genres=genres)
    print(processed_df.head())
