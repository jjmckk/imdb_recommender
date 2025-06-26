"""
main.py

Main entry point for the IMDb-style movie recommendation pipeline.
This script:
1. Loads a movie dataset,
2. Processes the data using RecProcessor (embedding, similarity, and genre classification),
3. Runs a content-based recommender using SBERT embeddings and metadata,
4. Displays the top N movie recommendations.
"""

import pandas as pd
import numpy as np
from utils.processor import RecProcessor
from utils.recommender import Recommender

def main():
    # Load dataset locally
    path = r"./TMDB_updated.csv"
    df = pd.read_csv(path)

    # Create test dataset
    df = df.head(20).copy()

    # Process dataframe
    processor = RecProcessor(df)
    processed_df = processor.run_pipeline()

    # Generate target film for recommender
    # Specific title
    title = 'Cold Blood'
    target_urn = processed_df.loc[processed_df['title'] == title]['urn'].values[0]

    # Run recommender
    top_n = 3
    recommender = Recommender(processed_df, target_urn=target_urn, system_col='sbert_scores')
    recommendations = recommender.show_recommendations(top_n=top_n)

    # Display recommendations
    print("\nTop Recommendations:")
    print(recommendations)

if __name__ == "__main__":
    main()
