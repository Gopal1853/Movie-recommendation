import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from datetime import datetime
from typing import Dict, List, Union, Optional
import pathlib

class MovieSearchEngine:
    def __init__(self, 
                 movies_path: str = 'movies.csv',
                 genome_scores_path: str = 'genome-scores.csv',
                 genome_tags_path: str = 'genome-tags.csv',
                 ratings_path: str = 'ratings.csv',
                 vectorizer_path: str = 'model/vectorizer.pkl',
                 tfidf_path: str = 'model/tfidf_matrix.pkl'):
        
        # Validate file paths
        self._validate_file_paths(movies_path, genome_scores_path, genome_tags_path, ratings_path)
        
        # Load all datasets with error handling
        try:
            self.movies = pd.read_csv(movies_path)
            self.genome_scores = pd.read_csv(genome_scores_path)
            self.genome_tags = pd.read_csv(genome_tags_path)
            self.ratings = pd.read_csv(ratings_path)
        except Exception as e:
            raise ValueError(f"Error loading datasets: {e}")
        
        # Validate required columns
        self._validate_required_columns()
        
        # Extract year from title and convert to integer
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)').astype(float)
        self.movies['clean_title'] = self.movies['title'].apply(self.clean_title)

        # Create genre list
        self.movies['genre_list'] = self.movies['genres'].str.split('|')
        
        # Calculate average ratings
        self.movie_ratings = self.ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
        self.movie_ratings.columns = ['movieId', 'avg_rating', 'rating_count']
        
        # Merge ratings with movies
        self.movies = self.movies.merge(self.movie_ratings, on='movieId', how='left')
        
        # Initialize TF-IDF with secure model loading
        self._initialize_tfidf(vectorizer_path, tfidf_path)

    def _validate_file_paths(self, *paths: str) -> None:
        """Validate that all required files exist."""
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")
            if not os.path.isfile(path):
                raise ValueError(f"Path is not a file: {path}")

    def _validate_required_columns(self) -> None:
        """Validate that all required columns exist in the datasets."""
        required_columns = {
            'movies': ['movieId', 'title', 'genres'],
            'genome_scores': ['movieId', 'tagId', 'relevance'],
            'genome_tags': ['tagId', 'tag'],
            'ratings': ['userId', 'movieId', 'rating']
        }
        
        for df_name, columns in required_columns.items():
            df = getattr(self, df_name)
            missing = [col for col in columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns in {df_name}: {missing}")

    def _initialize_tfidf(self, vectorizer_path: str, tfidf_path: str) -> None:
        """Initialize TF-IDF with secure model loading."""
        try:
        if os.path.exists(vectorizer_path) and os.path.exists(tfidf_path):
                # Validate file sizes
                if os.path.getsize(vectorizer_path) > 10 * 1024 * 1024:  # 10MB limit
                    raise ValueError("Vectorizer file too large")
                if os.path.getsize(tfidf_path) > 100 * 1024 * 1024:  # 100MB limit
                    raise ValueError("TF-IDF matrix file too large")
                
            self.vectorizer = joblib.load(vectorizer_path)
            self.tfidf = joblib.load(tfidf_path)
        else:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            self.tfidf = self.vectorizer.fit_transform(self.movies['clean_title'])
                
                # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
                
                # Save models
            joblib.dump(self.vectorizer, vectorizer_path)
            joblib.dump(self.tfidf, tfidf_path)
        except Exception as e:
            raise ValueError(f"Error initializing TF-IDF: {e}")

    @staticmethod
    def clean_title(title: str) -> str:
        """Clean and sanitize movie title."""
        if not isinstance(title, str):
            raise ValueError("Title must be a string")
        title = title.lower()
        title = re.sub("[^a-z0-9 ]", "", title)
        return title

    def search(self, 
              title: str, 
              top_n: int = 10, 
              min_year: Optional[int] = None, 
              max_year: Optional[int] = None, 
              genres: Optional[List[str]] = None, 
              min_rating: Optional[float] = None) -> pd.DataFrame:
        """Search movies with input validation."""
        # Validate inputs
        if not isinstance(title, str) or len(title.strip()) < 3:
            raise ValueError("Title must be a string with at least 3 characters")
        
        if not isinstance(top_n, int) or top_n <= 0 or top_n > 100:
            raise ValueError("top_n must be a positive integer <= 100")
        
        # Basic title search
        cleaned = self.clean_title(title)
        query_vec = self.vectorizer.transform([cleaned])
        similarity = cosine_similarity(query_vec, self.tfidf).flatten()
        
        # Apply filters
        mask = pd.Series(True, index=self.movies.index)
        
        # Apply year filter
        if min_year is not None and max_year is not None:
            if min_year > max_year:
                raise ValueError("min_year cannot be greater than max_year")
            mask &= (self.movies['year'] >= min_year) & (self.movies['year'] <= max_year)
        elif min_year is not None:
            mask &= self.movies['year'] >= min_year
        elif max_year is not None:
            mask &= self.movies['year'] <= max_year
            
        # Apply genre filter
        if genres:
            if not isinstance(genres, list):
                raise ValueError("genres must be a list")
            genre_mask = self.movies['genre_list'].apply(lambda x: any(g in x for g in genres))
            mask &= genre_mask
            
        # Apply rating filter
        if min_rating is not None:
            if not isinstance(min_rating, (int, float)) or min_rating < 0 or min_rating > 5:
                raise ValueError("min_rating must be a number between 0 and 5")
            mask &= self.movies['avg_rating'] >= min_rating
        
        # Apply mask to similarity scores
        similarity[~mask] = 0
        
        # Get top results
        indices = np.argpartition(similarity, -top_n)[-top_n:]
        results = self.movies.iloc[indices].iloc[np.argsort(similarity[indices])[::-1]]
        
        return results[['movieId', 'title', 'year', 'genres', 'avg_rating', 'rating_count']].reset_index(drop=True)

    def get_movie_details(self, movie_id: int) -> Dict:
        """Get movie details with input validation."""
        if not isinstance(movie_id, int) or movie_id <= 0:
            raise ValueError("movie_id must be a positive integer")
            
        if movie_id not in self.movies['movieId'].values:
            raise ValueError("Movie not found")
            
        movie = self.movies[self.movies['movieId'] == movie_id].iloc[0]
        
        # Get genome tags for the movie
        movie_tags = self.genome_scores[self.genome_scores['movieId'] == movie_id]
        movie_tags = movie_tags.merge(self.genome_tags, on='tagId')
        movie_tags = movie_tags.sort_values('relevance', ascending=False).head(10)
        
        # Get user ratings
        movie_ratings = self.ratings[self.ratings['movieId'] == movie_id]
        
        return {
            'movie': movie.to_dict(),
            'tags': movie_tags[['tag', 'relevance']].to_dict('records'),
            'rating_stats': {
                'mean': movie_ratings['rating'].mean(),
                'count': len(movie_ratings),
                'distribution': movie_ratings['rating'].value_counts().to_dict()
            }
        }
