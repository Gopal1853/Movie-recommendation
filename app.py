import gradio as gr
import pandas as pd
from movie_search import MovieSearchEngine
import json
from datetime import datetime
import os
import pathlib
from typing import Dict, List, Union, Optional

# Constants
MAX_HISTORY_SIZE = 1000  # Maximum number of search history entries
MAX_FILE_SIZE = 1024 * 1024  # 1MB maximum file size
HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'search_history.json')

def validate_numeric_input(value: Union[int, float], min_val: float, max_val: float) -> bool:
    """Validate numeric input is within range."""
    try:
        num_val = float(value)
        return min_val <= num_val <= max_val
    except (ValueError, TypeError):
        return False

def load_search_history() -> Dict:
    """Load search history with proper error handling."""
    try:
        if not os.path.exists(HISTORY_FILE):
            return {"searches": []}
            
        if os.path.getsize(HISTORY_FILE) > MAX_FILE_SIZE:
            raise ValueError("Search history file too large")
            
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
            
        if not isinstance(history, dict) or 'searches' not in history:
            raise ValueError("Invalid search history format")
            
        return history
    except (json.JSONDecodeError, ValueError, IOError) as e:
        print(f"Error loading search history: {e}")
        return {"searches": []}

def save_search_history(history: Dict) -> None:
    """Save search history with proper error handling."""
    try:
        if not isinstance(history, dict) or 'searches' not in history:
            raise ValueError("Invalid history format")
            
        # Limit history size
        if len(history['searches']) > MAX_HISTORY_SIZE:
            history['searches'] = history['searches'][-MAX_HISTORY_SIZE:]
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f)
    except (IOError, ValueError) as e:
        print(f"Error saving search history: {e}")

def get_all_genres() -> List[str]:
    """Get all available genres."""
    try:
        genres = set()
        for movie_genres in engine.movies['genres']:
            if isinstance(movie_genres, str):
                genres.update(movie_genres.split('|'))
        return sorted(list(genres))
    except Exception as e:
        print(f"Error getting genres: {e}")
        return []

def get_year_range() -> tuple:
    """Get valid year range from the dataset."""
    try:
        years = engine.movies['year'].dropna().astype(int)
        return int(years.min()), int(years.max())
    except Exception as e:
        print(f"Error getting year range: {e}")
        return 1900, 2024  # Default fallback range

def search_movies(title: str, min_year: int, max_year: int, genres: str, min_rating: float) -> Union[pd.DataFrame, str]:
    """Search movies with input validation."""
    # Validate title
    if not isinstance(title, str) or len(title.strip()) < 3:
        return "Please enter a valid movie title (minimum 3 characters)."
    
    # Validate numeric inputs
    if not validate_numeric_input(min_year, 1900, 2024):
        return "Invalid minimum year."
    if not validate_numeric_input(max_year, 1900, 2024):
        return "Invalid maximum year."
    if not validate_numeric_input(min_rating, 0, 5):
        return "Invalid minimum rating."
    
    if min_year > max_year:
        return "Minimum year cannot be greater than maximum year."
    
    # Validate and clean genres
    genre_list = None
    if genres:
        try:
            genre_list = [g.strip() for g in genres.split(',') if g.strip()]
            if not genre_list:
                genre_list = None
        except Exception:
            genre_list = None
    
    try:
        # Search movies
        results = engine.search(
            title=title,
            min_year=min_year,
            max_year=max_year,
            genres=genre_list,
            min_rating=min_rating
        )
        
        # Store search history
        history = load_search_history()
        history['searches'].append({
            'query': title,
            'timestamp': datetime.now().isoformat(),
            'filters': {
                'min_year': min_year,
                'max_year': max_year,
                'genres': genre_list,
                'min_rating': min_rating
            }
        })
        save_search_history(history)
        
        return results
    except Exception as e:
        print(f"Error during search: {e}")
        return "An error occurred during the search. Please try again."

def get_movie_details(movie_id: Union[int, str]) -> str:
    """Get movie details with input validation."""
    try:
        if not movie_id:
            return "Please select a movie."
        
        movie_id = int(movie_id)
        details = engine.get_movie_details(movie_id)
        
        # Format the details for display
        movie = details['movie']
        tags = details['tags']
        rating_stats = details['rating_stats']
        
        # Create HTML for movie details
        html = f"""
        <div style='padding: 20px;'>
            <h2>{movie['title']} ({movie['year']})</h2>
            <p><strong>Genres:</strong> {movie['genres']}</p>
            <p><strong>Average Rating:</strong> {movie['avg_rating']:.2f} ({movie['rating_count']} ratings)</p>
            
            <h3>Top Tags:</h3>
            <ul>
            {''.join(f"<li>{tag['tag']} (relevance: {tag['relevance']:.2f})</li>" for tag in tags)}
            </ul>
            
            <h3>Rating Distribution:</h3>
            <ul>
            {''.join(f"<li>{rating} stars: {count} ratings</li>" for rating, count in sorted(rating_stats['distribution'].items()))}
            </ul>
        </div>
        """
        
        return html
    except (ValueError, TypeError):
        return "Invalid movie ID."
    except Exception as e:
        print(f"Error getting movie details: {e}")
        return "An error occurred while fetching movie details."

def get_search_history() -> str:
    """Get search history with proper error handling."""
    try:
        history = load_search_history()
        if not history or 'searches' not in history or not history['searches']:
            return "No search history found."
        
        # Format search history
        html = "<div style='padding: 20px;'><h2>Search History</h2><ul>"
        for entry in reversed(history['searches']):
            html += f"""
            <li>
                <strong>Query:</strong> {entry['query']}<br>
                <strong>Time:</strong> {entry['timestamp']}<br>
                <strong>Filters:</strong> {json.dumps(entry['filters'], indent=2)}
            </li>
            """
        html += "</ul></div>"
        
        return html
    except Exception as e:
        print(f"Error getting search history: {e}")
        return "An error occurred while fetching search history."

# Initialize the search engine
engine = MovieSearchEngine()

# Get available genres and year range
all_genres = get_all_genres()
min_year, max_year = get_year_range()

# Create the Gradio interface
with gr.Blocks(title="🎬 Movie Recommendation System for Movie Buffs") as demo:
    gr.Markdown("# 🎬 Movie Recommendation System for Movie Buffs")
    
    with gr.Tab("Search Movies"):
        with gr.Row():
            with gr.Column():
                search_input = gr.Textbox(label="Enter Movie Name")
                min_year_input = gr.Slider(min_year, max_year, value=min_year, label="Minimum Year")
                max_year_input = gr.Slider(min_year, max_year, value=max_year, label="Maximum Year")
                genres_input = gr.Textbox(label="Genres (comma-separated)")
                min_rating_input = gr.Slider(0, 5, value=0, label="Minimum Rating")
                search_button = gr.Button("Search")
            
            with gr.Column():
                search_results = gr.Dataframe(
                    label="Search Results",
                    interactive=False,
                    wrap=True,
                    column_widths=["100px", "300px", "100px", "200px", "100px", "100px"]
                )
                movie_details = gr.HTML(label="Movie Details")
    
    with gr.Tab("Search History"):
        with gr.Row():
            with gr.Column():
                history_button = gr.Button("View History")
            with gr.Column():
                history_output = gr.HTML(label="Search History")
    
    # Set up event handlers
    search_button.click(
        fn=search_movies,
        inputs=[search_input, min_year_input, max_year_input, genres_input, min_rating_input],
        outputs=search_results
    )
    
    search_results.select(
        fn=get_movie_details,
        inputs=[search_results],
        outputs=movie_details
    )
    
    history_button.click(
        fn=get_search_history,
        inputs=[],
        outputs=history_output
)

if __name__ == "__main__":
    demo.launch()
