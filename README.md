# 🎬 Movie Search Engine

This is a lightweight content-based movie recommendation system using TF-IDF on movie titles. You can search for similar movies by title, and get recommendations from the [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/).

---

## 🛠️ How It Works

- **Input**: A movie title (e.g. `"The Dark Knight"`)
- **Vectorizer**: `TfidfVectorizer(ngram_range=(1, 2))` on cleaned titles
- **Similarity**: Cosine similarity with other titles
- **Output**: Top 5 most similar movies

---

## 📦 Files

- `movie_search.py`: Main class and logic
- `movies.csv`: Source data (from MovieLens)
- `model/vectorizer.pkl`: Trained vectorizer (auto-generated)
- `model/tfidf_matrix.pkl`: TF-IDF matrix (auto-generated)

---

## 🔍 Example

```python
from movie_search import MovieSearchEngine

engine = MovieSearchEngine()
results = engine.search("Iron Man")
print(results)
```

---

## 🧠 Future Ideas

- Switch to sentence-transformers for semantic search
- Add genre or plot-based search
- Integrate with TMDb or IMDb API
- Build a Gradio UI or API backend

---

## 🤗 Hosted on Hugging Face

This model is designed to be uploaded and shared via the Hugging Face Hub. You can integrate it with `Spaces`, `Gradio`, or `Streamlit`.
