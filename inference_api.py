from movie_search import MovieSearchEngine

class MovieRecommender:
    def __init__(self):
        self.engine = MovieSearchEngine()

    def __call__(self, inputs):
        if isinstance(inputs, dict):
            title = inputs.get("title", "")
        else:
            title = str(inputs)
        results = self.engine.search(title)
        return results.to_dict(orient="records")
