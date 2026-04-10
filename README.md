# 🎬 Movie Recommender Web App

A content-based movie recommendation engine built with Python, powered by TF-IDF and cosine similarity on a dataset of 9,000+ films. Integrated with the TMDb API for real-time movie data and deployed as an interactive Streamlit web app.

---

## Features

- **Content-based filtering** using TF-IDF vectorization and cosine similarity
- **Real-time movie data** via TMDb API (posters, descriptions, cast, and crew)
- **Interactive UI** built with Streamlit
- **Genre and decade filters** to narrow recommendations
- **Similarity scoring** displayed alongside each recommendation
- **Surprise me** feature for random discovery

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python |
| ML / NLP | scikit-learn (TF-IDF, cosine similarity) |
| Data | MovieLens dataset (9,000+ films) |
| API | TMDb API |
| Frontend | Streamlit |

---

## How It Works

1. Movie metadata (title, genre, overview, etc.) is vectorized using **TF-IDF** to convert text into numerical feature vectors
2. **Cosine similarity** is computed across all films to measure how closely related each movie is to a given input
3. The top N most similar films are returned and enriched with live data from the **TMDb API**
4. Users can further filter results by genre or decade, or explore a random recommendation via the surprise feature

---

## Getting Started

### Prerequisites

```bash
pip install streamlit scikit-learn pandas requests
```

### API Key Setup

1. Create a free account at [themoviedb.org](https://www.themoviedb.org/)
2. Generate an API key from your account settings
3. Add it to your environment:

```bash
export TMDB_API_KEY=your_api_key_here
```

### Run the App

```bash
streamlit run app.py
```

---

## Dataset

This project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/) provided by GroupLens Research. The dataset includes metadata for 9,000+ films including titles, genres, and descriptions.

---

## Author

**Sanjeev Venkat**
- GitHub: [github.com/SanjeevVenkat](https://github.com/SanjeevVenkat)
- LinkedIn: [linkedin.com/in/sanjeev-venkat-91893a326](https://www.linkedin.com/in/sanjeev-venkat-91893a326/)
