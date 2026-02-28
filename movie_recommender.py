import os
import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
DATA_PATH = "ml-latest-small/movies.csv"

@st.cache_data
def load_movies(path):
    df = pd.read_csv(path)
    df["genres"] = df["genres"].str.replace("|", " ", regex=False)
    df["genres"] = df["genres"].str.replace("(no genres listed)", "", regex=False)
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$")
    df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()
    df["soup"] = df["genres"] + " " + df["genres"]
    return df.reset_index(drop=True)

@st.cache_data
def build_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["soup"])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

@st.cache_data
def fetch_poster(title, year=None):
    """Fetch movie poster URL from TMDb API."""
    try:
        query = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}&year={year or ''}"
        response = requests.get(query).json()
        results = response.get("results", [])
        if results and results[0].get("poster_path"):
            return f"https://image.tmdb.org/t/p/w300{results[0]['poster_path']}"
    except:
        pass
    return None

def get_recommendations(title_clean, df, similarity, n=8):
    matches = df[df["title_clean"] == title_clean]
    if matches.empty:
        return []
    idx = matches.index[0]
    scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)[1:n+1]
    results = []
    for i, score in scores:
        row = df.iloc[i]
        results.append({
            "title": row["title_clean"],
            "genres": row["genres"],
            "year": row["year"],
            "score": f"{round(score * 100, 1)}%"
        })
    return results

# ── UI ──────────────────────────────────────
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #00b4d8, #0077b6, #00f5c4);
    }
    .stApp h1, .stApp h2, .stApp h3 {
        color: #ffffff !important;
    }
    .movie-card {
        background: rgba(0,0,0,0.4);
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        height: 100%;
    }
    .movie-card img { border-radius: 8px; width: 100%; }
    .movie-title { color: white; font-weight: bold; margin-top: 8px; font-size: 14px; }
    .movie-meta { color: #ccc; font-size: 11px; margin-top: 4px; }
    .movie-score { color: #00f5c4; font-weight: bold; font-size: 13px; margin-top: 4px; }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommender")
st.caption("9,000+ movies · MovieLens · TMDb Posters")
st.divider()

try:
    movies = load_movies(DATA_PATH)
    similarity_matrix = build_similarity_matrix(movies)
    all_titles = sorted(movies["title_clean"].dropna().unique().tolist())

    selected = st.selectbox("Search for a movie you like:", options=all_titles)
    n_recs = st.slider("How many recommendations?", min_value=5, max_value=100, value=6)
    if n_recs > 20:
        st.warning("⚠️ Fetching posters for many movies may take a moment!")

    if st.button("Find Similar Movies →", type="primary"):
        recs = get_recommendations(selected, movies, similarity_matrix, n=n_recs)

        # Show selected movie poster at the top
        st.subheader(f"Because you liked **{selected}**:")
        selected_row = movies[movies["title_clean"] == selected].iloc[0]
        selected_poster = fetch_poster(selected, selected_row["year"])
        if selected_poster:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(selected_poster, width=120)
            with col2:
                st.markdown(f"### {selected}")
                st.caption(f"🎭 {selected_row['genres']} · 📅 {selected_row['year']}")
        st.divider()

        # Show recommendations as a poster grid
        cols = st.columns(min(n_recs, 4))
        for i, rec in enumerate(recs):
            poster = fetch_poster(rec["title"], rec["year"])
            with cols[i % 4]:
                if poster:
                    st.image(poster, use_column_width=True)
                else:
                    st.markdown("🎬 *No poster*")
                st.markdown(f"**{rec['title']}**")
                st.caption(f"{rec['genres'][:30]}... · {rec['year']}")
                st.markdown(f"Match: `{rec['score']}`")

except FileNotFoundError:
    st.error("⚠️ Dataset not found! Make sure ml-latest-small/ is in the same folder as this file.")