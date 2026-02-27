"""
🎬 Movie Recommender — MovieLens Edition
-----------------------------------------
Content-based filtering using TF-IDF + Cosine Similarity.
Dataset: MovieLens Small (9,000+ movies)
Download: https://grouplens.org/datasets/movielens/latest/

To run:
    pip3 install pandas scikit-learn streamlit
    streamlit run movie_recommender.py
"""

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "ml-latest-small/movies.csv"

@st.cache_data
def load_movies(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["genres"] = df["genres"].str.replace("|", " ", regex=False)
    df["genres"] = df["genres"].str.replace("(no genres listed)", "", regex=False)
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$")
    df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()
    df["soup"] = df["genres"] + " " + df["genres"]
    return df.reset_index(drop=True)

@st.cache_data
def build_similarity_matrix(df: pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["soup"])
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity

def get_recommendations(title_clean, df, similarity, n=10):
    matches = df[df["title_clean"] == title_clean]
    if matches.empty:
        return pd.DataFrame({"Error": [f"'{title_clean}' not found."]})
    idx = matches.index[0]
    scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)[1:n+1]
    rec_indices = [i[0] for i in scores]
    rec_scores  = [f"{round(i[1] * 100, 1)}%"for i in scores]
    result = df.iloc[rec_indices][["title_clean", "genres", "year"]].copy()
    result.columns = ["Title", "Genres", "Year"]
    result["Match Score"] = rec_scores
    result = result.reset_index(drop=True)
    result.index += 1
    return result

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #00b4d8, #0077b6, #00f5c4);
    }
    h1 {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)


st.title("🎬 Movie Recommender")
st.caption("9,000+ movies · MovieLens dataset · Content-based filtering")
st.divider()

try:
    movies = load_movies(DATA_PATH)
    similarity_matrix = build_similarity_matrix(movies)
    all_titles = sorted(movies["title_clean"].dropna().unique().tolist())
    selected = st.selectbox("Search for a movie you like:", options=all_titles)
    n_recs = st.slider("How many recommendations?", min_value=3, max_value=20, value=8)

    if st.button("Find Similar Movies →", type="primary"):
        with st.spinner("Finding the best matches..."):
            recs = get_recommendations(selected, movies, similarity_matrix, n=n_recs)
        st.subheader(f"Because you liked **{selected}**:")
        st.dataframe(recs, use_container_width=True)
        st.caption("💡 Match Score is based on genre similarity (0% = no match, 100% = identical).")

except FileNotFoundError:
    st.error("⚠️ Dataset not found! Make sure ml-latest-small/ is in the same folder as this file.")