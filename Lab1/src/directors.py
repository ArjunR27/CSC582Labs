import pandas as pd
import ast
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

nltk.download('punkt')
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def combined_to_df():
    df = pd.read_csv("../movie_data/combined.csv")
    return df

def get_directors(crew_str):
    crew = ast.literal_eval(crew_str)
    return [member['name'] for member in crew if member['job'] == 'Director']

def parse_genres(genre_str):
    try:
        return [item['name'] for item in ast.literal_eval(genre_str)]
    except Exception:
        return []

def encode_text(text):
    if pd.isna(text) or text.strip() == '':
        return None
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return None
    return embedding_model.encode(sentences).mean(axis=0)

def embed_overviews(df):
    tqdm.pandas(desc="Embedding overviews")
    df['overview_vector'] = df['overview'].progress_apply(encode_text)
    return df

# --- Genre embedding ---

def embed_genres(genre_names):
    """Embed each genre label string. Returns (genre_names, genre_matrix)."""
    vectors = embedding_model.encode(genre_names)
    return genre_names, vectors

def get_top_genres(query_vec, genre_names, genre_matrix, top_k=3):
    """Return top-k genre names closest to the query vector."""
    scores = cosine_similarity(query_vec.reshape(1, -1), genre_matrix)[0]
    top_indices = scores.argsort()[::-1][:top_k]
    return [genre_names[i] for i in top_indices]

# --- Genre → directors map ---

def build_genre_director_map(train_df):
    """Map each genre name to the set of directors who worked in that genre."""
    genre_director_map = defaultdict(set)
    for _, row in train_df.iterrows():
        for genre in row['genres_list']:
            for director in row['directors']:
                genre_director_map[genre].add(director)
    return genre_director_map

# --- Movie-level index ---

def build_movie_index(train_df):
    """Build a per-movie index for similarity search."""
    valid_df = train_df[train_df['overview_vector'].notna()].reset_index(drop=True)
    movie_directors = valid_df['directors'].tolist()
    vector_matrix = np.vstack(valid_df['overview_vector'].tolist())
    return movie_directors, vector_matrix

def majority_vote(director_lists):
    all_directors = [d for directors in director_lists for d in directors]
    if not all_directors:
        return None
    return Counter(all_directors).most_common(1)[0][0]

def predict_director(query_vec, movie_directors, movie_matrix, candidate_directors, top_k=10):
    """
    Find top-k most similar movies restricted to candidate_directors,
    then majority vote on the director.
    """
    # Filter movie index to candidates
    indices = [i for i, dirs in enumerate(movie_directors)
               if any(d in candidate_directors for d in dirs)]
    if not indices:
        # Fallback: use full index
        indices = list(range(len(movie_directors)))

    filtered_dirs = [movie_directors[i] for i in indices]
    filtered_matrix = movie_matrix[np.array(indices)]

    scores = cosine_similarity(query_vec.reshape(1, -1), filtered_matrix)[0]
    top_indices = scores.argsort()[::-1][:top_k]
    top_director_lists = [filtered_dirs[i] for i in top_indices]

    return majority_vote(top_director_lists)

# --- Train/test split ---

def split_by_repeated_directors(df, test_size=0.1, random_state=42):
    director_counts = Counter(d for directors in df['directors'] for d in directors)

    def is_eligible(directors):
        return bool(directors) and all(director_counts[d] >= 2 for d in directors)

    eligible_mask = df['directors'].apply(is_eligible)
    eligible_df = df[eligible_mask]
    ineligible_df = df[~eligible_mask]

    test_df = eligible_df.sample(frac=test_size, random_state=random_state)
    train_df = pd.concat([ineligible_df, eligible_df.drop(test_df.index)])

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    print(f"Eligible for test (director has 2+ movies): {eligible_mask.sum()} movies")
    return train_df, test_df

# --- Main ---

def main():
    df = combined_to_df()
    df['directors'] = df['crew'].apply(get_directors)
    df['genres_list'] = df['genres'].apply(parse_genres)
    df = embed_overviews(df)

    train_df, test_df = split_by_repeated_directors(df, test_size=0.1, random_state=42)

    # Build genre embeddings
    all_genres = sorted({g for genres in df['genres_list'] for g in genres})
    genre_names, genre_matrix = embed_genres(all_genres)
    print(f"{len(genre_names)} genre vectors built: {genre_names}")

    # Build genre → directors map from training data
    genre_director_map = build_genre_director_map(train_df)
    for g in genre_names:
        print(f"  {g}: {len(genre_director_map[g])} directors")

    # Build movie-level index from training data
    print("Building movie index...")
    movie_directors, movie_matrix = build_movie_index(train_df)
    print(f"Movie index built: {len(movie_directors)} movies")

    # Evaluate
    test_df = test_df[test_df['overview_vector'].notna()].reset_index(drop=True)
    correct_count = 0
    total = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        correct_directors = row['directors']
        query_vec = row['overview_vector']

        top_genres = get_top_genres(query_vec, genre_names, genre_matrix, top_k=3)
        candidate_directors = set()
        for g in top_genres:
            candidate_directors |= genre_director_map[g]

        predicted_director = predict_director(
            query_vec, movie_directors, movie_matrix, candidate_directors, top_k=10
        )

        match = predicted_director in correct_directors
        if match:
            correct_count += 1
        total += 1

        print(f"Test movie:         {row['title_x']}")
        print(f"Matched genres:     {top_genres}")
        print(f"Candidates:         {len(candidate_directors)} directors")
        print(f"Correct director:   {correct_directors}")
        print(f"Predicted director: {predicted_director}")
        print(f"Match: {match}")
        print()

    print(f"Accuracy: {correct_count}/{total} ({correct_count/total*100:.1f}%)")

def main2():
    df = combined_to_df()
    df['directors'] = df['crew'].apply(get_directors)
    print(f'Number of Unique Directors: {df["directors"].apply(tuple).nunique()}')

if __name__ == "__main__":
    main()
    # main2()
