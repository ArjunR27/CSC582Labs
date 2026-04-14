import pandas as pd
import ast
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score


def credits_to_df():
    df = pd.read_csv('../movie_data/tmdb_5000_credits.csv')
    return df


def movies_to_df():
    df = pd.read_csv('../movie_data/tmdb_5000_movies.csv')
    return df


def get_director(crew_text):
    if pd.isna(crew_text):
        return None

    try:
        crew_items = ast.literal_eval(crew_text)
    except (ValueError, SyntaxError):
        return None

    for item in crew_items:
        if item.get("job") == "Director":
            return item.get("name")

    return None


def get_cast_members(cast_text):
    if pd.isna(cast_text):
        return ()

    try:
        cast_items = ast.literal_eval(cast_text)
    except (ValueError, SyntaxError):
        return ()

    cast_with_order = []
    for item in cast_items:
        name = item.get("name")
        order = item.get("order")
        if name is not None and order is not None:
            cast_with_order.append((name, order))

    cast_with_order.sort(key=lambda x: x[1])
    return tuple(cast_with_order)


def filter_directors_with_min_movies(df, min_movies):
    director_counts = df["director"].value_counts(dropna=True)
    eligible_directors = director_counts[director_counts >= min_movies].index
    filtered_df = df[df["director"].isin(eligible_directors)].reset_index(drop=True)
    return filtered_df


def predict_director_for_test_row(test_row_position, train_df, train_embeddings, test_embeddings, top_k=5):
    test_vector = test_embeddings[test_row_position]
    if len(test_vector.shape) == 1:
        test_vector = test_vector.reshape(1, -1)
    similarity_scores = cosine_similarity(test_vector, train_embeddings).flatten()

    top_indices = similarity_scores.argsort()[-top_k:][::-1]
    top_directors = []
    for idx in top_indices:
        director = train_df.iloc[idx]["director"]
        if pd.notna(director):
            top_directors.append((director, similarity_scores[idx]))

    if not top_directors:
        return None, []

    director_counts = Counter([d for d, _ in top_directors])
    majority_director, majority_count = director_counts.most_common(1)[0]

    top_director_names = [d for d, _ in top_directors]
    if majority_count > top_k / 2:
        return majority_director, top_director_names

    return top_directors[0][0], top_director_names


def evaluate_pipeline(name, combined_df, feature_matrix, top_k=5):
    train_df, test_df, train_features, test_features = train_test_split(
        combined_df,
        feature_matrix,
        test_size=0.1,
        random_state=42
    )

    print(f"\n{name}")
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
    print(f"Train feature shape: {train_features.shape} | Test feature shape: {test_features.shape}")

    predicted_directors = []
    actual_directors = []
    top5_hit_count = 0

    for i in range(len(test_df)):
        actual = test_df.iloc[i]["director"]
        if pd.isna(actual):
            continue

        predicted, top5 = predict_director_for_test_row(
            test_row_position=i,
            train_df=train_df,
            train_embeddings=train_features,
            test_embeddings=test_features,
            top_k=top_k
        )

        if actual in top5:
            top5_hit_count += 1


        # if actual in top5:
        #     movie_title = test_df.iloc[i].get("title_x", test_df.iloc[i].get("title", "Unknown"))
        #     print(f"Movie: {movie_title}")
        #     print(f"Predicted: {predicted}")
        #     print(f"Actual: {actual}")
        #     print(f"Top 5: {top5}")
        #     print("-----")
        predicted_directors.append(predicted)
        actual_directors.append(actual)

    accuracy = accuracy_score(actual_directors, predicted_directors)
    precision = precision_score(actual_directors, predicted_directors, average="weighted", zero_division=0)
    recall = recall_score(actual_directors, predicted_directors, average="weighted", zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(
        f"Top-5 director hit rate: {top5_hit_count}/{len(actual_directors)} "
        f"({top5_hit_count / len(actual_directors):.4f})"
    )


def run_tfidf_pipeline(combined_df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(combined_df["overview"].fillna(""))
    evaluate_pipeline("TF-IDF Pipeline", combined_df, tfidf_matrix, top_k=5)


def run_sentence_bert_pipeline(combined_df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    overview_embeddings = model.encode(
        combined_df["overview"].fillna("").tolist(),
        convert_to_numpy=True,
        show_progress_bar=True
    )
    evaluate_pipeline("Sentence-BERT Pipeline", combined_df, overview_embeddings, top_k=5)

def main():
    # credits_df = credits_to_df()
    # movies_df = movies_to_df()

    # combined_df = movies_df.merge(
    #     credits_df,
    #     left_on='id',
    #     right_on='movie_id',
    #     how='inner'
    # )

    combined_df = pd.read_csv("../movie_data/combined.csv")
    combined_df["director"] = combined_df["crew"].apply(get_director)
    combined_df["cast_members"] = combined_df["cast"].apply(get_cast_members)
    min_movies_per_director = 4
    combined_df = filter_directors_with_min_movies(combined_df, min_movies_per_director)

    unique_directors = combined_df["director"].dropna().nunique()
    print(f"Unique directors (>= {min_movies_per_director} movies): {unique_directors}")
    print(f"Rows after filter: {len(combined_df)}")
    run_tfidf_pipeline(combined_df)
    print(" ========= ")
    run_sentence_bert_pipeline(combined_df)


if __name__ == "__main__":
    main()
