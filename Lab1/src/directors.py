import pandas as pd
import ast
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm                                                                      

nltk.download('punkt')
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def combined_to_df():
    df = pd.read_csv("../movie_data/combined.csv")
    return df

def get_directors(crew_str):
    crew = ast.literal_eval(crew_str)
    directors = [member['name'] for member in crew if member['job'] == 'Director']
    return directors

def encode_overview(text):
    if pd.isna(text) or text.strip() == '':
        return None
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return None
    vectors = embedding_model.encode(sentences)
    return vectors.mean(axis=0)

def build_index(df):
    valid_df = df[df['overview_vector'].notna()].reset_index(drop=True)
    titles = valid_df['title_x'].to_list()
    directors = valid_df['crew'].apply(get_directors).tolist()
    vector_matrix = np.vstack(valid_df['overview_vector'].tolist())
    return titles, directors, vector_matrix

def find_similar(input_overview, titles, directors, vector_matrix, top_k=5):
    query_vec = embedding_model.encode([input_overview])
    scores = cosine_similarity(query_vec, vector_matrix)[0]
    top_indices = scores.argsort()[::-1][:top_k]
    return [(titles[i], directors[i], scores[i]) for i in top_indices]

def embed_overviews(df):
    tqdm.pandas(desc="Embedding overviews")
    df['overview_vector'] = df['overview'].progress_apply(encode_overview)
    return df

def majority_vote(results):
    all_directors = [director for _, directors, _ in results for director in directors]
    if not all_directors:
        return []
    counts = Counter(all_directors)
    top_count = counts.most_common(1)[0][1]
    return [d for d, c in counts.items() if c == top_count]

def weighted_majority_vote(results):
    scores = defaultdict(float)
    for rank, (_, directors, _) in enumerate(results, start=1):
        weight = 1 / rank
        for director in directors:
            scores[director] += weight
    if not scores:
        return []
    top_score = max(scores.values())
    return [d for d, s in scores.items() if s == top_score]

def main2():
    df = combined_to_df()
    df['directors'] = df['crew'].apply(get_directors).tolist()
    print(f'Number of Unique Directors: {df['directors'].apply(tuple).nunique()}')

def main():
    df = combined_to_df()
    df = embed_overviews(df)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    titles, directors, vector_matrix = build_index(train_df)

    test_df = test_df[test_df['overview_vector'].notna()].reset_index(drop=True)
    correct_count = 0
    total = 0
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        correct_director = get_directors(row['crew'])
        results = find_similar(row['overview'], titles, directors, vector_matrix, top_k=25)

        predicted_director = majority_vote(results)
        match = bool(set(correct_director) & set(predicted_director))
        if match:
            correct_count += 1
        total += 1
        print(f"Test movie:         {row['title_x']}")
        print(f"Correct director:   {correct_director}")
        print(f"Predicted director: {predicted_director}")
        print(f"Match: {match}")
        print()

    print(f"Accuracy: {correct_count}/{total} ({correct_count/total*100:.1f}%)")

if __name__ == "__main__":
    main()
    # main2()