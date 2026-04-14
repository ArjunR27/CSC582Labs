import numpy as np
from tqdm import tqdm
from collections import defaultdict                                                                                                                                                                       
import pandas as pd
import nltk
import ast
import pickle
import os
from sentence_transformers import SentenceTransformer, CrossEncoder

from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


CACHE_PATH = "../movie_data/embeddings_cache.pkl"
TOP_K = 10
MINIMUM_NUM_MOVIES = 4
N_CAST = 20

def load_data():
    df = pd.read_csv("../movie_data/combined.csv")
    df['directors'] = df['crew'].apply(get_directors)
    df['cast_list'] = df['cast'].apply(get_cast)

    # counting and filtering for directors with at least N movies
    director_counts = df.explode('directors')['directors'].value_counts()                                                                                                                                     
    frequent_directors = director_counts[director_counts >= MINIMUM_NUM_MOVIES].index   

    # keep only rows where at least one director meets the threshold                                                                                                                                          
    df = df[df['directors'].apply(lambda dirs: any(d in frequent_directors for d in dirs))]
    df = df.reset_index(drop=True)  

    return df

# custom train/test split --> this ensures that every director in the test appears AT LEAST once in the train
def train_test_split(df):
    train_idx, test_idx = [], []
    director_counts = df.explode('directors')['directors'].value_counts()                                                                                                                                     

    for director in director_counts.index:
        movies = df[df['directors'].apply(lambda d: director in d)].index.tolist()
        if len(movies) < 2:
            train_idx.extend(movies)
            continue
        split = int(len(movies) * 0.80)
        train_idx.extend(movies[:split])
        test_idx.extend(movies[split:])
    
    train_df = df.loc[list(set(train_idx))].reset_index(drop=True)
    test_df = df.loc[list(set(test_idx))].reset_index(drop=True)
    return train_df, test_df

def get_directors(crew_str):
    crew = ast.literal_eval(crew_str)
    return [member['name'] for member in crew if member.get('job') == 'Director']

def get_cast(crew_str):
    cast = ast.literal_eval(crew_str)
    return [(member['name'], member.get('order', 99)) for member in cast]

def encode(text):
    sentences = nltk.sent_tokenize(text)
    vectors = model.encode(sentences)
    return vectors.mean(axis=0)

def load_index(df):
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            matrix = pickle.load(f)
    else:
        vectors = []
        for text in df['overview']:
            vectors.append(encode(text))
        matrix = np.vstack(vectors)
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(matrix, f)
    return matrix

def find_similar(query_overview, matrix, top_k=TOP_K):
    scores = cosine_similarity(query_overview.reshape(1, -1), matrix)[0]
    top_indices = scores.argsort()[::-1][:top_k]
    return top_indices, scores[top_indices]

def get_input_overview(file):
    with open(file, 'r') as file:
        input_overview = file.read()
        return input_overview

def suggest_director(overview, matrix, train_df):
    encoded_overview = encode(overview)

    top_indices, _ = find_similar(encoded_overview, matrix)
    candidates = train_df.iloc[top_indices].copy()

    top1 = candidates.iloc[0]
    return top1

def suggest_director_weighted_vote(overview, matrix, train_df):

    encoded_overview = encode(overview)
    top_indices, scores = find_similar(encoded_overview, matrix)
    candidates = train_df.iloc[top_indices].copy()

    director_scores = defaultdict(float)                                                                                                                                                                      
    for (_, movie), score in zip(candidates.iterrows(), scores):
        for director in movie['directors']:                                                                                                                                                                   
            director_scores[director] += score

    best_director = max(director_scores, key=director_scores.get)                                                                                                                                             
    top1 = candidates[candidates['directors'].apply(lambda d: best_director in d)].iloc[0]
    return top1  

def suggest_director_reranker(overview, matrix, train_df):
    encoded_overview = encode(overview)

    top_indices, _ = find_similar(encoded_overview, matrix)
    candidates = train_df.iloc[top_indices].copy()

    pairs = [[overview, c['overview']] for _, c in candidates.iterrows()]
    rerank_scores = reranker.predict(pairs)
    candidates['rerank_score'] = rerank_scores
    top1 = candidates.sort_values('rerank_score', ascending=False).iloc[0]
    return top1

def suggest_cast(overview, matrix, train_df, top_n=N_CAST):
    encoded_overview = encode(overview)
    top_indices, scores = find_similar(encoded_overview, matrix)
    candidates = train_df.iloc[top_indices].copy()

    actor_scores = defaultdict(float)
    for (_, movie), score in zip(candidates.iterrows(), scores):
        for actor_name, order in movie['cast_list']:
            actor_scores[actor_name] += score / (order + 1)

    ranked_cast = sorted(actor_scores.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked_cast[:top_n]]


def suggest_cast_reranker(overview, matrix, train_df, top_n=N_CAST, candidate_movies_k=TOP_K):
    """
    Rerank cast candidates with a cross-encoder on (query_overview, actor_context) pairs.
    Actor contexts are built from the top retrieved training movies.
    """
    encoded_overview = encode(overview)
    top_indices, scores = find_similar(encoded_overview, matrix, top_k=candidate_movies_k)
    candidates = train_df.iloc[top_indices].copy()

    actor_context = defaultdict(list)
    for (_, movie), sim_score in zip(candidates.iterrows(), scores):
        movie_title = movie.get('original_title', movie.get('title', 'Unknown'))
        movie_overview = str(movie.get('overview', ''))
        overview_snippet = movie_overview[:220].replace("\n", " ")

        # Use top-billed actors from each candidate movie to keep candidate pool focused.
        for actor_name, order in movie['cast_list']:
            if order > 9:
                continue
            actor_context[actor_name].append(
                f"title: {movie_title}; billing_order: {order}; sim: {sim_score:.4f}; overview: {overview_snippet}"
            )

    if not actor_context:
        return []

    actor_names = list(actor_context.keys())
    pairs = []
    for actor_name in actor_names:
        context_text = " | ".join(actor_context[actor_name][:3])
        actor_profile = f"actor: {actor_name}. evidence: {context_text}"
        pairs.append([overview, actor_profile])

    rerank_scores = reranker.predict(pairs)
    ranked_actors = sorted(zip(actor_names, rerank_scores), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked_actors[:top_n]]


def score_test_overview(actual_directors, predicted_directors, actual_cast_list, guessed_cast):
    """
    Scoring rules per test overview:
      - +20 if predicted director is correct.
      - +10 per guessed cast member found in full cast list (max 50).
      - +5 per guessed cast member found in top-5 billed cast (order 0-4, max 25).
    """
    actual_director_set = set(actual_directors)
    predicted_director_set = set(predicted_directors)

    director_points = 20 if actual_director_set.intersection(predicted_director_set) else 0

    guessed_cast_set = set(guessed_cast)
    actual_cast_names = {name for name, _ in actual_cast_list}
    matched_cast_count = len(guessed_cast_set.intersection(actual_cast_names))
    cast_points = min(matched_cast_count * 10, 50)

    top5_cast_names = {name for name, order in actual_cast_list if order <= 4}
    matched_top5_count = len(guessed_cast_set.intersection(top5_cast_names))
    top5_points = min(matched_top5_count * 5, 25)

    total_points = director_points + cast_points + top5_points
    return {
        'director_points': director_points,
        'cast_points': cast_points,
        'top5_points': top5_points,
        'total_points': total_points,
    }


def evaluate_test_overview_scores(test_df, train_df, matrix):
    """Compute scoring breakdown for each test overview using reranked director + cast guesses."""
    scores = []
    actor_point_instances = 0

    for row_idx, row in test_df.iterrows():
        actual_directors = row['directors']
        actual_cast_list = row['cast_list']
        overview = row['overview']
        movie_title = row.get('original_title', row.get('title', 'Unknown'))

        predicted_director_row = suggest_director_reranker(overview, matrix, train_df)
        predicted_directors = predicted_director_row['directors']
        guessed_cast = suggest_cast_reranker(overview, matrix, train_df)

        breakdown = score_test_overview(
            actual_directors=actual_directors,
            predicted_directors=predicted_directors,
            actual_cast_list=actual_cast_list,
            guessed_cast=guessed_cast,
        )
        scores.append(breakdown)

    if scores:
        total_points = sum(s['total_points'] for s in scores)
        avg_points = total_points / len(scores)
        print(f"\nOverview score avg: {avg_points:.2f}/95")
        print(f"Overview score total: {total_points}/{95 * len(scores)}")

    return scores


def evaluate_cast_predictions(test_df, train_df, matrix, top_n=N_CAST):
    """
    Evaluate cast suggestion quality against truth on test overviews.
    Reports overlap and hit-rate metrics.
    """
    full_overlap_counts = []
    top5_overlap_counts = []
    full_hit_count = 0
    top5_hit_count = 0
    total_full_correct = 0
    total_top5_correct = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating cast"):
        actual_cast_list = row['cast_list']
        overview = row['overview']

        guessed_cast = suggest_cast_reranker(overview, matrix, train_df, top_n=top_n)
        guessed_cast_set = set(guessed_cast)

        actual_full_set = {name for name, _ in actual_cast_list}
        actual_top5_set = {name for name, order in actual_cast_list if order <= 4}

        full_overlap = len(guessed_cast_set.intersection(actual_full_set))
        top5_overlap = len(guessed_cast_set.intersection(actual_top5_set))

        full_overlap_counts.append(full_overlap)
        top5_overlap_counts.append(top5_overlap)
        total_full_correct += full_overlap
        total_top5_correct += top5_overlap

        if full_overlap > 0:
            full_hit_count += 1
        if top5_overlap > 0:
            top5_hit_count += 1

    n = len(test_df)
    if n == 0:
        return {}

    avg_full_overlap = np.mean(full_overlap_counts)
    avg_top5_overlap = np.mean(top5_overlap_counts)
    avg_precision_at_n = avg_full_overlap / top_n
    avg_top5_recall = np.mean([count / 5 for count in top5_overlap_counts])
    total_guesses = n * top_n
    total_top5_slots = n * 5

    print(f"\nCorrect actor guesses (full cast): {total_full_correct}/{total_guesses} ({100*total_full_correct/total_guesses:.1f}%)")
    print(f"Correct actor guesses (top-5 billed): {total_top5_correct}/{total_top5_slots} ({100*total_top5_correct/total_top5_slots:.1f}%)")
    print(f"\nCast overlap avg (full cast): {avg_full_overlap:.2f}/{top_n}")
    print(f"Cast overlap avg (top-5 billed): {avg_top5_overlap:.2f}/5")
    print(f"Avg precision@{top_n} vs full cast: {avg_precision_at_n:.4f}")
    print(f"Avg recall on top-5 billed cast: {avg_top5_recall:.4f}")
    print(f"Any full-cast hit rate: {full_hit_count}/{n} ({100*full_hit_count/n:.1f}%)")
    print(f"Any top-5 billed hit rate: {top5_hit_count}/{n} ({100*top5_hit_count/n:.1f}%)")

    return {
        'avg_full_overlap': avg_full_overlap,
        'avg_top5_overlap': avg_top5_overlap,
        'avg_precision_at_n': avg_precision_at_n,
        'avg_top5_recall': avg_top5_recall,
        'total_full_correct': total_full_correct,
        'total_top5_correct': total_top5_correct,
        'full_hit_rate': full_hit_count / n,
        'top5_hit_rate': top5_hit_count / n,
    }


def evaluate_director_retrieval(test_df, train_df, matrix):
    correct_top1 = 0
    correct_weighted = 0
    correct_reranker = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        actual_directors = row['directors']
        overview = row['overview']

        d_top1 = suggest_director(overview, matrix, train_df)
        d_weighted = suggest_director_weighted_vote(overview, matrix, train_df)
        d_reranker = suggest_director_reranker(overview, matrix, train_df)

        if any(d in d_top1['directors'] for d in actual_directors):
            correct_top1 += 1
        if any(d in d_weighted['directors'] for d in actual_directors):
            correct_weighted += 1
        if any(d in d_reranker['directors'] for d in actual_directors):
            correct_reranker += 1

    n = len(test_df)
    print(f"\nTop-1 cosine:    {correct_top1}/{n} ({100*correct_top1/n:.1f}%)")
    print(f"Weighted vote:   {correct_weighted}/{n} ({100*correct_weighted/n:.1f}%)")
    print(f"Reranker:        {correct_reranker}/{n} ({100*correct_reranker/n:.1f}%)")
    return correct_top1, correct_weighted, correct_reranker


def test():
    df = load_data()
    train_df, test_df = train_test_split(df)
    matrix = load_index(train_df)
    evaluate_director_retrieval(test_df, train_df, matrix)
    evaluate_cast_predictions(test_df, train_df, matrix)
    evaluate_test_overview_scores(test_df, train_df, matrix)

def main():
    df = load_data()
    train_df, test_df = train_test_split(df)
    matrix = load_index(train_df)
    # input_overview = get_input_overview('./test_input.txt')

    input_overview = test_df[['directors', 'overview']].iloc[200]
    print(input_overview.to_dict())
    encoded_input_overview = encode(input_overview['overview'])

    top_indices, scores = find_similar(encoded_input_overview, matrix)                                                                                                                                            
    candidates = train_df.iloc[top_indices].copy()
                                                                                                                                                                                                                    
    # rerank        
    pairs = [[input_overview['overview'], row['overview']] for _, row in candidates.iterrows()]                                                                                                                   
    rerank_scores = reranker.predict(pairs)                                                                                                                                                                       
    candidates['score'] = rerank_scores                                                                                                                                                                           
                                                                                                                                                                                                                    
    candidates = candidates.sort_values('score', ascending=False)                                                                                                                                                 
    results = candidates[['original_title', 'directors', 'score']]
    print(results)                

    # top_indices, scores = find_similar(encoded_input_overview, matrix)
    
    # results = train_df.iloc[top_indices][['original_title', 'directors']].copy()
    # results['score'] = scores
    
    # print(results)


if __name__ == "__main__":
    test()
