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
MINIMUM_NUM_MOVIES = 2

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

def suggest_cast(encoded_overview, matrix, train_df):
    top_indices, scores = find_similar(encoded_overview, matrix)
    candidates = train_df.iloc[top_indices].copy()


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

