import numpy as np
from collections import defaultdict
import pandas as pd
import nltk
import ast
import pickle
import os
import torch
import sys
import random
from sentence_transformers import SentenceTransformer, CrossEncoder

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

from evaluation import evaluate_cast_predictions, evaluate_director_retrieval, evaluate_test_overview_scores

stop_words = set(stopwords.words('english'))

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

ngram_n = 3

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
    encoded_overview = encode(overview)
    top_indices, _ = find_similar(encoded_overview, matrix, top_k=candidate_movies_k)
    candidates = train_df.iloc[top_indices].copy()

    if candidates.empty:
        return []

    pairs = [[overview, c['overview']] for _, c in candidates.iterrows()]
    rerank_scores = reranker.predict(pairs)
    candidates['rerank_score'] = rerank_scores
    candidates = candidates.sort_values('rerank_score', ascending=False)

    actor_scores = defaultdict(float)
    for _, movie in candidates.iterrows():
        movie_score = max(float(movie['rerank_score']), 0.0)
        cast_sorted = sorted(movie['cast_list'], key=lambda x: x[1])
        for actor_name, order in cast_sorted:
            actor_scores[actor_name] += movie_score / (order + 1)

    ranked_cast = sorted(actor_scores.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked_cast[:top_n]]

def ngram_model(titles, n=ngram_n):
    ngrams = []
    for title in titles:
        tokens = nltk.word_tokenize(title.lower()) + ['<END>']
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
    
    return ngrams

# need to ensure that the vocab between titles and overviews is the same because ngram is fit on titles
def generate_vocabulary(titles, overviews):
    title_vocab = set()
    overview_vocab = set()

    for title in titles:
        for word in nltk.word_tokenize(title.lower()):
            if word not in stop_words:
                title_vocab.add(word)
    
    for text in overviews:
        for word in nltk.word_tokenize(text.lower()):
            overview_vocab.add(word)
    
    return title_vocab & overview_vocab

def pick_seed_word(overview, matrix, train_df, lookup):
    valid_starters = {context[0] for context in lookup if context[0] not in stop_words}

    # prefer words taken directly from the input overview
    overview_tokens = [w for w in nltk.word_tokenize(overview.lower()) if w.isalpha() and w not in stop_words]
    overview_candidates = [w for w in overview_tokens if w in valid_starters]
    if overview_candidates:
        return random.choice(overview_candidates)

    # fallback: score first words of similar movies' titles by cosine similarity
    encoded = encode(overview)
    top_indices, scores = find_similar(encoded, matrix, top_k=10)
    similar = train_df.iloc[top_indices]

    word_scores = {}
    for (_, row), score in zip(similar.iterrows(), scores):
        tokens = nltk.word_tokenize(row['original_title'].lower())
        if not tokens:
            continue
        first_word = tokens[0]
        if first_word in valid_starters:
            word_scores[first_word] = word_scores.get(first_word, 0) + score

    if word_scores:
        words = list(word_scores.keys())
        weights = list(word_scores.values())
        return random.choices(words, weights=weights)[0]

    return max(valid_starters, key=lambda w: sum(s for (_, row), s in zip(similar.iterrows(), scores) if w in row['original_title'].lower()))


def create_title_bigram(overview, matrix, train_df):
    titles = train_df['original_title'].dropna().tolist()
    ngrams = ngram_model(titles, n=2)

    lookup = {}
    for ngram in ngrams:
        context = ngram[:-1]
        next_word = ngram[-1]
        if context not in lookup:
            lookup[context] = []
        lookup[context].append(next_word)

    seed = pick_seed_word(overview, matrix, train_df, lookup)
    generated = [seed]
    context = (seed,)

    while len(generated) < 20:
        if context not in lookup:
            break
        next_words = lookup[context]
        counts = {}
        for word in next_words:
            counts[word] = counts.get(word, 0) + 1
        next_word = random.choices(list(counts.keys()), weights=list(counts.values()))[0]
        if next_word == '<END>':
            break
        generated.append(next_word)
        context = (generated[-1],)

    return ' '.join(w.capitalize() for w in generated)


def create_title_trigram(overview, matrix, train_df):
    titles = train_df['original_title'].dropna().tolist()

    trigram_lookup = {}
    for ngram in ngram_model(titles, n=3):
        context = ngram[:-1]
        next_word = ngram[-1]
        if context not in trigram_lookup:
            trigram_lookup[context] = []
        trigram_lookup[context].append(next_word)

    # prefer a 2-word seed pair drawn directly from the input overview
    overview_tokens = nltk.word_tokenize(overview.lower())
    overview_pairs = [
        (overview_tokens[i], overview_tokens[i + 1])
        for i in range(len(overview_tokens) - 1)
        if (overview_tokens[i], overview_tokens[i + 1]) in trigram_lookup
        and overview_tokens[i] not in stop_words
        and overview_tokens[i].isalpha()
    ]

    if overview_pairs:
        seed_pair = random.choice(overview_pairs)
    else:
        # fallback: score first pairs of similar movies' titles by cosine similarity
        encoded = encode(overview)
        top_indices, scores = find_similar(encoded, matrix, top_k=10)
        similar = train_df.iloc[top_indices]

        pair_scores = {}
        for (_, row), score in zip(similar.iterrows(), scores):
            tokens = nltk.word_tokenize(row['original_title'].lower())
            if len(tokens) < 2:
                continue
            pair = (tokens[0], tokens[1])
            if pair in trigram_lookup and tokens[0] not in stop_words:
                pair_scores[pair] = pair_scores.get(pair, 0) + score

        if pair_scores:
            pairs = list(pair_scores.keys())
            weights = list(pair_scores.values())
            seed_pair = random.choices(pairs, weights=weights)[0]
        else:
            valid_pairs = [ctx for ctx in trigram_lookup if ctx[0] not in stop_words]
            seed_pair = random.choice(valid_pairs)

    generated = list(seed_pair)

    while len(generated) < 20:
        context = (generated[-2], generated[-1])
        if context not in trigram_lookup:
            break
        next_words = trigram_lookup[context]
        counts = {}
        for word in next_words:
            counts[word] = counts.get(word, 0) + 1
        next_word = random.choices(list(counts.keys()), weights=list(counts.values()))[0]
        if next_word == '<END>':
            break
        generated.append(next_word)

    return ' '.join(w.capitalize() for w in generated)


def generate_simple_title(overview):
    TEMPLATES = [
    "The {noun}",
    "{noun} of {noun2}",
    "The {noun} of {noun2}",
    "The Last {noun}",
    "The {noun} Returns",
    "Rise of the {noun}",
    ]

    tagged = nltk.pos_tag(nltk.word_tokenize(overview))

    nouns = [w for w, pos in tagged if pos in ('NN', 'NNP') and w.isalpha() and w.lower() not in stop_words]
    
    if not nouns:
        return "The Movie"

    template = random.choice(TEMPLATES)

    title = template.format(
        noun=random.choice(nouns).capitalize(),
        noun2=random.choice(nouns).capitalize(),
    )
    return title


def test():

    df = load_data()
    train_df, test_df = train_test_split(df)
    matrix = load_index(train_df)
    evaluate_director_retrieval(test_df, train_df, matrix)
    evaluate_cast_predictions(test_df, train_df, matrix)
    evaluate_test_overview_scores(test_df, train_df, matrix)

def read_overview(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def main():
    
    input_path = sys.argv[1]
    overview = read_overview(input_path)

    df = load_data()
    train_df, _ = train_test_split(df)     
    matrix = load_index(df)     

    trigram_title = create_title_trigram(overview, matrix, train_df)

    d_reranker = suggest_director_reranker(overview, matrix, train_df)
    d_reranker = ", ".join(d_reranker['directors'])
    cast_reranker = suggest_cast_reranker(overview, matrix, train_df, top_n=20)
    cast_reranker = ", ".join(cast_reranker)
    print(" ========================== ")

    print(f"Title suggestion: {trigram_title}")
    print(f"\nDirector suggestion: {d_reranker}")
    print(f"\nCast suggestions: {cast_reranker}")
       
    print(" ========================== ")

if __name__ == "__main__":
    main()
