"""
robotproducer.py — Given a movie overview, suggest a title, director, and cast.

Usage:
    python robotproducer.py input.txt

Embeddings are cached to disk on first run (~30s) and reloaded instantly after.
"""

import sys
import os
import ast
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('punkt', quiet=True)
# nltk.download('averaged_perceptron_tagger', quiet=True)
# nltk.download('punkt_tab', quiet=True)

DATA_PATH = "../movie_data/combined.csv"
CACHE_PATH = "../movie_data/embeddings_cache.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 20          # similar movies to retrieve
N_CAST = 10         # cast members to suggest
N_TITLE_WORDS = 4   # max words in generated title

model = SentenceTransformer(MODEL_NAME)

# ---------------------------------------------------------------------------
# Data loading & parsing
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['directors'] = df['crew'].apply(get_directors)
    df['cast_list'] = df['cast'].apply(get_cast)
    df = df[df['overview'].notna() & (df['overview'].str.strip() != '')].reset_index(drop=True)
    return df

def get_directors(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
        return [m['name'] for m in crew if m.get('job') == 'Director']
    except Exception:
        return []

def get_cast(cast_str):
    """Return list of (name, order) tuples sorted by billing order."""
    try:
        cast = ast.literal_eval(cast_str)
        return [(m['name'], m.get('order', 99)) for m in cast]
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Embedding & caching
# ---------------------------------------------------------------------------

def encode(text):
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        sentences = [text]
    vecs = model.encode(sentences)
    return vecs.mean(axis=0)

def load_or_build_index(df):
    if os.path.exists(CACHE_PATH):
        print("Loading cached embeddings...")
        with open(CACHE_PATH, 'rb') as f:
            matrix = pickle.load(f)
        if len(matrix) == len(df):
            return matrix
        print("Cache size mismatch, rebuilding...")

    print("Building embeddings (first run, ~30s)...")
    matrix = np.vstack([encode(text) for text in df['overview']])
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(matrix, f)
    print("Embeddings cached.")
    return matrix

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def find_similar(query_vec, matrix, top_k=TOP_K):
    scores = cosine_similarity(query_vec.reshape(1, -1), matrix)[0]
    top_indices = scores.argsort()[::-1][:top_k]
    return top_indices, scores[top_indices]

# ---------------------------------------------------------------------------
# Director prediction — weighted majority vote
# ---------------------------------------------------------------------------

def suggest_director(indices, scores, df):
    director_scores = defaultdict(float)
    for idx, score in zip(indices, scores):
        for director in df.iloc[idx]['directors']:
            director_scores[director] += score
    if not director_scores:
        return "Unknown"
    return max(director_scores, key=director_scores.get)

# ---------------------------------------------------------------------------
# Cast prediction — weighted by similarity × billing prominence
# ---------------------------------------------------------------------------

def suggest_cast(indices, scores, df, n=N_CAST):
    """
    Score each actor by: sum over similar movies of (cosine_score / (order + 1))
    This favours top-billed actors in the most similar movies.
    """
    actor_scores = defaultdict(float)
    for idx, score in zip(indices, scores):
        for name, order in df.iloc[idx]['cast_list']:
            actor_scores[name] += score / (order + 1)
    sorted_actors = sorted(actor_scores, key=actor_scores.get, reverse=True)
    return sorted_actors[:n]

# ---------------------------------------------------------------------------
# Title generation — n-gram model seeded with overview keywords (Option C)
# ---------------------------------------------------------------------------

def build_ngram_model(titles):
    """
    Build a bigram model and unigram frequency count from movie titles.
    Proper nouns (mid-title capitalised words) are excluded so generated
    titles don't contain character/place names baked into the training data.
    Returns (model, unigram_counts):
      model: {(word,): Counter({next_word: count})}
      unigram_counts: Counter({word: total occurrences across all titles})
    """
    from collections import Counter
    model = defaultdict(Counter)
    unigram_counts = Counter()
    for title in titles:
        raw_tokens = nltk.word_tokenize(str(title))
        # Drop mid-title capitalised tokens (proxy for proper nouns)
        tokens = [
            t.lower() for i, t in enumerate(raw_tokens)
            if t.isalpha() and (i == 0 or not t[0].isupper())
        ]
        unigram_counts.update(tokens)
        for i in range(len(tokens) - 1):
            model[(tokens[i],)][tokens[i + 1]] += 1
    return model, unigram_counts

def extract_seed_words(overview):
    """
    Extract common nouns and adjectives from the overview (no proper nouns).
    Returns deduplicated list in order of appearance.
    """
    stopwords = {'a', 'an', 'the', 'his', 'her', 'their', 'its', 'of', 'in',
                 'on', 'at', 'to', 'for', 'and', 'or', 'but', 'from', 'with'}
    tokens = nltk.word_tokenize(overview)
    tagged = nltk.pos_tag(tokens)
    seeds = [
        word.lower() for word, tag in tagged
        if tag in ('NN', 'NNS', 'JJ', 'VBG')   # no NNP/NNPS (proper nouns)
        and word.lower() not in stopwords
        and word.isalpha()
        and len(word) > 3
    ]
    seen = set()
    return [w for w in seeds if not (w in seen or seen.add(w))]

def generate_from_ngram(seed, ngram_model, max_words=4):
    """Walk the bigram chain from seed, picking the most frequent next word."""
    tokens = [seed]
    current = (seed,)
    for _ in range(max_words - 1):
        candidates = ngram_model.get(current)
        if not candidates:
            break
        next_word = candidates.most_common(1)[0][0]
        if next_word == tokens[-1]:
            break
        tokens.append(next_word)
        current = (next_word,)
    return ' '.join(w.capitalize() for w in tokens)

def suggest_title(overview, ngram_model, unigram_counts):
    seeds = extract_seed_words(overview)
    title_vocab = {w for key in ngram_model for w in key}

    # Keep only seeds that appear in the title vocabulary
    candidates = [w for w in seeds if w in title_vocab]
    if not candidates:
        return seeds[0].capitalize() if seeds else "Untitled"

    # Pick the seed with lowest title-corpus frequency — more thematically specific
    seed = min(candidates, key=lambda w: unigram_counts.get(w, 0))
    return generate_from_ngram(seed, ngram_model)

# ---------------------------------------------------------------------------
# Evaluation (for testing against known movies)
# ---------------------------------------------------------------------------

def evaluate(query_overview, true_director, true_cast_ordered, df, matrix):
    """
    true_cast_ordered: list of actor names in billing order (index 0 = top billed)
    Returns score breakdown.
    """
    query_vec = encode(query_overview)
    indices, scores = find_similar(query_vec, matrix)

    pred_director = suggest_director(indices, scores, df)
    pred_cast = suggest_cast(indices, scores, df)
    _ngram, _uni = build_ngram_model(df['title_x'])
    title = suggest_title(query_overview, _ngram, _uni)

    director_score = 20 if pred_director in true_director else 0

    cast_set = set(true_cast_ordered)
    top5_set = set(true_cast_ordered[:5])
    pred_cast_set = set(pred_cast)

    cast_hits = pred_cast_set & cast_set
    top5_hits = pred_cast_set & top5_set

    cast_score = min(len(cast_hits) * 10, 50)
    top5_score = min(len(top5_hits) * 5, 25)
    total = director_score + cast_score + top5_score

    return {
        'title': title,
        'director': pred_director,
        'cast': pred_cast,
        'director_score': director_score,
        'cast_score': cast_score,
        'top5_score': top5_score,
        'total': total,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python robotproducer.py input.txt")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        overview = f.read().strip()

    df = load_data()
    matrix = load_or_build_index(df)
    ngram_model, unigram_counts = build_ngram_model(df['title_x'])

    query_vec = encode(overview)
    indices, scores = find_similar(query_vec, matrix)

    title = suggest_title(overview, ngram_model, unigram_counts)
    director = suggest_director(indices, scores, df)
    cast = suggest_cast(indices, scores, df)

    print(f"\nTitle suggestion: {title}")
    print(f"Director suggestion: {director}")
    print(f"Cast suggestions: {', '.join(cast)}")

    # Score against the closest matching movie in the dataset as ground truth
    top_match = df.iloc[indices[0]]
    true_directors = top_match['directors']
    true_cast = [name for name, order in sorted(top_match['cast_list'], key=lambda x: x[1])]

    director_score = 20 if director in true_directors else 0
    cast_hits = set(cast) & set(true_cast)
    top5_hits = set(cast) & set(true_cast[:5])
    cast_score = min(len(cast_hits) * 10, 50)
    top5_score = min(len(top5_hits) * 5, 25)
    total = director_score + cast_score + top5_score

    print(f"\n--- Score vs closest match: '{top_match['title_x']}' ---")
    print(f"Director: {director_score}/20  (true: {', '.join(true_directors)})")
    print(f"Cast hits: {cast_score}/50  ({len(cast_hits)} matched: {', '.join(cast_hits) or 'none'})")
    print(f"Top-5 cast bonus: {top5_score}/25  ({len(top5_hits)} matched: {', '.join(top5_hits) or 'none'})")
    print(f"Total: {total}/95")

if __name__ == "__main__":
    main()
