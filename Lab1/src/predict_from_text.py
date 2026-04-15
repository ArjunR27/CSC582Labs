import sys
import os

from robotproducer import (
    load_data,
    load_index,
    suggest_cast_reranker,
    suggest_director_reranker,
    train_test_split,
    create_title_bigram,
    create_title_trigram,
    generate_simple_title
)
from evaluation import score_test_overview


def read_overview(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    director_name = None
    cast_names = []
    for line in lines:
        if line.startswith("Director:"):
            director_name = line.split(":", 1)[1].strip()
        elif line.startswith("Cast List:"):
            cast_text = line.split(":", 1)[1].strip()
            cast_names = [name.strip() for name in cast_text.split(",") if name.strip()]

    if director_name is None:
        raise ValueError(f"Missing 'Director:' line in {path}")
    if not cast_names:
        raise ValueError(f"Missing or empty 'Cast List:' line in {path}")

    cast_list_with_order = [(name, order) for order, name in enumerate(cast_names)]
    return [director_name], cast_list_with_order


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 predict_from_text.py <input_file>")

    input_path = sys.argv[1]
    cast_top_n = 20
    ground_truth_path = os.path.join(os.path.dirname(__file__), "ground_truth.txt")

    overview = read_overview(input_path)
    if not overview:
        raise ValueError(f"Input file is empty: {input_path}")
    actual_directors, actual_cast_list = read_ground_truth(ground_truth_path)

    df = load_data()
    train_df, _ = train_test_split(df)
    matrix = load_index(train_df)

    # bigram and trigram do a vector similairty lookup of similar movies and use those similar movie titles as a seed phrase and then generate
    trigram_title = create_title_trigram(overview, matrix, train_df)
    bigram_title = create_title_bigram(overview, matrix, train_df)
    basic_title = generate_simple_title(overview)
    print("Trigram Title:", trigram_title)
    print("Bigram Title:", bigram_title)
    print("Simple Noun Title:", basic_title)

    d_reranker = suggest_director_reranker(overview, matrix, train_df)
    cast_reranker = suggest_cast_reranker(
        overview, matrix, train_df, top_n=cast_top_n
    )

    print("Input file:", input_path)
    print("\nDirector predictions")
    print("- Reranker:", d_reranker["directors"])

    print(f"\nCast predictions (top {cast_top_n})")
    print("- Reranker:", cast_reranker)

    points_reranker_reranker_cast = score_test_overview(
        actual_directors=actual_directors,
        predicted_directors=d_reranker["directors"],
        actual_cast_list=actual_cast_list,
        guessed_cast=cast_reranker,
    )

    print("\nPoints (ground_truth.txt, max 95)")
    print("- Reranker director + reranker cast:", points_reranker_reranker_cast)


if __name__ == "__main__":
    main()
