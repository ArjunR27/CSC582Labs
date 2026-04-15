from robotproducer import N_CAST, suggest_cast, suggest_cast_reranker, suggest_director_reranker, suggest_director, suggest_director_weighted_vote

# according to scoring rubric on assignment link, 20 for director, 50 for cast, 25 for in top5
def score_test_overview(actual_directors, predicted_directors, actual_cast_list, guessed_cast):

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
        "director_points": director_points,
        "cast_points": cast_points,
        "top5_points": top5_points,
        "total_points": total_points,
    }


def evaluate_test_overview_scores(test_df, train_df, matrix):
    scores = []

    for _, row in test_df.iterrows():
        actual_directors = row["directors"]
        actual_cast_list = row["cast_list"]
        overview = row["overview"]

        predicted_director_row = suggest_director_reranker(overview, matrix, train_df)
        predicted_directors = predicted_director_row["directors"]
        guessed_cast = suggest_cast_reranker(overview, matrix, train_df)

        breakdown = score_test_overview(
            actual_directors=actual_directors,
            predicted_directors=predicted_directors,
            actual_cast_list=actual_cast_list,
            guessed_cast=guessed_cast,
        )
        scores.append(breakdown)

    if scores:
        total_points = sum(s["total_points"] for s in scores)
        total_director_points = sum(s["director_points"] for s in scores)
        total_cast_points = sum(s["cast_points"] for s in scores)
        total_top5_points = sum(s["top5_points"] for s in scores)
        avg_points = total_points / len(scores)
        avg_director_points = total_director_points / len(scores)
        avg_cast_points = total_cast_points / len(scores)
        avg_top5_points = total_top5_points / len(scores)
        print(f"\nAverage director points: {avg_director_points:.2f}/20")
        print(f"Average cast points: {avg_cast_points:.2f}/50")
        print(f"Average top5 points: {avg_top5_points:.2f}/25")
        print(f"\nOverview score avg: {avg_points:.2f}/95")

    return scores


def evaluate_cast_predictions(test_df, train_df, matrix, top_n=N_CAST):
    regular_total_cast_correct = 0
    regular_total_top5_hits = 0
    reranker_total_cast_correct = 0
    reranker_total_top5_hits = 0

    for _, row in test_df.iterrows():
        actual_cast_list = row["cast_list"]
        overview = row["overview"]

        actual_full_set = {name for name, _ in actual_cast_list}
        actual_top5_set = {name for name, order in actual_cast_list if order <= 4}

        regular_guessed_cast = suggest_cast(overview, matrix, train_df, top_n=top_n)
        regular_guessed_cast_set = set(regular_guessed_cast)
        regular_full_overlap = len(regular_guessed_cast_set.intersection(actual_full_set))
        regular_top5_overlap = len(regular_guessed_cast_set.intersection(actual_top5_set))
        regular_total_cast_correct += regular_full_overlap
        regular_total_top5_hits += regular_top5_overlap

        reranker_guessed_cast = suggest_cast_reranker(overview, matrix, train_df, top_n=top_n)
        reranker_guessed_cast_set = set(reranker_guessed_cast)
        reranker_full_overlap = len(reranker_guessed_cast_set.intersection(actual_full_set))
        reranker_top5_overlap = len(reranker_guessed_cast_set.intersection(actual_top5_set))
        reranker_total_cast_correct += reranker_full_overlap
        reranker_total_top5_hits += reranker_top5_overlap

    n = len(test_df)
    if n == 0:
        return {}

    total_guesses = n * top_n
    total_top5_slots = n * 5

    print("Cast eval summary")
    print("Regular suggest_cast")
    print(
        f"total cast correct: {regular_total_cast_correct}/{total_guesses} "
        f"({100 * regular_total_cast_correct / total_guesses:.1f}%)"
    )
    print(
        f"total top5 hits: {regular_total_top5_hits}/{total_top5_slots} "
        f"({100 * regular_total_top5_hits / total_top5_slots:.1f}%)"
    )
    print("Reranker suggest_cast")
    print(
        f"total cast correct: {reranker_total_cast_correct}/{total_guesses} "
        f"({100 * reranker_total_cast_correct / total_guesses:.1f}%)"
    )
    print(
        f"total top5 hits: {reranker_total_top5_hits}/{total_top5_slots} "
        f"({100 * reranker_total_top5_hits / total_top5_slots:.1f}%)"
    )

    return {
        "regular": {
            "total_cast_correct": regular_total_cast_correct,
            "total_top5_hits": regular_total_top5_hits,
        },
        "reranker": {
            "total_cast_correct": reranker_total_cast_correct,
            "total_top5_hits": reranker_total_top5_hits,
        },
    }


def evaluate_director_retrieval(test_df, train_df, matrix):
    correct_top1 = 0
    correct_weighted = 0
    correct_reranker = 0

    for _, row in test_df.iterrows():
        actual_directors = row["directors"]
        overview = row["overview"]

        d_top1 = suggest_director(overview, matrix, train_df)
        d_weighted = suggest_director_weighted_vote(overview, matrix, train_df)
        d_reranker = suggest_director_reranker(overview, matrix, train_df)

        if any(d in d_top1["directors"] for d in actual_directors):
            correct_top1 += 1
        if any(d in d_weighted["directors"] for d in actual_directors):
            correct_weighted += 1
        if any(d in d_reranker["directors"] for d in actual_directors):
            correct_reranker += 1

    n = len(test_df)
    print(f"\nTop-1 cosine:    {correct_top1}/{n} ({100*correct_top1/n:.1f}%)")
    print(f"Weighted vote:   {correct_weighted}/{n} ({100*correct_weighted/n:.1f}%)")
    print(f"Reranker:        {correct_reranker}/{n} ({100*correct_reranker/n:.1f}%)")
    return correct_top1, correct_weighted, correct_reranker
