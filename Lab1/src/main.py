import pandas as pd


def credits_to_df():
    df = pd.read_csv('../movie_data/tmdb_5000_credits.csv')
    return df


def movies_to_df():
    df = pd.read_csv('../movie_data/tmdb_5000_movies.csv')
    return df


def main():
    credits_df = credits_to_df()
    movies_df = movies_to_df()

    combined_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='inner')
    combined_df.to_csv("../movie_data/combined.csv")
    print(combined_df.head())


if __name__ == "__main__":
    main()
