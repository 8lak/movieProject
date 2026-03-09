import pandas as pd
import re

DATA_DIR = "movienetwork"

EDGE_CAST = f"{DATA_DIR}/edge-cast.csv"
EDGE_CREDIT = f"{DATA_DIR}/edge-credit.csv"
VERTEX_ACTOR = f"{DATA_DIR}/vertex-actor.csv"
VERTEX_CREW = f"{DATA_DIR}/vertex-crew.csv"
VERTEX_MOVIES = f"{DATA_DIR}/vertex-movies.csv"
TITLE_PRINCIPALS = "title.principals.tsv"
NAME_BASICS = "name.basics.tsv"

# Importance filter settings (IMDb billing order; lower = more prominent)
BILLING_CUTOFF = 5
USE_IMPORTANCE_FILTER = True


def load_data():
    movies = pd.read_csv(VERTEX_MOVIES, low_memory=False)
    cast = pd.read_csv(EDGE_CAST, low_memory=False)
    credit = pd.read_csv(EDGE_CREDIT, low_memory=False)
    actors = pd.read_csv(VERTEX_ACTOR, low_memory=False)
    crew = pd.read_csv(VERTEX_CREW, low_memory=False)
    return movies, cast, credit, actors, crew


def split_genres(genre_value: str):
    if pd.isna(genre_value):
        return []
    # Split on '|' or ',' and normalize whitespace
    parts = re.split(r"\s*[|,]\s*", str(genre_value))
    return [p for p in (s.strip() for s in parts) if p]


def build_actor_name_to_nconst(actors: pd.DataFrame, name_basics_path: str) -> dict:
    actor_names = set(actors["Label"].dropna().astype(str))
    name_counts = {}
    name_first = {}

    for chunk in pd.read_csv(
        name_basics_path,
        sep="\t",
        usecols=["nconst", "primaryName"],
        dtype=str,
        chunksize=500_000,
        low_memory=False,
    ):
        sub = chunk[chunk["primaryName"].isin(actor_names)]
        for name, nconst in zip(sub["primaryName"], sub["nconst"]):
            name_counts[name] = name_counts.get(name, 0) + 1
            if name not in name_first:
                name_first[name] = nconst

    # Keep only unique name -> nconst mappings
    unique_map = {name: nconst for name, nconst in name_first.items() if name_counts.get(name, 0) == 1}
    dropped = len(name_first) - len(unique_map)
    print(f"Name->nconst mapping: {len(unique_map)} unique, {dropped} dropped (ambiguous).")
    return unique_map


def build_top_billed_keys(movies: pd.DataFrame, principals_path: str, nconst_set: set, cutoff: int) -> set:
    movie_tconsts = set(movies["imdbId"].dropna().astype(str))
    keys = set()
    usecols = ["tconst", "ordering", "nconst", "category"]

    for chunk in pd.read_csv(
        principals_path,
        sep="\t",
        usecols=usecols,
        dtype=str,
        chunksize=1_000_000,
        low_memory=False,
    ):
        sub = chunk[
            chunk["tconst"].isin(movie_tconsts)
            & chunk["nconst"].isin(nconst_set)
            & chunk["category"].isin(["actor", "actress"])
        ].copy()
        if sub.empty:
            continue
        sub["ordering"] = pd.to_numeric(sub["ordering"], errors="coerce")
        sub = sub[sub["ordering"] <= cutoff]
        if sub.empty:
            continue
        keys.update((tconst + "|" + nconst) for tconst, nconst in zip(sub["tconst"], sub["nconst"]))

    return keys


def filter_cast_by_importance(cast: pd.DataFrame, movies: pd.DataFrame, actors: pd.DataFrame) -> pd.DataFrame:
    if not USE_IMPORTANCE_FILTER:
        return cast

    name_to_nconst = build_actor_name_to_nconst(actors, NAME_BASICS)
    actor_to_nconst = (
        actors[["Id", "Label"]]
        .assign(nconst=lambda d: d["Label"].map(name_to_nconst))
        .dropna(subset=["nconst"])
    )
    actor_to_nconst = actor_to_nconst.set_index("Id")["nconst"].to_dict()

    top_billed_keys = build_top_billed_keys(movies, TITLE_PRINCIPALS, set(actor_to_nconst.values()), BILLING_CUTOFF)

    cast_imdb = cast.merge(
        movies[["Id", "imdbId"]].rename(columns={"Id": "MovieId"}),
        left_on="Source",
        right_on="MovieId",
        how="left",
    )
    cast_imdb["nconst"] = cast_imdb["Target"].map(actor_to_nconst)
    cast_imdb = cast_imdb.dropna(subset=["imdbId", "nconst"])

    keys = cast_imdb["imdbId"].astype(str) + "|" + cast_imdb["nconst"].astype(str)
    cast_imdb = cast_imdb[keys.isin(top_billed_keys)]

    # Return in original shape
    return cast_imdb[cast.columns].copy()


def analyze_action_actor_genres(movies, cast, actors, action_keyword="Action"):
    action_mask = movies["Genre"].astype(str).str.contains(action_keyword, case=False, na=False)
    action_movies = movies[action_mask].copy()
    action_movie_ids = set(action_movies["Id"])

    action_cast = cast[cast["Source"].isin(action_movie_ids)].copy()
    action_actor_ids = set(action_cast["Target"])

    # All movies for actors who appear in action movies
    action_actor_cast = cast[cast["Target"].isin(action_actor_ids)].copy()
    action_actor_movies = action_actor_cast.merge(
        movies[["Id", "Label", "Year", "Genre"]],
        left_on="Source",
        right_on="Id",
        how="left",
    )

    # Genre distribution across all movies those actors appear in
    exploded = action_actor_movies.copy()
    exploded["GenreList"] = exploded["Genre"].apply(split_genres)
    exploded = exploded.explode("GenreList")
    genre_counts = (
        exploded["GenreList"]
        .dropna()
        .value_counts()
        .rename_axis("Genre")
        .reset_index(name="MovieCredits")
    )

    # Helpful summaries
    num_action_movies = len(action_movies)
    num_action_actors = len(action_actor_ids)
    num_total_credits = len(action_actor_movies)

    action_actor_names = actors[actors["Id"].isin(action_actor_ids)][["Id", "Label"]]

    return {
        "action_movies": action_movies,
        "action_movie_ids": action_movie_ids,
        "action_actor_ids": action_actor_ids,
        "action_actor_names": action_actor_names,
        "action_actor_movies": action_actor_movies,
        "genre_counts": genre_counts,
        "summary": {
            "action_movies": num_action_movies,
            "action_actors": num_action_actors,
            "total_movie_credits_for_action_actors": num_total_credits,
        },
    }


def add_role_flags(credit: pd.DataFrame) -> pd.DataFrame:
    tokens = (
        credit["Role"]
        .astype(str)
        .str.split(",")
        .apply(lambda xs: [x.strip() for x in xs if x.strip()])
    )
    credit = credit.copy()
    credit["RoleTokens"] = tokens
    credit["IsDirector"] = credit["RoleTokens"].apply(
        lambda xs: any(t in {"Director", "Directing"} for t in xs)
    )
    credit["IsProducer"] = credit["RoleTokens"].apply(
        lambda xs: any(t in {"Producer"} for t in xs)
    )
    return credit


def analyze_relationships(movies, cast, credit, actors, crew, action_actor_ids=None):
    credit = add_role_flags(credit)

    directors = credit[credit["IsDirector"]].copy()
    producers = credit[credit["IsProducer"]].copy()

    # Join crew labels
    directors = directors.merge(
        crew[["Id", "Label"]].rename(columns={"Label": "DirectorLabel"}),
        left_on="Target",
        right_on="Id",
        how="left",
    )
    producers = producers.merge(
        crew[["Id", "Label"]].rename(columns={"Label": "ProducerLabel"}),
        left_on="Target",
        right_on="Id",
        how="left",
    )

    # Join cast
    cast_small = cast[["Source", "Target"]].rename(columns={"Target": "ActorId"})
    if action_actor_ids is not None:
        cast_small = cast_small[cast_small["ActorId"].isin(action_actor_ids)]

    # Loyalty: Actor + Director
    ad = cast_small.merge(
        directors[["Source", "Target", "DirectorLabel"]].rename(columns={"Target": "DirectorId"}),
        on="Source",
        how="inner",
    )
    ad = ad[ad["ActorId"] != ad["DirectorId"]]
    loyalty_pairs = (
        ad.groupby(["DirectorId", "DirectorLabel", "ActorId"]).size()
        .reset_index(name="SharedMovies")
        .sort_values("SharedMovies", ascending=False)
    )

    # Employment: Actor + Producer
    ap = cast_small.merge(
        producers[["Source", "Target", "ProducerLabel"]].rename(columns={"Target": "ProducerId"}),
        on="Source",
        how="inner",
    )
    ap = ap[ap["ActorId"] != ap["ProducerId"]]
    employment_pairs = (
        ap.groupby(["ProducerId", "ProducerLabel", "ActorId"]).size()
        .reset_index(name="SharedMovies")
        .sort_values("SharedMovies", ascending=False)
    )

    # Power Triangle: Actor + Director + Producer
    adp = ad.merge(
        producers[["Source", "Target", "ProducerLabel"]].rename(columns={"Target": "ProducerId"}),
        on="Source",
        how="inner",
    )
    adp = adp[adp["ActorId"] != adp["ProducerId"]]
    triplet_counts = (
        adp.groupby(["ActorId", "DirectorId", "ProducerId"]).size()
        .reset_index(name="TripletMovies")
        .sort_values("TripletMovies", ascending=False)
    )

    # Preference score: Actor + Director, producer diversity
    ad_producer_diversity = (
        ad.merge(
            producers[["Source", "Target"]].rename(columns={"Target": "ProducerId"}),
            on="Source",
            how="left",
        )
        .groupby(["ActorId", "DirectorId"])
        .agg(SharedMovies=("Source", "nunique"), UniqueProducers=("ProducerId", "nunique"))
        .reset_index()
        .sort_values("SharedMovies", ascending=False)
    )

    # Franchise score: Actor + Producer, director diversity
    ap_director_diversity = (
        ap.merge(
            directors[["Source", "Target"]].rename(columns={"Target": "DirectorId"}),
            on="Source",
            how="left",
        )
        .groupby(["ActorId", "ProducerId"])
        .agg(SharedMovies=("Source", "nunique"), UniqueDirectors=("DirectorId", "nunique"))
        .reset_index()
        .sort_values("SharedMovies", ascending=False)
    )

    # Recurring Director+Producer pairs and their actors
    dp_pairs = (
        directors[["Source", "Target"]]
        .rename(columns={"Target": "DirectorId"})
        .merge(
            producers[["Source", "Target"]].rename(columns={"Target": "ProducerId"}),
            on="Source",
            how="inner",
        )
    )
    dp_pair_counts = (
        dp_pairs.groupby(["DirectorId", "ProducerId"])
        .agg(SharedMovies=("Source", "nunique"))
        .reset_index()
        .sort_values("SharedMovies", ascending=False)
    )
    recurring_dp = dp_pair_counts[dp_pair_counts["SharedMovies"] >= 2]
    recurring_dp_actors = (
        adp.merge(recurring_dp, on=["DirectorId", "ProducerId"], how="inner")
        .groupby(["ActorId"])
        .agg(TripletMovies=("Source", "nunique"))
        .reset_index()
        .sort_values("TripletMovies", ascending=False)
    )

    # Attach names for readability
    crew_names = crew[["Id", "Label"]].rename(columns={"Id": "CrewId", "Label": "CrewLabel"})
    actor_names = actors[["Id", "Label"]].rename(columns={"Id": "ActorId", "Label": "ActorLabel"})
    loyalty_pairs = loyalty_pairs.merge(actor_names, on="ActorId", how="left")
    employment_pairs = employment_pairs.merge(actor_names, on="ActorId", how="left")
    triplet_counts = (
        triplet_counts.merge(actor_names, on="ActorId", how="left")
        .merge(crew_names.rename(columns={"CrewId": "DirectorId", "CrewLabel": "DirectorLabel"}), on="DirectorId", how="left")
        .merge(crew_names.rename(columns={"CrewId": "ProducerId", "CrewLabel": "ProducerLabel"}), on="ProducerId", how="left")
    )
    ad_producer_diversity = (
        ad_producer_diversity.merge(actor_names, on="ActorId", how="left")
        .merge(crew_names.rename(columns={"CrewId": "DirectorId", "CrewLabel": "DirectorLabel"}), on="DirectorId", how="left")
    )
    ap_director_diversity = (
        ap_director_diversity.merge(actor_names, on="ActorId", how="left")
        .merge(crew_names.rename(columns={"CrewId": "ProducerId", "CrewLabel": "ProducerLabel"}), on="ProducerId", how="left")
    )
    recurring_dp_actors = recurring_dp_actors.merge(actor_names, on="ActorId", how="left")

    return {
        "loyalty_pairs": loyalty_pairs,
        "employment_pairs": employment_pairs,
        "triplet_counts": triplet_counts,
        "preference_scores": ad_producer_diversity,
        "franchise_scores": ap_director_diversity,
        "dp_pair_counts": dp_pair_counts,
        "recurring_dp_actors": recurring_dp_actors,
    }


def main():
    movies, cast, credit, actors, crew = load_data()

    if USE_IMPORTANCE_FILTER:
        cast = filter_cast_by_importance(cast, movies, actors)
        print(f"Applied importance filter: billing <= {BILLING_CUTOFF}. Cast rows remaining: {len(cast)}")

    action_results = analyze_action_actor_genres(movies, cast, actors)

    print("=== Action Actor Summary ===")
    for k, v in action_results["summary"].items():
        print(f"{k}: {v}")

    print("\nTop 20 genres for actors who appear in action movies:")
    print(action_results["genre_counts"].head(20).to_string(index=False))

    # Relationship analysis restricted to action actors
    rel = analyze_relationships(
        movies,
        cast,
        credit,
        actors,
        crew,
        action_actor_ids=action_results["action_actor_ids"],
    )

    print("\nTop 20 Loyalty pairs (Director ↔ Actor, by shared movies):")
    print(rel["loyalty_pairs"][["DirectorLabel", "ActorLabel", "SharedMovies"]].head(20).to_string(index=False))

    print("\nTop 20 Employment pairs (Producer ↔ Actor, by shared movies):")
    print(rel["employment_pairs"][["ProducerLabel", "ActorLabel", "SharedMovies"]].head(20).to_string(index=False))

    print("\nTop 20 Power Triangle triplets (Actor + Director + Producer):")
    print(
        rel["triplet_counts"]
        .head(20)[["ActorLabel", "DirectorLabel", "ProducerLabel", "TripletMovies"]]
        .to_string(index=False)
    )

    print("\nTop 20 Preference scores (Actor + Director, producer diversity):")
    print(
        rel["preference_scores"]
        .head(20)[["ActorLabel", "DirectorLabel", "SharedMovies", "UniqueProducers"]]
        .to_string(index=False)
    )

    print("\nTop 20 Franchise scores (Actor + Producer, director diversity):")
    print(
        rel["franchise_scores"]
        .head(20)[["ActorLabel", "ProducerLabel", "SharedMovies", "UniqueDirectors"]]
        .to_string(index=False)
    )

    print("\nTop 20 actors in recurring Director+Producer pairs (>=2 shared movies):")
    print(rel["recurring_dp_actors"][["ActorLabel", "TripletMovies"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
