import argparse
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-m3")
    return _model


def encode_query(text: str) -> list[float]:
    vec = get_model().encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


def retrieve_candidates(
    query_embedding: list[float],
    top_k: int = 30,
    languages: list[str] | None = None,
) -> list[dict]:
    """Stage 1: Dense cosine similarity retrieval via pgvector RPC."""
    resp = supabase.rpc(
        "match_movies",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "min_similarity": 0.0,
            "filter_languages": languages,
        },
    ).execute()
    return resp.data or []


def rerank_candidates(
    candidates: list[dict],
    preferred_languages: list[str] | None = None,
    limit: int = 10,
) -> list[dict]:
    """Stage 2: Blend semantic similarity with quality/popularity signals and language preference."""
    preferred_set = {lang.lower() for lang in (preferred_languages or [])}
    scored = []

    for item in candidates:
        sim = float(item.get("similarity") or 0.0)
        vote = float(item.get("vote_average") or 0.0) / 10.0  # normalize 0..1
        pop = math.log1p(max(float(item.get("popularity") or 0.0), 0.0)) / 10.0
        pop = min(pop, 1.0)

        lang_bonus = (
            0.05
            if (preferred_set and (item.get("original_language") or "").lower() in preferred_set)
            else 0.0
        )

        # Weighted score: semantic match leads, boosted by ratings, popularity, and preferred language
        final_score = (0.72 * sim) + (0.18 * vote) + (0.10 * pop) + lang_bonus
        enriched = dict(item)
        enriched["score"] = round(final_score, 4)
        scored.append(enriched)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


def recommend_from_prompt(
    prompt: str,
    languages: list[str] | None = None,
    limit: int = 10,
) -> list[dict]:
    q_vec = encode_query(prompt)
    candidates = retrieve_candidates(q_vec, top_k=max(limit * 3, 25), languages=languages)
    return rerank_candidates(candidates, preferred_languages=languages, limit=limit)


def recommend_similar_to_movie(
    tmdb_id: int,
    languages: list[str] | None = None,
    limit: int = 10,
) -> list[dict]:
    row = (
        supabase.table("movies")
        .select("tmdb_id, title, embedding")
        .eq("tmdb_id", tmdb_id)
        .single()
        .execute()
    ).data
    if not row or not row.get("embedding"):
        raise ValueError(f"Movie {tmdb_id} not found or missing embedding.")

    emb = row["embedding"]
    if isinstance(emb, str):
        emb = [float(x) for x in emb.strip().strip("[]").split(",") if x.strip()]

    candidates = retrieve_candidates(emb, top_k=max(limit * 3, 25), languages=languages)
    filtered = [c for c in candidates if int(c["tmdb_id"]) != tmdb_id]
    return rerank_candidates(filtered, preferred_languages=languages, limit=limit)


def recommend_for_user(
    user_id: str,
    limit: int = 10,
    exclude_interacted: bool = True,
) -> list[dict]:
    """Recommend movies using the user's `taste_embedding` and `preferred_languages`."""
    user_row = (
        supabase.table("users")
        .select("id, taste_embedding, preferred_languages")
        .eq("id", user_id)
        .single()
        .execute()
    ).data
    if not user_row:
        raise ValueError(f"User {user_id} not found.")

    raw_vec = user_row.get("taste_embedding")
    if not raw_vec:
        raise ValueError(f"User {user_id} has no taste_embedding yet. Log interactions first.")

    if isinstance(raw_vec, str):
        taste_vec = [float(x) for x in raw_vec.strip().strip("[]").split(",") if x.strip()]
    else:
        taste_vec = list(raw_vec)

    preferred_langs = user_row.get("preferred_languages") or None

    seen_ids: set[int] = set()
    if exclude_interacted:
        inter_rows = (
            supabase.table("user_interactions")
            .select("tmdb_id")
            .eq("user_id", user_id)
            .execute()
        ).data or []
        seen_ids = {int(r["tmdb_id"]) for r in inter_rows if r.get("tmdb_id") is not None}

    candidates = retrieve_candidates(
        taste_vec,
        top_k=max(limit * 4, 30),
        languages=None,
    )
    filtered = [c for c in candidates if int(c["tmdb_id"]) not in seen_ids]
    return rerank_candidates(filtered, preferred_languages=preferred_langs, limit=limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NextMovie retrieve-and-rerank engine")
    parser.add_argument("--query", type=str, default="mind-bending sci-fi thriller with twist endings")
    parser.add_argument("--tmdb-id", type=int, default=None, help="Recommend similar to a TMDB movie ID")
    parser.add_argument("--langs", type=str, default=None, help="Comma-separated language codes, e.g. en,ml,hi")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--user-id", type=str, default=None, help="Recommend using user's taste_embedding")
    args = parser.parse_args()

    lang_list = [x.strip() for x in args.langs.split(",")] if args.langs else None

    if args.user_id:
        results = recommend_for_user(args.user_id, limit=args.limit)
        header = f"Personalized recommendations for user {args.user_id}"
    elif args.tmdb_id:
        results = recommend_similar_to_movie(args.tmdb_id, languages=lang_list, limit=args.limit)
        header = f"Similar to TMDB #{args.tmdb_id}"
    else:
        results = recommend_from_prompt(args.query, languages=lang_list, limit=args.limit)
        header = f"Results for: '{args.query}'"

    print(f"\n=== {header} ===")
    for idx, m in enumerate(results, 1):
        yr = (m.get("release_date") or "????")[:4]
        print(
            f"{idx:2d}. {m['title']} ({yr}) [{m.get('original_language')}] "
            f"| score={m['score']} | sim={round(float(m.get('similarity') or 0), 4)} "
            f"| rating={m.get('vote_average')}"
        )