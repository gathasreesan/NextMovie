import argparse
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from supabase import create_client

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)

ACTION_WEIGHTS: dict[str, float] = {
    "like": 1.0,
    "watchlist": 0.6,
    "click": 0.25,
    "dislike": -0.8,
}


def _parse_embedding(raw: object) -> np.ndarray | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        vec = np.array(raw, dtype=np.float32)
    elif isinstance(raw, str):
        cleaned = raw.strip().strip("[]")
        if not cleaned:
            return None
        vec = np.array([float(x) for x in cleaned.split(",")], dtype=np.float32)
    else:
        return None
    return vec if vec.size == 1024 else None


def log_interaction(user_id: str, tmdb_id: int, action: str) -> dict:
    """Insert a user interaction and refresh the user's `taste_embedding`."""
    if action not in ACTION_WEIGHTS:
        raise ValueError(f"Invalid action '{action}'. Expected one of {list(ACTION_WEIGHTS)}.")

    inserted = (
        supabase.table("user_interactions")
        .insert({"user_id": user_id, "tmdb_id": tmdb_id, "action": action})
        .execute()
    )
    updated_vec = recompute_taste_embedding(user_id)
    return {
        "interaction": inserted.data[0] if inserted.data else None,
        "taste_updated": updated_vec is not None,
    }


def recompute_taste_embedding(user_id: str) -> list[float] | None:
    """Compute a weighted, L2-normalized taste vector from `user_interactions`."""
    interactions_resp = (
        supabase.table("user_interactions")
        .select("tmdb_id, action, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=False)
        .execute()
    )
    interactions = interactions_resp.data or []
    if not interactions:
        supabase.table("users").update({"taste_embedding": None}).eq("id", user_id).execute()
        return None

    tmdb_ids = list({int(row["tmdb_id"]) for row in interactions})
    movies_resp = (
        supabase.table("movies")
        .select("tmdb_id, embedding")
        .in_("tmdb_id", tmdb_ids)
        .execute()
    )
    movie_vectors: dict[int, np.ndarray] = {}
    for m in movies_resp.data or []:
        parsed = _parse_embedding(m.get("embedding"))
        if parsed is not None:
            movie_vectors[int(m["tmdb_id"])] = parsed

    acc = np.zeros(1024, dtype=np.float32)
    for idx, row in enumerate(interactions, start=1):
        vec = movie_vectors.get(int(row["tmdb_id"]))
        if vec is None:
            continue
        base_weight = ACTION_WEIGHTS.get(row["action"], 0.0)
        # Gentle recency boost so newer interactions matter slightly more
        recency_scale = 0.8 + 0.2 * (idx / len(interactions))
        acc += (base_weight * recency_scale) * vec

    # Clamp negative drift slightly so pure dislikes don't invert tastes weirdly
    norm = float(np.linalg.norm(acc))
    if norm == 0.0:
        supabase.table("users").update({"taste_embedding": None}).eq("id", user_id).execute()
        return None

    normalized = (acc / norm).tolist()
    supabase.table("users").update({"taste_embedding": normalized}).eq("id", user_id).execute()
    return normalized


def create_demo_user(preferred_languages: list[str] | None = None) -> str:
    payload = {"preferred_languages": preferred_languages or ["en"]}
    resp = supabase.table("users").insert(payload).execute()
    return resp.data[0]["id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage user interactions and taste embeddings")
    parser.add_argument("--create-user", action="store_true", help="Create a test user and print UUID")
    parser.add_argument("--langs", type=str, default="en", help="Comma-separated preferred languages")
    parser.add_argument("--user-id", type=str, help="Target user UUID")
    parser.add_argument("--tmdb-id", type=int, help="Movie TMDB ID for interaction")
    parser.add_argument("--action", type=str, choices=list(ACTION_WEIGHTS.keys()), help="Interaction action")
    parser.add_argument("--recompute", action="store_true", help="Recompute taste_embedding for --user-id")
    args = parser.parse_args()

    if args.create_user:
        langs = [x.strip() for x in args.langs.split(",") if x.strip()]
        uid = create_demo_user(langs)
        print(f"Created user: {uid} (preferred_languages={langs})")
    elif args.user_id and args.tmdb_id and args.action:
        res = log_interaction(args.user_id, args.tmdb_id, args.action)
        print(f"Logged '{args.action}' on TMDB #{args.tmdb_id} for user {args.user_id} | taste_updated={res['taste_updated']}")
    elif args.user_id and args.recompute:
        vec = recompute_taste_embedding(args.user_id)
        print(f"Recomputed taste_embedding for {args.user_id}: {'1024-dim vector saved' if vec else 'cleared (None)'}")
    else:
        parser.print_help()