import argparse
import asyncio
import gzip
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from supabase import create_client

# Ensure project root is on sys.path so `from ml.encoder import run_encoder` works
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.encoder import run_encoder

load_dotenv(ROOT_DIR / ".env")

TMDB_TOKEN = os.environ["TMDB_READ_ACCESS_TOKEN"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

TMDB_HEADERS = {
    "Authorization": f"Bearer {TMDB_TOKEN}",
    "accept": "application/json",
}

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


async def download_daily_export(client: httpx.AsyncClient) -> list[dict]:
    """Download TMDB's latest daily movie ID export (checks up to 3 days back)."""
    now_utc = datetime.now(timezone.utc)
    for days_back in range(3):
        date_str = (now_utc - timedelta(days=days_back)).strftime("%m_%d_%Y")
        url = f"https://files.tmdb.org/p/exports/movie_ids_{date_str}.json.gz"
        resp = await client.get(url, timeout=60.0)
        if resp.status_code == 200:
            print(f"Downloaded TMDB export: movie_ids_{date_str}.json.gz")
            lines = gzip.decompress(resp.content).splitlines()
            return [
                json.loads(line)
                for line in lines
                if not json.loads(line).get("adult") and not json.loads(line).get("video")
            ]
    raise RuntimeError("Could not download TMDB daily export for the last 3 days.")


def get_existing_tmdb_ids() -> set[int]:
    """Fetch all tmdb_id values currently stored in Supabase `movies`."""
    existing: set[int] = set()
    page_size = 1000
    start = 0

    while True:
        resp = (
            supabase.table("movies")
            .select("tmdb_id")
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = resp.data or []
        for row in rows:
            if "tmdb_id" in row:
                existing.add(int(row["tmdb_id"]))
        if len(rows) < page_size:
            break
        start += page_size

    return existing


def build_record(data: dict) -> dict:
    """Format a TMDB movie payload into a Supabase `movies` row."""
    title = data.get("title") or data.get("original_title") or "Untitled"
    release_date = data.get("release_date") or None
    year = release_date[:4] if release_date else "Unknown"
    orig_lang = data.get("original_language") or "unknown"
    genres = [g["name"] for g in data.get("genres", [])]
    overview = (data.get("overview") or "").strip()

    crew = data.get("credits", {}).get("crew", [])
    directors = [c["name"] for c in crew if c.get("job") == "Director"]
    director_str = ", ".join(directors[:2]) if directors else "Unknown"

    cast = data.get("credits", {}).get("cast", [])
    top_cast = ", ".join(c["name"] for c in cast[:5]) if cast else "Unknown"

    kw_list = data.get("keywords", {}).get("keywords", [])
    keywords = ", ".join(k["name"] for k in kw_list[:10]) if kw_list else "None"

    metadata_soup = (
        f"{title} ({year}). Language: {orig_lang}. "
        f"Genres: {', '.join(genres) or 'Unknown'}. "
        f"Director: {director_str}. Cast: {top_cast}. "
        f"Themes: {keywords}. Plot: {overview}"
    )

    return {
        "tmdb_id": data["id"],
        "title": title,
        "original_language": orig_lang,
        "release_date": release_date,
        "popularity": float(data.get("popularity") or 0.0),
        "vote_average": float(data.get("vote_average") or 0.0),
        "genres": genres,
        "poster_path": data.get("poster_path"),
        "metadata_soup": metadata_soup,
    }


async def fetch_movie(client: httpx.AsyncClient, sem: asyncio.Semaphore, movie_id: int) -> dict | None:
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"append_to_response": "credits,keywords"}
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.get(url, headers=TMDB_HEADERS, params=params, timeout=15.0)
                if resp.status_code == 200:
                    return build_record(resp.json())
                if resp.status_code == 429:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                return None
            except httpx.HTTPError:
                await asyncio.sleep(1.0)
    return None


async def run_daily_sync(max_new: int = 50, skip_embed: bool = False):
    async with httpx.AsyncClient() as client:
        export_movies = await download_daily_export(client)
        existing_ids = get_existing_tmdb_ids()
        print(f"Existing movies in Supabase: {len(existing_ids)}")

        new_candidates = [m for m in export_movies if int(m["id"]) not in existing_ids]
        new_candidates.sort(key=lambda x: x.get("popularity", 0), reverse=True)
        target_ids = [int(m["id"]) for m in new_candidates[:max_new]]

        if not target_ids:
            print("No new candidate movies to sync.")
            return

        print(f"Syncing {len(target_ids)} new movies from TMDB...")
        sem = asyncio.Semaphore(15)
        batch_size = 50
        inserted = 0

        for i in range(0, len(target_ids), batch_size):
            batch_ids = target_ids[i : i + batch_size]
            tasks = [fetch_movie(client, sem, mid) for mid in batch_ids]
            results = await asyncio.gather(*tasks)
            rows = [r for r in results if r is not None]

            if rows:
                supabase.table("movies").upsert(rows, on_conflict="tmdb_id").execute()
                inserted += len(rows)
                print(f"Upserted {inserted}/{len(target_ids)} new movies...")

        print(f"Daily sync complete. Added {inserted} new movies.")

    if inserted > 0 and not skip_embed:
        run_encoder()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental daily TMDB sync + embedding")
    parser.add_argument("--max-new", type=int, default=50, help="Max new movies to fetch per run")
    parser.add_argument("--skip-embed", action="store_true", help="Skip running ml/encoder.py afterward")
    args = parser.parse_args()
    asyncio.run(run_daily_sync(max_new=args.max_new, skip_embed=args.skip_embed))