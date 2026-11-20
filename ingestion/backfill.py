import argparse
import asyncio
import gzip
import json
import os
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

TMDB_TOKEN = os.environ["TMDB_READ_ACCESS_TOKEN"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

TMDB_HEADERS = {
    "Authorization": f"Bearer {TMDB_TOKEN}",
    "accept": "application/json",
}

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


async def download_daily_export(client: httpx.AsyncClient) -> list[dict]:
    """Download TMDB's latest daily movie ID dump (falls back to yesterday if today's isn't published yet)."""
    now_utc = datetime.now(timezone.utc)
    for days_back in range(3):
        date_str = (now_utc - timedelta(days=days_back)).strftime("%m_%d_%Y")
        url = f"https://files.tmdb.org/p/exports/movie_ids_{date_str}.json.gz"
        resp = await client.get(url, timeout=60.0)
        if resp.status_code == 200:
            print(f"Downloaded TMDB export: movie_ids_{date_str}.json.gz")
            lines = gzip.decompress(resp.content).splitlines()
            movies = []
            for line in lines:
                item = json.loads(line)
                if not item.get("adult") and not item.get("video"):
                    movies.append(item)
            return movies
    raise RuntimeError("Could not download TMDB daily export for the last 3 days.")


def build_record(data: dict) -> dict:
    """Format a TMDB movie payload into a Supabase row + metadata soup for embeddings."""
    title = data.get("title") or data.get("original_title") or "Untitled"
    release_date = data.get("release_date") or None  # Postgres date rejects empty strings
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


async def run_backfill(limit: int):
    async with httpx.AsyncClient() as client:
        export_movies = await download_daily_export(client)
        # Sort by popularity descending so your catalog starts with the most recognized global films
        export_movies.sort(key=lambda x: x.get("popularity", 0), reverse=True)
        target_ids = [m["id"] for m in export_movies[:limit]]
        print(f"Fetching metadata for top {len(target_ids)} global movies...")

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
                print(f"Upserted {inserted}/{len(target_ids)} movies...")

    print(f"Done! Total movies synced to Supabase: {inserted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200, help="Number of movies to backfill")
    args = parser.parse_args()
    asyncio.run(run_backfill(args.limit))