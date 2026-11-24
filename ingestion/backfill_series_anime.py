import os
import time
import requests
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
TMDB_BASE = "https://api.themoviedb.org/3"

TV_GENRE_MAP = {
    10759: "Action & Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    10762: "Kids",
    9648: "Mystery",
    10763: "News",
    10764: "Reality",
    10765: "Sci-Fi & Fantasy",
    10766: "Soap",
    10767: "Talk",
    10768: "War & Politics",
    37: "Western",
}

# Categories to ingest
CATEGORIES = [
    # 1. Top Anime Series
    {
        "name": "Anime Series",
        "params": {
            "with_original_language": "ja",
            "with_genres": "16",
            "sort_by": "vote_count.desc",
            "vote_count.gte": 50,
        },
        "extra_tags": ["Anime", "TV Series"],
        "pages": 4,
    },
    # 2. Top K-Dramas & Asian Dramas
    {
        "name": "K-Dramas & Asian Dramas",
        "params": {
            "with_original_language": "ko",
            "sort_by": "popularity.desc",
            "vote_count.gte": 30,
        },
        "extra_tags": ["K-Drama", "TV Series"],
        "pages": 4,
    },
    # 3. Top Indian Web Series
    {
        "name": "Indian Web Series",
        "params": {
            "with_origin_country": "IN",
            "sort_by": "vote_count.desc",
            "vote_count.gte": 10,
        },
        "extra_tags": ["Indian Series", "TV Series"],
        "pages": 3,
    },
    # 4. Top Acclaimed Global TV Series & Limited Dramas
    {
        "name": "Global Acclaimed Series",
        "params": {
            "sort_by": "vote_count.desc",
            "vote_count.gte": 500,
        },
        "extra_tags": ["TV Series"],
        "pages": 5,
    },
    # 5. Trending & Binge-Worthy TV Series
    {
        "name": "Trending Series",
        "params": {
            "sort_by": "popularity.desc",
            "vote_count.gte": 100,
        },
        "extra_tags": ["TV Series"],
        "pages": 4,
    },
]

def backfill_series():
    total_added = 0
    seen_tv_ids = set()
    print("Starting TV Series, Anime & Drama backfill from TMDB...")

    for cat in CATEGORIES:
        cat_name = cat["name"]
        params = cat["params"]
        extra_tags = cat["extra_tags"]
        pages = cat["pages"]
        cat_count = 0

        for page in range(1, pages + 1):
            query_params = {**params, "page": page, "include_adult": "false"}
            try:
                resp = requests.get(
                    f"{TMDB_BASE}/discover/tv",
                    headers=TMDB_HEADERS,
                    params=query_params,
                    timeout=15,
                )
                if resp.status_code != 200:
                    print(f"Error status {resp.status_code} on {cat_name} p{page}")
                    break
                results = resp.json().get("results", [])
                if not results:
                    break

                rows = []
                for s in results:
                    tv_id = s.get("id")
                    if not tv_id or tv_id in seen_tv_ids:
                        continue
                    seen_tv_ids.add(tv_id)

                    title = s.get("name") or s.get("original_name")
                    overview = (s.get("overview") or "").strip()
                    if not title:
                        continue

                    air_date = s.get("first_air_date") or None
                    year = air_date[:4] if air_date else "????"
                    lang = s.get("original_language") or "en"

                    # Genres with specialized tags
                    raw_genres = [TV_GENRE_MAP[gid] for gid in s.get("genre_ids", []) if gid in TV_GENRE_MAP]
                    genres = list(dict.fromkeys(extra_tags + raw_genres))

                    type_label = "Anime Series" if "Anime" in extra_tags else ("K-Drama" if "K-Drama" in extra_tags else "TV Series")
                    soup = f"{title} ({year}). Format: {type_label}. Language: {lang}. Genres: {', '.join(genres)}. Synopsis: {overview or 'Acclaimed television drama.'}"

                    # Unique offset ID for TV shows to avoid movie TMDB ID collision
                    offset_id = tv_id + 100_000_000

                    rows.append({
                        "tmdb_id": offset_id,
                        "title": title,
                        "original_language": lang,
                        "release_date": air_date,
                        "popularity": float(s.get("popularity", 0.0)),
                        "vote_average": float(s.get("vote_average", 0.0)),
                        "genres": genres,
                        "poster_path": s.get("poster_path"),
                        "metadata_soup": soup,
                    })

                if rows:
                    supabase.table("movies").upsert(rows, on_conflict="tmdb_id").execute()
                    cat_count += len(rows)
                    total_added += len(rows)

                time.sleep(0.15)
            except Exception as e:
                print(f"Exception on {cat_name} p{page}: {e}")
                break

        print(f"Upserted {cat_count} titles for {cat_name}")

    print(f"\nSeries/Anime backfill complete! Total television titles upserted: {total_added}")

if __name__ == "__main__":
    backfill_series()
