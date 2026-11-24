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

GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}

TARGET_LANGUAGES = [
    {"code": "ml", "name": "Malayalam", "pages": 3},
    {"code": "ta", "name": "Tamil", "pages": 3},
    {"code": "te", "name": "Telugu", "pages": 3},
    {"code": "hi", "name": "Hindi", "pages": 3},
    {"code": "kn", "name": "Kannada", "pages": 2},
    {"code": "ko", "name": "Korean", "pages": 3},
    {"code": "ja", "name": "Japanese", "pages": 3},
    {"code": "es", "name": "Spanish", "pages": 3},
    {"code": "fr", "name": "French", "pages": 3},
    {"code": "de", "name": "German", "pages": 2},
    {"code": "it", "name": "Italian", "pages": 2},
]


def backfill_languages():
    total_added = 0
    print("Starting multi-language movie backfill from TMDB...")

    for target in TARGET_LANGUAGES:
        code = target["code"]
        name = target["name"]
        pages = target["pages"]
        lang_movies = 0

        # Fetch both top-voted (critically acclaimed) and trending popular
        for sort_metric in ["vote_count.desc", "popularity.desc"]:
            for page in range(1, pages + 1):
                params = {
                    "with_original_language": code,
                    "sort_by": sort_metric,
                    "vote_count.gte": 5 if code in ["kn", "ml"] else 15,
                    "page": page,
                    "include_adult": "false",
                }
                try:
                    resp = requests.get(
                        f"{TMDB_BASE}/discover/movie",
                        headers=TMDB_HEADERS,
                        params=params,
                        timeout=15,
                    )
                    if resp.status_code != 200:
                        break
                    results = resp.json().get("results", [])
                    if not results:
                        break

                    rows = []
                    for m in results:
                        title = m.get("title") or m.get("original_title")
                        overview = (m.get("overview") or "").strip()
                        if not title:
                            continue
                        
                        year = (m.get("release_date") or "????")[:4]
                        genres = [GENRE_MAP[gid] for gid in m.get("genre_ids", []) if gid in GENRE_MAP]
                        soup = f"{title} ({year}). Language: {code}. Genres: {', '.join(genres) or 'Cinema'}. Plot: {overview or 'Acclaimed film.'}"

                        rows.append({
                            "tmdb_id": m["id"],
                            "title": title,
                            "original_language": code,
                            "release_date": m.get("release_date") or None,
                            "popularity": float(m.get("popularity", 0.0)),
                            "vote_average": float(m.get("vote_average", 0.0)),
                            "genres": genres,
                            "poster_path": m.get("poster_path"),
                            "metadata_soup": soup,
                        })

                    if rows:
                        supabase.table("movies").upsert(rows, on_conflict="tmdb_id").execute()
                        lang_movies += len(rows)
                        total_added += len(rows)

                    time.sleep(0.15)
                except Exception as e:
                    print(f"Error fetching {name} p{page}: {e}")
                    break

        print(f"Upserted {lang_movies} titles for {name} ({code})")

    print(f"\nAll done! Total movie records upserted: {total_added}")


if __name__ == "__main__":
    backfill_languages()
