import argparse
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

INDUSTRY_BUCKETS = [
    # India — State & Language Industries
    {"name": "Mollywood (Kerala / Malayalam)", "lang": "ml", "country": "IN"},
    {"name": "Kollywood (Tamil Nadu / Tamil)", "lang": "ta", "country": "IN"},
    {"name": "Tollywood (Andhra & Telangana / Telugu)", "lang": "te", "country": "IN"},
    {"name": "Bollywood (Hindi)", "lang": "hi", "country": "IN"},
    {"name": "Sandalwood (Karnataka / Kannada)", "lang": "kn", "country": "IN"},
    {"name": "Marathi Cinema (Maharashtra)", "lang": "mr", "country": "IN"},
    {"name": "Tollygunge (West Bengal / Bengali)", "lang": "bn", "country": "IN"},
    {"name": "Pollywood (Punjab / Punjabi)", "lang": "pa", "country": "IN"},
    {"name": "Dhollywood (Gujarat / Gujarati)", "lang": "gu", "country": "IN"},
    {"name": "Ollywood (Odisha / Odia)", "lang": "or", "country": "IN"},
    {"name": "Jollywood (Assam / Assamese)", "lang": "as", "country": "IN"},
    # Global Regions & Industries
    {"name": "Hollywood & North America (English)", "lang": "en", "country": "US"},
    {"name": "British & Irish Cinema (UK/IE)", "lang": "en", "country": "GB"},
    {"name": "Korean Cinema (Hallyu)", "lang": "ko", "country": "KR"},
    {"name": "Japanese Cinema & Anime", "lang": "ja", "country": "JP"},
    {"name": "Mandarin & Sinophone Cinema", "lang": "zh", "country": "CN"},
    {"name": "Cantonese / Hong Kong Cinema", "lang": "cn", "country": "HK"},
    {"name": "French Cinema", "lang": "fr", "country": "FR"},
    {"name": "Spanish & Latin American Cinema", "lang": "es", "country": "ES"},
    {"name": "German-Language Cinema", "lang": "de", "country": "DE"},
    {"name": "Italian Cinema", "lang": "it", "country": "IT"},
    {"name": "Brazilian & Portuguese Cinema", "lang": "pt", "country": "BR"},
    {"name": "Arabic & Middle Eastern Cinema", "lang": "ar", "country": "EG"},
    {"name": "Iranian / Persian Cinema", "lang": "fa", "country": "IR"},
    {"name": "Turkish Cinema", "lang": "tr", "country": "TR"},
    {"name": "Nordic Cinema (SE/NO/DK)", "lang": "sv", "country": "SE"},
    {"name": "African & Nollywood Cinema", "lang": "en", "country": "NG"},
]


def fetch_genre_map() -> dict[int, str]:
    resp = requests.get(
        f"{TMDB_BASE}/genre/movie/list",
        headers=TMDB_HEADERS,
        params={"language": "en-US"},
        timeout=20,
    )
    resp.raise_for_status()
    return {g["id"]: g["name"] for g in resp.json().get("genres", [])}


def backfill_global(per_industry_limit: int = 80):
    genre_map = fetch_genre_map()
    pages_per_bucket = max(1, (per_industry_limit + 19) // 20)
    total_upserted = 0

    for bucket in INDUSTRY_BUCKETS:
        print(f"Fetching {bucket['name']} ({bucket['lang']})...")
        fetched_for_bucket = 0

        for page in range(1, pages_per_bucket + 1):
            if fetched_for_bucket >= per_industry_limit:
                break

            params = {
                "with_original_language": bucket["lang"],
                "sort_by": "popularity.desc",
                "page": page,
                "include_adult": "false",
            }
            if bucket.get("country"):
                params["with_origin_country"] = bucket["country"]

            resp = requests.get(
                f"{TMDB_BASE}/discover/movie",
                headers=TMDB_HEADERS,
                params=params,
                timeout=20,
            )
            if resp.status_code != 200:
                print(f"  Skipped page {page} (HTTP {resp.status_code})")
                break

            results = resp.json().get("results", [])
            if not results:
                break

            rows = []
            for m in results:
                if not m.get("overview") or not m.get("title"):
                    continue
                rows.append({
                    "tmdb_id": m["id"],
                    "title": m["title"],
                    "original_title": m.get("original_title"),
                    "overview": m.get("overview"),
                    "original_language": m.get("original_language") or bucket["lang"],
                    "release_date": m.get("release_date") or None,
                    "popularity": m.get("popularity", 0.0),
                    "vote_average": m.get("vote_average", 0.0),
                    "vote_count": m.get("vote_count", 0),
                    "poster_path": m.get("poster_path"),
                    "genres": [genre_map[gid] for gid in m.get("genre_ids", []) if gid in genre_map],
                })

            if rows:
                supabase.table("movies").upsert(rows, on_conflict="tmdb_id").execute()
                fetched_for_bucket += len(rows)
                total_upserted += len(rows)

            time.sleep(0.25)

        print(f"  -> Upserted {fetched_for_bucket} movies for {bucket['name']}")

    print(f"\nDone! Total upserted across all global & Indian regional industries: {total_upserted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill global & regional movies from TMDB")
    parser.add_argument("--per-industry", type=int, default=80, help="Movies to fetch per industry/sub-region")
    args = parser.parse_args()
    backfill_global(per_industry_limit=args.per_industry)