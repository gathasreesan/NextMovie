import os
import time
import requests
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
token = os.environ['TMDB_READ_ACCESS_TOKEN']
sb = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_SERVICE_KEY'])
headers = {'Authorization': f'Bearer {token}', 'accept': 'application/json'}
GENRE_MAP = {28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy', 80: 'Crime', 18: 'Drama', 9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction', 53: 'Thriller'}

for sort_by in ['vote_count.desc', 'popularity.desc']:
    for p in range(1, 4):
        try:
            r = requests.get(f'https://api.themoviedb.org/3/discover/movie?with_original_language=ta&sort_by={sort_by}&vote_count.gte=5&page={p}', headers=headers, timeout=20)
            if r.status_code == 200:
                rows = []
                for m in r.json().get('results', []):
                    title = m.get('title') or m.get('original_title')
                    if not title:
                        continue
                    genres = [GENRE_MAP[g] for g in m.get('genre_ids', []) if g in GENRE_MAP]
                    y = (m.get('release_date') or '????')[:4]
                    soup = f"{title} ({y}). Language: ta. Genres: {', '.join(genres)}. Plot: {m.get('overview', '')}"
                    rows.append({
                        'tmdb_id': m['id'],
                        'title': title,
                        'original_language': 'ta',
                        'release_date': m.get('release_date') or None,
                        'popularity': float(m.get('popularity', 0.0)),
                        'vote_average': float(m.get('vote_average', 0.0)),
                        'genres': genres,
                        'poster_path': m.get('poster_path'),
                        'metadata_soup': soup
                    })
                if rows:
                    sb.table('movies').upsert(rows, on_conflict='tmdb_id').execute()
            time.sleep(0.3)
        except Exception as e:
            print('Err:', e)

c = sb.table('movies').select('tmdb_id', count='exact', head=True).eq('original_language', 'ta').execute().count
print('Tamil movie count now:', c)
