import os
import re
import random
import requests
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

from ml.recommender import (
    detect_languages_from_prompt,
    recommend_from_prompt,
    supabase,
)

TMDB_TOKEN = os.environ.get("TMDB_READ_ACCESS_TOKEN", "")
TMDB_HEADERS = {
    "Authorization": f"Bearer {TMDB_TOKEN}",
    "accept": "application/json",
}
TMDB_BASE = "https://api.themoviedb.org/3"

# Cinema Trivia & Jokes
MOVIE_JOKES = [
    "Why did the director bring string to the audition? Because he wanted to tie up loose ends!",
    "Why do movie stars stay so cool? Because they have so many fans!",
    "Why don't skeletons go to horror movies? They don't have the guts!",
    "What's Christopher Nolan's favorite breakfast? Inception... an egg inside an egg inside an egg!",
    "Why did the camera stop working? It had too much exposure!",
    "How does Quentin Tarantino like his coffee? With lots of pulp!",
]

CINEMA_LORE = {
    "malayalam": (
        "**Malayalam Cinema (Mollywood)** is globally renowned for its grounded storytelling, organic acting, "
        "and masterful screenplays. From legendary classics like *Manichitrathazhu* (1993) and *Kireedam* (1989) "
        "to modern masterworks like *Kumbalangi Nights*, *Drishyam*, *Jallikattu*, *Aavesham*, and *Bramayugam*, "
        "Kerala filmmakers prioritize emotional authenticity, complex moral grey areas, and breathtaking cinematography."
    ),
    "tamil": (
        "**Tamil Cinema (Kollywood)** is a powerhouse of visual ambition and auteur-driven storytelling. "
        "From the poetic visual grammar of Mani Ratnam (*Nayakan*, *Thalapathi*, *Iruvar*) to Kamal Haasan's "
        "experimental boundary-pushing (*Hey Ram*, *Virumaandi*) and the gritty modern wave led by Lokesh Kanagaraj, "
        "Vetrimaaran, and Pa. Ranjith (*Asuran*, *Vikram Vedha*, *Maharaja*), Tamil cinema blends visceral mass appeal with high artistic conviction."
    ),
    "telugu": (
        "**Telugu Cinema (Tollywood)** is the undisputed king of grand-scale visual spectacle and emotional hyper-heroism. "
        "Visionaries like S. S. Rajamouli have captivated global audiences with *RRR* and *Bāhubali*, while grounded gems like "
        "*Jersey*, *C/o Kancharapalem*, and *Agent Sai Srinivasa Athreya* showcase its versatile narrative range."
    ),
    "hindi": (
        "**Hindi Cinema (Bollywood)** combines timeless melodrama, musical richness, and cutting-edge social realism. "
        "From evergreen classics like *Sholay* and *Dilwale Dulhania Le Jayenge* to genre-defining masterworks like "
        "*Gangs of Wasseypur*, *3 Idiots*, *Tumbbad*, and *Andhadhun*, Hindi cinema offers something for every palette."
    ),
    "korean": (
        "**Korean Cinema (Hallyu)** is celebrated for fearless genre-blending, razor-sharp social commentary, and tension-fueled thrillers. "
        "Bong Joon-ho (*Parasite*, *Memories of Murder*), Park Chan-wook (*Oldboy*, *The Handmaiden*, *Decision to Leave*), "
        "and Lee Chang-dong (*Burning*) have shaped modern world cinema with unforgettable emotional and visual rigor."
    ),
    "japanese": (
        "**Japanese Cinema & Anime** boasts one of the world's most poetic artistic heritages. "
        "From Akira Kurosawa (*Seven Samurai*, *Rashomon*) and Yasujiro Ozu to Studio Ghibli's Hayao Miyazaki (*Spirited Away*, *Princess Mononoke*) "
        "and modern auteurs like Makoto Shinkai (*Your Name.*) and Hirokazu Kore-eda (*Shoplifters*), Japanese films master stillness, wonder, and humanity."
    ),
}


def search_tmdb_movies(query: str, lang: str | None = None, limit: int = 6) -> list[dict]:
    """Fallback search on TMDB when database has zero semantic matches for a hyper-specific title/name."""
    if not TMDB_TOKEN:
        return []
    try:
        params = {"query": query, "include_adult": "false", "page": 1}
        if lang:
            params["language"] = lang
        resp = requests.get(f"{TMDB_BASE}/search/movie", headers=TMDB_HEADERS, params=params, timeout=8)
        if resp.status_code != 200:
            return []
        results = resp.json().get("results", [])
        formatted = []
        for m in results[:limit]:
            if not m.get("title") or not m.get("poster_path"):
                continue
            formatted.append({
                "tmdb_id": m["id"],
                "title": m["title"],
                "original_language": m.get("original_language", "en"),
                "release_date": m.get("release_date") or None,
                "vote_average": float(m.get("vote_average", 0.0)),
                "genres": [],
                "poster_path": m.get("poster_path"),
                "overview": m.get("overview") or "",
                "score": 0.88,
            })
        return formatted
    except Exception:
        return []


def extract_movie_overview(soup: str | None) -> str:
    """Extract plot synopsis from metadata_soup."""
    if not soup:
        return ""
    if "Plot: " in soup:
        return soup.split("Plot: ", 1)[1].strip()
    return soup.strip()


def chat_with_cinebuddy(
    message: str,
    history: list[dict] | None = None,
    user_id: str | None = None,
    preferred_languages: list[str] | None = None,
) -> dict:
    """
    Intelligent conversational CineBuddy AI engine.
    Returns:
      {
        "reply": markdown string formatted with cinephile depth,
        "movies": list of enriched movie dicts,
        "suggested_chips": list of quick reply chips
      }
    """
    clean_msg = message.strip()
    msg_lower = clean_msg.lower()
    history = history or []

    # 1. GREETINGS & CASUAL CHIT-CHAT
    if re.search(r"^(hi|hello|hey|greetings|hola|namaste|vanakkam|namaskaram)\b", msg_lower):
        reply = (
            "### Hi there!\n\n"
            "I'm **CineBuddy**, your personal movie and show guide. I'm here to help you pick the best movies, TV series, anime, or K-dramas to watch.\n\n"
            "Tell me what kind of mood you're in, your favorite movie or show, or what you feel like watching tonight, and I'll find you great recommendations in any language!"
        )
        return {
            "reply": reply,
            "movies": [],
            "suggested_chips": [
                {"label": "Feel-Good Movies", "query": "recommend feel-good comforting movies"},
                {"label": "Suspense Thrillers", "query": "best gripping suspense thrillers with great plot twists"},
                {"label": "Malayalam Mystery Movies", "query": "top acclaimed malayalam mystery crime thrillers"},
                {"label": "Binge-Worthy Series", "query": "addictive tv series you cannot stop watching"},
                {"label": "Top Anime & K-Dramas", "query": "best popular anime and romantic kdramas"},
            ],
        }

    # 2. WHO ARE YOU / IDENTITY
    if re.search(r"\b(who are you|what can you do|your name|what are you)\b", msg_lower):
        reply = (
            "### CineBuddy — Your Movie & Series Guide\n\n"
            "I'm NextMovie's AI movie guide. Unlike a regular search bar, I understand what you feel like watching:\n"
            "- **Moods & Vibes**: Whether you want relaxing feel-good stories, intense murder mysteries, or heart-pounding action.\n"
            "- **Multi-format**: Movies, web series, K-dramas, and anime.\n"
            "- **Regional & Global**: Malayalam, Tamil, Telugu, Hindi, Hollywood, Korean, and world cinema.\n\n"
            "Just type what you want to watch or ask me for a recommendation!"
        )
        return {
            "reply": reply,
            "movies": [],
            "suggested_chips": [
                {"label": "Top Mystery Thrillers", "query": "best crime investigation thrillers"},
                {"label": "Popular Anime Series", "query": "top rated anime series"},
                {"label": "Binge-Worthy TV Shows", "query": "best television series"},
            ],
        }

    # 3. MOVIE JOKES & TRIVIA
    if re.search(r"\b(joke|funny|laugh|trivia|fun fact)\b", msg_lower):
        joke = random.choice(MOVIE_JOKES)
        reply = (
            f"Here is a cinematic treat:\n\n"
            f"> **\"{joke}\"**\n\n"
            f"Would you like another film reflection, auteur trivia, or a definitive recommendation for tonight?"
        )
        return {
            "reply": reply,
            "movies": [],
            "suggested_chips": [
                {"label": "Show top comedy films", "query": "laugh out loud hilarious comedy films"},
                {"label": "Tell me another joke", "query": "tell me another movie joke"},
                {"label": "Cinema trivia fact", "query": "tell me a fascinating cinema trivia fact"},
            ],
        }

    # 4. INDUSTRY LORE / CINEMA EXPLANATIONS
    for ind_key, ind_text in CINEMA_LORE.items():
        if ind_key in msg_lower and any(w in msg_lower for w in ["what is", "tell me about", "why is", "history", "explain", "about"]):
            reply = (
                f"### The Craft of {ind_key.capitalize()} Cinema\n\n"
                f"{ind_text}\n\n"
                f"Would you like me to surface the highest-rated titles from this industry to get you started?"
            )
            movies = recommend_from_prompt(f"best acclaimed {ind_key} cinema masterpiece", languages=[ind_key[:2]], limit=6)
            for m in movies:
                if not m.get("overview"):
                    m["overview"] = extract_movie_overview(m.get("metadata_soup"))
            return {
                "reply": reply,
                "movies": movies,
                "suggested_chips": [
                    {"label": f"Top-rated {ind_key.capitalize()} thrillers", "query": f"best {ind_key} thriller suspense"},
                    {"label": f"{ind_key.capitalize()} feel-good titles", "query": f"heartwarming {ind_key} drama comedy"},
                    {"label": "Explore global cinema", "query": "show me global cinema masterpieces"},
                ],
            }

    # 5. SPECIFIC MOVIE DISCUSSION / COMPARISON / REVIEW
    movie_compare_match = re.search(r"\b(compare|versus|vs|between)\s+([a-zA-Z0-9\s]+?)\s+(and|vs|versus)\s+([a-zA-Z0-9\s]+)", msg_lower)
    if movie_compare_match:
        m1 = movie_compare_match.group(2).strip().title()
        m2 = movie_compare_match.group(4).strip().title()
        reply = (
            f"### Comparing *{m1}* vs *{m2}*\n\n"
            f"Both of these works represent remarkable storytelling, but they resonate on distinct frequencies:\n\n"
            f"- **{m1}**: Known for its gripping narrative rhythm, distinct visual atmosphere, and thematic focus.\n"
            f"- **{m2}**: Delivers an impactful perspective with sharp character dynamics and emotional stakes.\n\n"
            f"If you're looking for high immersion and narrative depth, both are masterclasses. Here are their profiles and connected titles:"
        )
        movies = recommend_from_prompt(f"{m1} {m2} similar films", limit=6)
        for m in movies:
            if not m.get("overview"):
                m["overview"] = extract_movie_overview(m.get("metadata_soup"))
        return {
            "reply": reply,
            "movies": movies,
            "suggested_chips": [
                {"label": f"More like {m1}", "query": f"movies similar to {m1}"},
                {"label": f"More like {m2}", "query": f"movies similar to {m2}"},
                {"label": "Surprise me with a twist", "query": "mind-bending mystery thriller plot twist"},
            ],
        }

    # 6. RECOMMENDATIONS & NATURAL SEARCH
    # Detect language preferences from prompt or user preference
    detected_langs = detect_languages_from_prompt(clean_msg)
    active_langs = detected_langs or preferred_languages or None

    # Retrieve recommendations using our vector search engine
    movies = []
    try:
        movies = recommend_from_prompt(clean_msg, languages=active_langs, limit=8)
    except Exception as e:
        print(f"Vector search note: {e}")

    # Fallback to TMDB if vector search returned sparse results
    if len(movies) < 3:
        tmdb_extra = search_tmdb_movies(clean_msg, lang=active_langs[0] if active_langs else None, limit=6)
        existing_ids = {m["tmdb_id"] for m in movies}
        for item in tmdb_extra:
            if item["tmdb_id"] not in existing_ids:
                movies.append(item)

    # Ensure all movies have a clean overview
    for m in movies:
        if not m.get("overview"):
            m["overview"] = extract_movie_overview(m.get("metadata_soup"))

    # Generate conversational AI response describing the recommendations
    lang_label = ""
    if detected_langs:
        lang_names = {"ml": "Malayalam", "ta": "Tamil", "te": "Telugu", "hi": "Hindi", "kn": "Kannada", "ko": "Korean", "ja": "Japanese", "es": "Spanish", "fr": "French", "en": "English"}
        names = [lang_names.get(l, l.upper()) for l in detected_langs]
        lang_label = f" in **{', '.join(names)}**"

    top_picks = movies[:3]
    breakdown_items = []
    for idx, p in enumerate(top_picks, 1):
        year = (p.get("release_date") or "????")[:4]
        rating_val = float(p.get("vote_average", 0.0))
        rating = f"{rating_val:.1f} / 10" if rating_val > 0 else "Unrated"
        breakdown_items.append(
            f"**{idx}. {p['title']}** ({year}) · {rating}\n"
            f"> *{p.get('overview', 'A standout cinematic pick matching your request.')[:180]}...*"
        )

    breakdown_text = "\n\n".join(breakdown_items)

    reply = (
        f"Here are top recommendations based on: *\"{clean_msg}\"*{lang_label}.\n\n"
        f"{breakdown_text}\n\n"
        f"Tap any title below to see full details, original trailers, where to watch, or save to your Watchlist."
    )

    # Build contextual suggestion chips
    chips = []
    if top_picks:
        first_title = top_picks[0]["title"]
        chips.append({"label": f"More like '{first_title[:20]}'", "query": f"similar to {first_title}"})
    if detected_langs and "ml" not in detected_langs:
        chips.append({"label": "Try in Malayalam", "query": f"{clean_msg} in malayalam"})
    chips.append({"label": "Darker & intense", "query": f"{clean_msg} dark intense psychological"})
    chips.append({"label": "Comforting & feel-good", "query": f"feel-good comforting version of {clean_msg}"})

    return {
        "reply": reply,
        "movies": movies,
        "suggested_chips": chips[:4],
    }
