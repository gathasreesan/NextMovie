import os
import sys
from pathlib import Path
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.auth_store import (
    save_user_credentials,
    verify_user_credentials,
    user_has_password,
)
from ml.cinebuddy import chat_with_cinebuddy, extract_movie_overview
from ml.recommender import (
    recommend_for_user,
    recommend_from_prompt,
    recommend_similar_to_movie,
    retrieve_candidates,
    rerank_candidates,
)
from ml.taste_profile import (
    create_demo_user,
    log_interaction,
    recompute_taste_embedding,
    supabase,
)

TMDB_TOKEN = os.environ.get("TMDB_READ_ACCESS_TOKEN", "")
TMDB_HEADERS = {
    "Authorization": f"Bearer {TMDB_TOKEN}",
    "accept": "application/json",
}
TMDB_BASE = "https://api.themoviedb.org/3"

app = FastAPI(title="NextMovie API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Request Models ──────────────────────────────────────────────────

class PromptRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language movie prompt")
    languages: list[str] | None = Field(default=None, description="Optional ISO language codes, e.g. ['en', 'ml']")
    limit: int = Field(default=10, ge=1, le=50)


class CreateUserRequest(BaseModel):
    preferred_languages: list[str] = Field(default=["en"])


class InteractionRequest(BaseModel):
    user_id: str
    tmdb_id: int
    action: Literal["like", "watchlist", "click", "dislike"]


class MinimalAuthRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=32)
    password: str = Field(..., min_length=4, max_length=64)
    preferred_languages: list[str] = Field(default=[])


class PreferencesSetupRequest(BaseModel):
    preferred_languages: list[str] = Field(..., min_length=1)
    vibes: list[str] = Field(default=[])
    liked_tmdb_ids: list[int] = Field(default=[])


class DeckSwipeRequest(BaseModel):
    user_id: str
    tmdb_id: int
    action: Literal["approve", "reject", "like", "dislike", "watchlist"]


class GuidedDecisionRequest(BaseModel):
    user_id: str | None = None
    mood: str
    language: str | None = None
    company: str | None = None
    format: str | None = None
    pacing: str | None = None
    exclude_ids: list[int] = Field(default=[])


class ChatPayload(BaseModel):
    user_id: str | None = None
    message: str = Field(..., min_length=1)
    history: list[dict] = Field(default=[])
    preferred_languages: list[str] | None = None


# ── System Routes & Static Mounting ──────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.1.0", "platform": "NextMovie Recommendation & Decision Engine"}


@app.get("/")
def root_redirect():
    return RedirectResponse(url="/app/index.html")


WEB_DIR = ROOT_DIR / "web"
if WEB_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


# ── Strict Authentication Routes (Minimal Auth Gate with Password) ─────────

@app.post("/auth/signup")
def signup(payload: MinimalAuthRequest):
    """Strictly minimal sign-up: username + password credentials."""
    username = payload.username.strip().lower()
    if len(payload.password) < 4:
        raise HTTPException(status_code=422, detail="Password must be at least 4 characters long.")

    existing = supabase.table("users").select("id").eq("username", username).execute().data
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists. Please sign in.")

    resp = (
        supabase.table("users")
        .insert({
            "username": username,
            "preferred_languages": payload.preferred_languages or [],
        })
        .execute()
    )
    user = resp.data[0]
    user_id = user["id"]
    save_user_credentials(user_id=user_id, username=username, password=payload.password)

    return {
        "user_id": user_id,
        "username": user["username"],
        "is_demo": False,
        "preferred_languages": user.get("preferred_languages", []),
        "needs_onboarding": True,
    }


@app.post("/auth/login")
def login(payload: MinimalAuthRequest):
    """Strictly minimal login: verifies username and password credentials."""
    username = payload.username.strip().lower()
    resp = (
        supabase.table("users")
        .select("id, username, preferred_languages, taste_embedding")
        .eq("username", username)
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="User not found. Please create an account.")

    user = resp.data[0]
    user_id = user["id"]

    if user_has_password(username):
        if not verify_user_credentials(username, payload.password):
            raise HTTPException(status_code=401, detail="Invalid password. Please check your credentials.")
    else:
        # Pre-existing user without password sets password upon this login
        save_user_credentials(user_id=user_id, username=username, password=payload.password)

    has_prefs = bool(user.get("preferred_languages"))
    has_taste = user.get("taste_embedding") is not None
    needs_onboarding = not (has_prefs and has_taste)

    return {
        "user_id": user_id,
        "username": user["username"],
        "is_demo": False,
        "preferred_languages": user.get("preferred_languages", []),
        "needs_onboarding": needs_onboarding,
    }


@app.post("/auth/demo")
def demo_auth():
    """Provides a restricted guest preview session."""
    return {
        "user_id": "demo-preview-user",
        "username": "guest_critic",
        "is_demo": True,
        "preferred_languages": ["en", "ml"],
        "needs_onboarding": False,
    }


# ── Post-First-Login Preferences & Persistence ───────────────────────────────

@app.post("/users/{user_id}/preferences")
def save_user_preferences(user_id: str, payload: PreferencesSetupRequest):
    """Persist post-login language and vibe preferences to the database, compute initial taste vector."""
    # 1. Update preferred languages in users table
    supabase.table("users").update({
        "preferred_languages": payload.preferred_languages,
    }).eq("id", user_id).execute()

    # 2. Record initial liked anchor movies (validate foreign key existence)
    if payload.liked_tmdb_ids:
        valid_resp = (
            supabase.table("movies")
            .select("tmdb_id")
            .in_("tmdb_id", payload.liked_tmdb_ids)
            .execute()
        )
        valid_ids = {int(r["tmdb_id"]) for r in (valid_resp.data or []) if r.get("tmdb_id") is not None}
        if valid_ids:
            rows = [
                {"user_id": user_id, "tmdb_id": mid, "action": "seed"}
                for mid in valid_ids
            ]
            supabase.table("user_interactions").insert(rows).execute()
            recompute_taste_embedding(user_id)

    return {
        "user_id": user_id,
        "preferences_saved": True,
        "preferred_languages": payload.preferred_languages,
        "taste_ready": bool(payload.liked_tmdb_ids),
    }


@app.get("/movies/onboarding")
def get_onboarding_movies(limit: int = Query(default=36, ge=6, le=60)):
    """Return diverse, popular multi-language movies for the post-login setup anchor picker."""
    resp = (
        supabase.table("movies")
        .select("tmdb_id, title, original_language, release_date, vote_average, genres, poster_path, metadata_soup")
        .not_.is_("poster_path", "null")
        .order("popularity", desc=True)
        .limit(limit)
        .execute()
    )
    results = []
    for m in resp.data or []:
        item = dict(m)
        item["overview"] = extract_movie_overview(item.get("metadata_soup"))
        results.append(item)
    return {"count": len(results), "results": results}


# ── Tinder-Style "Match With Your Film" Swipe Deck ───────────────────────────

@app.get("/deck/candidates")
def get_deck_candidates(
    user_id: str = Query(..., description="Active user ID"),
    limit: int = Query(default=20, ge=1, le=50),
):
    """Return a curated candidate deck for the Tinder-style swipe match mode.
    Excludes all movies previously swiped or saved by this user.
    """
    # 1. Exclude previously interacted movies
    seen_ids = set()
    user_row = {}
    if not user_id.startswith("demo"):
        try:
            inter_resp = (
                supabase.table("user_interactions")
                .select("tmdb_id")
                .eq("user_id", user_id)
                .execute()
            )
            seen_ids = {int(r["tmdb_id"]) for r in (inter_resp.data or []) if r.get("tmdb_id") is not None}
            user_row = (
                supabase.table("users")
                .select("taste_embedding, preferred_languages")
                .eq("id", user_id)
                .single()
                .execute()
            ).data or {}
        except Exception:
            pass

    taste_vec = user_row.get("taste_embedding")
    pref_langs = user_row.get("preferred_languages") or None

    candidates = []
    if taste_vec:
        if isinstance(taste_vec, str):
            taste_vec = [float(x) for x in taste_vec.strip("[]").split(",") if x.strip()]
        raw = retrieve_candidates(taste_vec, top_k=max(limit * 3, 60), languages=pref_langs)
        filtered = [c for c in raw if int(c["tmdb_id"]) not in seen_ids]
        candidates = rerank_candidates(filtered, preferred_languages=pref_langs, limit=limit)

    # 3. Fallback or top-up if candidates pool is smaller than limit
    if len(candidates) < limit:
        needed = limit - len(candidates)
        existing_cand_ids = {int(c["tmdb_id"]) for c in candidates} | seen_ids
        q = (
            supabase.table("movies")
            .select("tmdb_id, title, original_language, release_date, vote_average, popularity, genres, poster_path, metadata_soup")
            .not_.is_("poster_path", "null")
        )
        if pref_langs:
            q = q.in_("original_language", pref_langs)
        extra = q.order("popularity", desc=True).limit(needed * 4).execute().data or []
        for m in extra:
            if int(m["tmdb_id"]) not in existing_cand_ids:
                candidates.append(m)
                existing_cand_ids.add(int(m["tmdb_id"]))
                if len(candidates) >= limit:
                    break

    # 4. Format cards with dynamic compelling hook and match percentage (Zero Emojis)
    deck = []
    for c in candidates:
        soup = c.get("metadata_soup")
        overview = c.get("overview") or extract_movie_overview(soup)
        score_val = float(c.get("score") or 0.85)
        pct = min(99, max(75, int(score_val * 100)))

        genres = c.get("genres") or []
        genre_str = genres[0] if genres else "Cinema"
        lang_code = (c.get("original_language") or "en").upper()
        rating = float(c.get("vote_average") or 0.0)
        is_tv = int(c.get("tmdb_id", 0)) >= 100_000_000

        # Editorial hooks and format classification with clean typography
        if "Anime" in genres:
            format_tag = "ANIME"
            hook = f"ACCLAIMED ANIME · {pct}% MATCH"
        elif "K-Drama" in genres:
            format_tag = "K-DRAMA"
            hook = f"PREMIER K-DRAMA · {pct}% MATCH"
        elif "TV Series" in genres or is_tv:
            format_tag = "SERIES"
            hook = f"PRESTIGE SERIES · {pct}% MATCH"
        elif rating >= 8.0:
            format_tag = "FILM"
            hook = f"CRITIC'S CHOICE · {pct}% MATCH"
        elif "ml" in lang_code.lower():
            format_tag = "FILM"
            hook = f"MOLLYWOOD ACCLAIMED · {pct}% MATCH"
        elif "ta" in lang_code.lower():
            format_tag = "FILM"
            hook = f"KOLLYWOOD ESSENTIAL · {pct}% MATCH"
        elif "te" in lang_code.lower():
            format_tag = "FILM"
            hook = f"TOLLYWOOD SHOWPIECE · {pct}% MATCH"
        elif "ko" in lang_code.lower():
            format_tag = "FILM"
            hook = f"HALLYU CONTEMPORARY · {pct}% MATCH"
        elif "ja" in lang_code.lower():
            format_tag = "FILM"
            hook = f"JAPANESE CINEMA · {pct}% MATCH"
        else:
            format_tag = "FILM"
            hook = f"{lang_code} CINEMA · {pct}% MATCH"

        deck.append({
            "tmdb_id": int(c["tmdb_id"]),
            "title": c.get("title", ""),
            "original_language": c.get("original_language", "en"),
            "release_date": c.get("release_date"),
            "vote_average": rating,
            "genres": genres,
            "poster_path": c.get("poster_path"),
            "overview": overview,
            "match_pct": pct,
            "hook": hook,
            "format_tag": format_tag,
        })

    return {"count": len(deck), "candidates": deck}


@app.post("/deck/swipe")
def record_deck_swipe(payload: DeckSwipeRequest):
    """Record a rapid swipe action (approve/reject). Trains taste embeddings in real-time."""
    if payload.user_id.startswith("demo"):
        return {
            "status": "ok",
            "tmdb_id": payload.tmdb_id,
            "action": payload.action,
            "saved_to_watchlist": False,
            "taste_updated": True,
        }
    try:
        # Map approve -> like, reject -> dislike for taste vector training
        vector_action = "like" if payload.action in ["approve", "like"] else "dislike"
        if payload.action == "watchlist":
            vector_action = "watchlist"
        res = log_interaction(payload.user_id, payload.tmdb_id, vector_action)
        return {
            "status": "ok",
            "tmdb_id": payload.tmdb_id,
            "action": payload.action,
            "saved_to_watchlist": payload.action == "watchlist",
            "taste_updated": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Recommendation-First Guided Decision Assistant ──────────────────────────

@app.post("/recommend/guided")
def get_guided_recommendation(payload: GuidedDecisionRequest):
    """Actively guides confused or indecisive users to their perfect movie, series, or anime match in 3 steps."""
    lang_filter = None
    if payload.language and payload.language.lower() not in ["all", "any", "none"]:
        lang_filter = [payload.language.lower()]

    format_keyword = ""
    if payload.format and payload.format.lower() not in ["all", "any"]:
        format_keyword = f"{payload.format} "

    prompt = f"{format_keyword}{payload.mood} {payload.pacing or ''} {payload.company or ''} acclaimed masterclass"
    try:
        results = recommend_from_prompt(prompt, languages=lang_filter, limit=36)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # If format requested, filter/prioritize matching items
    if payload.format and payload.format.lower() not in ["all", "any"]:
        fmt = payload.format.lower()
        if fmt in ["series", "tv"]:
            matched = [r for r in results if "TV Series" in (r.get("genres") or [])]
            if matched:
                results = matched
        elif fmt == "anime":
            matched = [r for r in results if "Anime" in (r.get("genres") or [])]
            if matched:
                results = matched
        elif fmt in ["kdrama", "k-drama"]:
            matched = [r for r in results if "K-Drama" in (r.get("genres") or [])]
            if matched:
                results = matched
        elif fmt == "movie":
            matched = [r for r in results if int(r.get("tmdb_id", 0)) < 100_000_000]
            if matched:
                results = matched

    for r in results:
        if not r.get("overview"):
            r["overview"] = extract_movie_overview(r.get("metadata_soup"))

    if not results:
        raise HTTPException(status_code=404, detail="No matching titles found for this combination.")

    # Filter out excluded/already seen titles so regeneration produces fresh results
    exclude_set = {int(x) for x in (payload.exclude_ids or [])}
    available = [r for r in results if int(r.get("tmdb_id", 0)) not in exclude_set]
    if not available:
        # If exhausted all retrieved candidates, cycle back
        available = results

    primary = available[0]
    runners_up = available[1:4]

    primary_title = primary.get("title", "")
    setting_text = f" while {payload.company.lower()}" if payload.company else ""
    rationale = (
        f"Based on what you feel like watching ({payload.mood.lower()}{setting_text}), "
        f"***{primary_title}*** is a top match for you. It has great ratings, an engaging story, and is sure to keep you entertained."
    )

    return {
        "primary_match": primary,
        "runners_up": runners_up,
        "rationale": rationale,
        "query_prompt": prompt,
        "remaining_count": len(available) - 1,
    }


# ── AI CineBuddy Conversational Chat ─────────────────────────────────────────

@app.post("/chat")
def chat_endpoint(payload: ChatPayload):
    """Conversational AI bot for cinema guidance, reviews, lore, and recommendations."""
    try:
        res = chat_with_cinebuddy(
            message=payload.message,
            history=payload.history,
            user_id=payload.user_id,
            preferred_languages=payload.preferred_languages,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Core Recommendation & Search Endpoints ───────────────────────────────────

@app.post("/recommend/prompt")
def get_prompt_recommendations(payload: PromptRequest):
    try:
        results = recommend_from_prompt(
            prompt=payload.query,
            languages=payload.languages,
            limit=payload.limit,
        )
        for r in results:
            if not r.get("overview"):
                r["overview"] = extract_movie_overview(r.get("metadata_soup"))
        return {"query": payload.query, "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/similar/{tmdb_id}")
def get_similar_recommendations(
    tmdb_id: int,
    langs: str | None = Query(default=None, description="Comma-separated language codes, e.g. en,ml"),
    limit: int = Query(default=10, ge=1, le=50),
):
    try:
        lang_list = [x.strip() for x in langs.split(",") if x.strip()] if langs else None
        results = recommend_similar_to_movie(tmdb_id=tmdb_id, languages=lang_list, limit=limit)
        for r in results:
            if not r.get("overview"):
                r["overview"] = extract_movie_overview(r.get("metadata_soup"))
        return {"tmdb_id": tmdb_id, "count": len(results), "results": results}
    except ValueError as e:
        try:
            resp = requests.get(f"{TMDB_BASE}/movie/{tmdb_id}/recommendations", headers=TMDB_HEADERS, timeout=8)
            if resp.status_code == 200:
                raw = resp.json().get("results", [])[:limit]
                fallback = [
                    {
                        "tmdb_id": m["id"],
                        "title": m.get("title", ""),
                        "original_language": m.get("original_language", "en"),
                        "release_date": m.get("release_date") or None,
                        "vote_average": float(m.get("vote_average", 0.0)),
                        "popularity": float(m.get("popularity", 0.0)),
                        "poster_path": m.get("poster_path"),
                        "overview": m.get("overview", ""),
                        "genres": [],
                    }
                    for m in raw if m.get("poster_path")
                ]
                return {"tmdb_id": tmdb_id, "count": len(fallback), "results": fallback}
        except Exception:
            pass
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/user/{user_id}")
def get_user_recommendations(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=50),
    exclude_interacted: bool = Query(default=True),
):
    try:
        results = recommend_for_user(
            user_id=user_id,
            limit=limit,
            exclude_interacted=exclude_interacted,
        )
        for r in results:
            if not r.get("overview"):
                r["overview"] = extract_movie_overview(r.get("metadata_soup"))
        return {"user_id": user_id, "count": len(results), "results": results}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Secondary Exploration & Watchlist ────────────────────────────────────────

@app.get("/users/{user_id}/watchlist")
def get_user_watchlist(user_id: str):
    """Return all movies the user saved or liked with metadata for sorting."""
    inter_resp = (
        supabase.table("user_interactions")
        .select("tmdb_id, action, created_at")
        .eq("user_id", user_id)
        .eq("action", "watchlist")
        .order("created_at", desc=True)
        .execute()
    )
    interactions = inter_resp.data or []
    if not interactions:
        return {"count": 0, "results": []}

    latest_by_id: dict[int, dict] = {}
    for row in interactions:
        mid = int(row["tmdb_id"])
        if mid not in latest_by_id:
            latest_by_id[mid] = row

    tmdb_ids = list(latest_by_id.keys())
    movies_resp = (
        supabase.table("movies")
        .select("tmdb_id, title, original_language, release_date, vote_average, popularity, genres, poster_path, metadata_soup")
        .in_("tmdb_id", tmdb_ids)
        .execute()
    )

    results = []
    for m in movies_resp.data or []:
        mid = int(m["tmdb_id"])
        meta = latest_by_id.get(mid, {})
        enriched = dict(m)
        enriched["overview"] = extract_movie_overview(enriched.get("metadata_soup"))
        enriched["saved_action"] = meta.get("action", "watchlist")
        enriched["saved_at"] = meta.get("created_at")
        results.append(enriched)

    results.sort(key=lambda x: x.get("saved_at") or "", reverse=True)
    return {"count": len(results), "results": results, "watchlist": results}


@app.delete("/users/{user_id}/interactions/{tmdb_id}")
def remove_user_interaction(user_id: str, tmdb_id: int):
    """Remove saved/liked interactions for a movie so users can unpin or reset."""
    supabase.table("user_interactions").delete().eq("user_id", user_id).eq("tmdb_id", tmdb_id).execute()
    recompute_taste_embedding(user_id)
    return {"user_id": user_id, "tmdb_id": tmdb_id, "removed": True}


@app.get("/movies/languages")
def get_languages():
    """Return catalog language options with native labels and regional industry metadata."""
    langs = [
        {"code": "all", "name": "All Languages", "native": "Global Cinema", "industry": "Worldwide"},
        {"code": "ml", "name": "Malayalam", "native": "മലയാളം", "industry": "Mollywood"},
        {"code": "ta", "name": "Tamil", "native": "தமிழ்", "industry": "Kollywood"},
        {"code": "te", "name": "Telugu", "native": "తెలుగు", "industry": "Tollywood"},
        {"code": "hi", "name": "Hindi", "native": "हिन्दी", "industry": "Bollywood"},
        {"code": "kn", "name": "Kannada", "native": "ಕನ್ನಡ", "industry": "Sandalwood"},
        {"code": "en", "name": "English", "native": "English", "industry": "Hollywood / UK"},
        {"code": "ko", "name": "Korean", "native": "한국어", "industry": "Hallyu Cinema"},
        {"code": "ja", "name": "Japanese", "native": "日本語", "industry": "Japanese Cinema"},
        {"code": "es", "name": "Spanish", "native": "Español", "industry": "Spain & LatAm"},
        {"code": "fr", "name": "French", "native": "Français", "industry": "French Cinema"},
        {"code": "de", "name": "German", "native": "Deutsch", "industry": "German Cinema"},
        {"code": "it", "name": "Italian", "native": "Italiano", "industry": "Italian Cinema"},
    ]
    return {"languages": langs}


@app.get("/movies/browse")
def browse_movies(
    langs: str | None = Query(default=None, description="Comma-separated ISO 639-1 language codes"),
    genre: str | None = Query(default=None, description="Optional genre filter"),
    format_type: str | None = Query(default=None, description="all | movie | series | anime | kdrama"),
    sort_by: str = Query(default="popularity", pattern="^(popularity|vote_average|release_date|title)$"),
    min_rating: float | None = Query(default=None, ge=0.0, le=10.0),
    search: str | None = Query(default=None, description="Title search keyword"),
    limit: int = Query(default=36, ge=1, le=100),
    page: int = Query(default=1, ge=1),
):
    """Secondary catalog explorer: browse and sort global movies, series, dramas, and anime."""
    q = supabase.table("movies").select(
        "tmdb_id, title, original_language, release_date, vote_average, popularity, genres, poster_path, metadata_soup"
    )

    if langs and langs.lower() != "all":
        lang_list = [x.strip() for x in langs.split(",") if x.strip()]
        if lang_list:
            q = q.in_("original_language", lang_list)

    if format_type and format_type.lower() != "all":
        ft = format_type.lower()
        if ft in ["series", "tv"]:
            q = q.contains("genres", ["TV Series"])
        elif ft == "anime":
            q = q.contains("genres", ["Anime"])
        elif ft in ["kdrama", "k-drama"]:
            q = q.contains("genres", ["K-Drama"])
        elif ft == "movie":
            q = q.lt("tmdb_id", 100_000_000)

    if genre and genre.upper() != "ALL":
        q = q.contains("genres", [genre])

    if min_rating is not None and min_rating > 0:
        q = q.gte("vote_average", min_rating)

    if search and search.strip():
        q = q.ilike("title", f"%{search.strip()}%")

    q = q.not_.is_("poster_path", "null")

    desc = sort_by != "title"
    offset = (page - 1) * limit
    q = q.order(sort_by, desc=desc).range(offset, offset + limit - 1)

    resp = q.execute()
    results = []
    for m in resp.data or []:
        item = dict(m)
        item["overview"] = extract_movie_overview(item.get("metadata_soup"))
        results.append(item)

    return {
        "count": len(results),
        "page": page,
        "limit": limit,
        "results": results,
    }


@app.get("/movies/details/{tmdb_id}")
def get_movie_details(tmdb_id: int):
    """Get rich title dossier for movie, series, or anime, including synopsis, cast, and trailer preview."""
    local = (
        supabase.table("movies")
        .select("tmdb_id, title, original_language, release_date, vote_average, popularity, genres, poster_path, metadata_soup")
        .eq("tmdb_id", tmdb_id)
        .execute()
    ).data

    movie_data = {}
    if local:
        movie_data = dict(local[0])
        movie_data["overview"] = extract_movie_overview(movie_data.get("metadata_soup"))

    if TMDB_TOKEN:
        try:
            is_tv = tmdb_id >= 100_000_000
            real_id = tmdb_id - 100_000_000 if is_tv else tmdb_id
            endpoint = f"{TMDB_BASE}/tv/{real_id}" if is_tv else f"{TMDB_BASE}/movie/{real_id}"

            r = requests.get(
                endpoint,
                headers=TMDB_HEADERS,
                params={"append_to_response": "credits,videos,similar,watch/providers"},
                timeout=8,
            )
            if r.status_code == 200:
                t_data = r.json()
                if not movie_data.get("title"):
                    movie_data["title"] = t_data.get("name") if is_tv else (t_data.get("title") or t_data.get("original_title"))
                    movie_data["tmdb_id"] = tmdb_id
                    movie_data["original_language"] = t_data.get("original_language", "en")
                    movie_data["release_date"] = t_data.get("first_air_date") if is_tv else t_data.get("release_date")
                    movie_data["vote_average"] = float(t_data.get("vote_average", 0.0))
                    movie_data["genres"] = [g["name"] for g in t_data.get("genres", [])]
                    movie_data["poster_path"] = t_data.get("poster_path")

                if not movie_data.get("overview"):
                    movie_data["overview"] = t_data.get("overview", "")

                movie_data["backdrop_path"] = t_data.get("backdrop_path")
                movie_data["tagline"] = t_data.get("tagline")
                movie_data["runtime"] = t_data.get("episode_run_time", [None])[0] if is_tv else t_data.get("runtime")

                credits = t_data.get("credits", {})
                movie_data["cast"] = [c["name"] for c in credits.get("cast", [])[:6]]
                directors = [c["name"] for c in credits.get("crew", []) if c.get("job") in ["Director", "Executive Producer"]]
                if not directors and is_tv:
                    directors = [c["name"] for c in t_data.get("created_by", [])]
                movie_data["director"] = directors[0] if directors else None

                # Videos & Trailers (robust detection with YouTube fallback)
                videos = t_data.get("videos", {}).get("results", [])
                yt_trailers = [v["key"] for v in videos if v.get("site") == "YouTube" and v.get("type") == "Trailer"]
                yt_teasers = [v["key"] for v in videos if v.get("site") == "YouTube" and v.get("type") in ["Teaser", "Clip"]]
                yt_all = [v["key"] for v in videos if v.get("site") == "YouTube"]

                if yt_trailers:
                    movie_data["trailer_key"] = yt_trailers[0]
                elif yt_teasers:
                    movie_data["trailer_key"] = yt_teasers[0]
                elif yt_all:
                    movie_data["trailer_key"] = yt_all[0]
                else:
                    movie_data["trailer_key"] = None

                # OTT Streaming Providers (Where to Watch)
                wp = t_data.get("watch/providers", {}).get("results", {})
                in_prov = wp.get("IN", {})
                us_prov = wp.get("US", {})

                flatrate_in = [p.get("provider_name") for p in in_prov.get("flatrate", [])]
                flatrate_us = [p.get("provider_name") for p in us_prov.get("flatrate", [])]
                rent_in = [p.get("provider_name") for p in in_prov.get("rent", [])]

                providers = []
                for p in flatrate_in + flatrate_us + rent_in:
                    if p and p not in providers:
                        providers.append(p)

                movie_data["watch_providers"] = providers[:6]
                movie_data["watch_link"] = in_prov.get("link") or us_prov.get("link")
        except Exception:
            pass

    if not movie_data:
        raise HTTPException(status_code=404, detail="Title not found")

    return movie_data
