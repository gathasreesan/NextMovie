import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.recommender import (
    recommend_for_user,
    recommend_from_prompt,
    recommend_similar_to_movie,
)
from ml.taste_profile import create_demo_user, log_interaction

app = FastAPI(title="NextMovie API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/users")
def create_user(payload: CreateUserRequest):
    try:
        user_id = create_demo_user(payload.preferred_languages)
        return {"user_id": user_id, "preferred_languages": payload.preferred_languages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interactions")
def record_interaction(payload: InteractionRequest):
    try:
        return log_interaction(payload.user_id, payload.tmdb_id, payload.action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/prompt")
def get_prompt_recommendations(payload: PromptRequest):
    try:
        results = recommend_from_prompt(
            prompt=payload.query,
            languages=payload.languages,
            limit=payload.limit,
        )
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
        return {"tmdb_id": tmdb_id, "count": len(results), "results": results}
    except ValueError as e:
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
        return {"user_id": user_id, "count": len(results), "results": results}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))