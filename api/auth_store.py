import os
import json
import hashlib
import secrets
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CREDENTIALS_FILE = DATA_DIR / "auth_credentials.json"

def _ensure_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CREDENTIALS_FILE.exists():
        with open(CREDENTIALS_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)

def _load_credentials() -> dict:
    _ensure_dir()
    try:
        with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_credentials(data: dict):
    _ensure_dir()
    with open(CREDENTIALS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    ).hex()
    return hashed, salt

def save_user_credentials(user_id: str, username: str, password: str):
    clean_username = username.strip().lower()
    hashed, salt = hash_password(password)
    creds = _load_credentials()
    creds[clean_username] = {
        "user_id": user_id,
        "username": clean_username,
        "password_hash": hashed,
        "salt": salt,
    }
    _save_credentials(creds)

def verify_user_credentials(username: str, password: str) -> bool:
    clean_username = username.strip().lower()
    creds = _load_credentials()
    record = creds.get(clean_username)
    if not record:
        return False
    hashed, _ = hash_password(password, record["salt"])
    return secrets.compare_digest(hashed, record["password_hash"])

def user_has_password(username: str) -> bool:
    clean_username = username.strip().lower()
    creds = _load_credentials()
    return clean_username in creds

def set_user_password(user_id: str, username: str, password: str):
    save_user_credentials(user_id, username, password)
