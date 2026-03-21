import os
import json
import secrets
import sqlite3
import hashlib
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from passlib.context import CryptContext
import google.generativeai as genai

load_dotenv()

app = FastAPI(title="Medical Document Extractor")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_PATH = os.getenv("DB_PATH", "users.db")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _prepare(password: str) -> str:
    """SHA-256 the password so bcrypt never sees more than 44 bytes."""
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


# In-memory sessions: { session_id -> { user_id, username } }
sessions: dict = {}

PROMPT = (
    "This is a Spanish medical pre-operative consultation document. "
    "Extract the following fields and return a JSON object with ONLY these keys:\n"
    '- "nhc": the NHC number (the code next to "NHC", e.g. "S00288482") (string)\n'
    '- "nombre": the full patient name printed in large bold text below the document title '
    '(e.g. "ROCA PRADO, MARIA DEL CARMEN") (string)\n'
    '- "medico": the value next to "Médico" or "Medico" (string)\n'
    '- "entidad": the value next to "Entidad" (string)\n\n'
    "Use an empty string for any field not found. "
    "Return ONLY valid JSON — no markdown, no explanation."
)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                username         TEXT UNIQUE NOT NULL,
                email            TEXT UNIQUE NOT NULL,
                hashed_password  TEXT NOT NULL,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS extractions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                fecha      TEXT NOT NULL,
                nhc        TEXT,
                nombre     TEXT,
                ptc        TEXT DEFAULT '',
                medico     TEXT,
                entidad    TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()


def get_user_by_username(username: str) -> Optional[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        return dict(row) if row else None


def create_user(username: str, email: str, password: str):
    hashed = pwd_context.hash(_prepare(password))
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO users (username, email, hashed_password) VALUES (?, ?, ?)",
            (username, email, hashed),
        )
        conn.commit()


def save_extraction(user_id: int, fecha: str, nhc: str, nombre: str,
                    ptc: str, medico: str, entidad: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO extractions (user_id, fecha, nhc, nombre, ptc, medico, entidad)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, fecha, nhc, nombre, ptc, medico, entidad),
        )
        conn.commit()


def get_extractions(user_id: int) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM extractions WHERE user_id = ? ORDER BY created_at ASC",
            (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]


init_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_current_user(request: Request) -> Optional[dict]:
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        return None
    return sessions[session_id]


# ---------------------------------------------------------------------------
# Static files & root
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    return Path("static/index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.post("/auth/register")
async def register(request: Request):
    body = await request.json()
    username = body.get("username", "").strip()
    email = body.get("email", "").strip()
    password = body.get("password", "")
    confirm = body.get("confirm_password", "")

    if not username or not email or not password:
        raise HTTPException(status_code=400, detail="Todos los campos son obligatorios")
    if password != confirm:
        raise HTTPException(status_code=400, detail="Las contraseñas no coinciden")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="La contraseña debe tener al menos 6 caracteres")
    if get_user_by_username(username):
        raise HTTPException(status_code=400, detail="El usuario ya existe")
    if get_user_by_email(email):
        raise HTTPException(status_code=400, detail="El email ya está registrado")

    create_user(username, email, password)
    user = get_user_by_username(username)

    session_id = secrets.token_urlsafe(32)
    sessions[session_id] = {"user_id": user["id"], "username": user["username"]}

    resp = JSONResponse({"success": True, "username": username})
    resp.set_cookie("session_id", session_id, httponly=True, samesite="lax")
    return resp


@app.post("/auth/login")
async def login(request: Request):
    body = await request.json()
    username = body.get("username", "").strip()
    password = body.get("password", "")

    user = get_user_by_username(username)
    if not user or not pwd_context.verify(_prepare(password), user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos")

    session_id = secrets.token_urlsafe(32)
    sessions[session_id] = {"user_id": user["id"], "username": user["username"]}

    resp = JSONResponse({"success": True, "username": username})
    resp.set_cookie("session_id", session_id, httponly=True, samesite="lax")
    return resp


@app.post("/auth/logout")
async def logout(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in sessions:
        del sessions[session_id]
    resp = JSONResponse({"success": True})
    resp.delete_cookie("session_id")
    return resp


@app.get("/auth/me")
async def me(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="No autenticado")
    return {"username": user["username"]}


# ---------------------------------------------------------------------------
# Extractions
# ---------------------------------------------------------------------------

@app.get("/extractions")
async def list_extractions(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="No autenticado")
    rows = get_extractions(user["user_id"])
    return rows


@app.post("/extract")
async def extract(request: Request, file: UploadFile = File(...)):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="No autenticado")

    content_type = file.content_type or "image/jpeg"
    image_data = base64.b64encode(await file.read()).decode("utf-8")

    try:
        response = gemini_model.generate_content([
            {"mime_type": content_type, "data": image_data},
            PROMPT,
        ])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {e}")

    raw_text = response.text.strip()
    start = raw_text.find("{")
    end = raw_text.rfind("}") + 1
    if start != -1 and end > start:
        raw_text = raw_text[start:end]

    try:
        extracted = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"JSON parse error: {e}\nRaw: {raw_text}")

    fecha = datetime.now().strftime("%d/%m/%Y")
    nhc = extracted.get("nhc", "")
    nombre = extracted.get("nombre", "")
    ptc = ""
    medico = extracted.get("medico", "")
    entidad = extracted.get("entidad", "")

    save_extraction(user["user_id"], fecha, nhc, nombre, ptc, medico, entidad)

    return {
        "fecha": fecha,
        "nhc": nhc,
        "nombre": nombre,
        "ptc": ptc,
        "medico": medico,
        "entidad": entidad,
    }
