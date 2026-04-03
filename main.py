import os
import json
import secrets
import hashlib
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import psycopg2
from psycopg2.extras import RealDictCursor

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from passlib.context import CryptContext
import google.generativeai as genai

load_dotenv()

app = FastAPI(title="Medical Document Extractor")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL   = os.getenv("DATABASE_URL")

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
# Database helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id               SERIAL PRIMARY KEY,
                    username         TEXT UNIQUE NOT NULL,
                    email            TEXT UNIQUE NOT NULL,
                    hashed_password  TEXT NOT NULL,
                    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS extractions (
                    id         SERIAL PRIMARY KEY,
                    user_id    INTEGER NOT NULL REFERENCES users(id),
                    tipo       TEXT NOT NULL DEFAULT 'Etiqueta',
                    fecha      TEXT NOT NULL,
                    nhc        TEXT,
                    nombre     TEXT,
                    ptc        TEXT DEFAULT '',
                    medico     TEXT,
                    entidad    TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Migrations for existing deployments
            cur.execute("""
                ALTER TABLE extractions
                ADD COLUMN IF NOT EXISTS tipo TEXT NOT NULL DEFAULT 'Etiqueta'
            """)
            cur.execute("""
                ALTER TABLE extractions
                ADD COLUMN IF NOT EXISTS copied BOOLEAN NOT NULL DEFAULT FALSE
            """)
        conn.commit()
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[dict]:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_email(email: str) -> Optional[dict]:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def create_user(username: str, email: str, password: str):
    hashed = pwd_context.hash(_prepare(password))
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, email, hashed_password) VALUES (%s, %s, %s)",
                (username, email, hashed),
            )
        conn.commit()
    finally:
        conn.close()


def save_extraction(user_id: int, tipo: str, fecha: str, nhc: str,
                    nombre: str, ptc: str, medico: str, entidad: str) -> dict:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """INSERT INTO extractions (user_id, tipo, fecha, nhc, nombre, ptc, medico, entidad)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id, tipo, fecha, nhc, nombre, ptc, medico, entidad""",
                (user_id, tipo, fecha, nhc, nombre, ptc, medico, entidad),
            )
            row = cur.fetchone()
        conn.commit()
        return dict(row)
    finally:
        conn.close()


def get_extractions(user_id: int, tipo: Optional[str] = None,
                    copied: Optional[bool] = None) -> list:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query  = "SELECT * FROM extractions WHERE user_id = %s"
            params = [user_id]
            if tipo is not None:
                query += " AND tipo = %s"
                params.append(tipo)
            if copied is not None:
                query += " AND copied = %s"
                params.append(copied)
            query += " ORDER BY created_at ASC"
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def mark_extractions_copied(user_id: int, ids: List[int]) -> int:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE extractions SET copied = TRUE WHERE user_id = %s AND id = ANY(%s)",
                (user_id, ids)
            )
            count = cur.rowcount
        conn.commit()
        return count
    finally:
        conn.close()


def delete_extractions_by_ids(user_id: int, ids: List[int]) -> int:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM extractions WHERE user_id = %s AND id = ANY(%s)",
                (user_id, ids)
            )
            count = cur.rowcount
        conn.commit()
        return count
    finally:
        conn.close()


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
    email    = body.get("email", "").strip()
    password = body.get("password", "")
    confirm  = body.get("confirm_password", "")

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
async def list_extractions(
    request: Request,
    tipo:   Optional[str]  = Query(None),
    copied: Optional[bool] = Query(None),
):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="No autenticado")
    return get_extractions(user["user_id"], tipo, copied)


@app.patch("/extractions/mark-copied")
async def mark_copied(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="No autenticado")
    body = await request.json()
    ids  = body.get("ids", [])
    if not ids:
        raise HTTPException(status_code=400, detail="No se proporcionaron IDs")
    count = mark_extractions_copied(user["user_id"], ids)
    return {"marked": count}


@app.delete("/extractions")
async def delete_extractions(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="No autenticado")
    body = await request.json()
    ids = body.get("ids", [])
    if not ids:
        raise HTTPException(status_code=400, detail="No se proporcionaron IDs")
    count = delete_extractions_by_ids(user["user_id"], ids)
    return {"deleted": count}


async def _do_extract(request: Request, file: UploadFile, tipo: str):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="No autenticado")

    content_type = file.content_type or "image/jpeg"
    image_data   = base64.b64encode(await file.read()).decode("utf-8")

    try:
        response = gemini_model.generate_content([
            {"mime_type": content_type, "data": image_data},
            PROMPT,
        ])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {e}")

    raw_text = response.text.strip()
    start = raw_text.find("{")
    end   = raw_text.rfind("}") + 1
    if start != -1 and end > start:
        raw_text = raw_text[start:end]

    try:
        extracted = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"JSON parse error: {e}\nRaw: {raw_text}")

    fecha   = datetime.now().strftime("%d/%m/%Y")
    nhc     = extracted.get("nhc", "")
    nombre  = extracted.get("nombre", "")
    ptc     = ""
    medico  = extracted.get("medico", "")
    entidad = extracted.get("entidad", "")

    row = save_extraction(user["user_id"], tipo, fecha, nhc, nombre, ptc, medico, entidad)
    return row


@app.post("/extract")
async def extract(request: Request, file: UploadFile = File(...)):
    return await _do_extract(request, file, tipo="Etiqueta")


@app.post("/extract/ficha")
async def extract_ficha(request: Request, file: UploadFile = File(...)):
    return await _do_extract(request, file, tipo="Ficha Medica")
