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
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from passlib.context import CryptContext
import google.generativeai as genai

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build

load_dotenv()

app = FastAPI(title="Medical Document Extractor")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY")
DATABASE_URL         = os.getenv("DATABASE_URL")
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/google/auth/callback")
GOOGLE_SCOPES        = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

# Allow OAuth over HTTP for local development (http:// redirect URI)
if GOOGLE_REDIRECT_URI.startswith("http://"):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _prepare(password: str) -> str:
    """SHA-256 the password so bcrypt never sees more than 44 bytes."""
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


# In-memory sessions: { session_id -> { user_id, username, google_oauth? } }
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
                    user_id    TEXT NOT NULL,
                    tipo       TEXT NOT NULL DEFAULT 'Etiqueta',
                    fecha      TEXT NOT NULL,
                    nhc        TEXT,
                    nombre     TEXT,
                    ptc        TEXT DEFAULT '',
                    medico     TEXT,
                    entidad    TEXT,
                    copied     BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS google_connections (
                    id            SERIAL PRIMARY KEY,
                    user_id       TEXT UNIQUE NOT NULL,
                    access_token  TEXT NOT NULL,
                    refresh_token TEXT,
                    token_expiry  TIMESTAMP,
                    scopes        TEXT,
                    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Migrations for existing deployments
            cur.execute("ALTER TABLE extractions ADD COLUMN IF NOT EXISTS tipo   TEXT NOT NULL DEFAULT 'Etiqueta'")
            cur.execute("ALTER TABLE extractions ADD COLUMN IF NOT EXISTS copied BOOLEAN NOT NULL DEFAULT FALSE")
            # Migrate user_id columns from INTEGER to TEXT (handles UUID-based Supabase users)
            cur.execute("""
                ALTER TABLE extractions
                    DROP CONSTRAINT IF EXISTS extractions_user_id_fkey
            """)
            cur.execute("""
                DO $$ BEGIN
                    IF (SELECT data_type FROM information_schema.columns
                        WHERE table_name='extractions' AND column_name='user_id') = 'integer' THEN
                        ALTER TABLE extractions ALTER COLUMN user_id TYPE TEXT USING user_id::TEXT;
                    END IF;
                END $$
            """)
            cur.execute("""
                DO $$ BEGIN
                    IF (SELECT data_type FROM information_schema.columns
                        WHERE table_name='google_connections' AND column_name='user_id') = 'integer' THEN
                        ALTER TABLE google_connections ALTER COLUMN user_id TYPE TEXT USING user_id::TEXT;
                    END IF;
                END $$
            """)
        conn.commit()
    finally:
        conn.close()


# ── Users ──────────────────────────────────────────────────────────────────

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


# ── Extractions ────────────────────────────────────────────────────────────

def save_extraction(user_id: str, tipo: str, fecha: str, nhc: str,
                    nombre: str, ptc: str, medico: str, entidad: str) -> dict:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """INSERT INTO extractions (user_id, tipo, fecha, nhc, nombre, ptc, medico, entidad)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id, tipo, fecha, nhc, nombre, ptc, medico, entidad, copied""",
                (user_id, tipo, fecha, nhc, nombre, ptc, medico, entidad),
            )
            row = cur.fetchone()
        conn.commit()
        return dict(row)
    finally:
        conn.close()


def get_extractions(user_id: str, tipo: Optional[str] = None,
                    copied: Optional[bool] = None) -> list:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query  = "SELECT * FROM extractions WHERE user_id = %s"
            params = [user_id]
            if tipo is not None:
                query += " AND tipo = %s"; params.append(tipo)
            if copied is not None:
                query += " AND copied = %s"; params.append(copied)
            query += " ORDER BY created_at ASC"
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_extractions_by_ids(user_id: str, ids: List[int]) -> list:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM extractions WHERE user_id = %s AND id = ANY(%s) ORDER BY created_at ASC",
                (user_id, ids)
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def mark_extractions_copied(user_id: str, ids: List[int]) -> int:
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


def delete_extractions_by_ids(user_id: str, ids: List[int]) -> int:
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


# ── Google connections ─────────────────────────────────────────────────────

def upsert_google_tokens(user_id: str, access_token: str,
                         refresh_token: Optional[str],
                         token_expiry: Optional[str],
                         scopes: Optional[str]):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO google_connections (user_id, access_token, refresh_token, token_expiry, scopes)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    access_token  = EXCLUDED.access_token,
                    refresh_token = COALESCE(EXCLUDED.refresh_token, google_connections.refresh_token),
                    token_expiry  = EXCLUDED.token_expiry,
                    scopes        = EXCLUDED.scopes,
                    updated_at    = NOW()
            """, (user_id, access_token, refresh_token, token_expiry, scopes))
        conn.commit()
    finally:
        conn.close()


def get_google_tokens(user_id: str) -> Optional[dict]:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM google_connections WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def delete_google_tokens(user_id: str):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM google_connections WHERE user_id = %s", (user_id,))
        conn.commit()
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


def _build_flow() -> Flow:
    client_config = {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [GOOGLE_REDIRECT_URI],
        }
    }
    return Flow.from_client_config(
        client_config, scopes=GOOGLE_SCOPES, redirect_uri=GOOGLE_REDIRECT_URI
    )


def _get_credentials(user_id: str) -> Optional[Credentials]:
    tokens = get_google_tokens(user_id)
    if not tokens:
        return None

    scopes = tokens["scopes"].split(",") if tokens.get("scopes") else GOOGLE_SCOPES
    creds  = Credentials(
        token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=scopes,
    )

    if creds.expired and creds.refresh_token:
        creds.refresh(GoogleRequest())
        upsert_google_tokens(
            user_id,
            creds.token,
            creds.refresh_token,
            creds.expiry.isoformat() if creds.expiry else None,
            ",".join(creds.scopes) if creds.scopes else None,
        )

    return creds


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
    body     = await request.json()
    username = body.get("username", "").strip()
    email    = body.get("email", "").strip()
    password = body.get("password", "")
    confirm  = body.get("confirm_password", "")

    if not username or not email or not password:
        raise HTTPException(400, "Todos los campos son obligatorios")
    if password != confirm:
        raise HTTPException(400, "Las contraseñas no coinciden")
    if len(password) < 6:
        raise HTTPException(400, "La contraseña debe tener al menos 6 caracteres")
    if get_user_by_username(username):
        raise HTTPException(400, "El usuario ya existe")
    if get_user_by_email(email):
        raise HTTPException(400, "El email ya está registrado")

    create_user(username, email, password)
    user = get_user_by_username(username)

    session_id = secrets.token_urlsafe(32)
    sessions[session_id] = {"user_id": str(user["id"]), "username": user["username"]}

    resp = JSONResponse({"success": True, "username": username})
    resp.set_cookie("session_id", session_id, httponly=True, samesite="lax")
    return resp


@app.post("/auth/login")
async def login(request: Request):
    body     = await request.json()
    username = body.get("username", "").strip()
    password = body.get("password", "")

    user = get_user_by_username(username)
    if not user or not pwd_context.verify(_prepare(password), user["hashed_password"]):
        raise HTTPException(401, "Usuario o contraseña incorrectos")

    session_id = secrets.token_urlsafe(32)
    sessions[session_id] = {"user_id": str(user["id"]), "username": user["username"]}

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
        raise HTTPException(401, "No autenticado")
    return {"username": user["username"]}


# ---------------------------------------------------------------------------
# Extraction routes
# ---------------------------------------------------------------------------

@app.get("/extractions")
async def list_extractions(
    request: Request,
    tipo:   Optional[str]  = Query(None),
    copied: Optional[bool] = Query(None),
):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "No autenticado")
    return get_extractions(user["user_id"], tipo, copied)


@app.patch("/extractions/mark-copied")
async def mark_copied(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "No autenticado")
    body = await request.json()
    ids  = body.get("ids", [])
    if not ids:
        raise HTTPException(400, "No se proporcionaron IDs")
    count = mark_extractions_copied(user["user_id"], ids)
    return {"marked": count}


@app.delete("/extractions")
async def delete_extractions_route(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "No autenticado")
    body = await request.json()
    ids  = body.get("ids", [])
    if not ids:
        raise HTTPException(400, "No se proporcionaron IDs")
    count = delete_extractions_by_ids(user["user_id"], ids)
    return {"deleted": count}


async def _do_extract(request: Request, file: UploadFile, tipo: str):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "No autenticado")

    content_type = file.content_type or "image/jpeg"
    image_data   = base64.b64encode(await file.read()).decode("utf-8")

    try:
        response = gemini_model.generate_content([
            {"mime_type": content_type, "data": image_data},
            PROMPT,
        ])
    except Exception as e:
        raise HTTPException(502, f"Gemini error: {e}")

    raw_text = response.text.strip()
    start = raw_text.find("{")
    end   = raw_text.rfind("}") + 1
    if start != -1 and end > start:
        raw_text = raw_text[start:end]

    try:
        extracted = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"JSON parse error: {e}\nRaw: {raw_text}")

    row = save_extraction(
        user["user_id"], tipo,
        datetime.now().strftime("%d/%m/%Y"),
        extracted.get("nhc", ""),
        extracted.get("nombre", ""),
        "",
        extracted.get("medico", ""),
        extracted.get("entidad", ""),
    )
    return row


@app.post("/extract")
async def extract(request: Request, file: UploadFile = File(...)):
    return await _do_extract(request, file, tipo="Etiqueta")


@app.post("/extract/ficha")
async def extract_ficha(request: Request, file: UploadFile = File(...)):
    return await _do_extract(request, file, tipo="Ficha Medica")


# ---------------------------------------------------------------------------
# Google OAuth routes
# ---------------------------------------------------------------------------

@app.get("/google/auth/status")
async def google_auth_status(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "No autenticado")
    tokens = get_google_tokens(user["user_id"])
    return {"connected": tokens is not None}


@app.get("/google/auth/login")
async def google_auth_login(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "No autenticado")
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(500, "Google OAuth no está configurado en el servidor")

    flow = _build_flow()
    auth_url, state = flow.authorization_url(access_type="offline", prompt="consent")

    # Save state AND code_verifier (PKCE) in session so callback can use them
    session_id = request.cookies.get("session_id")
    if session_id and session_id in sessions:
        sessions[session_id]["google_oauth_state"]    = state
        sessions[session_id]["google_code_verifier"]  = getattr(flow, "code_verifier", None)

    return {"auth_url": auth_url}


@app.get("/google/auth/callback")
async def google_auth_callback(request: Request, code: str, state: str):
    session_id = request.cookies.get("session_id")
    session    = sessions.get(session_id) if session_id else None

    if not session:
        return RedirectResponse("/?google_error=session_lost")

    stored_state = session.pop("google_oauth_state", None)
    if stored_state != state:
        return RedirectResponse("/?google_error=state_mismatch")

    try:
        flow = _build_flow()

        # Restore PKCE code_verifier generated during login (new flow instance doesn't have it)
        code_verifier = session.pop("google_code_verifier", None)
        if code_verifier:
            flow.code_verifier = code_verifier

        # Render proxies internally as http:// even when the public URL is https://
        # The registered redirect URI uses https://, so we must upgrade the scheme here
        auth_response = str(request.url)
        if GOOGLE_REDIRECT_URI.startswith("https://") and auth_response.startswith("http://"):
            auth_response = "https://" + auth_response[7:]

        flow.fetch_token(authorization_response=auth_response)
        creds = flow.credentials
    except Exception as e:
        import urllib.parse
        err = urllib.parse.quote(str(e)[:120], safe="")
        return RedirectResponse(f"/?google_error={err}")

    upsert_google_tokens(
        session["user_id"],
        creds.token,
        creds.refresh_token,
        creds.expiry.isoformat() if creds.expiry else None,
        ",".join(creds.scopes) if creds.scopes else None,
    )

    return RedirectResponse("/?google_connected=1")


@app.post("/google/auth/disconnect")
async def google_auth_disconnect(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "No autenticado")
    delete_google_tokens(user["user_id"])
    return {"success": True}


# ---------------------------------------------------------------------------
# Google Sheets routes
# ---------------------------------------------------------------------------

@app.get("/google/sheets")
async def google_list_sheets(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "No autenticado")

    creds = _get_credentials(user["user_id"])
    if not creds:
        raise HTTPException(401, "Google no conectado")

    try:
        service = build("drive", "v3", credentials=creds)
        result  = service.files().list(
            q="mimeType='application/vnd.google-apps.spreadsheet' and trashed=false",
            fields="files(id, name)",
            orderBy="modifiedTime desc",
            pageSize=50,
        ).execute()
        return result.get("files", [])
    except Exception as e:
        raise HTTPException(500, f"Error listando hojas: {e}")


@app.post("/google/send")
async def google_send_to_sheet(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "No autenticado")

    body           = await request.json()
    spreadsheet_id = body.get("spreadsheet_id")
    ids            = body.get("ids", [])

    if not spreadsheet_id:
        raise HTTPException(400, "spreadsheet_id es obligatorio")
    if not ids:
        raise HTTPException(400, "No se proporcionaron registros")

    creds = _get_credentials(user["user_id"])
    if not creds:
        raise HTTPException(401, "Google no conectado")

    rows_data = get_extractions_by_ids(user["user_id"], ids)
    if not rows_data:
        raise HTTPException(404, "Registros no encontrados")

    EXPECTED_HEADERS = ["Fecha", "NHC", "Nombre", "PTC", "Médico", "Entidad"]

    try:
        sheets_svc = build("sheets", "v4", credentials=creds)

        # Get the first sheet tab name (avoids "Sheet1" vs "Hoja 1" issue)
        meta       = sheets_svc.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        sheet_name = meta["sheets"][0]["properties"]["title"]
        range_ref  = f"'{sheet_name}'!A:F"

        # Read the first row to check structure
        first_row_resp = sheets_svc.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f"'{sheet_name}'!A1:F1"
        ).execute()
        first_row = first_row_resp.get("values", [[]])[0] if first_row_resp.get("values") else []

        values = []
        if not first_row:
            # Sheet is empty — write headers first
            values.append(EXPECTED_HEADERS)
        else:
            # Sheet has content — validate headers match (case-insensitive, accent-tolerant)
            def normalize(s):
                return s.strip().lower().replace("é", "e").replace("á", "a").replace("í", "i")

            expected_norm = [normalize(h) for h in EXPECTED_HEADERS]
            actual_norm   = [normalize(str(c)) for c in first_row[:6]]

            if actual_norm != expected_norm:
                raise HTTPException(
                    400,
                    f"La hoja '{sheet_name}' tiene una estructura diferente. "
                    f"Se esperaban las columnas: {', '.join(EXPECTED_HEADERS)}. "
                    f"Se encontró: {', '.join(first_row[:6]) or '(vacío)'}. "
                    f"Selecciona una hoja vacía o una que ya tenga el formato correcto."
                )

        for r in rows_data:
            values.append([r["fecha"], r["nhc"] or "", r["nombre"] or "",
                           r["ptc"] or "", r["medico"] or "", r["entidad"] or ""])

        sheets_svc.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=range_ref,
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": values},
        ).execute()

        spreadsheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        return {"success": True, "rows_written": len(rows_data), "spreadsheet_url": spreadsheet_url}

    except HTTPException:
        raise  # re-raise our own validation errors as-is
    except Exception as e:
        raise HTTPException(500, f"Error enviando a Sheets: {e}")
