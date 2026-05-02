# MedExtract — Project Summary

## What It Does

**MedExtract** is a web application for extracting structured data from Spanish medical pre-operative consultation documents. Users upload photos of medical documents, and Google Gemini AI automatically extracts key fields (patient ID/NHC, patient name, doctor, insurance entity, and date). The results can be copied to clipboard, downloaded as CSV, or sent directly to Google Sheets.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3 · FastAPI |
| Frontend | Vanilla HTML/CSS/JS (single-page app) |
| AI | Google Gemini 2.5 Flash |
| Database | PostgreSQL via Supabase |
| Auth | Username/password + session cookies |
| Cloud | Google OAuth 2.0 · Sheets API · Drive API |
| Deployment | Render |

---

## File Structure

```
new project/
├── main.py              # 739 lines — entire FastAPI backend
├── static/index.html    # ~1,400 lines — SPA (HTML + CSS + JS)
├── requirements.txt     # 12 Python dependencies
├── .env                 # API keys & secrets
└── README.md            # Documentation
```

---

## Architecture

### Backend (main.py)

Monolithic single-file FastAPI app with four areas:

1. **Authentication** — Register/login with bcrypt-hashed passwords, HTTPOnly session cookies, in-memory session store
2. **Document Extraction** — Two endpoints (`/extract` and `/extract/ficha`) that base64-encode uploaded images, send them to Gemini with a hardcoded Spanish-language extraction prompt, and save results to PostgreSQL
3. **Data Management** — List, delete, and mark-as-copied endpoints; supports a "pending queue" pattern (records hidden after being processed)
4. **Google Integration** — Full OAuth 2.0 with PKCE, token refresh, Sheets list, and append-to-sheet logic with smart header validation

### Database Schema

Three PostgreSQL tables:

- `users` — id, username, email, hashed_password, created_at
- `extractions` — id, user_id, tipo, fecha, nhc, nombre, ptc, medico, entidad, copied, created_at
- `google_connections` — id, user_id, access_token, refresh_token, token_expiry, scopes

### Frontend (static/index.html)

Single-page app (~1,400 lines of HTML + CSS + JS) with four views:

- **Etiquetas** — Upload + pending queue for label-type documents
- **Ficha Médica** — Upload + pending queue for medical record documents
- **Historia** — Full archive with bulk select, copy, send-to-Sheets, and delete
- **Configuración** — Google account connection and default sheet selection

---

## Key Design Decisions

- **Pending queue pattern**: Etiquetas/Ficha show only unprocessed (`copied=false`) records; Historia shows everything
- **In-memory sessions**: Simple dict-based store — works for single-instance but lost on server restart
- **Auto-migration**: `init_db()` runs on startup and handles schema changes idempotently
- **Monolithic layout**: All backend logic in one `main.py` — manageable at current scale

---

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini AI access |
| `DATABASE_URL` | Yes | Supabase PostgreSQL connection string |
| `GOOGLE_CLIENT_ID` | No | OAuth (Google features disabled if absent) |
| `GOOGLE_CLIENT_SECRET` | No | OAuth |
| `GOOGLE_REDIRECT_URI` | No | OAuth callback URL (defaults to localhost) |

---

## Deployment

- Hosted on **Render** with auto-deploy on `git push`
- Database on **Supabase** (PostgreSQL with connection pooling)
- HTTPS-ready with transparent HTTP→HTTPS proxy handling for OAuth
