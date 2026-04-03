# MedExtract

A web application that uses **Google Gemini 2.5 Flash** to extract structured data from Spanish medical pre-operative consultation documents. Extracted data is displayed in a table that users can copy to the clipboard or download as CSV.

---

## Features

- **AI-powered extraction** — Upload a photo of a medical document and Gemini automatically extracts: Date, NHC, Patient Name, Doctor, and Insurance Entity
- **Two extraction types** — Etiquetas (labels) and Ficha Médica (medical record)
- **Pending queue** — Etiquetas and Ficha Médica show only unprocessed records; copying them marks them as done and clears the view
- **Historia** — Permanent archive of all extractions with status (Pending / Copied), bulk select, copy and delete
- **User authentication** — Register/login with username and password; each user sees only their own data
- **Responsive** — Works on desktop and mobile (bottom navigation + mobile header on small screens)
- **CSV export** — Download any table as a CSV file
- **Cloud-ready** — Deployed on Render with Supabase PostgreSQL

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python · FastAPI |
| AI Model | Google Gemini 2.5 Flash |
| Database | PostgreSQL (Supabase) |
| Auth | Username/password · bcrypt · session cookies |
| Frontend | Vanilla HTML/CSS/JS (single page) |
| Hosting | Render |

---

## Project Structure

```
├── main.py               # FastAPI application (routes, DB, auth, extraction)
├── static/
│   └── index.html        # Single-page frontend
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── .gitignore
└── README.md
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USER/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://user:password@host:port/dbname
```

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key — get it at [aistudio.google.com](https://aistudio.google.com) |
| `DATABASE_URL` | PostgreSQL connection string (Supabase or local) |

### 5. Run the server

```bash
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Database

Tables are created automatically on startup. No manual migration is needed.

### `users`
| Column | Type | Description |
|---|---|---|
| id | SERIAL | Primary key |
| username | TEXT | Unique username |
| email | TEXT | Unique email |
| hashed_password | TEXT | bcrypt hash |
| created_at | TIMESTAMP | Registration date |

### `extractions`
| Column | Type | Description |
|---|---|---|
| id | SERIAL | Primary key |
| user_id | INTEGER | Foreign key → users |
| tipo | TEXT | `Etiqueta` or `Ficha Medica` |
| fecha | TEXT | Extraction date (dd/mm/yyyy) |
| nhc | TEXT | Patient NHC number |
| nombre | TEXT | Patient full name |
| ptc | TEXT | PTC (always empty, reserved) |
| medico | TEXT | Doctor name |
| entidad | TEXT | Insurance entity |
| copied | BOOLEAN | True once the user has copied the record |
| created_at | TIMESTAMP | Extraction timestamp |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend |
| `POST` | `/auth/register` | Create a new user account |
| `POST` | `/auth/login` | Login and get session cookie |
| `POST` | `/auth/logout` | Logout and clear session |
| `GET` | `/auth/me` | Return current logged-in user |
| `POST` | `/extract` | Extract fields from a document (tipo=Etiqueta) |
| `POST` | `/extract/ficha` | Extract fields from a document (tipo=Ficha Medica) |
| `GET` | `/extractions` | List extractions (supports `?tipo=` and `?copied=` filters) |
| `PATCH` | `/extractions/mark-copied` | Mark records as copied by ID list |
| `DELETE` | `/extractions` | Delete records permanently by ID list |

---

## Deployment on Render

### 1. Push code to GitHub

```bash
git add .
git commit -m "Initial commit"
git push
```

### 2. Create a Web Service on Render

- **Runtime**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### 3. Set environment variables on Render

| Key | Value |
|---|---|
| `GEMINI_API_KEY` | Your Gemini API key |
| `DATABASE_URL` | Supabase connection pooler URL (port 6543) |

> Use the **Connection Pooler** URL from Supabase (port `6543`), not the direct connection (port `5432`), to avoid network issues on Render's free tier.

### 4. Deploy

Every `git push` to the main branch triggers an automatic redeploy.

---

## Extracted Fields

The app reads Spanish pre-operative consultation documents and extracts:

| App Field | Document Source |
|---|---|
| Fecha | Today's date (date of extraction) |
| NHC | Value next to "NHC" label |
| Nombre | Patient name in large bold text below the title |
| PTC | Always empty (reserved for future use) |
| Médico | Value next to "Médico" label |
| Entidad | Value next to "Entidad" label |

---

## Updating the App

After making code changes locally:

```bash
git add .
git commit -m "Description of changes"
git push
```

Render detects the push and redeploys automatically (takes ~1–2 minutes).
