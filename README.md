# Image Detector API

This project is a FastAPI backend that accepts uploaded media, analyzes it, and tells you whether it likely came from AI generation or a real capture.

Even though the repository name says “image detector”, the current API handles both images and videos through one shared pipeline.

## What this project gives you

You can upload a media file and get back:

- extracted metadata
- AI analysis text from OpenAI
- a final decision: `REAL` or `AI_GENERATED`
- where the processed file was saved
- a full processing log so you can follow each step

The repo also contains starter auth and user endpoints plus SQLAlchemy models for persistence.

## Tech stack (quick view)

- **API framework:** FastAPI
- **Workflow orchestration:** LangGraph
- **Image processing:** Pillow
- **AI analysis:** OpenAI API
- **ORM/data layer:** SQLAlchemy
- **Database target:** PostgreSQL
- **Container support:** Docker + Docker Compose

## How a request moves through the app

### 1) App boot

The app starts in `app.main` and mounts versioned routes under `/api/v1`.

### 2) Routers

Three router groups are registered:

- `/auth`
- `/users`
- `/media`

### 3) Main upload endpoint

Main flow entrypoint:

- `POST /api/v1/media/upload`

This endpoint:

1. validates file type
2. validates file size (image/video limits differ)
3. runs the LangGraph pipeline in a worker thread
4. tries to persist a record in DB
5. returns a structured response

### 4) Pipeline path

The pipeline lives in `app/services/media_pipeline_graph.py`.

```mermaid
flowchart TD
    A[Upload request] --> B[Upload Agent]
    B --> C[File Type Classifier]
    C -->|image| D[Image Agent]
    C -->|video| E[Video Agent]
    D --> F[AI Detection Agent]
    E --> F
    F --> G[Decision Agent]
    G -->|REAL| H[Store File Agent]
    G -->|AI_GENERATED| I[Watermark Agent]
    I --> J[Store Result Agent]
```

## Pipeline stages explained in plain English

### Agent 1: Upload Agent

Checks that bytes and filename exist, then starts the processing log.

### Agent 2: File Type Classifier

Classifies upload as `image`, `video`, or `unknown` based on MIME type.

### Agent 3: Image Agent

Extracts image details like dimensions, mode, EXIF presence/data, color stats, and SHA-256 hash.

### Agent 4: Video Agent

Performs lightweight binary/container inspection (no FFmpeg), then records file stats and SHA-256 hash.

### Agent 5: AI Detection Agent

- Images are sent for vision-style analysis.
- Videos are analyzed from extracted metadata text.

If `OPENAI_API_KEY` is missing, analysis is skipped gracefully and defaults to:

- `ai_detection_result = NOT_AI_GENERATED`
- decision = `REAL`

### Agent 6: Decision Agent

Maps AI output to business decision:

- `NOT_AI_GENERATED` → `REAL`
- `AI_GENERATED` → `AI_GENERATED`

### Agent 7: Store File Agent

If decision is `REAL`, saves the original media.

### Agent 8: Watermark Agent

If decision is `AI_GENERATED`:

- images get an `AI GENERATED` banner
- videos are saved with a JSON sidecar flag file

### Agent 9: Store Result Agent

Final bookkeeping step; confirms completion in the processing log.

## API endpoints

### Health

- `GET /`

Response:

```json
{
  "status": "ok"
}
```

### Media upload

- `POST /api/v1/media/upload`

Input:

- `multipart/form-data`
- field name: `file`

Allowed image MIME types:

- `image/jpeg`
- `image/png`
- `image/gif`
- `image/webp`

Allowed video MIME types:

- `video/mp4`
- `video/mpeg`
- `video/quicktime`
- `video/x-msvideo`
- `video/webm`

Size limits:

- images: **10 MB**
- videos: **100 MB**

### Auth (placeholder)

- `POST /api/v1/auth/login`

### Users (placeholder)

- `GET /api/v1/users/`
- `POST /api/v1/users/`

These user/auth endpoints are currently stubs.

## Example response from upload

```json
{
  "filename": "sample.png",
  "content_type": "image/png",
  "size_bytes": 184520,
  "media_type": "image",
  "metadata": {
    "format": "PNG",
    "mode": "RGBA",
    "width": 1024,
    "height": 1024,
    "megapixels": 1.05,
    "aspect_ratio": "1024:1024",
    "has_transparency": true,
    "has_exif": false,
    "exif_fields_count": 0,
    "exif_data": {},
    "colour_stats": {
      "mean": [123.4, 118.9, 121.2],
      "stddev": [41.6, 39.8, 40.5]
    },
    "sha256": "..."
  },
  "ai_analysis": "RESULT: AI_GENERATED\nANALYSIS: ...",
  "ai_detection_result": "AI_GENERATED",
  "decision": "AI_GENERATED",
  "stored_file_path": null,
  "watermarked_file_path": "uploads/sample_xxx_ai_watermarked.png",
  "processing_log": [
    "upload_agent: validated input",
    "file_type_classifier_agent: classified as 'image'",
    "image_agent: extracted metadata",
    "ai_detection_agent: detection=AI_GENERATED",
    "decision_agent: routing decision='AI_GENERATED'",
    "watermark_agent: watermarked image saved",
    "store_result_agent: pipeline complete"
  ]
}
```

## Project layout

```text
app/
├── main.py
├── api/
│   ├── deps.py
│   └── v1/
│       ├── router.py
│       └── endpoints/
│           ├── auth.py
│           ├── users.py
│           └── images.py
├── core/
│   ├── config.py
│   ├── database.py
│   ├── logging.py
│   └── security.py
├── models/
│   ├── image.py
│   └── user.py
├── schemas/
│   ├── image.py
│   └── user.py
└── services/
  └── media_pipeline_graph.py
docker/
├── Dockerfile
└── docker-compose.yml
uploads/
tests/
```

## Key files and what they do

### `app/main.py`

- creates FastAPI app
- exposes health route
- includes `/api/v1` router

### `app/api/v1/endpoints/images.py`

- validates upload constraints
- runs pipeline via thread pool
- tries DB persistence
- returns `MediaResponse`

### `app/services/media_pipeline_graph.py`

- defines all agents and routing
- performs metadata extraction
- calls OpenAI
- chooses real vs AI-generated storage path

### `app/models/image.py`

Stores media identity, metadata JSON, AI analysis output, decision, storage paths, user reference, and timestamp.

## Database behavior

Configured through `DATABASE_URL`.

Default:

```text
postgresql://postgres:password@localhost:5432/mydb
```

Important behavior:

- pipeline success is prioritized for user response
- DB write errors are intentionally not surfaced as upload failure

So a request can succeed even if DB is temporarily unavailable.

## Environment variables

Create a `.env` file in project root:

```env
UPLOAD_DIR=uploads
DATABASE_URL=postgresql://postgres:password@localhost:5432/mydb
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_IMAGE_MODEL=gpt-4.1-mini
```

## Run locally

### 1) Create virtual environment

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure `.env`

Add the environment variables shown above.

### 4) Start API

```bash
uvicorn app.main:app --reload
```

URLs:

- `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Run with Docker

This repo includes:

- `docker/Dockerfile`
- `docker/docker-compose.yml`

Start services:

```bash
docker compose -f docker/docker-compose.yml up --build
```

This brings up:

- API service
- PostgreSQL 16 service

## Internal upload sequence

When a file is uploaded, runtime sequence is:

1. FastAPI receives multipart payload
2. endpoint reads bytes
3. endpoint validates MIME type
4. endpoint validates size limit
5. LangGraph workflow runs in worker thread
6. metadata extraction happens
7. AI detection runs (or is skipped if no key)
8. decision is made (`REAL` or `AI_GENERATED`)
9. file is stored (plain or watermarked/flagged)
10. DB save is attempted
11. structured response is returned

## Error handling behavior

- Unsupported MIME type → HTTP `400`
- File too large → HTTP `400`
- Pipeline-level failure → HTTP `422`
- DB write failure after successful pipeline → response still succeeds

## Current limitations

- auth is not implemented yet
- user endpoints are placeholders
- no frontend in this repo
- `alembic/` exists but migrations are not defined yet
- video detection is metadata-based (not frame-level)
- AI detection is skipped when `OPENAI_API_KEY` is missing

## Good next steps

- implement real auth/token flow
- wire user endpoints to service/repository
- add DB migrations
- add upload test coverage
- add frame extraction for deeper video checks
- record richer scoring/audit detail

## Short summary

This is a practical FastAPI + LangGraph media analysis backend. Upload a file, get metadata + AI assessment + final decision, and keep a transparent processing log from start to finish.


