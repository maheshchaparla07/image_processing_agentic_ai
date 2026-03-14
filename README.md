# Image Detector API

FastAPI backend for image uploads and AI-assisted image analysis.

## Features

- Health check endpoint
- Versioned API routes under `/api/v1`
- Image upload endpoint with file validation
- OpenAI-based analysis to detect if an image appears AI-generated

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

	`pip install -r requirements.txt`

3. Set environment variables in `.env`.
4. Start server:

	`uvicorn app.main:app --reload`

## Key endpoint

- `POST /api/v1/images/upload`


