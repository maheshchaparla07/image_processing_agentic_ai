from fastapi import FastAPI

from app.api.v1.router import api_router

app = FastAPI(title="API", version="1.0.0")

@app.get("/")
async def health_check():
    return {"status": "ok"}

app.include_router(api_router, prefix="/api/v1")