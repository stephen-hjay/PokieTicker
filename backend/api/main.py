from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.database import init_db
from backend.api.routers import stocks, news, analysis, pipeline, predict

IMAGES_DIR = Path(__file__).resolve().parent.parent / "static" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PokieTicker", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:7777", "http://127.0.0.1:7777"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(news.router, prefix="/api/news", tags=["news"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(predict.router, prefix="/api/predict", tags=["predict"])
app.mount("/api/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


@app.on_event("startup")
def startup():
    init_db()


@app.get("/api/health")
def health():
    return {"status": "ok"}
