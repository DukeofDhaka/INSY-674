from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api import api_router
from app.config import get_logger, settings, setup_logging
from src.processing.data_manager import model_artifact_path

setup_logging(settings)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        f"Starting API | env={settings.ENVIRONMENT} host={settings.HOST} port={settings.PORT}"
    )
    artifact_path = model_artifact_path()
    if artifact_path.exists():
        logger.info(f"Model artifact found: {artifact_path.name}")
    else:
        logger.warning(
            "Model artifact is missing. Run `python -m src.train_pipeline` before inference."
        )
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

root_router = APIRouter()


@root_router.get("/", response_class=HTMLResponse)
def index(request: Request) -> str:
    _ = request
    return (
        "<h3>INSY-674 ML API</h3>"
        "<p>OpenAPI docs are available at <a href='/docs'>/docs</a>.</p>"
    )


app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(root_router)

if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
