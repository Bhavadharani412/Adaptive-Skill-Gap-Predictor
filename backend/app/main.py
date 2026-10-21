import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.database.init_db import init_and_seed_database
from backend.app.api.routers import (
    health, occupations, employees, gaps,
    recommendations, simulator, experiments, admin
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing ASGP backend services and database...")
    try:
        init_and_seed_database(force_reseed=False)
        logger.info("Database verified and seeded.")
    except Exception as e:
        logger.error("Failed to initialize database during startup: %s", e)
    yield
    logger.info("Shutting down ASGP backend services...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Backend API for Adaptive Skill Gap Predictor using Occupation-Skill Knowledge Graphs and Adaptive Recommendation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware for modern frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Modular API Routers
app.include_router(health.router, prefix="/api")
app.include_router(occupations.router, prefix="/api")
app.include_router(employees.router, prefix="/api")
app.include_router(gaps.router, prefix="/api")
app.include_router(recommendations.router, prefix="/api")
app.include_router(simulator.router, prefix="/api")
app.include_router(experiments.router, prefix="/api")
app.include_router(admin.router, prefix="/api")


@app.get("/")
def root():
    return {
        "message": "Welcome to Adaptive Skill Gap Predictor (ASGP) API",
        "docs_url": "/docs",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)
