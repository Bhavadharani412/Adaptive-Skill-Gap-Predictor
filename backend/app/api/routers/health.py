from datetime import datetime
from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.schemas.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
def get_health():
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )
