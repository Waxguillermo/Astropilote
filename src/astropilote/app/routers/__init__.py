"""Modular FastAPI API routers for Astropilote v1.2."""

from astropilote.app.routers.auth import router as auth_router
from astropilote.app.routers.logbook import router as logbook_router

__all__ = ["auth_router", "logbook_router"]
