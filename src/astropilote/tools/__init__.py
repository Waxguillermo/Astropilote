"""Six deterministic and retrieval tools exposed to the Google ADK agent."""

from astropilote.tools.catalog import get_object, search_catalog
from astropilote.tools.sky import night, visibility
from astropilote.tools.world import geocode, weather

ALL_TOOLS = [search_catalog, get_object, night, visibility, geocode, weather]
TOOLS_BY_NAME = {t.__name__: t for t in ALL_TOOLS}

__all__ = [
    "ALL_TOOLS",
    "TOOLS_BY_NAME",
    "search_catalog",
    "get_object",
    "night",
    "visibility",
    "geocode",
    "weather",
]
