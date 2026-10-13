"""Six deterministic and retrieval tools exposed to the AstroPilote agent."""

from astropilote.tools.catalog import get_object, search_catalog
from astropilote.tools.sky import night, visibility
from astropilote.tools.world import geocode, weather

ALL_TOOLS = [search_catalog, get_object, night, visibility, geocode, weather]
for _fn in ALL_TOOLS:
    setattr(_fn, "name", _fn.__name__)
    setattr(_fn, "invoke", lambda args=None, _f=_fn, **kw: _f(**(args or {}), **kw))

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
