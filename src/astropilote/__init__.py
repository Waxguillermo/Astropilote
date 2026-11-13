"""AstroPilote — Hybrid RAG & Agentic Copilot for Visual Astronomy and Astrophotography."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("astropilote")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.2"
