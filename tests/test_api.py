"""Integration tests for the FastAPI backend and custom Observatory UI (`src/astropilote/app/main.py`)."""

from fastapi.testclient import TestClient

from astropilote.app.main import app

client = TestClient(app)


def test_health_and_static_index() -> None:
    """Verify `/api/health` and `/` HTML delivery."""
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    payload = r_health.json()
    assert payload["status"] == "ok"
    assert payload["corpus_documents"] == 506
    assert payload["indexed_chunks"] >= 2000

    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "AstroPilote" in r_index.text
    assert "<dialog" in r_index.text


def test_presets_and_catalog_endpoint() -> None:
    """Verify `/api/presets` and `/api/catalog/{object_id}`."""
    r_presets = client.get("/api/presets")
    assert r_presets.status_code == 200
    data = r_presets.json()
    assert len(data["presets"]) >= 6

    r_obj = client.get("/api/catalog/M031?focal_length_mm=480")
    assert r_obj.status_code == 200
    obj = r_obj.json()
    assert obj["doc_id"] == "M031"
    assert obj["framing_advice"] is not None


def test_ephemeris_and_chat_endpoints() -> None:
    """Verify `/api/ephemeris` and `/api/chat` end-to-end execution."""
    r_eph = client.post(
        "/api/ephemeris",
        json={
            "place_name": "Fontainebleau",
            "obs_date": "2026-10-17",
            "focal_length_mm": 480.0,
            "aperture_mm": 80.0,
            "object_ids": ["M031", "M013"],
        },
    )
    assert r_eph.status_code == 200
    eph = r_eph.json()
    assert eph["night"]["has_astronomical_darkness"] is True
    assert "M031" in eph["visibility"]["objects"]

    r_chat = client.post(
        "/api/chat",
        json={
            "question": "In which constellation is Messier 13 located?",
            "focal_length_mm": 480.0,
            "aperture_mm": 80.0,
        },
    )
    assert r_chat.status_code == 200
    chat = r_chat.json()
    assert chat["answer"]["verified"] is True
    assert "M013" in chat["answer"]["sources"]
