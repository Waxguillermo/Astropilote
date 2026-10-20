"""Integration tests for the FastAPI backend and Argon/Alter-Ego Web UI (`src/astropilote/app/main.py`)."""

from fastapi.testclient import TestClient

from astropilote.app.main import app

client = TestClient(app)


def test_health_and_static_index() -> None:
    """Verify `/api/health` and `/` HTML delivery."""
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    payload = r_health.json()
    assert payload["status"] == "ok"
    assert payload["corpus_documents"] == 567
    assert payload["indexed_chunks"] >= 2000

    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "Astro" in r_index.text and "Pilote" in r_index.text
    assert "Sky Vision" in r_index.text
    assert "<dialog" in r_index.text


def test_presets_and_catalog_endpoint() -> None:
    """Verify `/api/presets`, `/api/vision/presets`, and `/api/catalog/{object_id}`."""
    r_presets = client.get("/api/presets")
    assert r_presets.status_code == 200
    data = r_presets.json()
    assert len(data["presets"]) >= 6
    assert len(data["vision_presets"]) >= 4

    r_vis_presets = client.get("/api/vision/presets")
    assert r_vis_presets.status_code == 200
    assert len(r_vis_presets.json()["presets"]) >= 4

    r_obj = client.get("/api/catalog/M031?focal_length_mm=480")
    assert r_obj.status_code == 200
    obj = r_obj.json()
    assert obj["doc_id"] == "M031"
    assert obj["framing_advice"] is not None
    assert "Messier" in obj["hierarchy"]
    assert obj["wikipedia_url"] == "https://en.wikipedia.org/wiki/Messier_31"
    assert len(obj["chunks"]) >= 1

    r_corpus = client.get("/api/corpus")
    assert r_corpus.status_code == 200
    corpus = r_corpus.json()
    assert corpus["total_documents"] == 567
    assert len(corpus["documents"]) == 567
    assert corpus["documents"][0]["wikipedia_url"].startswith("https://en.wikipedia.org/wiki/")


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
            "mode": "agent",
            "focal_length_mm": 480.0,
            "aperture_mm": 80.0,
        },
    )
    assert r_chat.status_code == 200
    chat = r_chat.json()
    assert chat["answer"]["verified"] is True
    assert "M013" in chat["answer"]["sources"]
    assert len(chat["sources_full"]) >= 1
    assert len(chat["tool_observations"]) >= 1


def test_vision_identify_endpoint() -> None:
    """Verify `/api/vision/identify` executes Alessandro's catalog projection and `visualize.py` overlay."""
    r_vis = client.post(
        "/api/vision/identify",
        json={"preset_id": "andromeda_widefield"},
    )
    assert r_vis.status_code == 200
    payload = r_vis.json()
    assert payload["status"] == "ok"
    vis = payload["vision"]
    assert vis["schema_version"] == "1.0"
    assert vis["status"] == "ok"
    found_ids = [o["doc_id"] for o in vis["objects"]]
    assert "M031" in found_ids
    assert "M032" in found_ids
    assert "M110" in found_ids
    assert payload["annotated_image_data_url"].startswith("data:image/png;base64,")
    assert "doc_id,title,object_type,magnitude,x_px,y_px,distance_to_edge_px" in payload["csv_content"]
    assert len(payload["trace_steps"]) == 3
    assert len(payload["sources_full"]) == vis["object_count"]
    assert payload["sources_full"][0]["vision_pixel"] is not None
