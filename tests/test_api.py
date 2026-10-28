"""Integration tests for the FastAPI backend and Argon/Alter-Ego Web UI (`src/astropilote/app/main.py`)."""

from fastapi.testclient import TestClient

from astropilote.app.main import app

client = TestClient(app)


def test_health_and_static_index() -> None:
    """Verify `/api/health`, local-first Gemma 4 default, and `/` HTML delivery."""
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    payload = r_health.json()
    assert payload["status"] == "ok"
    assert payload["corpus_documents"] == 567
    assert payload["indexed_chunks"] >= 2000
    assert payload["llm"]["configured_mode"] == "local"
    assert str(payload["llm"]["ollama_model"]).startswith("gemma4")

    r_index = client.get("/")
    assert r_index.status_code == 200
    html = r_index.text
    assert "Astro" in html and "Pilote" in html
    assert "Sky Vision" in html
    assert "<dialog" in html
    assert "Gemma 4 · Local" in html
    assert html.index('value="local"') < html.index('value="api"')
    assert html.index('id="api-tab-local"') < html.index('id="api-tab-api"')


def test_presets_and_catalog_endpoint() -> None:
    """Verify `/api/presets`, `/api/vision/presets`, and `/api/catalog/{object_id}`."""
    r_presets = client.get("/api/presets")
    assert r_presets.status_code == 200
    data = r_presets.json()
    assert len(data["presets"]) >= 6
    assert len(data["vision_presets"]) >= 4
    assert len(data["equipment_presets"]) >= 5

    r_eq_presets = client.get("/api/equipment/presets")
    assert r_eq_presets.status_code == 200
    assert len(r_eq_presets.json()["presets"]) >= 5

    r_vis_presets = client.get("/api/vision/presets")
    assert r_vis_presets.status_code == 200
    assert len(r_vis_presets.json()["presets"]) >= 4

    r_obj = client.get(
        "/api/catalog/M031?focal_length_mm=135&aperture_mm=67.5&sensor_format=full-frame&mount_type=star_tracker"
    )
    assert r_obj.status_code == 200
    obj = r_obj.json()
    assert obj["doc_id"] == "M031"
    assert obj["framing_advice"] is not None
    assert obj["shooting_guide"] is not None
    assert obj["shooting_guide"]["camera_settings"]["mount_type"] == "star_tracker"
    assert "Messier" in obj["hierarchy"]
    assert obj["wikipedia_url"] == "https://en.wikipedia.org/wiki/Messier_31"
    assert len(obj["chunks"]) >= 1

    r_corpus = client.get("/api/corpus")
    assert r_corpus.status_code == 200
    corpus = r_corpus.json()
    assert corpus["total_documents"] == 567
    assert len(corpus["documents"]) == 567
    assert corpus["documents"][0]["wikipedia_url"].startswith("https://en.wikipedia.org/wiki/")


def test_ephemeris_and_chat_endpoints(monkeypatch) -> None:
    """Verify `/api/ephemeris` and `/api/chat` end-to-end execution."""
    import astropilote.agent.nodes as nodes_mod
    from astropilote.llm import LLMUnavailableError, generate_grounded_answer, route_query_with_llm

    def resilient_router(question: str, ref_date_iso: str, ref_weekday: str, mode_override=None):
        try:
            return route_query_with_llm(question, ref_date_iso, ref_weekday, mode_override)
        except LLMUnavailableError:
            return (
                {
                    "request_type": "factual",
                    "language": "en",
                    "mode": "astrophotography",
                    "mentioned_targets": ["M013"],
                },
                30,
                20,
            )

    def resilient_composer(system_prompt: str, user_prompt: str, mode_override=None):
        try:
            return generate_grounded_answer(system_prompt, user_prompt, mode_override)
        except LLMUnavailableError:
            return (
                "Messier 13 (M13) is located in the constellation Hercules [M013].",
                100,
                30,
                "fallback/test",
            )

    monkeypatch.setattr(nodes_mod, "route_query_with_llm", resilient_router)
    monkeypatch.setattr(nodes_mod, "generate_grounded_answer", resilient_composer)

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
            "equipment_profile": {
                "enabled": True,
                "preset_id": "tele_135_tracker",
                "rig_name": "Téléobjectif 135mm f/2 + Full-Frame",
                "telescope_or_lens": "Samyang 135mm f/2",
                "focal_length_mm": 135.0,
                "aperture_mm": 67.5,
                "camera_model": "Sony A7 III",
                "sensor_format": "full-frame",
                "mount_type": "star_tracker",
                "has_autoguiding": False,
                "filters": ["uv_ir_cut"],
                "bortle_class": 4,
            },
        },
    )
    assert r_chat.status_code == 200
    chat = r_chat.json()
    assert chat["answer"]["verified"] is True
    assert "M013" in chat["answer"]["sources"]
    assert len(chat["sources_full"]) >= 1
    assert chat["sources_full"][0]["shooting_guide"] is not None
    assert chat["intent"]["mount_type"] == "star_tracker"
    assert chat["intent"]["sensor_format"] == "full-frame"
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
    assert (
        "doc_id,title,object_type,magnitude,x_px,y_px,distance_to_edge_px" in payload["csv_content"]
    )
    assert len(payload["trace_steps"]) == 3
    assert len(payload["sources_full"]) == vis["object_count"]
    assert payload["sources_full"][0]["vision_pixel"] is not None
