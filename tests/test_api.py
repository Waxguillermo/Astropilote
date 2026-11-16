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


def test_ephemeris_and_chat_endpoints() -> None:
    """Verify `/api/ephemeris` and `/api/chat` end-to-end execution offline without external HTTP leaks."""
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
    assert eph["location"]["source"] == "fixture"
    assert eph["weather"]["source"] == "fixture"
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


def test_observatory_mcp_studio_endpoints_offline(tmp_path, monkeypatch) -> None:
    """Verify isolated `/api/observatory/*` routes (state, reset, plan/trace run, and HITL approve/reject) offline."""
    import astropilote.app.main as main_mod

    monkeypatch.setattr(main_mod, "OBSERVATORY_DB_PATH", tmp_path / "observatory.sqlite3")
    monkeypatch.setattr(main_mod, "OBSERVATORY_CHECKPOINT_PATH", tmp_path / "checkpoints.sqlite3")
    monkeypatch.setattr(main_mod, "OBSERVATORY_MEMORY_PATH", tmp_path / "memory.sqlite3")

    r_index = client.get("/")
    assert r_index.status_code == 200
    assert 'id="observatory-pill-btn"' in r_index.text

    # Natural instrument naming without raw T-xx ID maps cleanly to T-02
    r_pref = client.post(
        "/api/observatory/run",
        json={
            "text": "Je préfère la lunette 120mm",
            "user_id": "astro-club",
            "llm_mode": "offline",
            "reference_date": "2026-11-14",
        },
    )
    assert r_pref.status_code == 200
    assert r_pref.json()["preferred_telescope_id"] == "T-02"
    assert r_pref.json()["interpreted_text"] == "My preferred telescope is T-02"

    r_state = client.get("/api/observatory/state?user_id=astro-club")
    assert r_state.status_code == 200
    initial = r_state.json()
    assert len(initial["state"]["telescopes"]) == 2
    assert len(initial["state"]["bookings"]) == 1

    # 1. Multi-step weather + safe booking in offline mode ($0 API spend)
    r_book = client.post(
        "/api/observatory/run",
        json={
            "text": "Check the night weather in Lyon on 2026-11-17 and if the sky is clear, book telescope slot S-01 for me",
            "user_id": "alice",
            "llm_mode": "offline",
            "reference_date": "2026-11-14",
        },
    )
    assert r_book.status_code == 200
    book_payload = r_book.json()
    assert "interpreted_text" not in book_payload
    assert book_payload["simulation"] == {"reference_date": "2026-11-14", "offline_sky": True}
    assert book_payload["response"]["status"] == "completed"
    assert [c["tool"] for c in book_payload["tool_calls"]] == [
        "geocode",
        "weather",
        "list_slots",
        "create_booking",
    ]
    assert len(book_payload["plan"]) >= 2
    assert len(book_payload["state"]["bookings"]) == 2

    # 2. Dangerous cancellation pauses at checkpointed HITL interrupt before touching SQLite
    r_cancel = client.post(
        "/api/observatory/run",
        json={
            "text": "Annule ma session d'observatoire BK-001",
            "user_id": "astro-club",
            "llm_mode": "offline",
            "reference_date": "2026-11-14",
        },
    )
    assert r_cancel.status_code == 200
    cancel_payload = r_cancel.json()
    assert cancel_payload["response"]["status"] == "awaiting_human_approval"
    assert cancel_payload["response"]["pending_approval"] is True
    assert cancel_payload["response"]["pending_booking_id"] == "BK-001"
    thread_id = cancel_payload["response"]["thread_id"]
    bk1_before = next(b for b in cancel_payload["state"]["bookings"] if b["id"] == "BK-001")
    assert bk1_before["status"] == "confirmed"

    # 3. Operator rejection leaves BK-001 confirmed
    r_reject = client.post(
        "/api/observatory/approve",
        json={"thread_id": thread_id, "approved": False, "user_id": "astro-club"},
    )
    assert r_reject.status_code == 200
    rej_payload = r_reject.json()
    assert rej_payload["response"]["status"] == "human_rejected"
    bk1_after_rej = next(b for b in rej_payload["state"]["bookings"] if b["id"] == "BK-001")
    assert bk1_after_rej["status"] == "confirmed"

    # 4. Second cancellation + explicit operator approval cancels & refunds BK-001
    r_cancel_2 = client.post(
        "/api/observatory/run",
        json={
            "text": "Cancel telescope reservation BK-001",
            "user_id": "astro-club",
            "llm_mode": "offline",
            "reference_date": "2026-11-14",
        },
    )
    thread_id_2 = r_cancel_2.json()["response"]["thread_id"]
    r_approve = client.post(
        "/api/observatory/approve",
        json={"thread_id": thread_id_2, "approved": True, "user_id": "astro-club"},
    )
    assert r_approve.status_code == 200
    app_payload = r_approve.json()
    assert app_payload["response"]["status"] == "completed"
    assert [c["tool"] for c in app_payload["tool_calls"]] == ["get_booking", "cancel_booking"]
    bk1_after_app = next(b for b in app_payload["state"]["bookings"] if b["id"] == "BK-001")
    assert bk1_after_app["status"] == "cancelled"
    assert bk1_after_app["refunded"] is True

    # 5. Reset restores canonical seed
    r_reset = client.post(
        "/api/observatory/reset",
        json={"seed_name": "observatory_demo.json", "user_id": "astro-club", "clear_memory": True},
    )
    assert r_reset.status_code == 200
    assert len(r_reset.json()["state"]["bookings"]) == 1


def test_observatory_approval_is_user_bound_single_use_and_dies_with_the_dataset(
    tmp_path, monkeypatch
) -> None:
    """A paused cancellation can only be resolved once, by its requester, on the dataset it targets."""
    import astropilote.app.main as main_mod

    monkeypatch.setattr(main_mod, "OBSERVATORY_DB_PATH", tmp_path / "observatory.sqlite3")
    monkeypatch.setattr(main_mod, "OBSERVATORY_CHECKPOINT_PATH", tmp_path / "checkpoints.sqlite3")
    monkeypatch.setattr(main_mod, "OBSERVATORY_MEMORY_PATH", tmp_path / "memory.sqlite3")

    def request_cancellation() -> str:
        response = client.post(
            "/api/observatory/run",
            json={"text": "Cancel telescope reservation BK-001", "user_id": "astro-club"},
        )
        assert response.json()["response"]["pending_approval"] is True
        return response.json()["response"]["thread_id"]

    def decide(thread_id: str, *, approved: bool, user_id: str = "astro-club"):
        return client.post(
            "/api/observatory/approve",
            json={"thread_id": thread_id, "approved": approved, "user_id": user_id},
        )

    def booking_status() -> str:
        state = client.get("/api/observatory/state").json()["state"]
        return next(b["status"] for b in state["bookings"] if b["id"] == "BK-001")

    # Another user cannot approve, and the attempt does not consume the request.
    thread_id = request_cancellation()
    gateway = main_mod._observatory_gateway
    assert decide(thread_id, approved=True, user_id="alice").status_code == 403

    # A reset swaps the dataset: the paused approval must not act on the new BK-001.
    client.post("/api/observatory/reset", json={"seed_name": "observatory_demo.json"})
    assert decide(thread_id, approved=True).status_code == 404
    assert booking_status() == "confirmed"

    # A decision is single-use.
    thread_id = request_cancellation()
    assert decide(thread_id, approved=False).json()["response"]["status"] == "human_rejected"
    assert decide(thread_id, approved=True).status_code == 404
    assert booking_status() == "confirmed"

    # An expired request is refused and its paused thread is discarded, even when another
    # request was registered in the meantime.
    monkeypatch.setattr(main_mod, "_APPROVAL_TTL_S", 0)
    thread_id = request_cancellation()
    request_cancellation()
    assert decide(thread_id, approved=True).status_code == 410
    assert decide(thread_id, approved=True).status_code == 404
    assert booking_status() == "confirmed"

    # Every request above went through the same MCP session instead of respawning servers.
    assert main_mod._observatory_gateway is gateway
