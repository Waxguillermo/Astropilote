"""HTTP integration tests for the v1.2 modular FastAPI routers (`/api/auth/*`, `/api/logbook/*`)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from astropilote.app.main import app
from astropilote.config import settings
from astropilote.db import get_db, get_engine, init_db
from astropilote.db.session import get_session_factory


def _override_db(tmp_path: Path, name: str):
    url = f"sqlite:///{tmp_path / name}"
    init_db(url)
    factory = get_session_factory(url)

    def _dep():
        sess = factory()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    return url, _dep


def test_auth_and_logbook_routers_end_to_end_local_and_cloud(tmp_path: Path, monkeypatch) -> None:
    url, dep = _override_db(tmp_path, "api_v12.sqlite3")
    app.dependency_overrides[get_db] = dep
    monkeypatch.setattr(settings, "app_secret_key", "x" * 48)
    monkeypatch.setattr(settings, "auth_mode", "local")
    try:
        client = TestClient(app)

        # Health includes vision_solvers and auth_mode
        h = client.get("/api/health")
        assert h.status_code == 200
        assert "vision_solvers" in h.json()
        assert h.json()["auth_mode"] == "local"

        # Local mode: /api/auth/me works with zero login
        me = client.get("/api/auth/me")
        assert me.status_code == 200
        assert me.json()["user"]["id"] == "local-solo"

        # Save BYOK key & verify masked display
        saved_k = client.put("/api/auth/keys/anthropic", json={"api_key": "sk-ant-test-12345678"})
        assert saved_k.status_code == 200
        assert saved_k.json()["key_hint"] == "sk-ant...5678"

        # Rigs + Sites + Observing Session + Targets + Exports + Messier Progress
        r_rig = client.post(
            "/api/logbook/rigs",
            json={
                "name": "RedCat 51",
                "focal_length_mm": 250,
                "aperture_mm": 51,
                "is_default": True,
            },
        )
        assert r_rig.status_code == 200
        rig_id = r_rig.json()["rig"]["id"]

        r_site = client.post(
            "/api/logbook/sites",
            json={"name": "Fontainebleau", "lat": 48.4047, "lon": 2.7016, "bortle_class": 5},
        )
        assert r_site.status_code == 200
        site_id = r_site.json()["site"]["id"]

        r_sess = client.post(
            "/api/logbook/sessions",
            json={
                "title": "Clear November Night",
                "obs_date": "2026-11-28",
                "site_id": site_id,
                "rig_id": rig_id,
                "targets": [
                    {
                        "object_id": "M031",
                        "object_name": "Andromeda",
                        "exposure_minutes": 60,
                        "status": "imaged",
                    }
                ],
            },
        )
        assert r_sess.status_code == 200
        sid = r_sess.json()["session"]["id"]

        # Export session in all 3 formats (.ics, N.I.N.A. .csv, Stellarium .json)
        assert "BEGIN:VCALENDAR" in client.get(f"/api/logbook/sessions/{sid}/export/ics").text
        assert "M031" in client.get(f"/api/logbook/sessions/{sid}/export/nina_csv").text
        stel = client.get(f"/api/logbook/sessions/{sid}/export/stellarium").json()
        assert stel["objects"]["obj-001"]["designation"] == "M031"

        # Vision 1-click log bridge
        v_log = client.post(
            "/api/logbook/vision-log",
            json={
                "title": "Orion Plate Solve",
                "obs_date": "2026-11-29",
                "targets": [
                    {"object_id": "M042", "object_name": "Orion Nebula", "exposure_minutes": 45}
                ],
            },
        )
        assert v_log.status_code == 200
        assert v_log.json()["progress"]["messier"]["completed"] == 2

        # Cloud mode: anonymous blocked with 401; default Magic Link response NEVER leaks dev_token
        monkeypatch.setattr(settings, "auth_mode", "cloud")
        monkeypatch.setattr(settings, "allow_dev_magic_link_response", False)
        assert client.get("/api/auth/me").status_code == 401
        ml_prod = client.post("/api/auth/magic-link", json={"email": "cloud@astro.org"})
        assert ml_prod.status_code == 200
        assert "dev_token" not in ml_prod.json()

        # Explicit opt-in dev preview allows verifying Magic Link without SMTP
        monkeypatch.setattr(settings, "allow_dev_magic_link_response", True)
        ml = client.post("/api/auth/magic-link", json={"email": "cloud@astro.org"})
        assert ml.status_code == 200
        tok = ml.json()["dev_token"]
        verified = client.get(f"/api/auth/magic-link/verify?token={tok}")
        assert verified.status_code == 200
        assert client.get("/api/auth/me").json()["user"]["email"] == "cloud@astro.org"
        # Cloud user starts with empty sessions (isolated from local-solo)
        assert client.get("/api/logbook/sessions").json()["sessions"] == []

        # Cross-user thread_id collision check: passing local-solo's thread_id never raises 500 or leaks it
        from astropilote.db import LOCAL_USER_ID, db_session
        from astropilote.logbook import append_chat_message

        with db_session(url) as db:
            alice_msg = append_chat_message(
                db, user_id=LOCAL_USER_ID, role="user", content="Private local chat"
            )
            cloud_user_id = client.get("/api/auth/me").json()["user"]["id"]
            bob_msg = append_chat_message(
                db,
                user_id=cloud_user_id,
                thread_id=alice_msg["thread_id"],
                role="user",
                content="Should allocate fresh thread",
            )
            assert bob_msg["thread_id"] != alice_msg["thread_id"]
    finally:
        app.dependency_overrides.pop(get_db, None)
        get_engine.cache_clear()
