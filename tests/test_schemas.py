"""Unit tests for AstroPilote Pydantic schemas and configuration."""

from datetime import time

from astropilote.agent.schemas import AstroAnswer, Chunk, ChunkMetadata, PlanItem


def test_chunk_and_answer_schemas() -> None:
    """Verify that Chunk and AstroAnswer schemas validate properly."""
    meta = ChunkMetadata(
        doc_id="M031",
        catalog="Messier",
        object_type="galaxy",
        constellation="AND",
        ra_deg=10.6847,
        dec_deg=41.2687,
        magnitude=3.4,
        size_major_arcmin=178.0,
        size_minor_arcmin=63.0,
        narrowband_friendly=False,
        aliases=["M31", "NGC 224", "Andromeda Galaxy"],
    )
    chunk = Chunk(
        chunk_id="M031#observation-0",
        doc_id="M031",
        title="Andromeda Galaxy (Messier 31)",
        section="Observation",
        text="[M031 | Andromeda Galaxy] Visible to the naked eye under dark skies.",
        score=0.92,
        metadata=meta,
    )
    assert chunk.doc_id == "M031"

    item = PlanItem(
        object_id="M031",
        name="Andromeda Galaxy",
        start=time(23, 0),
        end=time(0, 30),
        altitude_deg=70.0,
        direction="E",
        difficulty="easy",
        imaging_notes="Broadband target after Moonset; requires ~480 mm focal length for full disk.",
        sources=["M031"],
    )
    answer = AstroAnswer(
        answer="Observe M031 at 23:00 when it reaches 70° altitude in the east [M031].",
        plan=[item],
        conditions="Astronomical darkness 20:40-06:29 | Moon 42% (sets 22:58) | Clouds 10%",
        sources=["M031"],
        abstained=False,
        verified=True,
    )
    assert answer.verified is True
    assert answer.plan[0].object_id == "M031"
