"""Tests for the corpus v2 extension: Astrophoto targets, practical guides and catalog fixes."""

from collections import Counter

from astropilote.agent.graph import run_agent
from astropilote.agent.schemas import AstroAnswer, Chunk, ChunkMetadata
from astropilote.agent.validator import verify_astro_answer
from astropilote.config import settings
from astropilote.ingest.catalog import (
    ASTROPHOTO_TARGETS,
    OBSERVING_GUIDES,
    build_curated_catalog,
)
from astropilote.ingest.chunker import chunk_target_sections
from astropilote.retrieval.hybrid import MetadataFilter


def _catalog():
    return build_curated_catalog(settings.data_dir / "raw")


def test_catalog_v2_composition_and_no_duplicates() -> None:
    """All subsets are present, every doc_id is unique, and M102 completes the Messier list."""
    targets = _catalog()
    ids = [t.doc_id for t in targets]
    assert len(ids) == len(set(ids)), "duplicate doc_id in the corpus"

    counts = Counter(t.catalog for t in targets)
    assert counts["Messier"] == 110
    assert counts["Caldwell"] == 109
    assert counts["Constellation"] == 88
    assert counts["Astrophoto"] == len(ASTROPHOTO_TARGETS)
    assert counts["Guide"] == len(OBSERVING_GUIDES)
    assert "M102" in ids


def test_every_sky_object_has_coordinates() -> None:
    """Visibility needs RA/Dec: only guides may lack coordinates (fixes C009, C041, C099)."""
    missing = [t.doc_id for t in _catalog() if t.catalog != "Guide" and t.ra_deg is None]
    assert missing == []


def test_herschel_subset_contains_no_stars() -> None:
    """The bright-NGC subset must not contain single/double stars (v1 had 11, e.g. NGC1990)."""
    herschel = [t for t in _catalog() if t.catalog == "Herschel"]
    assert len(herschel) == 200
    assert not [t.doc_id for t in herschel if t.object_type in {"star", "double_star"}]


def test_v1_flags_rebuild_the_original_subsets() -> None:
    """`include_astrophoto=False, include_guides=False` gives a v1-style corpus for ablations."""
    v1 = build_curated_catalog(
        settings.data_dir / "raw", include_astrophoto=False, include_guides=False
    )
    assert {t.catalog for t in v1} == {"Messier", "Caldwell", "Herschel", "Constellation"}


def test_guide_chunks_have_a_guide_header() -> None:
    """Guide chunks carry identity + aliases, never fake sky fields like constellation or magnitude."""
    guide = next(t for t in _catalog() if t.doc_id == "GUIDE_BORTLE")
    chunks = chunk_target_sections(
        guide, [("Overview", "The Bortle scale measures the brightness of the night sky.")]
    )
    assert chunks[0].text.startswith("[GUIDE_BORTLE | Bortle scale (Observing guide)")
    assert "Constellation:" not in chunks[0].text
    assert chunks[0].metadata.catalog == "Guide"


def test_metadata_filter_can_exclude_guides() -> None:
    """Object searches exclude guides; the dedicated guide search keeps only guides."""
    guide_meta = ChunkMetadata(doc_id="GUIDE_BORTLE", catalog="Guide", object_type="guide", constellation="")
    obj_meta = ChunkMetadata(doc_id="IC0434", catalog="Astrophoto", object_type="emission_nebula", constellation="ORI")
    assert not MetadataFilter(exclude_catalog="Guide").matches(guide_meta)
    assert MetadataFilter(exclude_catalog="Guide").matches(obj_meta)
    assert MetadataFilter(catalog="Guide").matches(guide_meta)
    assert not MetadataFilter(catalog="Guide").matches(obj_meta)


def test_validator_checks_guide_citations() -> None:
    """An inline [GUIDE_*] citation must have been retrieved, like any other source."""
    meta = ChunkMetadata(doc_id="GUIDE_BORTLE", catalog="Guide", object_type="guide", constellation="")
    chunk = Chunk(chunk_id="GUIDE_BORTLE#overview-0", doc_id="GUIDE_BORTLE", title="Bortle scale",
                  section="Overview", text="...", metadata=meta)
    state = {"retrieved": [chunk], "calculated_altitudes": {}, "night_window": None}
    ok = AstroAnswer(answer="City skies are class 8-9 [GUIDE_BORTLE].", sources=["GUIDE_BORTLE"])
    bad = AstroAnswer(answer="Use dark frames [GUIDE_DARK_FRAME].", sources=["GUIDE_BORTLE"])
    assert verify_astro_answer(ok, state).verified is True  # type: ignore[arg-type]
    assert verify_astro_answer(bad, state).verified is False  # type: ignore[arg-type]


def test_new_ic_object_is_answered_and_unknown_ic_still_abstains() -> None:
    """IC 434 is now in the corpus; IC 2118 (not indexed) must still trigger an abstention."""
    answer, _, _ = run_agent("What is IC 434 and in which constellation is it?", question_id="t1")
    assert answer.abstained is False
    assert "IC0434" in answer.sources

    trap, _, row = run_agent("What is the magnitude of the Witch Head Nebula IC 2118?", question_id="t2")
    assert trap.abstained is True
    assert trap.sources == []
    assert row.stop_reason == "abstained"
