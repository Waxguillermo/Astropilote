"""Unit tests for the AstroPilote ingestion, cleaning, and section-chunking pipeline."""

from astropilote.ingest.catalog import CatalogTarget, parse_dec_to_deg, parse_ra_to_deg
from astropilote.ingest.chunker import chunk_target_sections, estimate_tokens
from astropilote.ingest.cleaner import clean_wiki_text, split_into_sections


def test_parse_ra_and_dec() -> None:
    """Verify sexagesimal RA and Dec conversion to decimal degrees."""
    # M31: RA 00:42:44.35 -> ~10.68479 deg, Dec +41:16:08.6 -> ~41.26906 deg
    ra = parse_ra_to_deg("00:42:44.35")
    dec = parse_dec_to_deg("+41:16:08.6")
    assert ra is not None and abs(ra - 10.68479) < 1e-3
    assert dec is not None and abs(dec - 41.26906) < 1e-3

    # Southern target Dec -60:30:00 -> -60.5 deg
    south_dec = parse_dec_to_deg("-60:30:00")
    assert south_dec == -60.5


def test_cleaner_and_section_splitter() -> None:
    """Verify citation stripping, boilerplate filtering, and section splitting."""
    raw = (
        "The Andromeda Galaxy (M31)[1] is a barred spiral galaxy[citation needed] "
        "approximately 2.5 million light-years from Earth.\n\n"
        "== Observation ==\n"
        "In an 80 mm refractor, M31 shows a bright central core and an elongated disk[2]. "
        "Satellite galaxies M32 and M110 fit within the same wide field of view.\n\n"
        "== References ==\n"
        "1. Some reference that should be ignored because it is in STOP_SECTIONS.\n"
    )
    cleaned = clean_wiki_text(raw)
    assert "[1]" not in cleaned
    assert "[citation needed]" not in cleaned

    sections = split_into_sections(raw)
    titles = [title for title, _ in sections]
    assert titles == ["Overview", "Observation"]
    assert "References" not in titles


def test_section_aware_chunker_with_header() -> None:
    """Verify that chunks include the context header and respect max_chunk_tokens."""
    target = CatalogTarget(
        doc_id="M027",
        title="Dumbbell Nebula (Messier 27)",
        wiki_query="Messier 27",
        catalog="Messier",
        object_type="planetary_nebula",
        constellation="VUL",
        ra_deg=299.901,
        dec_deg=22.721,
        magnitude=7.4,
        size_major_arcmin=8.0,
        size_minor_arcmin=5.6,
        narrowband_friendly=True,
        aliases=["M027", "M27", "NGC6853", "Dumbbell Nebula"],
    )
    long_obs = " ".join(
        [
            f"Sentence {i}: The Dumbbell Nebula is bright in OIII and H-alpha narrowband filters."
            for i in range(45)
        ]
    )
    sections = [
        ("Overview", "Messier 27 is a planetary nebula in the constellation Vulpecula."),
        ("Astrophotography", long_obs),
    ]
    chunks = chunk_target_sections(target, sections, max_chunk_tokens=180, overlap_words=15)

    assert len(chunks) >= 3
    for ch in chunks:
        assert ch.doc_id == "M027"
        assert "[M027 | Dumbbell Nebula (Messier 27)" in ch.text
        assert "Narrowband: yes" in ch.text
        assert ch.metadata.narrowband_friendly is True
        assert estimate_tokens(ch.text) <= 220
