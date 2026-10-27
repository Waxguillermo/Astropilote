"""Pydantic schemas for chunks, user intent, structured observation plans, and evaluation logs."""

from datetime import date, time
from typing import Literal

from pydantic import BaseModel, Field

# Corpus subsets. "Astrophoto" = famous imaging targets missing from the brightness-based
# selection (mostly nebulae without a visual magnitude); "Guide" = practical observing and
# astrophotography guides (no sky coordinates).
CatalogName = Literal["Messier", "Caldwell", "Herschel", "Constellation", "Astrophoto", "Guide"]
CATALOG_NAMES: tuple[str, ...] = (
    "Messier",
    "Caldwell",
    "Herschel",
    "Constellation",
    "Astrophoto",
    "Guide",
)

SensorFormat = Literal["APS-C", "full-frame", "micro-4/3", "1-inch", "smartphone"]
MountType = Literal["equatorial_goto", "star_tracker", "altaz_goto", "tripod"]


class EquipmentProfile(BaseModel):
    """User astrophotography & optical equipment profile persisted in the UI and passed to the agent."""

    enabled: bool = Field(
        default=True,
        description="Whether this equipment profile is active for tailoring responses",
    )
    preset_id: str | None = Field(
        default=None,
        description="Identifier of the selected equipment preset if any",
    )
    rig_name: str | None = Field(
        default=None,
        description="Short human-readable name of the user's setup, e.g. 'Lunette APO 80ED + APS-C'",
    )
    telescope_or_lens: str | None = Field(
        default=None,
        description="Telescope or camera lens model, e.g. 'Lunette APO 80/480mm', 'Objectif 135mm f/2'",
    )
    focal_length_mm: float = Field(
        default=480.0,
        gt=0.0,
        description="Optical focal length in millimeters",
    )
    aperture_mm: float = Field(
        default=80.0,
        gt=0.0,
        description="Optical aperture diameter in millimeters",
    )
    camera_model: str | None = Field(
        default=None,
        description="Camera body or dedicated cooled astro camera name, e.g. 'APN APS-C défiltré', 'ZWO ASI2600MC'",
    )
    sensor_format: SensorFormat = Field(
        default="APS-C",
        description="Sensor size category ('APS-C', 'full-frame', 'micro-4/3', '1-inch', 'smartphone')",
    )
    mount_type: MountType = Field(
        default="equatorial_goto",
        description="Tracking mount category ('equatorial_goto', 'star_tracker', 'altaz_goto', 'tripod')",
    )
    has_autoguiding: bool = Field(
        default=False,
        description="True when the setup includes an autoguider scope/OAG for multi-minute sub-exposures",
    )
    filters: list[str] = Field(
        default_factory=list,
        description="Available optical filters, e.g. ['broadband', 'dual_band', 'uhc_cls', 'narrowband_sho']",
    )
    bortle_class: int | None = Field(
        default=None,
        ge=1,
        le=9,
        description="User's typical sky light-pollution rating on the Bortle scale (1..9)",
    )


class ChunkMetadata(BaseModel):
    """Structured astrometric & photometric metadata attached to every indexed chunk."""

    doc_id: str = Field(
        description="Canonical ID, e.g. 'M031', 'C014', 'NGC7000', 'IC0434', 'CON_ORI', 'GUIDE_BORTLE'"
    )
    catalog: CatalogName
    object_type: str = Field(
        description="Object category, e.g. 'galaxy', 'emission_nebula', 'globular_cluster'"
    )
    constellation: str = Field(description="3-letter IAU constellation code, e.g. 'AND', 'HER'")
    ra_deg: float | None = Field(default=None, description="Right Ascension J2000 in degrees")
    dec_deg: float | None = Field(default=None, description="Declination J2000 in degrees")
    magnitude: float | None = Field(default=None, description="Apparent visual magnitude")
    surface_brightness: float | None = Field(
        default=None, description="Surface brightness in mag/arcsec^2 for astrophotography"
    )
    size_major_arcmin: float | None = Field(
        default=None, description="Major angular size in arcminutes for FOV framing"
    )
    size_minor_arcmin: float | None = Field(
        default=None, description="Minor angular size in arcminutes"
    )
    narrowband_friendly: bool = Field(
        default=False,
        description="True for emission/planetary nebulae suitable for Ha/OIII narrowband imaging",
    )
    aliases: list[str] = Field(default_factory=list, description="Alternative designations")


class Chunk(BaseModel):
    """A single self-contained section passage retrieved from the hybrid index."""

    chunk_id: str = Field(description="Unique chunk ID, e.g. 'M031#observation-0'")
    doc_id: str = Field(description="Parent document canonical ID, e.g. 'M031'")
    title: str = Field(description="Document title, e.g. 'Andromeda Galaxy (Messier 31)'")
    section: str = Field(description="Section heading, e.g. 'Observation' or 'Astrophotography'")
    text: str = Field(description="Header-prefixed passage text")
    score: float = Field(default=0.0, description="Reranker or RRF relevance score")
    metadata: ChunkMetadata


class Intent(BaseModel):
    """Structured output of the `guard` node produced by the LLM Router and catalog/date invariants."""

    request_type: Literal[
        "factual",
        "session_plan",
        "off_topic",
        "impossible_forecast",
        "prompt_injection",
    ] = Field(description="Classification of the user query")
    language: str = Field(
        default="en",
        description="ISO 639-1 code of the language the user wrote in; the answer and refusals use it",
    )
    mode: Literal["visual", "astrophotography", "hybrid"] = Field(
        default="hybrid",
        description="Whether the user asks for visual observing, astrophotography imaging, or both",
    )
    is_fictional_location: bool = Field(
        default=False,
        description="True when the user asks to observe from a fictional, mythical, or extraterrestrial place",
    )
    location_name: str | None = Field(
        default=None,
        description="Explicit geographic place or coordinates mentioned in the query, or None",
    )
    target_date: date | None = Field(
        default=None, description="Extracted observation start date if any"
    )
    forecast_days: int = Field(
        default=1,
        ge=1,
        le=14,
        description="Number of consecutive nights to evaluate (1 for single night, 2..14 for multi-night planning)",
    )
    weather_only: bool = Field(
        default=False,
        description="True when the query exclusively asks about meteorological conditions without deep-sky targets",
    )
    mentioned_targets: list[str] = Field(
        default_factory=list,
        description="Specific object names or catalog designations mentioned in the query",
    )
    include_practical_guides: bool = Field(
        default=False,
        description="True when the query touches equipment, technique, sky quality, or observing methodology",
    )
    search_query: str | None = Field(
        default=None,
        description="Focused English query for hybrid catalog/guide retrieval",
    )
    constellation: str | None = Field(
        default=None,
        description="3-letter IAU constellation code filter (e.g. 'CYG', 'ORI') when restricting targets to a constellation",
    )
    catalog_filter: CatalogName | None = Field(
        default=None,
        description="Catalog subset filter when restricting targets to a specific catalog",
    )
    object_type: str | None = Field(
        default=None,
        description="Object category filter (e.g. 'galaxy', 'emission_nebula', 'globular_cluster', 'open_cluster', 'planetary_nebula')",
    )
    max_magnitude: float | None = Field(
        default=None,
        description="Upper visual magnitude threshold when requesting bright/binocular targets or an explicit magnitude limit",
    )
    narrowband_only: bool = Field(
        default=False,
        description="True when specifically requesting narrowband (Ha/OIII/SII) emission/planetary nebulae",
    )
    aperture_mm: float | None = Field(
        default=None, description="Telescope/lens aperture diameter in mm (e.g. 80.0)"
    )
    focal_length_mm: float | None = Field(
        default=None, description="Optical focal length in mm for astrophotography FOV calculations"
    )
    sensor_format: SensorFormat = Field(
        default="APS-C",
        description="Camera sensor format ('APS-C', 'full-frame', 'micro-4/3', '1-inch', 'smartphone') for FOV framing",
    )
    mount_type: MountType = Field(
        default="equatorial_goto",
        description="Tracking mount type ('equatorial_goto', 'star_tracker', 'altaz_goto', 'tripod')",
    )
    has_autoguiding: bool = Field(
        default=False,
        description="True when autoguiding is active for long sub-exposures",
    )
    filters: list[str] = Field(
        default_factory=list,
        description="Available optical filters from the user profile or query",
    )
    rig_summary: str | None = Field(
        default=None,
        description="Human-readable description of the user's configured photographic setup",
    )
    refusal_reason: str | None = Field(
        default=None,
        description="Polite explanation if request_type is off_topic/impossible_forecast/prompt_injection",
    )


class PlanItem(BaseModel):
    """A single scheduled target in an observation or astrophotography session plan."""

    object_id: str = Field(description="Canonical object ID, e.g. 'M031'")
    name: str = Field(description="Common name, e.g. 'Andromeda Galaxy'")
    start: time = Field(description="Recommended start time (local or UTC)")
    end: time = Field(description="Recommended end time")
    altitude_deg: float = Field(
        ge=0.0, le=90.0, description="Calculated target altitude in degrees during the slot"
    )
    direction: str = Field(default="", description="Cardinal direction, e.g. 'E', 'SW', 'NE'")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="Observing or imaging difficulty for the user's instrument"
    )
    imaging_notes: str | None = Field(
        default=None,
        description="Visual or astrophotography tip (eyepiece magnification, FOV framing, Ha/OIII filter)",
    )
    sources: list[str] = Field(description="Cited document IDs supporting this target")


class AstroAnswer(BaseModel):
    """Validated final response returned by the `compose` and `validate` nodes."""

    answer: str = Field(
        description="Natural-language response in the user's language with inline citations"
    )
    plan: list[PlanItem] = Field(
        default_factory=list,
        description="Ordered targets for session planning queries (empty for pure factual questions)",
    )
    conditions: str | None = Field(
        default=None,
        description="Summary of astronomical darkness window, Moon illumination/rise/set, and cloud cover",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Deduplicated list of cited canonical document IDs (must be a subset of retrieved_ids)",
    )
    abstained: bool = Field(
        default=False,
        description="True when the corpus does not contain the answer or the query is refused",
    )
    verified: bool = Field(
        default=True,
        description="Set by the deterministic Python `validate` node after checking citations & altitudes",
    )
    verification_errors: list[str] = Field(
        default_factory=list,
        description="List of any citation, altitude, or darkness window violations caught by `validate`",
    )


class ToolCallRecord(BaseModel):
    """Telemetry record for a single tool invocation."""

    name: str
    arguments: dict = Field(default_factory=dict)
    duration_s: float


class EvalResultRow(BaseModel):
    """Schema for each line written to `eval/results.jsonl` by `eval/run_eval.py`."""

    id: str
    answer: str
    cited_sources: list[str]
    retrieved_ids: list[str]
    tokens_in: int
    tokens_out: int
    latency_s: float
    tool_calls: list[ToolCallRecord]
    stop_reason: Literal["answered", "abstained", "refused", "max_steps", "error"]
    cost_usd: float = 0.0
