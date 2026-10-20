"""OpenNGC and IAU Constellation catalog loader and astrometric parser."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from astropilote.agent.schemas import CatalogName, ChunkMetadata

OPENNGC_CSV_URL = (
    "https://raw.githubusercontent.com/mattiaverga/OpenNGC/master/database_files/NGC.csv"
)
OPENNGC_ADDENDUM_URL = (
    "https://raw.githubusercontent.com/mattiaverga/OpenNGC/master/database_files/addendum.csv"
)

# Mapping from OpenNGC type codes to human-readable astro categories
OPENNGC_TYPE_MAP: dict[str, str] = {
    "G": "galaxy",
    "GPair": "galaxy_pair",
    "GTrpl": "galaxy_triplet",
    "GGroup": "galaxy_group",
    "PN": "planetary_nebula",
    "HII": "emission_nebula",
    "EmN": "emission_nebula",
    "Neb": "nebula",
    "RfN": "reflection_nebula",
    "Cl+N": "cluster_with_nebulosity",
    "GCl": "globular_cluster",
    "OCl": "open_cluster",
    "SNR": "supernova_remnant",
    "DrkN": "dark_nebula",
    "*": "star",
    "**": "double_star",
    "*Ass": "stellar_association",
}

NARROWBAND_TYPES: set[str] = {
    "planetary_nebula",
    "emission_nebula",
    "cluster_with_nebulosity",
    "supernova_remnant",
}

# Canonical Caldwell 1..109 mapping to OpenNGC Name ("NGCxxxx" / "ICxxxx")
CALDWELL_TO_OPENNGC: dict[int, tuple[str, str]] = {
    1: ("NGC0188", "Cluster NGC 188"),
    2: ("NGC0040", "Bow-Tie Nebula"),
    3: ("NGC4236", "Galaxy NGC 4236"),
    4: ("NGC7023", "Iris Nebula"),
    5: ("IC0342", "Hidden Galaxy IC 342"),
    6: ("NGC6543", "Cat's Eye Nebula"),
    7: ("NGC2403", "Galaxy NGC 2403"),
    8: ("NGC0559", "Cluster NGC 559"),
    9: ("Sh2-155", "Cave Nebula"),
    10: ("NGC0663", "Cluster NGC 663"),
    11: ("NGC7635", "Bubble Nebula"),
    12: ("NGC6946", "Fireworks Galaxy"),
    13: ("NGC0457", "Owl Cluster"),
    14: ("NGC0869", "Double Cluster"),
    15: ("NGC6826", "Blinking Planetary"),
    16: ("NGC7243", "Cluster NGC 7243"),
    17: ("NGC0147", "Dwarf Galaxy NGC 147"),
    18: ("NGC0185", "Dwarf Galaxy NGC 185"),
    19: ("IC5146", "Cocoon Nebula"),
    20: ("NGC7000", "North America Nebula"),
    21: ("NGC4449", "Box Galaxy NGC 4449"),
    22: ("NGC7662", "Blue Snowball Nebula"),
    23: ("NGC0891", "Silver Sliver Galaxy"),
    24: ("NGC1275", "Perseus A"),
    25: ("NGC2419", "Intergalactic Wanderer"),
    26: ("NGC4244", "Silver Needle Galaxy"),
    27: ("NGC6888", "Crescent Nebula"),
    28: ("NGC0752", "Cluster NGC 752"),
    29: ("NGC5005", "Galaxy NGC 5005"),
    30: ("NGC7331", "Deer Lick Galaxy"),
    31: ("IC0405", "Flaming Star Nebula"),
    32: ("NGC4631", "Whale Galaxy"),
    33: ("NGC6992", "Eastern Veil Nebula"),
    34: ("NGC6960", "Western Veil Nebula"),
    35: ("NGC4889", "Coma B Galaxy"),
    36: ("NGC4559", "Koi Fish Galaxy"),
    37: ("NGC6885", "Cluster NGC 6885"),
    38: ("NGC4565", "Needle Galaxy"),
    39: ("NGC2392", "Eskimo Nebula"),
    40: ("NGC3626", "Galaxy NGC 3626"),
    41: ("Mel025", "Hyades"),
    42: ("NGC7006", "Globular Cluster NGC 7006"),
    43: ("NGC7814", "Little Sombrero Galaxy"),
    44: ("NGC7479", "Superman Galaxy"),
    45: ("NGC5248", "Galaxy NGC 5248"),
    46: ("NGC2261", "Hubble's Variable Nebula"),
    47: ("NGC6934", "Globular Cluster NGC 6934"),
    48: ("NGC2775", "Caldwell 48 Galaxy"),
    49: ("NGC2237", "Rosette Nebula"),
    50: ("NGC2244", "Satellite Cluster"),
    51: ("IC1613", "Irregular Galaxy IC 1613"),
    52: ("NGC4697", "Elliptical Galaxy NGC 4697"),
    53: ("NGC3115", "Spindle Galaxy"),
    54: ("NGC2506", "Cluster NGC 2506"),
    55: ("NGC7009", "Saturn Nebula"),
    56: ("NGC0246", "Skull Nebula"),
    57: ("NGC6822", "Barnard's Galaxy"),
    58: ("NGC2360", "Caroline's Cluster"),
    59: ("NGC3242", "Ghost of Jupiter"),
    60: ("NGC4038", "Antennae Galaxies"),
    61: ("NGC4039", "Antennae Galaxies South"),
    62: ("NGC0247", "Claw Galaxy"),
    63: ("NGC7293", "Helix Nebula"),
    64: ("NGC2362", "Tau Canis Majoris Cluster"),
    65: ("NGC0253", "Sculptor Galaxy"),
    66: ("NGC5694", "Globular Cluster NGC 5694"),
    67: ("NGC1097", "Barred Spiral NGC 1097"),
    68: ("NGC6729", "R Coronae Australis Nebula"),
    69: ("NGC6302", "Bug Nebula"),
    70: ("NGC0300", "Southern Pinwheel Companion"),
    71: ("NGC2477", "Cluster NGC 2477"),
    72: ("NGC0055", "String of Pearls Galaxy"),
    73: ("NGC1851", "Globular Cluster NGC 1851"),
    74: ("NGC3132", "Eight-Burst Nebula"),
    75: ("NGC6124", "Cluster NGC 6124"),
    76: ("NGC6231", "Northern Jewel Box"),
    77: ("NGC5128", "Centaurus A"),
    78: ("NGC6541", "Globular Cluster NGC 6541"),
    79: ("NGC3201", "Globular Cluster NGC 3201"),
    80: ("NGC5139", "Omega Centauri"),
    81: ("NGC6352", "Globular Cluster NGC 6352"),
    82: ("NGC6193", "Cluster NGC 6193"),
    83: ("NGC4945", "Tweezers Galaxy"),
    84: ("NGC5286", "Globular Cluster NGC 5286"),
    85: ("IC2391", "Omicron Velorum Cluster"),
    86: ("NGC6397", "Globular Cluster NGC 6397"),
    87: ("NGC1261", "Globular Cluster NGC 1261"),
    88: ("NGC5823", "Cluster NGC 5823"),
    89: ("NGC6087", "S Normae Cluster"),
    90: ("NGC2867", "Planetary Nebula NGC 2867"),
    91: ("NGC3532", "Wishing Well Cluster"),
    92: ("NGC3372", "Carina Nebula"),
    93: ("NGC6752", "Great Peacock Globular"),
    94: ("NGC4755", "Jewel Box Cluster"),
    95: ("NGC6025", "Cluster NGC 6025"),
    96: ("NGC2516", "Southern Beehive Cluster"),
    97: ("NGC3766", "Pearl Cluster"),
    98: ("NGC4609", "Coalsack Cluster"),
    99: ("CoalSack", "Coalsack Nebula"),
    100: ("IC2944", "Running Chicken Nebula"),
    101: ("NGC6744", "Pavo Galaxy"),
    102: ("IC2602", "Southern Pleiades"),
    103: ("NGC2070", "Tarantula Nebula"),
    104: ("NGC0362", "Globular Cluster NGC 362"),
    105: ("NGC4833", "Globular Cluster NGC 4833"),
    106: ("NGC0104", "47 Tucanae"),
    107: ("NGC6101", "Globular Cluster NGC 6101"),
    108: ("NGC4372", "Globular Cluster NGC 4372"),
    109: ("NGC3195", "Planetary Nebula NGC 3195"),
}

# All 88 official IAU constellations: (3-letter code, Name, Wikipedia Title, approx RA deg, approx Dec deg)
IAU_CONSTELLATIONS: list[tuple[str, str, str, float, float]] = [
    ("AND", "Andromeda", "Andromeda (constellation)", 12.0, 38.0),
    ("ANT", "Antlia", "Antlia", 154.0, -32.0),
    ("APS", "Apus", "Apus", 242.0, -75.0),
    ("AQR", "Aquarius", "Aquarius (constellation)", 334.0, -11.0),
    ("AQL", "Aquila", "Aquila (constellation)", 295.0, 3.0),
    ("ARA", "Ara", "Ara (constellation)", 260.0, -56.0),
    ("ARI", "Aries", "Aries (constellation)", 39.0, 21.0),
    ("AUR", "Auriga", "Auriga (constellation)", 91.0, 42.0),
    ("BOO", "Bootes", "Boötes", 220.0, 31.0),
    ("CAE", "Caelum", "Caelum", 70.0, -38.0),
    ("CAM", "Camelopardalis", "Camelopardalis", 90.0, 69.0),
    ("CNC", "Cancer", "Cancer (constellation)", 130.0, 20.0),
    ("CVN", "Canes Venatici", "Canes Venatici", 196.0, 40.0),
    ("CMA", "Canis Major", "Canis Major", 102.0, -22.0),
    ("CMI", "Canis Minor", "Canis Minor", 115.0, 6.0),
    ("CAP", "Capricornus", "Capricornus", 315.0, -18.0),
    ("CAR", "Carina", "Carina (constellation)", 130.0, -63.0),
    ("CAS", "Cassiopeia", "Cassiopeia (constellation)", 18.0, 62.0),
    ("CEN", "Centaurus", "Centaurus", 196.0, -47.0),
    ("CEP", "Cepheus", "Cepheus (constellation)", 330.0, 71.0),
    ("CET", "Cetus", "Cetus", 25.0, -7.0),
    ("CHA", "Chamaeleon", "Chamaeleon", 160.0, -79.0),
    ("CIR", "Circinus", "Circinus", 223.0, -63.0),
    ("COL", "Columba", "Columba (constellation)", 88.0, -35.0),
    ("COM", "Coma Berenices", "Coma Berenices", 191.0, 23.0),
    ("CRA", "Corona Australis", "Corona Australis", 279.0, -41.0),
    ("CRB", "Corona Borealis", "Corona Borealis", 237.0, 33.0),
    ("CRV", "Corvus", "Corvus (constellation)", 186.0, -18.0),
    ("CRT", "Crater", "Crater (constellation)", 170.0, -16.0),
    ("CRU", "Crux", "Crux", 187.0, -60.0),
    ("CYG", "Cygnus", "Cygnus (constellation)", 309.0, 44.0),
    ("DEL", "Delphinus", "Delphinus", 310.0, 12.0),
    ("DOR", "Dorado", "Dorado", 78.0, -59.0),
    ("DRA", "Draco", "Draco (constellation)", 225.0, 65.0),
    ("EQU", "Equuleus", "Equuleus", 318.0, 8.0),
    ("ERI", "Eridanus", "Eridanus (constellation)", 50.0, -29.0),
    ("FOR", "Fornax", "Fornax", 42.0, -32.0),
    ("GEM", "Gemini", "Gemini (constellation)", 106.0, 22.0),
    ("GRU", "Grus", "Grus (constellation)", 337.0, -46.0),
    ("HER", "Hercules", "Hercules (constellation)", 258.0, 27.0),
    ("HOR", "Horologium", "Horologium (constellation)", 49.0, -53.0),
    ("HYA", "Hydra", "Hydra (constellation)", 150.0, -15.0),
    ("HYI", "Hydrus", "Hydrus", 35.0, -70.0),
    ("IND", "Indus", "Indus (constellation)", 315.0, -60.0),
    ("LAC", "Lacerta", "Lacerta", 337.0, 46.0),
    ("LEO", "Leo", "Leo (constellation)", 160.0, 13.0),
    ("LMI", "Leo Minor", "Leo Minor", 153.0, 32.0),
    ("LEP", "Lepus", "Lepus (constellation)", 84.0, -19.0),
    ("LIB", "Libra", "Libra (constellation)", 228.0, -15.0),
    ("LUP", "Lupus", "Lupus (constellation)", 230.0, -43.0),
    ("LYN", "Lynx", "Lynx (constellation)", 119.0, 47.0),
    ("LYR", "Lyra", "Lyra", 283.0, 37.0),
    ("MEN", "Mensa", "Mensa (constellation)", 82.0, -77.0),
    ("MIC", "Microscopium", "Microscopium", 315.0, -37.0),
    ("MON", "Monoceros", "Monoceros", 106.0, 0.0),
    ("MUS", "Musca", "Musca", 188.0, -70.0),
    ("NOR", "Norma", "Norma (constellation)", 241.0, -52.0),
    ("OCT", "Octans", "Octans", 330.0, -82.0),
    ("OPH", "Ophiuchus", "Ophiuchus", 260.0, -8.0),
    ("ORI", "Orion", "Orion (constellation)", 84.0, 3.0),
    ("PAV", "Pavo", "Pavo (constellation)", 294.0, -66.0),
    ("PEG", "Pegasus", "Pegasus (constellation)", 340.0, 19.0),
    ("PER", "Perseus", "Perseus (constellation)", 47.0, 45.0),
    ("PHE", "Phoenix", "Phoenix (constellation)", 14.0, -48.0),
    ("PIC", "Pictor", "Pictor", 85.0, -53.0),
    ("PSC", "Pisces", "Pisces (constellation)", 9.0, 14.0),
    ("PSA", "Piscis Austrinus", "Piscis Austrinus", 334.0, -31.0),
    ("PUP", "Puppis", "Puppis", 115.0, -31.0),
    ("PYX", "Pyxis", "Pyxis", 134.0, -27.0),
    ("RET", "Reticulum", "Reticulum", 59.0, -60.0),
    ("SGE", "Sagitta", "Sagitta", 295.0, 19.0),
    ("SGR", "Sagittarius", "Sagittarius (constellation)", 286.0, -25.0),
    ("SCO", "Scorpius", "Scorpius", 253.0, -27.0),
    ("SCL", "Sculptor", "Sculptor (constellation)", 7.0, -32.0),
    ("SCT", "Scutum", "Scutum (constellation)", 280.0, -10.0),
    ("SER", "Serpens", "Serpens", 254.0, 6.0),
    ("SEX", "Sextans", "Sextans", 154.0, -2.0),
    ("TAU", "Taurus", "Taurus (constellation)", 65.0, 15.0),
    ("TEL", "Telescopium", "Telescopium", 288.0, -51.0),
    ("TRI", "Triangulum", "Triangulum", 30.0, 31.0),
    ("TRA", "Triangulum Australe", "Triangulum Australe", 241.0, -65.0),
    ("TUC", "Tucana", "Tucana", 356.0, -66.0),
    ("UMA", "Ursa Major", "Ursa Major", 170.0, 56.0),
    ("UMI", "Ursa Minor", "Ursa Minor", 225.0, 78.0),
    ("VEL", "Vela", "Vela (constellation)", 144.0, -47.0),
    ("VIR", "Virgo", "Virgo (constellation)", 198.0, -4.0),
    ("VOL", "Volans", "Volans", 117.0, -69.0),
    ("VUL", "Vulpecula", "Vulpecula", 303.0, 24.0),
]


# ---------------------------------------------------------------------------------------------
# Corpus extension (v2): famous astrophotography targets + practical guides
# ---------------------------------------------------------------------------------------------

# Why this subset exists: the "Herschel" subset is chosen by visual magnitude, but OpenNGC gives
# no V-Mag for most diffuse nebulae, so iconic imaging targets (Horsehead, Heart, Pelican...)
# were silently excluded. Coordinates, sizes and constellations come from the OpenNGC row named
# in the second column; the object type is overridden when OpenNGC only says "Neb".
# (OpenNGC name, common name, Wikipedia title, object_type, narrowband_friendly)
ASTROPHOTO_TARGETS: list[tuple[str, str, str, str, bool]] = [
    # Emission / reflection nebulae (no visual magnitude in OpenNGC)
    ("IC0434", "Horsehead Nebula", "Horsehead Nebula", "emission_nebula", True),
    ("NGC2024", "Flame Nebula", "Flame Nebula", "emission_nebula", True),
    ("NGC1977", "Running Man Nebula", "NGC 1977", "reflection_nebula", False),
    ("NGC2174", "Monkey Head Nebula", "NGC 2174", "emission_nebula", True),
    ("IC0410", "Tadpoles Nebula", "IC 410", "emission_nebula", True),
    ("IC0417", "Spider Nebula", "IC 417", "emission_nebula", True),
    ("IC0443", "Jellyfish Nebula", "IC 443", "supernova_remnant", True),
    ("IC2177", "Seagull Nebula", "Seagull Nebula", "emission_nebula", True),
    ("NGC2359", "Thor's Helmet", "Thor's Helmet", "emission_nebula", True),
    ("IC1805", "Heart Nebula", "Heart Nebula", "emission_nebula", True),
    ("IC1848", "Soul Nebula", "Soul Nebula", "emission_nebula", True),
    ("NGC0281", "Pacman Nebula", "NGC 281", "emission_nebula", True),
    ("IC0063", "Ghost of Cassiopeia", "IC 63", "emission_nebula", True),
    ("NGC7822", "Question Mark Nebula", "NGC 7822", "emission_nebula", True),
    ("IC1396", "Elephant's Trunk Nebula", "IC 1396", "cluster_with_nebulosity", True),
    ("IC5070", "Pelican Nebula", "Pelican Nebula", "emission_nebula", True),
    ("NGC1491", "Fossil Footprint Nebula", "NGC 1491", "emission_nebula", True),
    ("NGC1333", "Embryo Nebula", "NGC 1333", "reflection_nebula", False),
    ("IC4592", "Blue Horsehead Nebula", "IC 4592", "reflection_nebula", False),
    ("IC4628", "Prawn Nebula", "IC 4628", "emission_nebula", True),
    ("NGC6188", "Fighting Dragons of Ara", "NGC 6188", "emission_nebula", True),
    # Galaxies just below the Herschel brightness cut-off but classic imaging targets
    ("NGC3628", "Hamburger Galaxy", "NGC 3628", "galaxy", False),
    ("NGC5907", "Splinter Galaxy", "NGC 5907", "galaxy", False),
    ("NGC4490", "Cocoon Galaxy", "NGC 4490", "galaxy", False),
    ("NGC2841", "Galaxy NGC 2841", "NGC 2841", "galaxy", False),
    ("NGC3184", "Galaxy NGC 3184", "NGC 3184", "galaxy", False),
    ("IC2574", "Coddington's Nebula", "IC 2574", "galaxy", False),
    ("NGC2158", "Open Cluster NGC 2158", "NGC 2158", "open_cluster", False),
]

# M102 has no "M" column in OpenNGC; it is most often identified with the Spindle Galaxy NGC 5866.
M102_OPENNGC_NAME = "NGC5866"

# Caldwell objects that have no usable OpenNGC row (Sharpless / Melotte / dark nebula entries).
# Approximate J2000 centres; double-check against the Wikipedia infobox if you edit them.
# c_num -> (constellation, ra_deg, dec_deg, magnitude, size_major_arcmin)
CALDWELL_MANUAL_ASTROMETRY: dict[int, tuple[str, float, float, float | None, float | None]] = {
    9: ("CEP", 344.2, 62.6, 7.7, 50.0),  # Cave Nebula (Sh2-155), RA 22h56.8m, Dec +62 37
    41: ("TAU", 66.7, 15.9, 0.5, 330.0),  # Hyades (Mel 25), RA 04h26.9m, Dec +15 52
    99: ("CRU", 192.5, -62.5, None, 420.0),  # Coalsack Nebula, RA 12h50m, Dec -62 30
}

# Kept out of the "Herschel" subset: single/double stars and rows with an unknown type
# (v1 contained 11 stars such as NGC 1990 because the subset is sorted by brightness).
HERSCHEL_EXCLUDED_TYPES: set[str] = {"star", "double_star", "deep_sky_object"}

# Practical guides (English Wikipedia). Topics deliberately avoid the out-of-scope domains the
# guard refuses (telescope collimation, sensor datasheets, exoplanets) so traps stay traps.
# (doc_id suffix, title, Wikipedia title, extra aliases)
OBSERVING_GUIDES: list[tuple[str, str, str, list[str]]] = [
    # Sky quality & observing conditions
    ("BORTLE", "Bortle scale", "Bortle scale", ["Bortle", "Bortle class"]),
    ("LIGHT_POLLUTION", "Light pollution", "Light pollution", ["skyglow"]),
    ("DARK_SKY", "Dark-sky preserves", "Dark-sky preserve", ["dark sky reserve", "dark-sky park"]),
    ("SEEING", "Astronomical seeing", "Astronomical seeing", ["seeing", "atmospheric turbulence"]),
    ("AIRMASS", "Air mass", "Air mass (astronomy)", ["airmass"]),
    ("TWILIGHT", "Twilight and astronomical darkness", "Twilight", ["astronomical twilight"]),
    ("LUNAR_PHASE", "Lunar phases", "Lunar phase", ["moon phase", "new moon", "full moon"]),
    # Visual observing technique
    ("DARK_ADAPTATION", "Dark adaptation of the eye", "Adaptation (eye)", ["dark adaptation", "night vision"]),
    ("AVERTED_VISION", "Averted vision", "Averted vision", []),
    ("STAR_HOPPING", "Star hopping", "Star hopping", []),
    ("LIMITING_MAGNITUDE", "Limiting magnitude", "Limiting magnitude", ["naked-eye limit"]),
    ("APPARENT_MAGNITUDE", "Apparent magnitude", "Apparent magnitude", ["magnitude scale"]),
    ("SURFACE_BRIGHTNESS", "Surface brightness", "Surface brightness", []),
    ("PLANISPHERE", "Planisphere (star wheel)", "Planisphere", ["star wheel", "star chart"]),
    ("MESSIER_MARATHON", "Messier marathon", "Messier marathon", []),
    ("HORIZONTAL_COORDS", "Altitude and azimuth", "Horizontal coordinate system", ["altitude", "azimuth"]),
    ("EQUATORIAL_COORDS", "Right ascension and declination", "Equatorial coordinate system", ["right ascension", "declination"]),
    # Equipment
    ("BINOCULARS", "Binoculars for astronomy", "Binoculars", ["10x50", "7x50"]),
    ("REFRACTOR", "Refracting telescope", "Refracting telescope", ["refractor", "apochromat"]),
    ("DOBSONIAN", "Dobsonian telescope", "Dobsonian telescope", ["dobsonian", "dob"]),
    ("EYEPIECE", "Eyepieces and magnification", "Eyepiece", ["magnification", "exit pupil"]),
    ("BARLOW", "Barlow lens", "Barlow lens", []),
    ("EQUATORIAL_MOUNT", "Equatorial mount", "Equatorial mount", ["polar alignment", "tracking mount"]),
    ("ALTAZIMUTH_MOUNT", "Altazimuth mount", "Altazimuth mount", ["alt-az mount"]),
    ("GOTO", "GoTo telescope mounts", "GoTo (telescopes)", ["goto mount"]),
    ("F_NUMBER", "Focal ratio (f-number)", "F-number", ["focal ratio", "f-ratio"]),
    # Astrophotography workflow
    ("ASTROPHOTOGRAPHY", "Astrophotography", "Astrophotography", ["astro imaging"]),
    ("DARK_FRAME", "Dark frames", "Dark-frame subtraction", ["dark frame", "hot pixels"]),
    ("FLAT_FIELD", "Flat frames", "Flat-field correction", ["flat frame", "flats", "vignetting"]),
    ("STAR_TRAILS", "Star trails", "Star trail", ["star trail"]),
    ("HYDROGEN_ALPHA", "H-alpha narrowband imaging", "H-alpha", ["h-alpha", "ha filter"]),
    ("OXYGEN_III", "OIII narrowband imaging", "Doubly ionized oxygen", ["oiii", "[o iii]"]),
]

# Long generic articles (Binoculars, Light pollution...) are cut to their first sections so
# they do not flood the index with off-topic history or health paragraphs.
MAX_GUIDE_SECTIONS = 6


@dataclass
class CatalogTarget:
    """A canonical astronomical object selected for the AstroPilote ~500-document corpus."""

    doc_id: str
    title: str
    wiki_query: str
    catalog: CatalogName
    object_type: str
    constellation: str
    ra_deg: float | None = None
    dec_deg: float | None = None
    magnitude: float | None = None
    surface_brightness: float | None = None
    size_major_arcmin: float | None = None
    size_minor_arcmin: float | None = None
    narrowband_friendly: bool = False
    aliases: list[str] = field(default_factory=list)

    def to_chunk_metadata(self) -> ChunkMetadata:
        """Convert this target into a Pydantic `ChunkMetadata` instance."""
        return ChunkMetadata(
            doc_id=self.doc_id,
            catalog=self.catalog,
            object_type=self.object_type,
            constellation=self.constellation,
            ra_deg=self.ra_deg,
            dec_deg=self.dec_deg,
            magnitude=self.magnitude,
            surface_brightness=self.surface_brightness,
            size_major_arcmin=self.size_major_arcmin,
            size_minor_arcmin=self.size_minor_arcmin,
            narrowband_friendly=self.narrowband_friendly,
            aliases=self.aliases,
        )


def parse_ra_to_deg(ra_str: str) -> float | None:
    """Convert OpenNGC sexagesimal RA ('HH:MM:SS.ss') to decimal degrees [0, 360)."""
    ra_str = ra_str.strip()
    if not ra_str:
        return None
    try:
        parts = [float(p) for p in ra_str.split(":")]
        hours = parts[0] + (parts[1] / 60.0 if len(parts) > 1 else 0.0) + (
            parts[2] / 3600.0 if len(parts) > 2 else 0.0
        )
        return round(hours * 15.0, 5)
    except ValueError:
        return None


def parse_dec_to_deg(dec_str: str) -> float | None:
    """Convert OpenNGC sexagesimal Dec ('±DD:MM:SS.s') to decimal degrees [-90, +90]."""
    dec_str = dec_str.strip()
    if not dec_str:
        return None
    try:
        sign = -1.0 if dec_str.startswith("-") else 1.0
        clean = dec_str.lstrip("+-")
        parts = [float(p) for p in clean.split(":")]
        deg = parts[0] + (parts[1] / 60.0 if len(parts) > 1 else 0.0) + (
            parts[2] / 3600.0 if len(parts) > 2 else 0.0
        )
        return round(sign * deg, 5)
    except ValueError:
        return None


def _parse_float(val: str) -> float | None:
    val = val.strip()
    if not val:
        return None
    try:
        return round(float(val), 2)
    except ValueError:
        return None


def download_openngc_csvs(raw_dir: Path) -> list[dict[str, str]]:
    """Download OpenNGC `NGC.csv` and `addendum.csv` (or read from local cache) and return rows."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    ngc_path = raw_dir / "NGC.csv"
    add_path = raw_dir / "addendum.csv"

    if not ngc_path.exists() or not add_path.exists():
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            if not ngc_path.exists():
                resp = client.get(OPENNGC_CSV_URL)
                resp.raise_for_status()
                ngc_path.write_text(resp.text, encoding="utf-8")
            if not add_path.exists():
                resp = client.get(OPENNGC_ADDENDUM_URL)
                resp.raise_for_status()
                add_path.write_text(resp.text, encoding="utf-8")

    rows: list[dict[str, str]] = []
    for path in (ngc_path, add_path):
        reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")), delimiter=";")
        rows.extend(reader)
    return rows


def build_curated_catalog(
    raw_dir: Path,
    herschel_limit: int = 200,
    include_astrophoto: bool = True,
    include_guides: bool = True,
) -> list[CatalogTarget]:
    """Select the canonical corpus targets.

    v1 (506 docs): Messier + Caldwell + 200 bright NGC ("Herschel") + 88 IAU constellations.
    v2 adds M102, the `ASTROPHOTO_TARGETS` subset and the `OBSERVING_GUIDES` subset, removes
    stars from the bright-NGC subset and fills the 3 Caldwell objects that had no coordinates.
    Set `include_astrophoto` / `include_guides` to False to rebuild the v1-style corpus for ablations.
    """
    rows = download_openngc_csvs(raw_dir)
    rows_by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        name = (row.get("Name") or "").strip()
        if name:
            rows_by_name[name] = row

    caldwell_openngc_names = {ngc_name for ngc_name, _ in CALDWELL_TO_OPENNGC.values()}
    # Objects handled by a dedicated subset must never also appear in "Herschel" (no duplicates)
    reserved_names = caldwell_openngc_names | {name for name, *_ in ASTROPHOTO_TARGETS} | {
        M102_OPENNGC_NAME
    }
    messier_by_num: dict[int, CatalogTarget] = {}
    herschel_candidates: list[CatalogTarget] = []

    for row in rows:
        name = (row.get("Name") or "").strip()
        obj_type_code = (row.get("Type") or "").strip()
        if obj_type_code in ("NonEx", "Dup", ""):
            continue

        obj_type = OPENNGC_TYPE_MAP.get(obj_type_code, "deep_sky_object")
        raw_const = (row.get("Const") or "UNK").strip().upper()
        const = "SER" if raw_const in ("SE1", "SE2") else raw_const
        ra_deg = parse_ra_to_deg(row.get("RA", ""))
        dec_deg = parse_dec_to_deg(row.get("Dec", ""))
        vmag = _parse_float(row.get("V-Mag", "")) or _parse_float(row.get("B-Mag", ""))
        surf_br = _parse_float(row.get("SurfBr", ""))
        maj_ax = _parse_float(row.get("MajAx", ""))
        min_ax = _parse_float(row.get("MinAx", ""))
        common = (row.get("Common names") or "").split(",")[0].strip()
        messier_raw = (row.get("M") or "").strip()

        if messier_raw.isdigit():
            m_num = int(messier_raw)
            if 1 <= m_num <= 110 and m_num not in messier_by_num:
                doc_id = f"M{m_num:03d}"
                m_short = f"M{m_num}"
                aliases = [doc_id, m_short, name] + ([common] if common else [])
                title = f"{common} (Messier {m_num})" if common else f"Messier {m_num} ({name})"
                messier_by_num[m_num] = CatalogTarget(
                    doc_id=doc_id,
                    title=title,
                    wiki_query=f"Messier {m_num}",
                    catalog="Messier",
                    object_type=obj_type,
                    constellation=const,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    magnitude=vmag,
                    surface_brightness=surf_br,
                    size_major_arcmin=maj_ax,
                    size_minor_arcmin=min_ax,
                    narrowband_friendly=obj_type in NARROWBAND_TYPES,
                    aliases=list(dict.fromkeys(a for a in aliases if a)),
                )
                continue

        # Candidate for Herschel / bright NGC subset (excluding Messier and Caldwell objects)
        if (
            name.startswith("NGC")
            and name not in reserved_names
            and obj_type not in HERSCHEL_EXCLUDED_TYPES
            and vmag is not None
            and vmag <= 11.5
            and ra_deg is not None
        ):
            ngc_num = name[3:].lstrip("0") or "0"
            title = f"{common} (NGC {ngc_num})" if common else f"NGC {ngc_num}"
            herschel_candidates.append(
                CatalogTarget(
                    doc_id=name,
                    title=title,
                    wiki_query=f"NGC {ngc_num}",
                    catalog="Herschel",
                    object_type=obj_type,
                    constellation=const,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    magnitude=vmag,
                    surface_brightness=surf_br,
                    size_major_arcmin=maj_ax,
                    size_minor_arcmin=min_ax,
                    narrowband_friendly=obj_type in NARROWBAND_TYPES,
                    aliases=list(
                        dict.fromkeys([name, f"NGC {ngc_num}"] + ([common] if common else []))
                    ),
                )
            )

    # Ensure M024 (Small Sagittarius Star Cloud) and M045 (Pleiades) are present if not in NGC rows
    if 24 not in messier_by_num:
        messier_by_num[24] = CatalogTarget(
            doc_id="M024",
            title="Sagittarius Star Cloud (Messier 24)",
            wiki_query="Messier 24",
            catalog="Messier",
            object_type="stellar_association",
            constellation="SGR",
            ra_deg=274.208,
            dec_deg=-18.55,
            magnitude=4.6,
            size_major_arcmin=90.0,
            size_minor_arcmin=35.0,
            narrowband_friendly=False,
            aliases=["M024", "M24", "IC4715", "Sagittarius Star Cloud"],
        )
    if 45 not in messier_by_num:
        messier_by_num[45] = CatalogTarget(
            doc_id="M045",
            title="Pleiades (Messier 45)",
            wiki_query="Pleiades",
            catalog="Messier",
            object_type="open_cluster",
            constellation="TAU",
            ra_deg=56.75,
            dec_deg=24.1167,
            magnitude=1.6,
            size_major_arcmin=110.0,
            size_minor_arcmin=110.0,
            narrowband_friendly=False,
            aliases=["M045", "M45", "Mel022", "Pleiades", "Seven Sisters"],
        )

    if 102 not in messier_by_num and M102_OPENNGC_NAME in rows_by_name:
        m102_row = rows_by_name[M102_OPENNGC_NAME]
        messier_by_num[102] = CatalogTarget(
            doc_id="M102",
            title="Spindle Galaxy (Messier 102)",
            wiki_query="NGC 5866",
            catalog="Messier",
            object_type="galaxy",
            constellation=(m102_row.get("Const") or "DRA").strip().upper(),
            ra_deg=parse_ra_to_deg(m102_row.get("RA", "")),
            dec_deg=parse_dec_to_deg(m102_row.get("Dec", "")),
            magnitude=_parse_float(m102_row.get("V-Mag", "")),
            surface_brightness=_parse_float(m102_row.get("SurfBr", "")),
            size_major_arcmin=_parse_float(m102_row.get("MajAx", "")),
            size_minor_arcmin=_parse_float(m102_row.get("MinAx", "")),
            narrowband_friendly=False,
            aliases=["M102", "Messier 102", "NGC5866", "NGC 5866", "Spindle Galaxy"],
        )

    messier_list = [messier_by_num[k] for k in sorted(messier_by_num)]

    # Build the 109 Caldwell objects (C001 .. C109) enriched with OpenNGC coordinates & sizes
    caldwell_list: list[CatalogTarget] = []
    for c_num in range(1, 110):
        doc_id = f"C{c_num:03d}"
        ngc_key, fallback_common = CALDWELL_TO_OPENNGC[c_num]
        row = rows_by_name.get(ngc_key, {})

        obj_type_code = (row.get("Type") or "").strip()
        obj_type = OPENNGC_TYPE_MAP.get(obj_type_code, "deep_sky_object")
        const = (row.get("Const") or "VAR").strip().upper()
        ra_deg = parse_ra_to_deg(row.get("RA", ""))
        dec_deg = parse_dec_to_deg(row.get("Dec", ""))
        vmag = _parse_float(row.get("V-Mag", "")) or _parse_float(row.get("B-Mag", ""))
        surf_br = _parse_float(row.get("SurfBr", ""))
        maj_ax = _parse_float(row.get("MajAx", ""))
        min_ax = _parse_float(row.get("MinAx", ""))
        common = (row.get("Common names") or "").split(",")[0].strip() or fallback_common
        if c_num in CALDWELL_MANUAL_ASTROMETRY and ra_deg is None:
            const, ra_deg, dec_deg, manual_mag, manual_size = CALDWELL_MANUAL_ASTROMETRY[c_num]
            vmag = vmag if vmag is not None else manual_mag
            maj_ax = maj_ax if maj_ax is not None else manual_size

        formatted_ngc = (
            f"NGC {ngc_key[3:].lstrip('0')}"
            if ngc_key.startswith("NGC")
            else (f"IC {ngc_key[2:].lstrip('0')}" if ngc_key.startswith("IC") else fallback_common)
        )
        wiki_query = formatted_ngc if ngc_key.startswith(("NGC", "IC")) else fallback_common
        title = f"{common} (Caldwell {c_num}, {formatted_ngc})"
        is_nb = obj_type in NARROWBAND_TYPES or c_num in {9, 11, 19, 20, 27, 31, 33, 34, 39, 49, 63, 92, 100, 103}

        caldwell_list.append(
            CatalogTarget(
                doc_id=doc_id,
                title=title,
                wiki_query=wiki_query,
                catalog="Caldwell",
                object_type=obj_type,
                constellation=const,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                magnitude=vmag,
                surface_brightness=surf_br,
                size_major_arcmin=maj_ax,
                size_minor_arcmin=min_ax,
                narrowband_friendly=is_nb,
                aliases=list(
                    dict.fromkeys([doc_id, f"C{c_num}", f"Caldwell {c_num}", ngc_key, formatted_ngc, common])
                ),
            )
        )

    # Sort Herschel candidates by brightness (lowest visual magnitude first) and take top N
    herschel_candidates.sort(
        key=lambda t: (t.magnitude if t.magnitude is not None else 99.0, t.doc_id)
    )
    herschel_list = herschel_candidates[:herschel_limit]

    # Add all 88 IAU constellations
    constellation_list: list[CatalogTarget] = [
        CatalogTarget(
            doc_id=f"CON_{code}",
            title=f"{cname} (Constellation)",
            wiki_query=wtitle,
            catalog="Constellation",
            object_type="constellation",
            constellation=code,
            ra_deg=ra,
            dec_deg=dec,
            narrowband_friendly=False,
            aliases=[f"CON_{code}", code, cname],
        )
        for code, cname, wtitle, ra, dec in IAU_CONSTELLATIONS
    ]

    astrophoto_list = build_astrophoto_targets(rows_by_name) if include_astrophoto else []
    guide_list = build_guide_targets() if include_guides else []

    return (
        messier_list
        + caldwell_list
        + herschel_list
        + astrophoto_list
        + constellation_list
        + guide_list
    )


def build_astrophoto_targets(rows_by_name: dict[str, dict[str, str]]) -> list[CatalogTarget]:
    """Build the `Astrophoto` subset from `ASTROPHOTO_TARGETS`, using OpenNGC rows for astrometry."""
    targets: list[CatalogTarget] = []
    for name, common, wiki_title, obj_type, is_nb in ASTROPHOTO_TARGETS:
        row = rows_by_name.get(name)
        if row is None:
            print(f"      [warn] {name} not found in OpenNGC, skipped.")
            continue
        prefix = "IC" if name.startswith("IC") else "NGC"
        number = name[len(prefix):].lstrip("0") or "0"
        formatted = f"{prefix} {number}"
        targets.append(
            CatalogTarget(
                doc_id=name,
                title=f"{common} ({formatted})",
                wiki_query=wiki_title,
                catalog="Astrophoto",
                object_type=obj_type,
                constellation=(row.get("Const") or "UNK").strip().upper(),
                ra_deg=parse_ra_to_deg(row.get("RA", "")),
                dec_deg=parse_dec_to_deg(row.get("Dec", "")),
                magnitude=_parse_float(row.get("V-Mag", "")),
                surface_brightness=_parse_float(row.get("SurfBr", "")),
                size_major_arcmin=_parse_float(row.get("MajAx", "")),
                size_minor_arcmin=_parse_float(row.get("MinAx", "")),
                narrowband_friendly=is_nb,
                aliases=list(dict.fromkeys([name, formatted, common])),
            )
        )
    return targets


def build_guide_targets() -> list[CatalogTarget]:
    """Build the `Guide` subset (practical observing & astrophotography guides, no coordinates)."""
    return [
        CatalogTarget(
            doc_id=f"GUIDE_{suffix}",
            title=f"{title} (Observing guide)",
            wiki_query=wiki_title,
            catalog="Guide",
            object_type="guide",
            constellation="",
            aliases=list(dict.fromkeys([f"GUIDE_{suffix}", title, *extra])),
        )
        for suffix, title, wiki_title, extra in OBSERVING_GUIDES
    ]
