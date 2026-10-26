"""Language detection helpers and localized (FR/EN) deterministic refusal messages."""

from __future__ import annotations

_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
}

_MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "off_topic": (
            "I am AstroPilote, an astronomy and night-sky observation assistant, so I can only help "
            "with astronomy, celestial observation, optics, and astrophotography."
        ),
        "prompt_injection": (
            "I cannot fulfill instructions that attempt to override system safety policies "
            "or expose internal configuration."
        ),
        "generic_refusal": (
            "I cannot fulfill this request as it falls outside AstroPilote's domain or safety policy."
        ),
        "impossible_forecast": (
            "I cannot predict cloud cover or meteorological weather for {date} ({delta_days} days ahead), "
            "as reliable atmospheric forecasts only extend up to {max_days} days. However, if you name "
            "specific astronomical targets, I can compute their exact sky altitudes and twilight windows "
            "for any future date."
        ),
        "messier_out_of_range": (
            "I must abstain: Messier {num} does not exist "
            "(the Messier catalog only contains 110 objects, M1 to M110)."
        ),
        "caldwell_out_of_range": (
            "I must abstain: Caldwell {num} does not exist "
            "(the Caldwell catalog only contains 109 objects, C1 to C109)."
        ),
        "not_indexed": (
            "I must abstain: '{code}' is not indexed in the AstroPilote corpus (which covers Messier, "
            "Caldwell, the 200 brightest NGC deep-sky objects, famous astrophotography nebulae and "
            "galaxies, the 88 IAU constellations, and practical observing guides)."
        ),
        "no_documents": "I must abstain: no matching documents were found in the AstroPilote corpus.",
        "fictional_location": (
            "I cannot compute sky ephemeris or weather because '{place}' is a fictional or "
            "non-terrestrial location. Please provide a real Earth city, region, or GPS coordinates."
        ),
        "geocode_failed": (
            "I cannot compute a night plan because '{place}' could not be geocoded as a real Earth "
            "location. Please provide a valid city, region, or GPS coordinates."
        ),
        "unknown_place": "the requested place",
    },
    "fr": {
        "off_topic": (
            "Je suis AstroPilote, un assistant d'astronomie et d'observation du ciel nocturne : je ne "
            "peux donc répondre qu'aux questions d'astronomie, d'observation céleste, d'optique et "
            "d'astrophotographie."
        ),
        "prompt_injection": (
            "Je ne peux pas suivre des instructions visant à contourner les règles de sécurité du "
            "système ou à révéler sa configuration interne."
        ),
        "generic_refusal": (
            "Je ne peux pas répondre à cette demande : elle sort du domaine d'AstroPilote ou de sa "
            "politique de sécurité."
        ),
        "impossible_forecast": (
            "Je ne peux pas prévoir la couverture nuageuse ni la météo pour le {date} (dans {delta_days} "
            "jours) : les prévisions atmosphériques fiables ne dépassent pas {max_days} jours. En "
            "revanche, si vous nommez des cibles astronomiques précises, je peux calculer leurs hauteurs "
            "exactes dans le ciel et les fenêtres de nuit noire pour n'importe quelle date future."
        ),
        "messier_out_of_range": (
            "Je dois m'abstenir : Messier {num} n'existe pas "
            "(le catalogue Messier ne compte que 110 objets, de M1 à M110)."
        ),
        "caldwell_out_of_range": (
            "Je dois m'abstenir : Caldwell {num} n'existe pas "
            "(le catalogue Caldwell ne compte que 109 objets, de C1 à C109)."
        ),
        "not_indexed": (
            "Je dois m'abstenir : « {code} » n'est pas indexé dans le corpus AstroPilote (qui couvre les "
            "catalogues Messier et Caldwell, les 200 objets NGC les plus brillants, les nébuleuses et "
            "galaxies célèbres en astrophotographie, les 88 constellations de l'UAI et des guides "
            "pratiques d'observation)."
        ),
        "no_documents": (
            "Je dois m'abstenir : aucun document correspondant n'a été trouvé dans le corpus AstroPilote."
        ),
        "fictional_location": (
            "Je ne peux pas calculer d'éphémérides ni de météo car « {place} » est un lieu fictif ou "
            "extraterrestre. Indiquez une vraie ville, une région ou des coordonnées GPS terrestres."
        ),
        "geocode_failed": (
            "Je ne peux pas établir de plan de nuit car « {place} » n'a pas pu être géolocalisé comme un "
            "lieu terrestre réel. Indiquez une ville, une région ou des coordonnées GPS valides."
        ),
        "unknown_place": "le lieu demandé",
    },
}


def normalize_language(raw: object) -> str:
    """Coerce the router's language hint into a lowercase ISO 639-1 code (default `'en'`)."""
    code = str(raw or "").strip().lower()[:2]
    return code if len(code) == 2 and code.isalpha() else "en"


def language_name(code: str) -> str:
    """Return a human-readable language name for LLM prompt directives."""
    return _LANGUAGE_NAMES.get(code, f"the language with ISO 639-1 code '{code}'")


def refusal_message(key: str, language: str, **fmt: object) -> str:
    """Return the deterministic refusal `key` in `language`, falling back to English."""
    catalog = _MESSAGES.get(language, _MESSAGES["en"])
    return catalog[key].format(**fmt)
