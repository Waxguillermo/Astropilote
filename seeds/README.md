# Observatory Backend Seeds

Deterministic JSON seed fixtures used to initialize or reset the SQLite observatory backend (`src/astropilote/backend/store.py`) before interactive runs and state-based scenario evaluations.

## Available Seeds

- [`observatory_demo.json`](observatory_demo.json): Default baseline seed (`2` Lyon telescopes `T-01`/`T-02`, `4` slots `S-01`..`S-04`, `1` paid reservation `BK-001` for `astro-club`).
- [`two_bookings.json`](two_bookings.json): Multi-booking & multi-site seed (`3` telescopes across Lyon and Fontainebleau, `5` slots across Tuesday/Wednesday/Friday, `2` confirmed reservations `BK-001` and `BK-002` for `alice` plus `BK-003` for `astro-club`).

## Usage

```bash
uv run python -m astropilote.backend.store --seed seeds/observatory_demo.json
```
