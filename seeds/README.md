# Observatoire simulé — phase 1 Agentic Workflows

Cette phase ajoute un backend **SQLite persistant et réinitialisable**, indépendant du LLM, sans paiement ni télescope réel. Le fichier `observatory_demo.json` contient deux télescopes, quatre créneaux, et une réservation payée préexistante (paiement simulé).

## Initialiser et inspecter

Depuis la racine du projet :

```bash
uv run python -m astropilote.backend.store --seed seeds/observatory_demo.json
```

Base locale créée sous `data/backend/observatory.sqlite3` (exclue de Git).

## Tester une réservation puis vérifier l'état réel

```bash
uv run python - <<'PY'
from astropilote.backend.store import ObservatoryStore
store = ObservatoryStore('data/backend/observatory.sqlite3')
print('Disponibles avant :', [s['id'] for s in store.list_slots(available_only=True)])
print('Réservation :', store.create_booking('S-01', 'observer-1'))
print('Disponibles après :', [s['id'] for s in store.list_slots(available_only=True)])
PY
```

La commande de reset remet l'état initial. Une seconde réservation sur le même créneau est refusée avec `SLOT_UNAVAILABLE` ; les erreurs exposent un `code` et des `details` structurés.

## Action dangereuse / limites de sécurité

`cancel_booking('BK-001', 'astro-club')` annule définitivement (dans cette simulation) une réservation payée et inscrit `refunded=True` pour **simuler** un remboursement. Aucun paiement réel n'est fait. **L'approbation humaine n'est pas encore implémentée : ne pas exposer cette écriture à un agent autonome avant l'étape LangGraph `interrupt()` + checkpointer.** Un paramètre `approved=True` fourni par le LLM ne constituerait pas une autorisation.

Prochaines étapes : MCP (deux serveurs) puis raccordement LangGraph/HITL, mémoire persistante et évaluation indépendante par état final. L'ajout de ce backend seul ne satisfait pas encore l'énoncé.
