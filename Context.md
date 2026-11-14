# CONTEXT — AstroPilote | Generative AI → Agentic Workflows

> **État vérifié le 14 novembre 2026 — branche `v1-agentic`.** Ce document fixe les objectifs et distingue le projet Generative AI hérité des ajouts Agentic Workflows réellement implémentés. Les phases 1 à 3 ont été commitées et testées sur le Mac ; l'agent conversationnel historique n'est pas encore relié aux opérations métier. Le journal, les preuves et les tâches restantes figurent dans [Avancement.md](Avancement.md). Les exigences ci-dessous proviennent de l'énoncé de l'enseignant **« Agentic Workflows group project: an agent team that acts, and a harness that proves it »**.

## 1. Identité, sources et organisation

- **Projet** : AstroPilote (nom du package : `astropilote`, version de référence : `0.1.2`).
- **Cours d'origine** : Generative AI — assistant d'astronomie et d'astrophotographie.
- **Nouveau cours** : Agentic Workflows — **Route B**, extension du projet Generative AI existant.
- **Dépôt initial (upstream)** : https://github.com/Waxguillermo/Astropilote
- **Dépôt principal (origin)** : https://github.com/Charles-Scannavino/Astropilote
- **Dépôt privé de développement/review** : https://github.com/Charles-Scannavino/Astropilote_review
- **Chemin local Mac actif** : `/Users/charles/Developer/Astropilote` (le dossier antérieur sous `Documents/Projets` n'est plus le clone de référence).
- **Branche de développement** : `v1-agentic`.
- **Remote du clone actif** : `origin` → dépôt **privé** `Charles-Scannavino/Astropilote_review`, confirmé par `git remote -v` et les pushes. Le dépôt **public** `Charles-Scannavino/Astropilote` est distinct et n'a pas été modifié par les commits des phases 1–3. Ne pas présumer que `upstream` ou `review` sont configurés dans ce nouveau clone.
- **Baseline obligatoire** : tag Git `v0.0-base` présent dans les deux dépôts personnels ; comparer `git diff v0.0-base..v1.0` lorsque `v1.0` aura été créé. Ne pas déplacer le tag de baseline.
- **Référence pédagogique** : https://agentic.lambz.cloud/ et laboratoires https://github.com/lambzcode/agentic-workflow-lab .
- **Licence** : conserver la licence MIT du code source et les attributions des données (voir `LICENSE`, `data/DATA_CARD.md`).

## 2. Partie A — Projet existant : Generative AI

### Problème métier et expérience utilisateur

AstroPilote aide un observateur à comprendre les objets du ciel profond, choisir des cibles selon le lieu et la date, préparer une séance d'observation / astrophotographie, adapter les conseils au matériel disponible et identifier des objets sur une photo du ciel. Jusqu'à la baseline, le rôle principal est **informatif et consultatif** : l'assistant calcule et conseille, mais ne gère pas des réservations persistantes.

### Fonctionnalités présentes dans le dépôt

1. **RAG hybride** sur un corpus documenté de **567 documents** et **2 204 chunks** : 110 Messier, 109 Caldwell, 200 NGC lumineux, 28 cibles d'astrophotographie, 88 constellations et 32 guides pratiques. Corpus Wikipedia/OpenNGC ; métadonnées astronomiques.
2. **Recherche** : BM25 + vecteurs sémantiques Qdrant + reranking ; références de documents contrôlées.
3. **Calculs déterministes et données externes** : géocodage Nominatim, météo Open-Meteo, nuit astronomique, éphémérides et visibilité, sélection de nuits, conseils de cadrage/optique ; calculs Python plutôt que chiffres inventés par le modèle.
4. **Six outils Python internes** : `search_catalog`, `get_object`, `geocode`, `night`, `weather`, `visibility` dans `src/astropilote/tools/`. Ils ne sont pas, à eux seuls, deux serveurs **MCP**.
5. **LangGraph réel** : `StateGraph(AstroState)` dans `src/astropilote/agent/graph.py`, avec `guard → agent ↔ tools → compose → validate`, chemins `refuse` et `scrub`. Le graphe a déjà une boucle de planification ReAct.
6. **Harness / limites** : `src/astropilote/agent/harness.py` contrôle notamment le nombre d'étapes et d'outils, le budget tokens/coût, les appels redondants, les timeouts et les erreurs consécutives.
7. **Modèles** : Ollama en local ou fournisseur cloud (Anthropic / OpenRouter), selon configuration ; ne pas présumer que l'exécution locale est déjà installée sur la machine.
8. **Application FastAPI et interface web** : API chat, streaming, graphe et vision astronomique dans `src/astropilote/app/`.
9. **Évaluation Generative AI existante** : `eval/golden.jsonl` et `eval/run_eval.py --questions ...` évaluent des réponses / citations et suivent tokens, coût et latence. **Ce n'est pas** l'évaluation sur état final du projet Agentic Workflows.

### Fichiers à connaître

| Chemin | Rôle |
|---|---|
| `README.md` | Installation, architecture existante et feuille de route Agentic Workflows |
| `src/astropilote/agent/graph.py` | Graphe LangGraph et exécution |
| `src/astropilote/agent/harness.py` | Budgets / garde-fous |
| `src/astropilote/tools/` | Six outils Python existants |
| `src/astropilote/retrieval/` ; `src/astropilote/ingest/` | RAG et constitution des données |
| `src/astropilote/tracking/` | Télémétrie |
| `src/astropilote/app/` ; `src/astropilote/vision/` | Interface et vision |
| `eval/` ; `tests/` | Évaluation historique et tests existants |
| `src/astropilote/backend/store.py` ; `backend/approvals.py` ; `seeds/observatory_demo.json` | SQLite réinitialisable, réservations et autorisations temporaires |
| `src/astropilote/mcp/sky_server.py` ; `mcp/observatory_server.py` ; `mcp/common.py` | Deux serveurs MCP distincts, 12 outils, résultats/erreurs structurés |
| `src/astropilote/agentic/workflow.py` ; `agentic/demo.py` | Graphe d'actions LangGraph, client MCP, checkpoints et CLI HITL |
| `tests/test_backend.py` ; `tests/test_mcp.py` ; `tests/test_agentic_approval.py` | Tests des phases 1, 2 et 3 (10 + 8 + 13) |
| `docs/AGENTIC_PHASE2_MCP.md` ; `docs/AGENTIC_PHASE3_HITL.md` | Installation et démonstrations techniques documentées |

## 3. Partie B — Nouveau projet Agentic Workflows : ce qui est attendu

### Principe central

**Un chatbot RAG seul ne suffit pas.** L'agent doit agir sur un **backend dont l'état observable change**. Les évaluations doivent prouver, en lisant le backend après l'action, que seules les modifications autorisées ont eu lieu. Les outils de recherche astronomique peuvent être conservés, mais ne suffisent pas comme action métier.

**Route B** : conserver le RAG de Generative AI et ajouter un backend modifiable, des outils MCP, une mémoire et un graphe agentique. **Le code réutilisé ne rapporte pas de points ; le jury note les nouveautés** depuis `v0.0-base`.

### Architecture métier — simulation retenue et implémentation partielle

Le choix retenu est un **observatoire simulé**, dont l'état est géré par SQLite et réinitialisé depuis `seeds/observatory_demo.json`. Le périmètre déjà implémenté comprend **2 télescopes, 4 créneaux et 1 réservation initiale**, les lectures d'état, la création de réservation et l'annulation avec remboursement **simulé**. La liste suivante décrit le **périmètre cible**, pas uniquement le code déjà livré :

- Créneaux et réservations : **implémentés**. Comptes utilisateurs complets, notifications et historique métier exhaustif : **non implémentés**.
- **Lectures** : lister les créneaux, vérifier disponibilités et réservations.
- **Écritures** : créer ou annuler une réservation (**implémenté**) ; modifier une séance ou envoyer une notification simulée (**pistes non implémentées**).
- Au moins une écriture **dangereuse ou difficilement réversible** (p. ex. annulation d'une réservation ou envoi d'une notification), toujours précédée d'une **approbation humaine** dans LangGraph.
- Séquences multi-étapes : rechercher un créneau → vérifier une cible et une date → demander confirmation → réserver → relire l'état final.
- Les scénarios ambigus ou impossibles ne doivent provoquer **aucun changement d'état**.
- Ne jamais envoyer de vrais messages, payer ou réserver des services externes dans les tests.

### Exigences officielles à démontrer

| Élément | Exigence |
|---|---|
| Backend | État observable réinitialisable depuis un seed ; lectures + écritures ; au moins une écriture dangereuse ; workflows multi-étapes ; variété suffisante pour ~30 scénarios |
| Outils | **Au moins 2 serveurs MCP** distincts, arguments typés et erreurs explicites |
| Graphe | **LangGraph**, budget d'étapes, **validation humaine avant toute action dangereuse** |
| Mémoire | **Au moins une mémoire long terme** ; test montrant qu'elle change un run ultérieur |
| Agents | **Un seul agent accepté** ; plusieurs seulement si bénéfice mesuré |
| Évaluation | Environ **30 scénarios** : ~14 `normal`, ~5 `ambiguous`, ~5 `impossible`, ~6 `adversarial` |
| Sécurité | Ignorer les instructions malveillantes cachées dans les **résultats d'outils** ; respecter `forbid_calls` |
| Traces | Par run : résultat pass/fail + motif, état final, outils dans l'ordre, étapes, tokens entrée/sortie, latence, coût, raison d'arrêt |
| Présentation | Démo d'un succès **et d'un échec**, traces, taux de réussite par type et analyse des erreurs ; chaque membre parle |

Le jury doit pouvoir exécuter **exactement** :

```bash
uv run python eval/run_eval.py --scenarios eval/scenarios.jsonl --out eval/
```

Ce runner doit réinitialiser le backend **avant chaque scénario**, comparer `expect_state` avec le **vrai état final du backend**, contrôler `forbid_calls` et `max_steps`, puis écrire `eval/results.jsonl`.

**Échéances officielles** : gel du code **jeudi 19 novembre 2026 à 23 h 59** (tag `v1.0`) ; soutenance **vendredi 20 novembre 2026**, 10 minutes de pitch + 5 minutes de questions ; travail de groupe **3 à 5 étudiants** ; projet = **50 %** de la note du cours.

### État d'implémentation prouvé au 14/11/2026

| Lot | Fichiers et fonctionnement | Preuves |
|---|---|---|
| **Phase 0 — baseline** | Tag `v0.0-base`, clone de travail `~/Developer/Astropilote`, branche privée `v1-agentic` | Baseline de Generative AI préservée ; suite initiale **74 passed, 1 skipped, 7 subtests passed** |
| **Phase 1 — backend** | `ObservatoryStore`, `reset_backend`, `get_state`, `list_telescopes`, `list_slots`, `create_booking`, `cancel_booking` | Commit `624c890` ; **10 nouveaux tests** ; reset, persistance, concurrence, double réservation, erreurs |
| **Phase 2 — MCP** | Serveurs **Sky** et **Observatory**, **6 outils chacun**, transport stdio et SDK `mcp==2.3.0` lors de l'installation | Commit `a0fd3f6` ; **8 nouveaux tests** ; `smoke_test` a découvert les 12 outils et obtenu `geocode` et `list_slots` sans erreur |
| **Phase 3 — HITL** | `src/astropilote/agentic/workflow.py` : graphe d'actions structuré avec `interrupt()`, `SqliteSaver`, `thread_id`, reprise ; jeton d'annulation lié à la réservation et consommable une fois | Commit `7720b40` ; **13 nouveaux tests**, refus et approbation observés en CLI |
| **Qualité globale** | Suite complète sur Mac ; lint Ruff ciblé | **105 passed, 1 skipped, 7 subtests passed** en **10,01 s** ; `All checks passed!` |

**Scénarios HITL observés sur le Mac, après réinitialisation indépendante du backend :**

- **Refus** (`phase3-reject-demo`) : `awaiting_human_approval` → `human_rejected` ; `BK-001` reste `confirmed`, `refunded=false` ; aucune écriture `cancel_booking`.
- **Accord** (`phase3-approve-demo`) : `awaiting_human_approval` → `completed` ; trace MCP `get_booking` puis `cancel_booking` ; `BK-001` devient `cancelled`, `paid=true`, `refunded=true`. Une commande `inspect` distincte a confirmé l'état final persistant et `S-02` de nouveau disponible.
- **Périmètre de sécurité** : simulation locale et CLI d'opérateur présumé fiable ; ne pas exposer `resume --approve` à un utilisateur non authentifié. Ni un prompt LLM, ni un champ `approved=true` ne suffisent à autoriser une annulation.

**Limitations non résolues :** le graphe `agentic/workflow.py` est **séparé** du graphe conversationnel `agent/graph.py` ; les actions ne sont pas encore routées depuis la langue naturelle. Mémoire utilisateur entre runs indépendants, `eval/scenarios.jsonl` avec 30 scénarios, évaluation `--scenarios`, injections indirectes et télémétrie finale tokens/coût ne sont pas démontrées. La suite de tests n'est **pas** une preuve que le projet répond déjà intégralement au barème officiel.

### Cours à mobiliser

- Sessions 1–2 : protocoles MCP, schémas typés, descriptions, erreurs, timeouts, idempotence.
- Session 3 : LangGraph, gardes, checkpoint, `interrupt()`, reprise après accord/refus.
- Session 4 : mémoire de court/long terme et persistance entre runs.
- Session 5 : architectures mono- et multi-agents ; mesurer avant de complexifier.
- Session 6 : évaluation de l'état final, tests adversariaux, analyse des échecs.
- Session 7 (selon énoncé) : observabilité, tokens, coût, latence.

### Bonus facultatifs

Interface montrant plan / outils / étape d'approbation ; comparaison de deux modèles sur les mêmes scénarios ; récupération après une panne simulée d'outil.

## 4. Règles de collaboration et de qualité

- Ne pas confondre **décrit dans le README**, **implémenté**, **testé**, et **validé avec l'évaluation officielle**.
- Préserver le RAG, les six outils et les tests de Generative AI ; privilégier une extension incrémentale avec tests.
- Développer dans `~/Developer/Astropilote` sur la branche privée `v1-agentic` : **dans ce clone, `origin` pointe vers `Astropilote_review` (privé)**. Ne pas toucher au dépôt public sans décision explicite.
- Avant toute écriture métier dangereuse, appliquer approbation explicite et reprise contrôlée ; ne pas confondre un simple paramètre `approved=True` avec une étape HITL `interrupt()`.
- Garder des fixtures déterministes ; isoler les états de tests pour éviter les interférences et doubles écritures.
- Protéger les clés API (`.env`, `.gitignore`) et conserver l'attribution MIT.
- Actualiser `Avancement.md` après chaque tâche avec date, commit, tests exécutés, résultats et blocages. Ne cocher une exigence que lorsqu'elle a été observée et testée.
- Pour collaborer, expliquer les choix techniques, présenter les commandes et identifier les critères de validation.

**État au 14/11/2026 :** sur la branche privée `v1-agentic`, les phases **backend SQLite, deux serveurs MCP et workflow HITL** sont implémentées et vérifiées. L'intégration conversationnelle, la mémoire long terme et l'évaluation officielle `--scenarios` restent à réaliser. Le `main` public n'est pas concerné par ces validations.
