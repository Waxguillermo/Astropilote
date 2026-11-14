# CONTEXT — AstroPilote | Generative AI → Agentic Workflows

> **Référence de projet (mise à jour : 14 novembre 2026).** Ce document décrit le contexte et les exigences. Pour le suivi des réalisations et les tâches ouvertes, voir [Avancement.md](Avancement.md). Les exigences d'évaluation ci-dessous proviennent de l'énoncé de l'enseignant **« Agentic Workflows group project: an agent team that acts, and a harness that proves it »** (2 pages, fourni dans la conversation). Les fonctionnalités de référence sont celles du code et du README accessibles au 14/11/2026 ; les tests d'exécution sur le Mac ne sont pas confirmés ici.

## 1. Identité, sources et organisation

- **Projet** : AstroPilote (nom du package : `astropilote`, version de référence : `0.1.2`).
- **Cours d'origine** : Generative AI — assistant d'astronomie et d'astrophotographie.
- **Nouveau cours** : Agentic Workflows — **Route B**, extension du projet Generative AI existant.
- **Dépôt initial (upstream)** : https://github.com/Waxguillermo/Astropilote
- **Dépôt principal (origin)** : https://github.com/Charles-Scannavino/Astropilote
- **Dépôt privé de développement/review** : https://github.com/Charles-Scannavino/Astropilote_review
- **Chemin local Mac** : `/Users/charles/Documents/Projets/Astropilote`.
- **Remotes** : `upstream` (projet source), `origin` (public), `review` (privé). **Un même dossier local**, sans seconde copie nécessaire.
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

## 3. Partie B — Nouveau projet Agentic Workflows : ce qui est attendu

### Principe central

**Un chatbot RAG seul ne suffit pas.** L'agent doit agir sur un **backend dont l'état observable change**. Les évaluations doivent prouver, en lisant le backend après l'action, que seules les modifications autorisées ont eu lieu. Les outils de recherche astronomique peuvent être conservés, mais ne suffisent pas comme action métier.

**Route B** : conserver le RAG de Generative AI et ajouter un backend modifiable, des outils MCP, une mémoire et un graphe agentique. **Le code réutilisé ne rapporte pas de points ; le jury note les nouveautés** depuis `v0.0-base`.

### Architecture métier proposée (choix d'implémentation, non exigence de domaine)

Construire un **observatoire distant simulé** avec un petit backend JSON ou SQLite réinitialisable à partir de `seeds/*.json` :

- Créneaux de télescope, réservations, utilisateurs/équipements, historique des opérations.
- **Lectures** : lister les créneaux, vérifier disponibilités et réservations.
- **Écritures** : créer ou annuler une réservation, modifier une séance, envoyer une notification **simulée**.
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
- Développer d'abord sur le dépôt privé `review` / une branche dédiée ; le dépôt public `origin` ne doit pas être modifié implicitement.
- Avant toute écriture métier dangereuse, appliquer approbation explicite et reprise contrôlée ; ne pas confondre un simple paramètre `approved=True` avec une étape HITL `interrupt()`.
- Garder des fixtures déterministes ; isoler les états de tests pour éviter les interférences et doubles écritures.
- Protéger les clés API (`.env`, `.gitignore`) et conserver l'attribution MIT.
- Actualiser `Avancement.md` après chaque tâche avec date, commit, tests exécutés, résultats et blocages. Ne cocher une exigence que lorsqu'elle a été observée et testée.
- Pour collaborer, expliquer les choix techniques, présenter les commandes et identifier les critères de validation.

**État à la création de ce document :** le code Generative AI est présent, le tag baseline existe et l'énoncé est analysé ; le backend d'observatoire, les 2 serveurs MCP, l'approbation `interrupt()`, la mémoire persistante et l'évaluation `--scenarios` ne sont **pas encore vérifiés comme implémentés sur la branche `main`**.
