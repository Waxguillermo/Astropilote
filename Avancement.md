# AVANCEMENT — AstroPilote | Agentic Workflows

> **Mise à jour vérifiée le 14 novembre 2026.** Suivi du dépôt privé `Charles-Scannavino/Astropilote_review`, branche **`v1-agentic`**, dossier Mac **`~/Developer/Astropilote`**. Contexte et critères : [Context.md](Context.md). Les phases 1–3 sont réellement installées, testées et publiées ; les exigences restantes sont identifiées séparément.

## 1. Résumé de l'existant — projet Generative AI

**Objectif originel** : assistant astronomique capable de répondre à des questions sur le ciel profond, rechercher des connaissances, calculer visibilité/météo/nuit astronomique, proposer des plans d'observation et annoter des photos.

### Acquis repérés dans les sources du dépôt

- [x] Dépôt initial et clone de travail configurés. Dans le **nouveau clone**, `origin` = dépôt privé `Astropilote_review` (ne pas réutiliser les anciennes consignes sur les remotes `review` et `upstream`).
- [x] Tag de référence `v0.0-base` poussé dans `origin` et `review` (ne pas le déplacer).
- [x] RAG hybride **BM25 + Qdrant + reranking**, corpus documenté **567 documents / 2 204 chunks**.
- [x] Six outils Python : `search_catalog`, `get_object`, `geocode`, `night`, `weather`, `visibility`.
- [x] Graphe **LangGraph** avec `guard`, boucle `agent ↔ tools`, `compose`, `validate`, `refuse`, `scrub`.
- [x] Budget d'étapes / d'appels d'outils, timeouts, limite tokens/coût et coupe-circuit dans le harness.
- [x] API / interface FastAPI, streaming et fonctions de vision.
- [x] Suite de tests et évaluation historique Generative AI sur `eval/golden.jsonl` (`--questions`) ; suivi existant tokens, coût et latence.
- [x] `README.md` contient déjà une **feuille de route** pour Agentic Workflows ; elle ne prouve pas que toutes les fonctionnalités visées sont codées.
- [x] Dépôt public et dépôt de review synchronisés au commit de documentation `c1b2964` au début du cadrage.

**Tests locaux désormais prouvés :** installation de `uv` et des dépendances, suite initiale **74 passed / 1 skipped**, puis suite des phases 1–3 **105 passed / 1 skipped / 7 subtests passed** (10,01 s). Ruff a validé les modules `agentic`, `backend`, `mcp` et les tests HITL. **À vérifier séparément :** exécution du runner historique `--questions` et démarrage complet de FastAPI ; la commande officielle `--scenarios` n'est pas encore implémentée.

## 2. Ce que demande précisément le professeur

**Route B** : prolonger Generative AI. Le système doit **modifier un backend observable et réinitialisable**, pas seulement répondre avec du RAG. Le travail réutilisé n'est pas noté : le jury doit distinguer les modifications intervenues depuis `v0.0-base`.

| Livrable / critère | État prouvé au 14/11/2026 | À compléter / preuve attendue |
|---|---|---|
| Tag `v0.0-base` | **Fait** | Préserver la baseline ; produire le diff avant `v1.0` |
| Backend réinitialisable | **Fait — Phase 1**, SQLite + seed, 2 télescopes / 4 créneaux / 1 réservation initiale ; 10 tests | Étendre les scénarios multi-étapes si nécessaire |
| Deux serveurs MCP distincts | **Fait — Phase 2**, Sky + Observatory, 12 outils, 8 tests et `smoke_test` | Tester encore les scénarios d'erreurs/timeout si requis |
| LangGraph et budget | **Fait pour un workflow d'actions typées — Phase 3**, limites d'étapes, client MCP, checkpoints SQLite | Raccorder au graphe **conversationnel** existant pour les demandes en langue naturelle |
| HITL avant annulation | **Fait et démontré — Phase 3**, `interrupt()`, refus/accord, jeton unique côté serveur ; 13 tests | Ne pas exposer une approbation opérateur sans authentification |
| Mémoire long terme | **Non fait** | Deux runs indépendants sur des `thread_id` différents, test montrant l'effet du souvenir |
| Architecture agentique | **Graphe conversationnel historique + workflow typé séparé** | Intégration claire sans ajouter inutilement des agents |
| 30 scénarios `--scenarios` | **Non fait** | ~14 normaux, 5 ambigus, 5 impossibles, 6 adversariaux ; état final réellement comparé |
| Protection injection indirecte | **À démontrer** | Résultats d'outils adversariaux, appels interdits non exécutés |
| Traces | **Partielles**, outils, étapes, statut, état SQLite dans la démo | Pass/fail par scénario, `stop_reason`, latence, tokens, coût |
| Démo et soutenance | **Deux démonstrations HITL réussies : refus et approbation** | Démo multi-étapes complète, échec évalué, tableau de résultats et présentation |
| Tag final `v1.0` | **À créer** | Gel du code : **jeudi 19/11/2026 23:59** |

**Commande incontournable** à implémenter, pas à confondre avec la commande d'évaluation de Generative AI :

```bash
uv run python eval/run_eval.py --scenarios eval/scenarios.jsonl --out eval/
```

Pour chaque scénario : réinitialiser depuis `seed`, lancer l'agent, contrôler `expect_state`, `forbid_calls`, `max_steps`, enregistrer pass/fail + raison, état final, séquence d'outils, étapes, tokens in/out, latence, coût et `stop_reason`.

## 3. Découpage des tâches — acquis et reste à faire

### Lot 0 — Baseline et environnement

- [x] Dépôts et tag `v0.0-base` configurés ; branche privée `v1-agentic` active.
- [x] Clone actif déplacé vers `~/Developer/Astropilote` après lenteurs/erreurs disque dans l'ancien dossier sous `Documents/Projets`.
- [x] Dépendances installées avec `uv` ; tests initiaux : **74 passed, 1 skipped, 7 subtests passed en 9,40 s**.
- [x] Documentation de cadrage et suivi corrigée pour le projet AstroPilote.
- [ ] Exécuter et consigner l'évaluation historique `--questions` et un démarrage FastAPI si utile pour la soutenance.

### Lot 1 — Backend SQLite (phase 1 : terminé pour le MVP)

- [x] `src/astropilote/backend/store.py` et `seeds/observatory_demo.json` : données et réinitialisation déterministes.
- [x] Lectures d'état et des disponibilités ; création et annulation de réservations simulées.
- [x] Gestion des conflits, créneaux fermés, validation du seed, refus d'écriture et double réservation concurrente.
- [x] **10 tests dédiés réussis**, Ruff validé ; commit privé **`624c890`**.
- [ ] Élargir les dépendances métiers multi-étapes, uniquement si nécessaire pour varier les 30 scénarios. Pas de notifications réelles ni de paiement externe.

### Lot 2 — MCP (phase 2 : terminé)

- [x] Serveur Sky : 6 outils `search_catalog`, `get_object`, `geocode`, `night`, `weather`, `visibility`.
- [x] Serveur Observatory : 6 outils `get_observatory_state`, `list_telescopes`, `list_slots`, `get_booking`, `create_booking`, `cancel_booking`.
- [x] Schémas typés, erreurs d'affaires structurées et test de découverte/invocation sur **deux processus stdio MCP**.
- [x] `uv run python -m astropilote.mcp.smoke_test` : `geocode` et `list_slots` réussis.
- [x] **8 tests MCP réussis**, Ruff validé ; commit privé **`a0fd3f6`**.
- [ ] Compléter, si nécessaire, des scénarios reproductibles de timeout MCP et d'indisponibilité de serveur.

### Lot 3 — LangGraph + HITL (phase 3 : validée pour le workflow structuré)

- [x] Graphe d'actions typées : `validate → read MCP → approve/interrupt (si cancel) → write MCP`.
- [x] `SqliteSaver` et `thread_id` : reprise possible dans un **nouveau processus CLI**, budget `max_steps`.
- [x] `cancel_booking` protégé côté serveur par un jeton d'autorisation éphémère, lié au couple utilisateur/réservation, consommable une fois et révoqué au reset.
- [x] **13 nouveaux tests réussis**, y compris absence/forgery de jeton, usage unique, refus et reprise.
- [x] Test CLI de **refus** : `phase3-reject-demo` suspendu puis `human_rejected` ; `BK-001` inchangée.
- [x] Test CLI d'**accord** : `phase3-approve-demo` suspendu puis `completed` ; `BK-001` annulée et remboursée **virtuellement** ; confirmation persistante par `inspect`.
- [x] Suite complète **105 passed, 1 skipped, 7 subtests passed en 10,01 s**, Ruff OK ; commit privé **`7720b40`**.
- [ ] **Travail distinct à réaliser** : compréhension des requêtes en langue naturelle et routage vers ce graphe depuis l'assistant historique `src/astropilote/agent/graph.py`.
- [ ] Si exposé par API/web : identité d'opérateur vérifiée, autorisations de rôles, protections CSRF et isolation. Pour l'instant, démonstration CLI locale uniquement.

### Lot 4 — Mémoire utilisateur entre runs (obligatoire)

- [ ] Choisir un stockage persistant isolé par utilisateur (préférence de lieu, télescope ou contraintes).
- [ ] Rendre la mémoire disponible à deux exécutions **indépendantes** (threads différents) ; ne pas confondre `SqliteSaver` de reprise et mémoire long terme.
- [ ] Test de preuve : la donnée retenue au run A modifie le choix du run B.

### Lot 5 — Évaluation, robustesse et sécurité (obligatoire)

- [ ] Construire `eval/scenarios.jsonl` avec ~30 cas : 14 normaux, 5 ambigus, 5 impossibles, 6 adversariaux.
- [ ] Implémenter `uv run python eval/run_eval.py --scenarios eval/scenarios.jsonl --out eval/`, tout en préservant la commande historique `--questions`.
- [ ] Reset du backend **pour chaque cas**, exécution isolée, lecture de `get_state()`, vérification de `expect_state`, `forbid_calls`, `max_steps`.
- [ ] Ajouter injections **dans le texte des résultats d'outils**, tests de non-mutation sur ambiguïté/impossibilité et refus d'actions interdites.
- [ ] Écrire `eval/results.jsonl` et métriques par catégorie ; traces ordonnées, étapes, erreur et `stop_reason`, latence, tokens, coût.

### Lot 6 — Documentation, gel et soutenance

- [x] Procédures des phases MCP et HITL documentées dans `docs/AGENTIC_PHASE2_MCP.md` et `docs/AGENTIC_PHASE3_HITL.md`.
- [x] Démo HITL : **refus sans mutation**, puis **approbation avec mutation contrôlée**.
- [ ] Démo complète en langue naturelle, planification et état final ; succès et échec analysés.
- [ ] Tableau d'évaluation par catégorie, traces et analyse d'erreurs ; répartition de la parole entre 3 à 5 membres.
- [ ] Exécuter les tests finaux, vérifier `git diff v0.0-base..v1.0`, taguer/pousser `v1.0` avant le **19/11/2026 à 23 h 59**.
- [ ] Soutenance le **20/11/2026**, 10 min de présentation + 5 min de questions.

## 4. État concret du dépôt et limites

**Clone de travail :** `/Users/charles/Developer/Astropilote` ; branche `v1-agentic` ; `origin` pointe vers **`Charles-Scannavino/Astropilote_review` (privé)**. Dernier commit de code vérifié avant cette actualisation documentaire : **`7720b40`**.

**Séquence des livraisons :** `624c890` → `a0fd3f6` → `7720b40`. Ces trois commits ont été poussés dans la branche privée. Le dépôt public n'est pas la destination de ces commits.

La commande `uv run python -m astropilote.backend.store --seed seeds/observatory_demo.json` montre un état initial de **2 télescopes, 4 créneaux, BK-001 confirmée et payée (simulée)**. `data/backend/` est exclu de Git pour protéger la base locale et les checkpoints.

**La phase 3 couvre un workflow d'actions déjà structuré (`book` / `cancel`), pas la compréhension libre de phrases utilisateur.** Le graphe conversationnel historique n'appelle pas encore ce workflow. Ne pas déclarer terminés mémoire long terme, évaluateur de 30 scénarios, protections contre injections d'outils ou télémétrie exhaustive.

**Sécurité :** une approbation locale par CLI suppose un opérateur de confiance ; ce modèle n'est pas suffisant pour une exposition réseau sans authentification. Les annulations remboursent uniquement en simulation.

## 5. Commandes de reproduction adaptées au clone actif

```bash
cd ~/Developer/Astropilote
git branch --show-current
git status --short
git log -4 --oneline
git diff --stat v0.0-base..HEAD

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run ruff check src/astropilote/agentic src/astropilote/backend src/astropilote/mcp tests/test_agentic_approval.py

# Démonstration de refus : nouveau thread à chaque répétition.
uv run python -m astropilote.backend.store --seed seeds/observatory_demo.json
uv run python -m astropilote.agentic.demo start --thread <nouveau-thread-refus> --action cancel --user astro-club --booking BK-001
uv run python -m astropilote.agentic.demo resume --thread <nouveau-thread-refus> --reject

# Démonstration d'accord : reset avant, et autre thread pour éviter les checkpoints précédents.
uv run python -m astropilote.backend.store --seed seeds/observatory_demo.json
uv run python -m astropilote.agentic.demo start --thread <nouveau-thread-accord> --action cancel --user astro-club --booking BK-001
uv run python -m astropilote.agentic.demo resume --thread <nouveau-thread-accord> --approve
uv run python -m astropilote.agentic.demo inspect --thread <nouveau-thread-accord>
```

**À ne pas exécuter encore comme une preuve de réussite :** la commande d'évaluation officielle `--scenarios` n'a pas été implémentée dans les phases 1–3. Préserver les données du clone antérieur sous `Documents/Projets` si elles ne sont pas encore comparées ; ne pas le supprimer automatiquement.

## 6. Journal d'avancement

### 14/11/2026 — Cadrage et reprise d'un clone stable

- Correction de `Context.md` / `Avancement.md` initialement associés par erreur à un autre projet ; préservation de la baseline `v0.0-base`.
- Ancienne copie sous `Documents/Projets` affectée par des blocages de lecture ; nouveau clone opérationnel sous `~/Developer/Astropilote`, branche `v1-agentic`, `origin` privé.
- Tests de référence sur Mac : `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q` → **74 passed, 1 skipped, 7 subtests passed en 9,40 s**.

### 14/11/2026 — Phase 1 : SQLite

- Commit **`624c890`** poussé sur `v1-agentic` ; backend dans `src/astropilote/backend/store.py`, seed `seeds/observatory_demo.json`.
- Démonstration du reset, lecture de l'état et réservations simulées ; **10 tests backend réussis**.
- Suite complète : **84 passed, 1 skipped, 7 subtests passed en 9,16 s** ; Ruff OK.

### 14/11/2026 — Phase 2 : deux serveurs MCP

- Commit **`a0fd3f6`** poussé sur `v1-agentic` ; SDK **MCP 2.3.0** installé.
- Sky et Observatory : **6 outils chacun** ; le `smoke_test` a réussi `geocode` et `list_slots` via les deux processus MCP.
- **8 nouveaux tests réussis** ; suite complète **92 passed, 1 skipped, 7 subtests passed en 9,74 s** ; Ruff OK.

### 14/11/2026 — Phase 3 : LangGraph, persistance et approbation

- Commit **`7720b40`** poussé sur `v1-agentic` ; `langgraph-checkpoint-sqlite==3.1.3` installé.
- Nouveau graphe d'actions avec `interrupt()`, checkpoints SQLite, reprise `Command(resume=...)`, client MCP, autorisation temporaire validée côté serveur.
- Premier essai : 10 réussis / 3 échecs de tests causés par `ToolFailure` levée à l'intérieur du contexte `Client`, puis enveloppée en `ExceptionGroup` par AnyIO. Correction du **fichier de test uniquement** : lever l'erreur après la fermeture du client.
- Après correctif : **13/13 tests HITL réussis** ; suite complète **105 passed, 1 skipped, 7 subtests passed en 10,01 s** ; Ruff `All checks passed!`.
- **Démonstration de refus** : `phase3-reject-demo` : état `awaiting_human_approval` → `human_rejected` ; `BK-001.status=confirmed`, `refunded=false`.
- **Démonstration d'approbation** (après reset) : `phase3-approve-demo` : `awaiting_human_approval` → `completed` en 4 étapes ; outils `get_booking` puis `cancel_booking` ; `BK-001.status=cancelled`, `paid=true`, `refunded=true` (**simulation**). La commande `inspect` séparée a confirmé l'état persistant et `S-02.available=true`.
- **Limites constatées** : workflow déclenché par paramètres CLI structurés ; pas encore de routeur conversationnel, de mémoire utilisateur cross-thread ou d'évaluation officielle `--scenarios`.

### Prochaine priorité

Intégrer les demandes en langue naturelle du graphe historique au workflow MCP (avec clarification des ambiguïtés), puis ajouter mémoire long terme et 30 scénarios avec vérification de l'état final réel.

## 7. Principes de suivi

- `[x]` = preuve visible dans les fichiers / Git / tests ; `[ ]` = non démontré.
- Consigner **date, chemin, commit, commande et sortie de test** avant de passer une tâche en « terminé ».
- Ne jamais présenter une fonctionnalité **proposée**, **documentée** ou **préparée en archive** comme **déjà intégrée et testée**.
- Dans le clone actif, `origin` désigne **le dépôt privé de review** ; ne pas pousser vers le dépôt public `Charles-Scannavino/Astropilote` sans décision explicite.
