# AVANCEMENT — AstroPilote | Agentic Workflows

> **Date de référence : 14 novembre 2026.** Suivi du dépôt privé `Charles-Scannavino/Astropilote_review`, branche `main`. Contexte et critères : [Context.md](Context.md). Ce document sépare le **projet Generative AI déjà existant** des **ajouts attendus en Agentic Workflows**. Les résultats d'exécution sur le Mac ne sont pas supposés acquis.

## 1. Résumé de l'existant — projet Generative AI

**Objectif originel** : assistant astronomique capable de répondre à des questions sur le ciel profond, rechercher des connaissances, calculer visibilité/météo/nuit astronomique, proposer des plans d'observation et annoter des photos.

### Acquis repérés dans les sources du dépôt

- [x] Dépôt GitHub et code initial clonés ; remotes `upstream`, `origin` et `review` configurés.
- [x] Tag de référence `v0.0-base` poussé dans `origin` et `review` (ne pas le déplacer).
- [x] RAG hybride **BM25 + Qdrant + reranking**, corpus documenté **567 documents / 2 204 chunks**.
- [x] Six outils Python : `search_catalog`, `get_object`, `geocode`, `night`, `weather`, `visibility`.
- [x] Graphe **LangGraph** avec `guard`, boucle `agent ↔ tools`, `compose`, `validate`, `refuse`, `scrub`.
- [x] Budget d'étapes / d'appels d'outils, timeouts, limite tokens/coût et coupe-circuit dans le harness.
- [x] API / interface FastAPI, streaming et fonctions de vision.
- [x] Suite de tests et évaluation historique Generative AI sur `eval/golden.jsonl` (`--questions`) ; suivi existant tokens, coût et latence.
- [x] `README.md` contient déjà une **feuille de route** pour Agentic Workflows ; elle ne prouve pas que toutes les fonctionnalités visées sont codées.
- [x] Dépôt public et dépôt de review synchronisés au commit de documentation `c1b2964` au début du cadrage.

**À revérifier à l'exécution** : `uv sync --extra dev`, `uv run pytest`, `uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/`, démarrage FastAPI. Aucun résultat de ces commandes n'a encore été fourni pour le Mac dans cette phase.

## 2. Ce que demande précisément le professeur

**Route B** : prolonger Generative AI. Le système doit **modifier un backend observable et réinitialisable**, pas seulement répondre avec du RAG. Le travail réutilisé n'est pas noté : le jury doit distinguer les modifications intervenues depuis `v0.0-base`.

| Livrable / critère | État au 14/11 | À faire / preuve attendue |
|---|---|---|
| Tag `v0.0-base` | **Fait** | Conserver le tag et documenter le diff vers `v1.0` |
| Backend réinitialisable avec lectures et écritures | **À développer** | Seed JSON/SQLite ; état lisible ; au moins une écriture dangereuse |
| Deux serveurs MCP distincts | **À développer** | Au moins 2 endpoints/serveurs ; arguments typés et erreurs testées |
| LangGraph et budget d'étapes | **Partiellement existant** | Réutiliser le graphe ; le raccorder aux outils MCP et aux écritures |
| Approbation humaine avant action dangereuse | **À développer** | `interrupt()` / checkpoint / reprise positive et négative |
| Mémoire long terme | **À développer** | Deux runs indépendants ; test prouvant l'effet du souvenir |
| Architecture agentique | **1 agent existant** | Suffisant si performant ; plusieurs seulement si gain mesuré |
| 30 scénarios évalués par état final | **À développer** | ~14 normaux, ~5 ambigus, ~5 impossibles, ~6 adversariaux |
| Protection contre injection indirecte | **À démontrer** | Outils pouvant renvoyer du texte malveillant ; pas d'appel interdit |
| Traces complètes des runs | **Partiellement existant** | Ajouter résultat, état final, outils ordonnés, étapes et stop reason |
| Rapport / soutenance | **À préparer** | Succès + échec, traces, tableau des résultats par catégorie, analyse d'erreurs |
| Tag final `v1.0` | **À créer lors du gel** | Publier avant jeudi 19/11/2026 23:59 |

**Commande incontournable** à implémenter, pas à confondre avec la commande d'évaluation de Generative AI :

```bash
uv run python eval/run_eval.py --scenarios eval/scenarios.jsonl --out eval/
```

Pour chaque scénario : réinitialiser depuis `seed`, lancer l'agent, contrôler `expect_state`, `forbid_calls`, `max_steps`, enregistrer pass/fail + raison, état final, séquence d'outils, étapes, tokens in/out, latence, coût et `stop_reason`.

## 3. Découpage du travail à réaliser — ordre recommandé

### Lot 0 — Initialisation et sécurité (priorité immédiate)

- [x] Cloner le projet ; connecter GitHub et créer le dépôt privé review.
- [x] Publier `v0.0-base`.
- [x] Analyser le README, le code principal et l'énoncé officiel.
- [x] Réécrire `Context.md` / `Avancement.md` pour retirer le contenu étranger (cette mise à jour).
- [ ] Sur le Mac : `git fetch --all --tags`, `git status`, puis `git pull --ff-only review main` **si le répertoire de travail est propre**.
- [ ] Tester les fonctionnalités initiales : installation, `pytest`, évaluation de Generative AI, puis consigner les résultats exacts.
- [ ] Créer une branche de travail dédiée dans `review` avant le développement, si nécessaire.

### Lot 1 — Backend d'observatoire simulé (priorité critique)

- [ ] Définir entités minimales : utilisateurs, créneaux, réservations, notifications simulées.
- [ ] Implémenter état observable et `reset(seed)`, avec opérations de lecture et d'écriture.
- [ ] Prévoir scénarios métiers multi-étapes : réserver / annuler une séance, vérifier disponibilité, choisir une cible.
- [ ] Une action dangereuse et une politique d'autorisation explicite ; protection contre doublons (idempotence).
- [ ] Tester : état avant/après, reset, refus, créneau occupé, double réservation, absence de mutation si erreur.

### Lot 2 — MCP (priorité critique)

- [ ] Serveur 1 : exposer outils de catalogue / visibilité / météo / géocodage existants.
- [ ] Serveur 2 : exposer outils de lecture et d'écriture du backend.
- [ ] Schémas typés et erreurs explicites : paramètres invalides, indisponibilité, timeout.
- [ ] Tester la découverte `tools/list` et les invocations via **un client MCP**, pas seulement des fonctions Python.

### Lot 3 — LangGraph + approbation (priorité critique)

- [ ] Raccorder le graphe existant aux serveurs MCP et au backend.
- [ ] Implémenter routage lecture / écriture et budget `max_steps`.
- [ ] Ajouter `interrupt()`, checkpoint persistant, `thread_id` et reprise après accord ou refus.
- [ ] Test : aucune mutation avant approbation ; aucun effet après refus ; après accord, effet unique même en cas de retry.

### Lot 4 — Mémoire entre exécutions (obligatoire)

- [ ] Persister un souvenir utile (p. ex. lieu préféré, matériel, contraintes horaires).
- [ ] Définir l'isolation par utilisateur et les règles de mise à jour.
- [ ] Test : sauvegarder au run A, relancer sur un **nouveau thread** au run B, constater une décision différente.

### Lot 5 — Harness d'évaluation / sécurité (obligatoire)

- [ ] Créer `eval/scenarios.jsonl` (~30 cas) avec les quatre catégories imposées.
- [ ] Pour chaque cas : seed déterministe, `expect_state`, `forbid_calls`, `max_steps`.
- [ ] Réinitialiser le backend **avant chaque cas**, isoler les runs, relire l'état réel après exécution.
- [ ] Mesurer pass rate par catégorie, refus corrects, appels interdits, timeouts et erreurs.
- [ ] Tester des **injections dans les réponses d'outils** (pas seulement dans l'entrée utilisateur).
- [ ] Produire `eval/results.jsonl` selon le format exigé et démontrer la commande exacte.
- [ ] Conserver les traces de tokens, coût, latence, appels ordonnés, état final et raison d'arrêt.

### Lot 6 — Livraison et soutenance

- [ ] Documenter les commandes d'installation, reproduction, reset et évaluation.
- [ ] Préparer une démonstration réelle réussie et une démonstration réelle échouée, en s'appuyant sur les traces.
- [ ] Présenter performances par type, erreurs analysées, corrections et limites.
- [ ] Vérifier que tous les membres parlent (groupe de 3 à 5).
- [ ] Geler le code, exécuter les tests finaux, taguer et pousser `v1.0` avant le **19/11/2026 23:59**.
- [ ] Soutenance **vendredi 20/11/2026**, **10 min de présentation + 5 min de Q&A** ; projet = **50 %** de la note.

## 4. Périmètre et statut du lot de démarrage proposé

Une archive de **Phase 1** (backend simulé, serveurs MCP et tests) a été préparée dans la conversation. **Elle n'est pas automatiquement installée, exécutée ni commitée dans ce dépôt.** Au moment du présent bilan, le dépôt `main` contient encore principalement le code Generative AI.

Ne pas marquer les lots 1 à 5 « terminés » sans retrouver les fichiers sur la branche concernée, contrôler les diffs et exécuter les tests requis. Un jeton manuel d'approbation dans un backend ne prouverait pas à lui seul la présence de `LangGraph interrupt()`.

## 5. Prochaines commandes utiles (sur le Mac)

```bash
cd /Users/charles/Documents/Projets/Astropilote
git status
git fetch --all --tags
git pull --ff-only review main    # seulement si l'arbre local est propre
git log -3 --oneline
git diff --stat v0.0-base..HEAD

uv sync --extra dev
uv run pytest -q
```

**Attention** : `git pull` sur la branche courante doit être fait après lecture de `git status`; préserver tout changement local non commité. Pour les développements spécifiques au dépôt privé, travailler sur une branche `feat/...`, puis pousser `git push -u review <branche>`.

## 6. Journal d'avancement

### 14/11/2026 — Cadrage et correction des documents de suivi

- Dépôts `Astropilote` (public) et `Astropilote_review` (privé) disponibles ; la branche `main` avait le commit `c1b2964` de documentation.
- Tag `v0.0-base` présent dans les deux dépôts ; architecture Generative AI lue dans README et fichiers Python principaux.
- **Anomalie trouvée** : `Context.md` et `Avancement.md` contenaient le contexte d'un autre projet **Deep Learning legal-tech**. Le suivi a été réorienté vers AstroPilote et l'énoncé Agentic Workflows ; correction limitée au dépôt privé review, sans modification du dépôt public.
- Prochaine tâche : vérifier les tests de la baseline sur le Mac, mettre en place le backend, puis l'intégrer par MCP et LangGraph. Aucun résultat de tests locaux de Phase 1 n'a encore été communiqué.

## 7. Principes de suivi

- `[x]` = preuve visible dans les fichiers / Git / tests ; `[ ]` = non démontré.
- Consigner **date, chemin, commit, commande et sortie de test** avant de passer une tâche en « terminé ».
- Ne jamais présenter une fonctionnalité **proposée**, **documentée** ou **préparée en archive** comme **déjà intégrée et testée**.
- Ne pas modifier implicitement `origin` ni `upstream` ; le travail de revue est privé jusqu'à décision explicite de publication.
