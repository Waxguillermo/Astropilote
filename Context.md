# Contexte Persistant - Projet Deep Learning : révision de contrats (legal-tech)

## **Informations Générales**
- **Projet** : [Python_Charles_projet_deep_learning](https://github.com/yvespierrecabon/Python_Charles_projet_deep_learning)
- **Répertoire Sandbox** : `/workspace/github__yvespierrecabon__Python_Charles_projet_deep_learning`
- **Objectif** : Aider un étudiant en Master (projet deep learning) à comprendre et réaliser son projet final, afin de pouvoir lui expliquer ensuite.
- **Scénario métier** : ingénieur IA dans une start-up européenne de legal-tech. Construire, évaluer et auditer le cœur NLP d'un assistant de révision de contrats :
  1. classer chaque clause contractuelle par type (droit applicable, résiliation, indemnisation, confidentialité…) ;
  2. récupérer les clauses les plus similaires d'une banque de clauses (Sentence Transformers), à partir d'une clause ou d'une question du juriste ;
  3. documenter les preuves que ça marche (global, par type, par type de contrat/partie) + déclaration des usages proscrits (audit des biais, modes de défaillance).
- **Utilisateur** : yvespierrecabon (accompagne l'étudiant, veut comprendre pour expliquer).

---

## **Mission Technique**
1. **Fine-tuning** d'un modèle pré-entraîné (**BERT / CamemBERT**) sur un dataset de classification de clauses contractuelles.
2. **Pipeline complet** : tokenisation, boucle d'entraînement, évaluation sur un ensemble de test (held-out).
3. **Index de recherche documentaire** avec Sentence Transformers (similarité sémantique sur une banque de clauses).
4. **Audit des biais** : résultats par classe, matrice de confusion, analyse d'erreurs, limites d'usage.

## **Notation & Livraison**
- Livrable technique via Git : **40 %** — soutenance orale en groupe : **40 %** — délibération individuelle : **10 %** — évaluation par les pairs : **10 %**.
- Oral : 15 min (10 présentation + 5 Q&A). **Historique des commits Git obligatoire.**
- **Livraison : 23 octobre** — code source sur GitHub + présentation PowerPoint.

---

## **Environnement Technique**
- **Langage** : Python (environnement dédié géré avec `uv`).
- **Bibliothèques clés attendues** : `torch`, `transformers` (Hugging Face), `datasets`, `sentence-transformers`, `scikit-learn` (métriques), `evaluate`.
- **Modèles** : BERT (`bert-base-uncased`) pour l'anglais / **CamemBERT** (`camembert-base`) pour le français — choix à justifier selon la langue du dataset.
- **Outils** : Git, GitHub, VS Code, Jupyter (notebook `Charles_projet_DL_NLP.ipynb`).

---

## **Structure du Dépôt**
- `Docs/` : documentation du projet
  - `CONTEXT.md` : ce fichier (contexte persistant).
  - `AVANCEMENT.md` : historique d'avancement (à mettre à jour régulièrement).
- `sujet.md` : énoncé du projet traduit en français (contexte, mission, notation, livraison).
- `Charles_projet_DL_NLP.ipynb` : notebook de travail (données : Colab fourni dans `sujet.md`).
- `README.md`, `.gitignore`.

---

## **Données**
- Notebook Colab du cours (dataset de clauses contractuelles) : <https://colab.research.google.com/drive/1rVOdhKNtL9qk-6ZjuFc0ZcQlTLfvaUjg#scrollTo=9FwDLJi7Yodz>
- Dataset probable : clause types de contrats (ex. ContractNLI / LEDGAR selon le contenu du Colab — à confirmer à la lecture du notebook).

---

## **Règles de Travail**
1. **Persistance** : lire ce fichier **à chaque nouvelle session** pour maintenir le contexte.
2. **Synchronisation systématique** :
   ```bash
   cd /workspace/github__yvespierrecabon__Python_Charles_projet_deep_learning && git fetch --all && git pull origin main
   ```
3. **Mises à jour** : mettre à jour `AVANCEMENT.md` après chaque tâche significative ; vérifier la cohérence `CONTEXT.md` / `AVANCEMENT.md`.
4. **Travail direct sur `main`** (choix utilisateur, confirmé en session 2).
5. **Rôle pédagogique** : l'utilisateur veut **comprendre pour expliquer** à l'étudiant. Privilégier :
   - **Toujours expliquer ce qui est demandé et comment on va y répondre AVANT de proposer éventuellement le code correspondant**. Comprendre d'abord, coder ensuite.
   - Explications conceptuelles claires (shapes, tokenisation, fine-tuning, embeddings, paramètres, métriques).
   - **Exemples de code génériques et rappels de syntaxe** plutôt que corrections directes des TODO du notebook (l'étudiant doit chercher lui-même).
   - Vérification des invariants (comptes de paramètres, dimensions) et interprétation des résultats.
   - Répondre en **français**.
6. **Clarté** : sections claires (`##`, `###`), listes à puces.

---

## **Historique des Projets Précédents (contexte de l'étudiant)**
- **Python_IA_causal_transformer** : lab Transformer from scratch (decoder-only, PyTorch pur, comptes de paramètres 4 128 / 4 128 / 12 608 / 1 997 114) + projet RAG sur data.gouv.fr.
- **Python_IA_HLM_RAG** : assistant logement social (RAG hybride BM25+dense, agent Ollama, golden set, éval recall@5/MRR).

*Dernière modification : session 2 — conservation des instructions d'interaction dans les Règles de Travail*
