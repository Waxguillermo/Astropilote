# Historique d'Avancement - Projet Deep Learning : révision de contrats (legal-tech)

> Ce fichier suit le projet final de deep learning (**Python_Charles_projet_deep_learning**) : fine-tuning BERT/CamemBERT pour la classification de clauses contractuelles + index Sentence Transformers + audit des biais. Énoncé complet dans `sujet.md`, contexte dans `Docs/CONTEXT.md`. À mettre à jour régulièrement.

---

## **État Actuel du Projet**
*Snapshot de l'avancement global*

### **Projet deep learning legal-tech (ce dépôt)** — 🆕 session de lancement
- [x] Énoncé du projet reçu (FINAL PROJECT) et traduit en français dans `sujet.md` (commit `cd8456a`).
- [x] Choix de travail : **directement sur la branche `main`** (demande utilisateur).
- [x] Mise à jour de `Docs/CONTEXT.md` et de ce fichier pour suivre ce nouveau projet (remplace le suivi des projets Transformer/HLM RAG).
- [x] Notebook `Charles_projet_DL_NLP.ipynb` ajouté par l'utilisateur.
- [x] Lecture du notebook + du Colab de données : dataset **LEDGAR via LexGLUE** (`coastalcph/lex_glue`, config `ledgar`) — clauses contractuelles SEC américaines, en **anglais** → choix modèle : **BERT** (`bert-base-uncased`), pas CamemBERT.
- [x] Environnement Python dédié (`uv`) créé par l'utilisateur (torch, transformers, datasets, sentence-transformers, scikit-learn, evaluate, ipywidgets). Notebook s'exécute dans PyCharm.
- [x] Exploration des données : splits 60k train / 10k val / 10k test ; 100 classes ; déséquilibre fort — majoritaire `Governing Laws` (3 167), minoritaire `Books` (23), ratio ~140:1 ; 11 classes < 200 exemples, 3 classes < 100 (Books 23, Assigns 31, Qualifications 47). Choix documenté : garder les 100 classes (comparabilité LexGLUE), F1 par classe instables sur les rares → point d'audit.
- [x] Analyse des longueurs (2 000 clauses, tokenizer BERT) : p50=109, p90=298, p95=385 tokens, >1% à 512. Décision : **`max_length=256`** (tronque ~15 %, coupe la fin, information type en début de clause) ; ablation 256 vs 384 prévue pour la soutenance. Padding dynamique via `DataCollatorWithPadding`.
- [x] Mini-test pipeline validé en local CPU (2 000 clauses, 1 epoch, ~36 min, 0,94 clause/s) : loss 4,02, accuracy 21,2 %, F1 macro 3,9 % — écart accuracy/F1 macro = déséquilibre visible dès le mini-test.
- [x] **Fine-tuning complet sur Colab (T4 GPU)** : 60k clauses, 2 epochs, batch 16, lr 2e-5, ~1h40 (20 clauses/s). Résultats validation : **accuracy 84,4 %, F1 macro 73,9 %** (réf. LexGLUE BERT : ~71-76 % acc). Loss val baisse sur les 2 epochs → pas d'overfitting. Modèle sauvegardé `bert-ledgar/` et téléchargé en local (zip).
- [x] Évaluation sur le **test** (10k clauses, CPU ~54 min, 3,1 clauses/s) : **accuracy 84,7 %, F1 macro 73,9 %**, weighted 0,83 — cohérent avec la validation (84,4 %) → généralisation OK, au-dessus de la réf. LexGLUE (~71-76 %, version sous-échantillonnée). Prédictions sauvegardées `test_preds.npy`/`test_labels.npy`.
- [x] **Audit des biais** (rapport par classe + top confusions) : 15 classes < 0,5 F1. Deux causes distinctes : (1) sous-représentation (Assigns/Books/Qualifications F1=0 mais supports test 4/2/5 — quasi non évaluables) ; (2) **confusions sémantiques entre labels quasi-synonymes de la taxonomie LEDGAR** : Tax Withholdings→Withholdings (71), Applicable Laws→Governing Laws (45), Defined Terms→Definitions (40), Integration→Entire Agreements (31), Waivers↔No Waivers (41), Indemnity→Indemnifications (23)… Conclusion d'audit : ~2/3 des erreurs = ambiguïtés du benchmark, pas du modèle ; Applicable Laws (F1=0, 352 ex. train) échoue malgré une représentation correcte. À mettre en avant dans la soutenance ("where it cannot be trusted").
- [x] **Index de recherche Sentence Transformers** (`all-MiniLM-L6-v2`, banque = 10k clauses du test, similarité cosinus, top-k) : validé sur deux types de requêtes — clause collée (governing law → 5/5 `Governing Laws` pertinents) et question du juriste ("termination clause with 30-day notice period" → remonte une clause `General` de résiliation avec préavis 30 jours, non labellisée `Terminations` — démontre la complémentarité classification vs recherche). Limite observée : bruit notionnel (`Terms` durée du contrat) → perspective : recherche hybride similarité + filtre de label prédit. Embeddings bank sauvegardés (`bank_emb.npy`, commité).
- [x] README de reproductibilité écrit (modèles exacts, données, résultats test, audit, `uv sync`, stockage du modèle, structure du dépôt).
- [ ] Présentation PowerPoint (10 min + 5 Q&A) : **à préparer** — points clés disponibles (résultats, audit, complémentarité classif/recherche, reproductibilité).
- [ ] Index de recherche Sentence Transformers (banque de clauses) : **non commencé**.
- [ ] Audit des biais (métriques par classe, matrice de confusion, analyse d'erreurs, limites) : **non commencé**.
- [ ] Présentation PowerPoint (10 min + 5 Q&A) : **non commencée**.
- ⚠️ Échéance : **23 octobre** (code sur GitHub + PPT). Historique de commits obligatoire. Missions techniques 1-4 : **terminées**.

---

## **Objectifs Actuels**
- [ ] **Priorité 1** : lire le notebook `Charles_projet_DL_NLP.ipynb` et le Colab de données pour identifier le dataset exact (langue, classes, taille) — conditionne BERT vs CamemBERT.
- [ ] **Priorité 2** : créer l'environnement dédié avec `uv` (`uv init` / `uv add torch transformers datasets sentence-transformers scikit-learn evaluate`).
- [ ] **Priorité 3** : exploration des données (distribution des classes, longueurs de clauses) avant tout entraînement.

---

## **Plan prévisionnel (mission)**
1. **Données** : charger le dataset de clauses, split train/validation/test, explorer (classes, longueurs, déséquilibres).
2. **Classification** : fine-tuner BERT (`bert-base-uncased`, EN) ou CamemBERT (`camembert-base`, FR) — tokenisation, Trainer/boucle custom, métriques (accuracy, F1 macro/par classe).
3. **Retrievabilité** : encoder la banque de clauses avec Sentence Transformers (ex. `all-MiniLM-L6-v2` / multilingue) et construire l'index de similarité (recherche top-k).
4. **Audit des biais** : performance par type de clause et par type de contrat, matrice de confusion, analyse d'erreurs, déclaration des usages proscrits.
5. **Livrables** : code GitHub (commits réguliers), README reproductible, PPT 10 min + 5 Q&A.

---

## **Points d'Attention**
- **Historique Git obligatoire** : commits réguliers et lisibles dès maintenant (40 % de la note = livrable Git).
- **Choix du modèle** : CamemBERT si clauses françaises, BERT si anglaises — à trancher après lecture du dataset.
- **Évaluation honnête** : held-out set jamais vu, métriques par classe (pas seulement l'accuracy globale), audit documenté.
- **Rôle pédagogique** : expliquer avant de coder ; l'étudiant implémente lui-même (cf. `CONTEXT.md`).

---

## **Journal des Actions**
*(Ordre chronologique inverse)*

---

### **Session 2 — Lancement du projet deep learning legal-tech (ce dépôt)**

#### **Actions Réalisées**
- Synchronisation du dépôt (`main`, fast-forward `cd8456a..2db90a9` : ajout du notebook, `Docs/CONTEXT.md`, `Docs/AVANCEMENT.md`).
- Traduction de l'énoncé FINAL PROJECT en français → `sujet.md` (contexte, données, mission, notation 40/40/10/10, soutenance 15 min, livraison 23 octobre) — commit `cd8456a`, fusionné sur `main`.
- Bascule du suivi depuis le projet HLM RAG vers ce projet : réécriture de `Docs/CONTEXT.md` et de ce fichier.
- Rappel à l'utilisateur de la commande `git clone` et des commandes `uv` pour l'environnement dédié.

#### **Contexte de la Session**
- **Demande Utilisateur** : traduire le sujet en français (`sujet.md`), travailler directement sur `main`, mettre à jour `Docs/CONTEXT.md` et `Docs/AVANCEMENT.md` pour ce nouveau projet, rappel `git clone` + `uv`.
- **Répertoire de Travail** : `/workspace/github__yvespierrecabon__Python_Charles_projet_deep_learning`
- **Outils Utilisés** : `git`, `gh`.

#### **Problèmes Rencontrés**
- Aucun notable.

#### **Prochaines Actions Immédiates (reprise session 3)**
- Préparation soutenance (PPT 10 min + 5 Q&A) : résultats, audit, complémentarité classif/recherche, reproductibilité. Plan de soutenance à structurer.
- Committer le notebook après chaque séance (historique noté).
---

### **Session 1 (historique, dépôts précédents)**
- Lab Transformer from scratch (`Python_IA_causal_transformer`) puis projet RAG logement social (`Python_IA_HLM_RAG`) : contexte, spec, arborescence, corpus, golden set — détail dans l'historique git de ce fichier.

---

## **Calendrier des Sessions**
- **Session 1** : lab Transformer + projet RAG (dépôts précédents).
- **Session 2** : lancement du projet deep learning legal-tech — traduction du sujet, mise à jour de la doc.

---

## **Tags Utilisés**
- ✅ : Tâche terminée
- 🔍 : En cours
- ❌ : Bloquée
- 💡 : Contexte important
- ⚠️ : Problème / attention
- 👀 : Observation
- 📝 : Documentation

---

*Dernière modification : session 2 — missions techniques 1-4 terminées, reste la soutenance*
