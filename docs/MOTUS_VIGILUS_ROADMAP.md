# ANIMA-MECHANICUS — PLAN DE CONQUETE

## PHASE 1 : FONDATIONS [TERMINEE]

- [x] Structure repo GitHub (ANIMA-MECHANICUS)
- [x] Template R15 Blender (r15_template.blend)
- [x] U-GAMMA motus_forge.py v3 fonctionnel (world→local fix)
- [x] Architecture pivot decidee : Gemini + WHAM + SMPL→R15
- [x] Contrat JSON Gemini defini et documente
- [x] Contrat .npz inchange et valide
- [x] README v2 complet (pipeline, contrats, stack, Quick Start)
- [x] PRD v2 mis a jour
- [x] Roadmap v2 mise a jour
- [x] State v3 mis a jour

---

## PHASE 2 : FREGATE U-ALPHA v2 — L'AUSPEX COGITATEUR [TERMINEE]

### 2.1 Module Gemini
- [x] Appel API Gemini 2.0 Flash avec prompt video
- [x] Parsing et validation du JSON retourne
- [x] Filtre automatique segments (qualite < 0.6 → exclu)
- [x] Affichage rapport utilisateur dans le notebook

### 2.2 Module WHAM (remplace par GVHMR en Phase 6)
- [x] Installation WHAM sur Colab (torch, detectron2, mmcv, ffmpeg)
- [x] Decoupe video par segments valides (OpenCV)
- [x] Appel WHAM inference par segment + par personne
- [x] Lecture sortie WHAM : poses (N,24,3) + transl (N,3)

### 2.3 Module SMPL→R15
- [x] Mapping fixe 15 joints SMPL → 15 os R15
- [x] Conversion axis-angle → quaternion WXYZ (scipy Rotation)
- [x] Application convention axes Roblox (X=-X, Z=-Z)
- [x] Validation : pas de NaN, pas de quaternion nul

### 2.4 Module Lissage + Interpolation
- [x] Savitzky-Golay (conserver logique ancienne version)
- [x] Renforcement extremites (mains, pieds, avant-bras)
- [x] Resampling FPS cible (30/60/120)
- [x] Interpolation gaps occlusion (<10 frames)

### 2.5 Export .npz
- [x] Respect strict du contrat (rotations, root_position, bone_names, fps...)
- [x] 1 fichier .npz par personne
- [x] Validation shape avant sauvegarde

---

## PHASE 3 : NOTEBOOK COLAB U-ALPHA [TERMINEE]

- [x] Cellule 1 : Installation (WHAM, torch, detectron2, mmcv, google-generativeai)
- [x] Cellule 2 : Config (cle Gemini API, FPS cible, niveau lissage)
- [x] Cellule 3 : Upload video .mp4
- [x] Cellule 4 : Analyse Gemini → affichage rapport segments
- [x] Cellule 5 : Validation utilisateur des segments
- [x] Cellule 6 : Extraction WHAM → SMPL→R15 → .npz
- [x] Cellule 7 : Telechargement des .npz

---

## PHASE 4 : NOTEBOOK COLAB U-GAMMA [TERMINEE]

- [x] motus_forge.py v4 — rig R15 genere programmatiquement (plus de r15_template.blend)
- [x] Cellule 1 : Installation Blender headless
- [x] Cellule 2 : Upload .npz
- [x] Cellule 3 : Forge FBX (sans template, rig auto-genere)
- [x] Cellule 4 : Telechargement .fbx

---

## PHASE 5 : VALIDATION IMPERIALE [BLOQUEE — en attente Phase 6]

Bloquee par echec installation WHAM sur Colab T4 (mmcv + detectron2 incompatibles).
Sera re-executee integralement apres migration GVHMR.

- [ ] Test "Danse" — 1 personne, corps complet face camera
- [ ] Verifier rapport Gemini : segments detectes correctement
- [ ] Verifier .npz : shapes (N,15,4), (N,3), pas de NaN
- [ ] Verifier .fbx forge sans erreur Blender
- [ ] Test "Combat" — 2 personnes, occlusions partielles
- [ ] Test "Foule" — 4 personnes simultanees
- [ ] Test "Camera instable" — verifier warning root motion
- [ ] Test "Zoom" — verifier warning root motion fausse
- [ ] Import FBX dans Roblox Studio — validation finale animation

---

## PHASE 6 : MIGRATION GVHMR [EN COURS]

**Raison** : WHAM bloque par dependances mmcv/detectron2 sur Colab T4 actuel.
**GVHMR** (SIGGRAPH Asia 2024) confirme fonctionnel sur T4 via notebook officiel.
**Impact code** : seul `run_wham()` dans motus_extract.py est a remplacer. Tout le reste est intact.

### 6.1 motus_extract.py — Module extraction
- [ ] Remplacer `run_wham()` par `run_gvhmr()` (appel subprocess demo.py GVHMR)
- [ ] Adapter lecture sortie GVHMR → meme format passes/transl que WHAM
- [ ] Implementer cas `FrankMocap_upper` : masquage joints inferieurs (LeftUpperLeg, LeftLowerLeg, LeftFoot, RightUpperLeg, RightLowerLeg, RightFoot → quaternion identite)
- [ ] Supprimer references WHAM/mmcv/detectron2

### 6.2 ANIMA_MECHANICUS_ALPHA.ipynb — Cellule 1 uniquement
- [ ] Refaire Cellule 1 installation : GVHMR (git clone + pip install -r requirements.txt + pip install -e .)
- [ ] Adapter telechargement checkpoints : aria2c depuis HuggingFace (camenduru/GVHMR)
- [ ] Supprimer installation mmcv/detectron2/WHAM
- [ ] Cellules 2 a 7 : INCHANGEES

### 6.3 Documentation
- [x] STATE.md mis a jour (V4 — GVHMR)
- [x] ROADMAP.md mis a jour
- [ ] PRD.md mis a jour (stack technique, pipeline step 2)
- [ ] README.md mis a jour

### 6.4 Re-validation (Phase 5 bis)
- [ ] Re-executer tous les tests de Phase 5 avec GVHMR

---

## NOTES D'ABANDON

| Version | Raison de l'abandon |
|---------|---------------------|
| V1 — 3 Fregates + BVH | Sur-ingenierie, format BVH inutile pour Roblox |
| V2 — MediaPipe | MediaPipe entraine sur humains reels : echoue totalement sur avatars Roblox 3D |
| V3 — WHAM | mmcv 1.3.9 + detectron2 impossibles a installer sur Colab T4 actuel (2026) |
