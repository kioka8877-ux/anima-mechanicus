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

### 2.2 Module WHAM
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

- [x] Renommer en ANIMA_MECHANICUS_GAMMA.ipynb
- [x] Cellule installation Blender headless
- [x] Cellule upload .npz
- [x] Cellule forge FBX
- [x] Cellule telechargement .fbx

---

## PHASE 5 : VALIDATION IMPERIALE [A FAIRE]

- [ ] Pousser r15_template.blend dans U-GAMMA/templates/ (prerequis U-GAMMA)
- [ ] Test "Danse" — 1 personne, corps complet face camera
- [ ] Test "Combat" — 2 personnes, occlusions partielles
- [ ] Test "Foule" — 4 personnes simultanees
- [ ] Test "Camera instable" — verifier warning root motion
- [ ] Test "Zoom" — verifier warning root motion fausse
- [ ] Import FBX dans Roblox Studio — validation finale animation

---

## NOTES D'ABANDON

| Version | Raison de l'abandon |
|---------|---------------------|
| V1 — 3 Fregates + BVH | Sur-ingenierie, format BVH inutile pour Roblox |
| V2 — MediaPipe | MediaPipe entraine sur humains reels : echoue totalement sur avatars Roblox 3D (cas d'usage initial incorrect) |
