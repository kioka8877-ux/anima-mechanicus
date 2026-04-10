# ANIMA-MECHANICUS — PLAN DE CONQUETE

## PHASE 1 : FONDATIONS [TERMINEE]

- [x] Structure repo GitHub (ANIMA-MECHANICUS)
- [x] Template R15 Blender (r15_template.blend)
- [x] U-GAMMA motus_forge.py v3 fonctionnel (world→local fix)
- [x] Architecture pivot décidée : Gemini + WHAM + SMPL→R15
- [x] Contrat JSON Gemini défini et documenté
- [x] Contrat .npz inchangé et validé
- [x] README v2 complet (pipeline, contrats, stack, Quick Start)
- [x] PRD v2 mis à jour
- [x] Roadmap v2 mise à jour
- [x] State v3 mis à jour

---

## PHASE 2 : FREGATE U-ALPHA v2 — L'AUSPEX COGITATEUR [EN COURS]

### 2.1 Module Gemini
- [ ] Appel API Gemini 2.0 Flash avec prompt vidéo
- [ ] Parsing et validation du JSON retourné
- [ ] Filtre automatique segments (qualite < 0.6 → exclu)
- [ ] Affichage rapport utilisateur dans le notebook

### 2.2 Module WHAM
- [ ] Installation WHAM sur Colab (torch, detectron2, ffmpeg)
- [ ] Découpe vidéo par segments validés (OpenCV)
- [ ] Appel WHAM inference par segment + par personne
- [ ] Lecture sortie WHAM : poses (N,24,3) + transl (N,3)

### 2.3 Module SMPL→R15
- [ ] Mapping fixe 15 joints SMPL → 15 os R15
- [ ] Conversion axis-angle → quaternion WXYZ (scipy Rotation)
- [ ] Application convention axes Roblox (X=-X, Z=-Z)
- [ ] Validation : pas de NaN, pas de quaternion nul

### 2.4 Module Lissage + Interpolation
- [ ] Savitzky-Golay (conserver logique ancienne version)
- [ ] Renforcement extrémités (mains, pieds, avant-bras)
- [ ] Resampling FPS cible (30/60/120)
- [ ] Interpolation gaps occlusion (<10 frames)

### 2.5 Export .npz
- [ ] Respect strict du contrat (rotations, root_position, bone_names, fps...)
- [ ] 1 fichier .npz par personne
- [ ] Validation shape avant sauvegarde

---

## PHASE 3 : NOTEBOOK COLAB U-ALPHA [A FAIRE]

- [ ] Cellule 1 : Installation (WHAM, torch, detectron2, google-generativeai)
- [ ] Cellule 2 : Config (clé Gemini API, FPS cible, niveau lissage)
- [ ] Cellule 3 : Upload vidéo .mp4
- [ ] Cellule 4 : Analyse Gemini → affichage rapport segments
- [ ] Cellule 5 : Validation utilisateur des segments
- [ ] Cellule 6 : Extraction WHAM → SMPL→R15 → .npz
- [ ] Cellule 7 : Téléchargement des .npz

---

## PHASE 4 : NOTEBOOK COLAB U-GAMMA [A FAIRE]

- [ ] Renommer en ANIMA_MECHANICUS_GAMMA.ipynb
- [ ] Cellule installation Blender headless (inchangée)
- [ ] Cellule upload .npz
- [ ] Cellule forge FBX
- [ ] Cellule téléchargement .fbx

---

## PHASE 5 : VALIDATION IMPERIALE [A FAIRE]

- [ ] Test "Danse" — 1 personne, corps complet face caméra
- [ ] Test "Combat" — 2 personnes, occlusions partielles
- [ ] Test "Foule" — 4 personnes simultanées
- [ ] Test "Caméra instable" — vérifier warning root motion
- [ ] Test "Zoom" — vérifier warning root motion faussé
- [ ] Import FBX dans Roblox Studio — validation finale animation

---

## NOTES D'ABANDON

| Version | Raison de l'abandon |
|---------|---------------------|
| V1 — 3 Frégates + BVH | Sur-ingénierie, format BVH inutile pour Roblox |
| V2 — MediaPipe | MediaPipe entraîné sur humains réels : échoue totalement sur avatars Roblox 3D (cas d'usage initial incorrect) |
