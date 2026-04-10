# ANIMA-MECHANICUS — PHYLACTERE DE RESURRECTION

| Champ | Valeur |
|-------|--------|
| STATUS | Phase 4 — En cours (Notebooks Colab a valider) |
| DATE | Avril 2026 |
| ARCHITECTURE | 2 Fregates (U-ALPHA + U-GAMMA) |
| VERSION | V3 — Option C : Gemini 2.0 Flash + WHAM + SMPL→R15 |
| CONTRAT .npz | Inchange : rotations (N,15,4) + root_position (N,3) + metadata |
| CONTRAT JSON GEMINI | Defini et documente dans README |

---

## [LAST_WORK]

- Projet renomme ANIMA-MECHANICUS (WH40K lore)
- Depot GitHub cree : kioka8877-ux/ANIMA-MECHANICUS
- Architecture pivot : MediaPipe abandonne → Gemini 2.0 Flash + WHAM + SMPL→R15
- README v2 redige avec pipeline complet Option C
- Contrat JSON Gemini defini et documente (segments, qualite, camera, warnings)
- U-GAMMA (motus_forge.py v3) : fonctionnel, inchange
- motus_extract.py v2 : COMPLETE — pipeline Gemini + WHAM + SMPL→R15 code
- ANIMA_MECHANICUS_ALPHA.ipynb : COMPLETE — notebook Colab 7 cellules cree
- ANIMA_MECHANICUS_GAMMA.ipynb : COMPLETE — notebook Colab 4 cellules cree

---

## [NEXT_TASK]

Phase 5 — Validation imperiale :

1. Test "Danse" — 1 personne, corps complet face camera
2. Test "Combat" — 2 personnes, occlusions partielles
3. Test "Foule" — 4 personnes simultanees
4. Test "Camera instable" — verifier warning root motion
5. Test "Zoom" — verifier warning root motion fausse
6. Import FBX dans Roblox Studio — validation finale animation

---

## [BLOCKERS]

- Modeles SMPL (body_models/) necessitent inscription gratuite sur mpg.de
  → Procedure documentee dans ANIMA_MECHANICUS_ALPHA.ipynb Cellule 1
- WHAM necessite ffmpeg + torch + detectron2 + mmcv sur Colab T4
  → Cellule d'installation complete dans ANIMA_MECHANICUS_ALPHA.ipynb
- Template r15_template.blend non encore pousse dans le repo
  → Necessaire pour que U-GAMMA fonctionne

---

## [SOLUTIONS]

| Probleme | Solution |
|----------|---------|
| Camera agitee → root motion fausse | Warning Gemini `camera.mouvement = agitee` + option desactivation root_position |
| Corps partiel → hallucination WHAM | Filtre Gemini `qualite_estimee < 0.6` → segment exclu automatiquement |
| Occlusions courtes (<10 frames) | Interpolation lineaire scipy interp1d (integre dans motus_extract.py) |
| SMPL 24 joints → R15 15 os | Mapping fixe defini dans le convertisseur |
| Blender headless lent | Template R15 pre-charge en .blend (inchange U-GAMMA) |
| Zoom video → root motion fausse | Warning Gemini `camera.zoom_detecte = true` |

---

## [HISTORIQUE DES VERSIONS]

| Version | Architecture | Statut |
|---------|-------------|--------|
| V1 | 3 Fregates + BVH | Abandonne |
| V2 | 2 Fregates + MediaPipe | Echoue (videos Roblox 3D incompatibles) |
| V3 | 2 Fregates + Gemini + WHAM | En cours — Phase 5 (Validation) |
