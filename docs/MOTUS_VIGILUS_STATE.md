# ANIMA-MECHANICUS — PHYLACTERE DE RESURRECTION

| Champ | Valeur |
|-------|--------|
| STATUS | Phase 5 — En cours (Validation imperiale) |
| DATE | Avril 2026 |
| ARCHITECTURE | 2 Fregates (U-ALPHA + U-GAMMA) |
| VERSION | V3 — Option C : Gemini 2.0 Flash + WHAM + SMPL→R15 |
| CONTRAT .npz | Inchange : rotations (N,15,4) + root_position (N,3) + metadata |
| CONTRAT JSON GEMINI | Defini et documente dans README |

---

## [LAST_WORK]

- Projet renomme ANIMA-MECHANICUS (WH40K lore)
- Architecture pivot : MediaPipe abandonne → Gemini 2.0 Flash + WHAM + SMPL→R15
- motus_extract.py v2 : COMPLETE — pipeline Gemini + WHAM + SMPL→R15
- ANIMA_MECHANICUS_ALPHA.ipynb : COMPLETE — notebook Colab 7 cellules
- motus_forge.py v4 : COMPLETE — rig R15 genere programmatiquement (plus de r15_template.blend)
- ANIMA_MECHANICUS_GAMMA.ipynb : COMPLETE — notebook Colab 4 cellules, sans template
- Toutes les dependances externes eliminees (pipeline 100% autonome)

---

## [NEXT_TASK]

Phase 5 — Validation imperiale :

1. Ouvrir ANIMA_MECHANICUS_ALPHA.ipynb sur Colab T4
2. Test "Danse" — 1 personne, corps complet face camera
3. Verifier les .npz exportes (shapes, valeurs)
4. Ouvrir ANIMA_MECHANICUS_GAMMA.ipynb
5. Forger les .fbx
6. Importer dans Roblox Studio — valider que l'animation joue correctement

---

## [BLOCKERS]

- Modeles SMPL (body_models/) necessitent inscription gratuite sur mpg.de
  → Procedure documentee dans ANIMA_MECHANICUS_ALPHA.ipynb Cellule 1
- WHAM necessite ffmpeg + torch + detectron2 + mmcv sur Colab T4
  → Cellule d'installation complete dans ANIMA_MECHANICUS_ALPHA.ipynb

---

## [SOLUTIONS]

| Probleme | Solution |
|----------|---------|
| Camera agitee → root motion fausse | Warning Gemini `camera.mouvement = agitee` + option desactivation root_position |
| Corps partiel → hallucination WHAM | Filtre Gemini `qualite_estimee < 0.6` → segment exclu automatiquement |
| Occlusions courtes (<10 frames) | Interpolation lineaire scipy interp1d |
| SMPL 24 joints → R15 15 os | Mapping fixe dans motus_extract.py |
| r15_template.blend manquant | Rig R15 genere programmatiquement dans motus_forge.py v4 |
| Blender headless lent | Template supprime → demarrage plus rapide |

---

## [HISTORIQUE DES VERSIONS]

| Version | Architecture | Statut |
|---------|-------------|--------|
| V1 | 3 Fregates + BVH | Abandonne |
| V2 | 2 Fregates + MediaPipe | Echoue (videos Roblox 3D incompatibles) |
| V3 | 2 Fregates + Gemini + WHAM | Phase 5 — Validation |
