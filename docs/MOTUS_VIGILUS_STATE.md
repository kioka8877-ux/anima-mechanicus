# ANIMA-MECHANICUS — PHYLACTERE DE RESURRECTION

| Champ | Valeur |
|-------|--------|
| STATUS | Phase 6 — En cours (Migration GVHMR) |
| DATE | Avril 2026 |
| ARCHITECTURE | 2 Fregates (U-ALPHA + U-GAMMA) |
| VERSION | V4 — Option D : Gemini + GVHMR + SMPL→R15 |
| CONTRAT .npz | Inchange : rotations (N,15,4) + root_position (N,3) + metadata |
| CONTRAT JSON GEMINI | Inchange — defini et documente dans README |

---

## [LAST_WORK]

### Decision de migration : WHAM → GVHMR (Avril 2026)

Phase 5 (Validation imperiale) a revele que WHAM est impossible a installer sur Colab T4 actuel :
- `mmcv-full 1.3.9` : echec de build systematique (subprocess-exited-with-error)
- `detectron2` : wheel torch2.10 indisponible, compilation source echoue
- `ViTPose` : 2.4 GB de poids, dependant de mmcv

**Solution retenue : GVHMR (SIGGRAPH Asia 2024)**
- Notebook Colab officiel confirme fonctionnel sur Tesla T4
- Zero mmcv, zero detectron2
- Detection via YOLO (deja present dans le projet)
- Poids sur HuggingFace (telechargement automatique via aria2c)
- Installation : 2 commandes pip uniquement
- Sortie : parametres SMPL world coordinates — meme format que WHAM
- `smpl_to_r15()` dans motus_extract.py : INCHANGE

### Travaux anterieurs (toujours valides)
- motus_extract.py v2 : pipeline Gemini + SMPL→R15 — CONSERVE, seul run_wham() a remplacer
- ANIMA_MECHANICUS_ALPHA.ipynb : Cellule 1 (installation) a refaire, reste intact
- motus_forge.py v4 : INTACT — non impacte par la migration
- ANIMA_MECHANICUS_GAMMA.ipynb : INTACT — non impacte

---

## [NEXT_TASK]

Phase 6 — Migration GVHMR :

1. Remplacer `run_wham()` par `run_gvhmr()` dans motus_extract.py
2. Adapter la lecture de sortie GVHMR (format pkl/npz GVHMR vs WHAM)
3. Implementer le cas `FrankMocap_upper` : masquage joints inferieurs quand Gemini detecte tronc seulement
4. Refaire Cellule 1 de ANIMA_MECHANICUS_ALPHA.ipynb (installation GVHMR)
5. Mettre a jour les checkpoints telecharges (GVHMR vs WHAM)
6. Re-valider Phase 5 avec la nouvelle stack

---

## [BLOCKERS]

- Modeles SMPL/SMPL-X necessitent inscription gratuite
  - SMPL_NEUTRAL.pkl → disponible sur HuggingFace (camenduru/SMPLer-X) sans inscription
  - SMPLX_NEUTRAL.npz → disponible sur HuggingFace (camenduru/GVHMR) sans inscription
  - NOTE : le notebook GVHMR officiel telecharge tout via aria2c depuis HuggingFace automatiquement
- `FrankMocap_upper` non implemente → migration est l'occasion de l'implementer proprement

---

## [SOLUTIONS]

| Probleme | Solution |
|----------|---------|
| WHAM mmcv/detectron2 echec Colab | Migration vers GVHMR (zero mmcv/detectron2) |
| Camera agitee → root motion fausse | Warning Gemini `camera.mouvement = agitee` + option desactivation root_position |
| Corps partiel → hallucination modele | Filtre Gemini `qualite_estimee < 0.6` → exclu + masquage joints inferieurs si `FrankMocap_upper` |
| Occlusions courtes (<10 frames) | Interpolation lineaire scipy interp1d |
| SMPL 24 joints → R15 15 os | Mapping fixe dans motus_extract.py (INCHANGE) |
| r15_template.blend manquant | Rig R15 genere programmatiquement dans motus_forge.py v4 |
| SMPL body models sans inscription | Disponibles sur HuggingFace via aria2c (notebook GVHMR officiel) |

---

## [HISTORIQUE DES VERSIONS]

| Version | Architecture | Statut |
|---------|-------------|--------|
| V1 | 3 Fregates + BVH | Abandonne |
| V2 | 2 Fregates + MediaPipe | Echoue (videos Roblox 3D incompatibles) |
| V3 | 2 Fregates + Gemini + WHAM | Bloque — WHAM impossible sur Colab T4 actuel |
| V4 | 2 Fregates + Gemini + GVHMR | Phase 6 — Migration en cours |

---

## [REFERENCE GVHMR]

| Element | Detail |
|---------|--------|
| Repo | https://github.com/zju3dv/GVHMR |
| Notebook Colab officiel | https://colab.research.google.com/drive/1N9WSchizHv2bfQqkE9Wuiegw_OT7mtGj |
| Publication | SIGGRAPH Asia 2024 |
| Detection | YOLOv8 (ultralytics) |
| Pose 2D | ViTPose-H |
| Poids | HuggingFace (camenduru/GVHMR) |
| CUDA requis | 12.1 (cu121) — compatible Colab T4 |
