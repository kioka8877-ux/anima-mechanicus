# ANIMA-MECHANICUS — PHYLACTERE DE RESURRECTION

| Champ | Valeur |
|-------|--------|
| STATUS | Phase 6 — Code termine, re-validation Colab en attente |
| DATE | Avril 2026 |
| ARCHITECTURE | 2 Fregates (U-ALPHA + U-GAMMA) |
| VERSION | V4 — Option D : Gemini + GVHMR + SMPL→R15 |
| CONTRAT .npz | Inchange : rotations (N,15,4) + root_position (N,3) + metadata |
| CONTRAT JSON GEMINI | Inchange — defini et documente dans README |

---

## [LAST_WORK]

### Phase 6 complete (code) — Avril 2026

**6.1 motus_extract.py — DONE**
- `run_wham()` remplace par `run_gvhmr()` (appel subprocess demo.py GVHMR)
- Lecture sortie GVHMR adaptee (format pkl/npz → meme contrat passes/transl)
- Cas `FrankMocap_upper` implemente : masquage joints inferieurs (LeftUpperLeg, LeftLowerLeg, LeftFoot, RightUpperLeg, RightLowerLeg, RightFoot → quaternion identite)
- References WHAM/mmcv/detectron2 supprimees
- Routing automatique selon `modele_recommande` Gemini : GVHMR / FrankMocap_upper / DECA / skip

**6.2 ANIMA_MECHANICUS_ALPHA.ipynb — DONE**
- Cellule 1 refaite : GVHMR via git clone + pip + aria2c HuggingFace (camenduru/GVHMR)
- Installation mmcv/detectron2/WHAM supprimee
- Cellules 2 a 7 : INCHANGEES

**6.3 Documentation — DONE (README + STATE + ROADMAP)**
- README mis a jour : GVHMR, stack YOLOv8/ViTPose-H/HMR2.0a, routing Gemini, historique V1→V4
- PRD reste a mettre a jour (stack technique step 2)

### Contexte migration (conserve pour reference)
- WHAM bloque : mmcv-full 1.3.9 echec build, detectron2 wheel torch2.10 indisponible
- GVHMR (SIGGRAPH Asia 2024) : zero mmcv, zero detectron2, poids HuggingFace via aria2c
- `smpl_to_r15()`, motus_forge.py v4, U-GAMMA : INCHANGES

---

## [NEXT_TASK]

Phase 6.4 — Re-validation sur Colab T4 reel :

1. Ouvrir `ANIMA_MECHANICUS_ALPHA.ipynb` sur Google Colab (GPU T4)
2. Executer Cellule 1 : verifier installation GVHMR sans erreur
3. Re-executer tous les tests de Phase 5 avec la nouvelle stack GVHMR :
   - Test "Danse" — 1 personne, corps complet
   - Test "Combat" — 2 personnes, occlusions
   - Test "Camera instable" — warning root motion
4. Valider .npz : shapes (N,15,4), (N,3), pas de NaN
5. Valider .fbx forge sans erreur Blender + import Roblox Studio
6. Mettre a jour PRD.md (stack technique pipeline step 2) — derniere doc en attente

---

## [BLOCKERS]

- Aucun blocker actif sur le code
- Re-validation Phase 6.4 requiert session Colab T4 avec GPU disponible
- Modeles SMPL/SMPL-X : disponibles sur HuggingFace sans inscription (telecharges automatiquement via aria2c au runtime)

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
| V4 | 2 Fregates + Gemini + GVHMR | Phase 6 — Code termine, re-validation en attente |

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
