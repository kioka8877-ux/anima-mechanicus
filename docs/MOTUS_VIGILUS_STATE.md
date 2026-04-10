# 📜 MOTUS-VIGILUS — PHYLACTÈRE DE RÉSURRECTION

| Champ | Valeur |
|-------|--------|
| STATUS | Phase 1 (Fondations) — EN COURS |
| DATE | Mars 2026 |
| ARCHITECTURE | 2 Frégates (U-ALPHA + U-GAMMA) |
| CONTRAT | Fichier .npz (rotations + root_position + metadata) |

## [LAST_WORK]
- Structure repo créée
- Template R15 Blender préparé
- Documentation technique rédigée (PRD, Roadmap)

## [NEXT_TASK]
- Frégate U-ALPHA : motus_extract.py (MediaPipe → .npz)

## [BLOCKERS]
- Aucun

## [SOLUTIONS]
- Si MediaPipe Z-depth imprécis → Normalisation par hip_center
- Si Blender headless lent → Template R15 pré-chargé en .blend
- Si occlusions >10 frames → Interpolation scipy interp1d

## [VERSION]
- Architecture V2 (2 Frégates)
- Précédent : V1 (3 Frégates + BVH) — ABANDONNÉ
