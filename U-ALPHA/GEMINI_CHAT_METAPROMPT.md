# GEMINI CHAT — META-PROMPT ANIMA-MECHANICUS U-ALPHA

> **Mode sans API** : colle ce prompt dans [gemini.google.com](https://gemini.google.com),
> uploade ta video `.mp4`, et recupere le JSON que la Cellule 4b utilisera directement.

---

## ETAPE 1 — Ce qu'il faut coller dans Gemini Chat

Uploade ta video dans le chat Gemini (bouton trombone / attache fichier),
puis colle **exactement** ce bloc de texte :

---

```
Tu es un analyseur de vidéo expert en motion capture humaine.
Analyse la vidéo que je t'envoie et retourne UNIQUEMENT un bloc JSON valide.
Ne mets aucun markdown, aucune explication, aucun texte avant ou après le JSON.
Commence ta réponse directement par { et termine par }.

FORMAT EXACT ATTENDU :
{
  "video_duration_seconds": <nombre décimal, durée totale en secondes>,
  "source_fps": <entier, images par seconde détectées>,
  "total_persons": <entier, nombre total de personnes dans la vidéo>,
  "persons": [
    {
      "person_id": <entier, commence à 1>,
      "segments_valides": [
        {
          "start_s": <décimal, début du segment en secondes>,
          "end_s": <décimal, fin du segment en secondes>,
          "corps_visible": "<EXACTEMENT l'une de ces valeurs : complet | tronc | tete_seulement>",
          "orientation": "<EXACTEMENT l'une de ces valeurs : face | profil | dos | mixte>",
          "distance_camera": "<EXACTEMENT l'une de ces valeurs : proche | moyen | lointain>",
          "type_mouvement": "<EXACTEMENT l'une de ces valeurs : marche | danse | combat | sport | statique | discussion>",
          "qualite_estimee": <décimal entre 0.0 et 1.0>,
          "problemes": [<liste de strings décrivant les problèmes, vide [] si aucun>],
          "extraction_possible": "<EXACTEMENT l'une de ces valeurs : corps_complet | haut_du_corps | tete_cou | aucune>",
          "modele_recommande": "<EXACTEMENT l'une de ces valeurs : WHAM | FrankMocap_upper | DECA | skip>"
        }
      ],
      "segments_exclus": [
        {
          "start_s": <décimal>,
          "end_s": <décimal>,
          "raison": "<EXACTEMENT l'une de ces valeurs : personne_absente | flou_mouvement | occlusion_totale | contre_jour | trop_court>"
        }
      ]
    }
  ],
  "camera": {
    "mouvement": "<EXACTEMENT l'une de ces valeurs : stable | panoramique | suivi | agitee>",
    "zoom_detecte": <true ou false>
  },
  "qualite_globale": "<EXACTEMENT l'une de ces valeurs : excellente | bonne | moyenne | mauvaise>",
  "recommandation": "<string, conseils pour améliorer la capture ou commentaire général>"
}

RÈGLES D'ÉVALUATION OBLIGATOIRES :

--- VISIBILITÉ DU CORPS ---
- corps_visible "complet"        = tête + torse + bras + jambes tous visibles simultanément
- corps_visible "tronc"          = tête + torse + bras visibles, jambes absentes ou partiellement coupées
- corps_visible "tete_seulement" = seule la tête ou le buste sans bras est visible (plan serré visage/épaules)

--- EXTRACTION POSSIBLE ET MODELE RECOMMANDE ---
Ces deux champs se déduisent directement de corps_visible + qualite_estimee :

| corps_visible    | qualite_estimee | extraction_possible | modele_recommande |
|------------------|-----------------|---------------------|-------------------|
| complet          | >= 0.6          | corps_complet       | WHAM              |
| tronc            | >= 0.6          | haut_du_corps       | FrankMocap_upper  |
| tete_seulement   | >= 0.5          | tete_cou            | DECA              |
| n'importe lequel | < seuil         | aucune              | skip              |

- Un segment avec extraction_possible "haut_du_corps" ou "tete_cou" est VALIDE et utile — ne pas le mettre dans segments_exclus
- segments_exclus est réservé aux cas où la personne est vraiment absente ou le corps totalement caché

--- QUALITE ---
- qualite_estimee 1.0  = corps complet, face caméra, lumière parfaite, sans flou, sans occlusion
- qualite_estimee 0.6  = seuil minimum pour corps_complet et tronc
- qualite_estimee 0.5  = seuil minimum pour tete_seulement
- qualite_estimee < seuil → extraction_possible "aucune" + modele_recommande "skip"

--- SEGMENTATION ---
- Découper en segments précis si la qualité ou la visibilité varie
- Les timestamps doivent couvrir toute la durée (pas de trous non justifiés)
- Si une personne n'est pas visible sur un passage → segment_exclu raison "personne_absente"
- Si la vidéo alterne entre plan large et plan serré → segments distincts par type
- Durée minimale d'un segment_valide : 1.5 secondes. En dessous → fusionner ou segments_exclus raison "trop_court"

--- AUTRES RÈGLES ---
- Créer un person_id distinct (1, 2, 3...) pour chaque personne présente
- zoom_detecte = true uniquement si la caméra zoome ou dé-zoome visiblement
- type_mouvement "discussion" = personne qui parle avec gestes, peu de déplacement

IMPORTANT :
- Retourne UNIQUEMENT le JSON brut. Pas de ```json```. Pas d'explication. Juste le JSON.
- Respecte EXACTEMENT les valeurs d'enum listées (minuscules, underscores comme indiqué).
- Un gros plan visage dans une vidéo de discussion est NORMAL → extraction_possible "tete_cou", pas exclu.
```

---

## ETAPE 2 — Injecter le JSON dans le notebook (Cellule 4b)

Une fois Gemini t'a retourne le JSON :

1. Dans le notebook Colab, **execute la Cellule 4b**
2. Une zone de texte apparait
3. **Colle le JSON complet** dans la zone
4. Clique **"Valider et injecter"**

Alpha va automatiquement :
- Valider le JSON
- Afficher le routing par segment (WHAM / FrankMocap / DECA / skip)
- Sauvegarder le cache
- Preparer `ALL_SEGMENTS` pour la suite

5. **Passe directement a la Cellule 6** — Cellule 5 est ignoree automatiquement

---

## Resume du workflow

```
1. gemini.google.com
   └─ Upload video + colle le prompt ci-dessus
   └─ Copie le JSON retourne

2. Colab — Cellule 4b
   └─ Colle le JSON → Valider et injecter
   └─ Routing affiche → ALL_SEGMENTS pret

3. Colab — Cellules 6, 7, 8
   └─ Validation → WHAM → Telechargement .npz
```

> La cle API Gemini n'est pas requise en mode chat.
> Tu peux laisser `GEMINI_API_KEY = ""` dans la Cellule 2.

---

## Tableau de routing

| extraction_possible | modele_recommande | Ce qu'Alpha fait                     |
|---------------------|-------------------|--------------------------------------|
| corps_complet       | WHAM              | Squelette complet (tous les os)      |
| haut_du_corps       | FrankMocap_upper  | Non implemente — segment en attente  |
| tete_cou            | DECA              | Non implemente — segment en attente  |
| aucune              | skip              | Segment ignore                       |

> FrankMocap et DECA seront implements dans une version future.
> Pour l'instant, seuls les segments WHAM (`corps_complet`) sont traites.

---

## Notes importantes

- **Taille video** : Gemini Chat accepte les videos jusqu'a ~1 Go
- **Qualite de l'analyse** : Gemini 2.5 Pro dans le chat est plus puissant que les modeles API free tier
- **Reutilisation** : le cache est lie a la video par MD5. Meme video = cache reutilise automatiquement
- **Plusieurs videos** : repete les etapes 1-2 pour chaque video
