# GEMINI CHAT — META-PROMPT ANIMA-MECHANICUS U-ALPHA

> **Mode sans API** : colle ce prompt dans [gemini.google.com](https://gemini.google.com),
> uploade ta video `.mp4`, et recupere le JSON que Alpha utilisera directement.

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
- corps_visible "complet"       = tête + torse + bras + jambes tous visibles simultanément
- corps_visible "tronc"         = tête + torse + bras visibles, jambes absentes ou partiellement coupées
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
- qualite_estimee 0.5  = seuil minimum pour tete_seulement (les modèles de tête tolèrent les gros plans)
- qualite_estimee < seuil → extraction_possible "aucune" + modele_recommande "skip"

--- SEGMENTATION ---
- Découper en segments précis : ne pas mettre un seul segment couvrant toute la durée si la qualité ou la visibilité varie
- Les timestamps doivent couvrir toute la durée de la vidéo (pas de trous non justifiés)
- Si une personne n'est pas visible sur un passage, créer un segment_exclu avec raison "personne_absente"
- Si la vidéo alterne entre plan large (corps complet) et plan serré (tête seulement), créer des segments distincts pour chaque type
- Durée minimale d'un segment_valide : 1.5 secondes. En dessous de 1.5s → fusionner avec le segment adjacent de même type, ou mettre dans segments_exclus avec raison "trop_court"
- "trop_court" est une valeur valide pour le champ "raison" des segments_exclus

--- AUTRES RÈGLES ---
- Créer un person_id distinct (1, 2, 3...) pour chaque personne présente dans la vidéo
- zoom_detecte = true uniquement si la caméra zoome ou dé-zoome visiblement pendant la vidéo
- Si une personne a qualite_estimee >= seuil sur toute la durée, mettre segments_exclus = []
- type_mouvement "discussion" = personne debout ou assise qui parle, gestes de conversation, peu de déplacement

IMPORTANT :
- Retourne UNIQUEMENT le JSON brut. Pas de ```json```. Pas d'explication. Juste le JSON.
- Respecte EXACTEMENT les valeurs d'enum listées (minuscules, underscores comme indiqué).
- Si tu ne sais pas exactement, estime plutôt que d'omettre un champ.
- Un gros plan visage dans une vidéo de discussion est NORMAL et doit produire extraction_possible "tete_cou", pas être exclu.
```

---

## ETAPE 2 — Recuperer le nom du fichier cache

Une fois que Gemini t'a donne le JSON, il faut le sauvegarder avec le bon nom
pour qu'Alpha le detecte automatiquement.

**Dans Colab, avant de lancer la Cellule 5, execute cette cellule temporaire :**

```python
# ══ Cellule temporaire : trouver le nom du fichier cache ══
import hashlib, os
with open(VIDEO_PATH, "rb") as f:
    chunk = f.read(1024 * 1024)  # premier Mo
h = hashlib.md5(chunk).hexdigest()[:10]
stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
cache_filename = f"gemini_cache_{stem}_{h}.json"
print(f"Nom du fichier : {cache_filename}")
print(f"Chemin complet : /content/gemini_cache/{cache_filename}")
```

Ce code t'affichera quelque chose comme :
```
Nom du fichier : gemini_cache_ma_video_a3f9b12c4d.json
Chemin complet : /content/gemini_cache/gemini_cache_ma_video_a3f9b12c4d.json
```

---

## ETAPE 3 — Sauvegarder le JSON dans le cache

Dans Colab, execute cette cellule en remplacant `TON_JSON_ICI` par le JSON
copie depuis Gemini Chat :

```python
# ══ Cellule temporaire : sauvegarder le JSON Gemini ══
import json, os

# Colle le JSON complet fourni par Gemini Chat entre les triples guillemets
JSON_GEMINI = """
{
  "video_duration_seconds": ...,
  "source_fps": ...,
  ...
}
"""

# Creer le dossier cache si besoin
os.makedirs("/content/gemini_cache", exist_ok=True)

# Valider que le JSON est bien forme
data = json.loads(JSON_GEMINI.strip())
print(f"  JSON valide : {len(data['persons'])} personne(s) detectee(s)")
print(f"  Duree : {data['video_duration_seconds']}s | Qualite : {data['qualite_globale']}")

# Afficher le routing par segment
for p in data['persons']:
    print(f"\n  Personne {p['person_id']} :")
    for seg in p['segments_valides']:
        print(f"    [{seg['start_s']}s → {seg['end_s']}s] "
              f"corps={seg['corps_visible']} | "
              f"extraction={seg['extraction_possible']} | "
              f"modele={seg['modele_recommande']}")

# Sauvegarder avec le bon nom de fichier cache
import hashlib
with open(VIDEO_PATH, "rb") as f:
    chunk = f.read(1024 * 1024)
h = hashlib.md5(chunk).hexdigest()[:10]
stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
cache_path = f"/content/gemini_cache/gemini_cache_{stem}_{h}.json"

with open(cache_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n  Cache sauvegarde : {cache_path}")
print("  Lance maintenant la Cellule 5 — Gemini ne sera PAS appele (0 quota consomme)")
```

---

## ETAPE 4 — Lancer la Cellule 5 normalement

Une fois le JSON sauvegarde dans le cache, lance la Cellule 5 comme d'habitude.
Alpha detectera le fichier cache et affichera :

```
[U-ALPHA][Gemini] Cache hit — 0 requete consommee
  (Supprimer /content/gemini_cache/gemini_cache_xxx.json pour forcer une re-analyse)
```

**La cle API n'est plus necessaire pour cette video.** Tu peux laisser `GEMINI_API_KEY = ""`
dans la Cellule 2 si tu utilises uniquement le mode cache.

---

## Resume du workflow

```
1. gemini.google.com
   └─ Upload video + colle le prompt ci-dessus
   └─ Copie le JSON retourne

2. Colab : execute la cellule "sauvegarder le JSON"
   └─ JSON valide → affiche le routing par segment
   └─ Sauvegarde automatique avec le bon nom de cache

3. Colab : lance Cellule 5
   └─ Alpha lit le cache → routing automatique par segment → pipeline adapte
```

---

## Tableau de routing — ce qu'Alpha fera selon le JSON

| extraction_possible | modele_recommande | FBX produit                          |
|---------------------|-------------------|--------------------------------------|
| corps_complet       | WHAM              | Squelette complet (tous les os)      |
| haut_du_corps       | FrankMocap_upper  | Torse + bras animes, jambes T-pose   |
| tete_cou            | DECA              | Rotation tete + cou uniquement       |
| aucune              | skip              | Segment ignore                       |

---

## Notes importantes

- **Taille video** : Gemini Chat accepte les videos jusqu'a ~1 Go en general
- **Qualite de l'analyse** : Gemini 2.5 Pro dans le chat est plus puissant que les modeles free tier API
- **Videos de discussion** : les gros plans visage sont normaux et produisent des segments `tete_cou` utiles — ils ne sont pas exclus
- **Reutilisation** : le cache est lie a la video par MD5. Si tu re-uploades la meme video,
  le cache est reutilise automatiquement. Si tu modifies la video, il faut regenerer le cache.
- **Plusieurs videos** : repete les etapes 1-3 pour chaque video. Les fichiers cache s'accumulent
  dans `/content/gemini_cache/` sans conflit.

