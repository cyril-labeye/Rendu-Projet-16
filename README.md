# Projet 16 : *Création automatique de graphes de relations à partir d’un rapport de mission*
----------------------------------------------------------------

Ce projet est une application web permettant d'extraire automatiquement des entités et des relations depuis un texte Markdown pour générer un graphe interactif.

### 1. Les Prérequis et Bibliothèques

* **Langage :** Python
* **Les bibliothèques principales à installer :**
    - `streamlit` (pour l'interface web)
    - `langchain`, `langgraph`, `langchain-google-genai` (pour l'utilisation de l'IA)
    - `pydantic` (pour la structuration des données)
    - `pyvis` (pour la cartographie du graphe)
    - `spacy` (pour le traitement du langage naturel)
    - `python-dotenv`
    - `sentence_transformers` (pour le module de vérification)
 
Commandes à exécuter pour l'installation des bibliothèques :
```text
pip install streamlit langchain langgraph langchain-google-genai pydantic pyvis spacy python-dotenv sentence-transformers
```
```text
python -m spacy download fr_core_news_md
```

### 2. L'Intelligence Artificielle utilisée

* **Le Modèle :** **Gemini** (plus précisément `gemini-3.1-flash-lite`)

* **La Clé API :** Pour des raisons de sécurité, il n'y a pas de clé API présente dans le code source. Vous devez configurer une variable d'environnement en local.

À la racine du projet, dans le fichier `.env`, juste après le `=` et sans mettre d'espace, copiez-collez votre clé API Google . Le fichier `.env` doit avoir ce format à la fin :

```text
Google_API="VOTRE_CLE_API_GOOGLE_ICI"
```

### 3. L'Exécution du code (l'outil d'extraction, le programme principal)

Pour exécuter l'outil d'extraction et ouvrir l'interface `Streamlit`, tapez dans un terminal la commande :

```text
streamlit run projet16_graphe_src.py
```

Si ça ne fonctionne pas, tapez cette commande : 

```text
python -m streamlit run projet16_graphe_src.py
```

Une fois sur l'interface web, téléversez un fichier au format Markdown via le chargeur de fichier.

### 4. Le Module de Vérification (Programme de test)

Ce projet inclut un programme de test (`module_verification.py`). Il permet de tester la précision de notre outil d'extraction en générant des graphes de relations aléatoires, en les bruitant avec du texte hors sujet, puis en comparant les résultats de l'IA avec les graphes de relations initiaux.

Pour lancer une série de tests, exécutez cette commande dans votre terminal :

```bash
python module_verification.py
```

Lors de l'exécution, le script vous demandera :
1. Le **nombre de sommets** souhaité pour le graphe de départ.
2. Le **nombre de tests (boucles)** à effectuer.

Le script calculera les taux de Faux Positifs (inventions) et de Faux Négatifs (oublis) pour les entités et les relations. Un bilan de précision globale sera affiché dans la console et l'historique sera sauvegardé automatiquement dans le fichier `historique_tests.csv`.

<br>
<br>
<br>

*Projet réalisé par : Akram Bougoffa, Haïtem El Harras, Cyril Labeye, Hannah Laïb, Rémy Maurice-Demourioux*
