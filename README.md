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
    - `sentence_transformers`
 
Commandes à exécuter pour l'installation des bibliothèques :
```text
pip install streamlit langchain langgraph langchain-google-genai pydantic pyvis spacy python-dotenv sentence-transformers

python -m spacy download fr_core_news_md
```

### 2. L'Intelligence Artificielle utilisée

* **Le Modèle :** **Gemini** (plus précisément `gemini-3.1-flash-lite`)

* **La Clé API :** Pour des raisons de sécurité, il n'y a pas de clé API présente dans le code source. Vous devez configurer une variable d'environnement en local.

À la racine du projet, juste après le `=` et sans mettre d'espace, copiez-collez votre clé API Google dans le fichier `.env`. Le fichier `.env` doit avoir ce format à la fin :

```text
Google_API="VOTRE_CLE_API_GOOGLE_ICI"
```

### 3. L'Exécution du code

Pour exécuter le code et ouvrir l'interface `Streamlit`, tapez dans un terminal la commande :

```text
streamlit run projet16_graphe_src.py
```

Si ça ne fonctionne pas, tapez cette commande : 

```text
python -m streamlit run projet16_graphe_src.py
```

Une fois sur l'interface web, téléversez un fichier au format Markdown via le chargeur de fichier.

<br>
<br>
<br>

*Projet réalisé par : Akram Bougoffa, Haïtem El Harras, Cyril Labeye, Hannah Laïb, Rémy Maurice-Demourioux*
