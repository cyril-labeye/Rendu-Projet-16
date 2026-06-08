import os
import json
import itertools
import unicodedata
from math import log2, floor
from typing import Annotated, List

import spacy
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from pyvis.network import Network
from pydantic import BaseModel, Field, AfterValidator, ValidationError

#Imports Langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader

#Imports LangGraph
from langgraph.graph import START, END, StateGraph, MessagesState

load_dotenv()
Google_API=os.getenv("Google_API")
model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite",
                             temperature=0,
                             google_api_key=Google_API)

def enlever_accents(texte):
    """Retire les accents d'une chaîne de caractères."""
    # Sépare la lettre de son accent
    texte_normalise = unicodedata.normalize('NFD', texte)
    # Garde uniquement la lettre de base
    return ''.join(c for c in texte_normalise if unicodedata.category(c) != 'Mn')

def forcer_format_nom(nom: str):
    """Fonction qui prend n'importe quel nom, le met en minuscule, trie les mots par ordre alphabétique, 
    puis remet des majuscules au debut du nom et du prenom.
    
    Entrées :
    =========
    nom (str) : Nom Prenom de l'entite

    Sortie :
    ========
    nom_normalise qui est le nom normalise"""
    mots = nom.lower().split()
    mots.sort()
    nom_normalise = " ".join([m.capitalize() for m in mots])
    return nom_normalise

def supprimer_doublons(l : list):
    vus = []
    resultat = []
    for element in l:
        cle = forcer_format_nom(element)
        if cle not in vus:
            vus.append(cle)
            resultat.append(element)
    return resultat

def supprimer_doublons_liens(liste_liens : list):
    """Fonction qui supprime les doublons d'une liste de liens
    Entrées :
    =========
    liste_liens : une liste de liens

    Sortie :
    ========
    liste_liens_nettoyee qui est la liste de liens avec les doublons retirés"""

    liste_liens_nettoyee=[]
    for lien in liste_liens:
        if lien not in liste_liens_nettoyee:
            liste_liens_nettoyee.append(lien)
    return liste_liens_nettoyee


class Lien(BaseModel):
    source: str = Field(description="Exact ID of the source entity. Must strictly match the 'id' attribute of a previously defined object in the 'entities' list to guarantee referential integrity.")
    target: str = Field(description="Exact ID of the target entity. Must strictly match the 'id' attribute of a previously defined object in the 'entities' list.")
    relation: str = Field(description="Descriptive label of the connection. Mandatory format: lowercase snake_case, purely ASCII with no accents or special characters (e.g., 'ceo_of', 'majority_shareholder', 'associated_with').")
    gravite: int = Field(ge=1, le=5, description="Criticality level of the interaction (1 = weak, indirect, or historical link; 5 = direct involvement, financial tie, or strong hierarchical control).")

class EntiteeNormalisee(BaseModel):
    id:str=Field(description="Unique and deterministic primary key. Mandatory format: strict snake_case (lowercase, spaces replaced by underscores), purely ASCII (e.g., 'enron_corp').")
    type:str=Field(description="Strict business categorization of the organization. Restricted values: 'company', 'institution', or 'group'.")
    name: str = Field(description="Full legal or official name, normalized to pure ASCII characters (stripped of any special typography).")
    Alias:Annotated[list[str], Field(default=[], description="Exhaustive and strictly deduplicated list of acronyms, merged subsidiaries, or alternative commercial names.")]

class PersonneNormalisee(BaseModel):
    id:str=Field(description="Unique and deterministic primary key. Mandatory format: strict snake_case (first_name_last_name), purely ASCII (e.g., 'jeffrey_skilling').")
    type:str=Field(default="person", description="Strict entity categorization. Static and mandatory literal value: 'person'.")
    name:str=Field(description="Normalized full legal name, converted to pure ASCII characters (no accented characters).")
    Alias:Annotated[list[str], Field(default=[], description="Exhaustive and strictly deduplicated list of identified pseudonyms, communication aliases, or diminutives." )]

class JSON_Normalise(BaseModel):
    entities: list[EntiteeNormalisee | PersonneNormalisee] = Field(description="Root collection (graph nodes) consolidating all extracted entities and persons. Must undergo absolute entity resolution: no real-world identity may appear as a duplicate.")
    Link: Annotated[list[Lien], Field(description="Collection of relationships (graph edges) connecting the entities. Must imperatively include bidirectional relationships (mirror links) inferred from the analysis."),AfterValidator(supprimer_doublons_liens)]

class GraphState(MessagesState):

   pages:list
   index:int
   Json_f:list
   tentatives_suggestions:int
   tentatives_fatales:int
nlp=spacy.load("fr_core_news_md")



parser_normalisation = PydanticOutputParser(pydantic_object=JSON_Normalise)

def text_splitter(nom_file):
   """
   divise un fichier en texte en plein de petit morceau
   """

   loader = UnstructuredMarkdownLoader(nom_file)
   doc=loader.load()
   decoupage=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=800,length_function=len)
   pages=decoupage.split_documents(doc)

   return pages

def text(text):
   """Transforme un texte dans un format json
      bien précis"""  
   template= """
    Context: I am conducting a forensic network analysis based on data extracted from various intelligence documents. The current raw dataset consists of fragmented, concatenated JSON objects. This data contains duplicates, inconsistent naming conventions, and non-standard characters that completely break Knowledge Graph ingestion pipelines. 

    Role: You are a Senior Forensic Data Engineer and Graph Architect. Your expertise lies in deterministic entity resolution, exhaustive connection mapping, and outputting strictly typed data that perfectly matches programmatic schemas.

    Action: Execute a rigorous audit, normalization, and relational mapping on the Target Data using the following logic:

    1. Normalization & Cleaning: Apply strict text normalization. Convert all special/accented characters to their bare ASCII equivalents (e.g., 'é' becomes 'e', 'ç' becomes 'c'). Trim trailing/leading whitespace and standardize string casing.
    2. Entity Resolution: Identify and merge duplicate entries that represent the same individual or organization, even across different keys. Create exactly one master record per unique entity.
    3. Attribute Consolidation: Merge and deduplicate all identified aliases into a single `nicknames` array for each entity. Drop null or empty arrays.
    4. Zero-Drop Relationship Extraction (CRITICAL): You are under a strict zero-drop policy. You must extract EVERY single relationship, interaction, or shared attribute mentioned or implied in the dataset. Scan the input systematically, entry by entry, ensuring no connection is left behind. Missing a link is considered a critical system failure.
    5. Relational Mapping (Graph Readiness):
    - Explicit & Implicit Links: Capture direct titles (e.g., CEO, subsidiary) AND implied connections (e.g., shared addresses, co-attendance, financial transactions).
    - Conciseness: Standardize relationship labels into concise, lowercase snake_case formats (e.g., "associate_of", "relative_of", "investor_in").
    - Bi-directional Integrity: Every relationship must be reciprocal. If Entity A links to Entity B, you must programmatically infer and explicitly write the mirror link from Entity B to Entity A.
    - Uniqueness: Strip all redundant or duplicate relationship objects from the final arrays.

    Format: Your output must perfectly conform to the provided JSON Schema. Return strictly the final JSON document and absolutely nothing else. Do not include preambles, post-analysis, conversational text, or markdown formatting blocks (do NOT use ```json or ```).

    Target Data: 
    {input_text}

    JSON Schema:
    {json_schema}
   """
   sortiejson=model.with_structured_output(JSON_Normalise)
   prompt = ChatPromptTemplate.from_template(template)
   chain=prompt|sortiejson
   
   return  chain.invoke({"input_text":text,"json_schema":JSON_Normalise})

def RepIA(PDF,REP,PRECJ):
   """
   Fonction qui corrige un resultat renvoyé par l'IA
   """

   template="""
   Context: We are in the final data validation phase of a high-stakes forensic audit. The integrity of our extracted data is paramount, as this JSON dataset feeds directly into an automated Knowledge Graph visualization pipeline. Any hallucinated nodes, orphaned edges, or structural deviations will cause fatal errors in the downstream compilation.

    Role: You are a Lead Forensic Data Auditor. You are renowned for your meticulous attention to detail, your zero-tolerance policy for data hallucinations, and your absolute adherence to programmatic constraints.

    Action: Execute a rigorous reconciliation of the Draft JSON against the Original Source Document, using the Validator's Critique as your mandatory correction punch-list.
        1. Ground Truth Enforcement: The Source Document is the absolute truth. If the Draft JSON contains entities, relations, or aliases not explicitly supported by the Source, you must delete them.
        2. Critique Resolution: Address every single failure point listed in the Critique. Remove invalid elements and add missing links exactly as requested, provided they align with the source text.
        3. Structural Integrity: You must preserve the exact hierarchical structure, naming conventions (e.g., strict snake_case for IDs), and data types of the original JSON. Do not alter the schema.

    Format: Your output must be pure, machine-readable JSON. Return strictly the final, corrected JSON document and absolutely nothing else. Do not include preambles, post-analysis, conversational text, or markdown formatting blocks (do NOT use ```json or ```).

    Target Data:

    [ORIGINAL SOURCE DOCUMENT]
    {source}

    [VALIDATOR CRITIQUE]
    {critique}

    [DRAFT JSON TO CORRECT]
    {PrecRes}"""
   prompt=ChatPromptTemplate.from_template(template)
   chain=prompt|model|StrOutputParser()

   return  chain.invoke({"source":PDF,"critique":REP,"PrecRes":PRECJ})

def generateur(state:GraphState):
   index=state.get("index",0)
   pages=state["pages"]
   texte_actuel=pages[index].page_content
   if len(state["messages"]) == 0 or "VALIDE" in state["messages"][-1].content:
        #On génère une nouvelle extraction
        res_pydantic = text(texte_actuel)
        #On convertit l'objet Pydantic en JSON string pour le stockage
        rep = res_pydantic.model_dump_json()
   else:
        #On corrige l'ancien JSON basé sur la critique
      AncienJSon = state["messages"][-2].content
      critique = state["messages"][-1].content
      rep = RepIA(texte_actuel, critique, AncienJSon)
   return {"messages": [AIMessage(content=rep)]}

def est_present(nom, phrase):
    nom_propre=enlever_accents(nom.lower())
    liste_mots=nom_propre.split()
    for mot in liste_mots:
        if len(mot) > 2:     
            if mot in phrase:
                return True 
    return False


    


import itertools
from pydantic import ValidationError
from langchain_core.messages import AIMessage


def Verificateur(state: GraphState):
    index = state.get("index", 0)
    tentatives_actuelles = state.get("tentatives_suggestions", 0)
    tentatives_fatales = state.get("tentatives_fatales", 0)
    texte_actu = state["pages"][index].page_content
    jsonn = state["messages"][-1].content           
    liste_critique = []
    liste_suggestions = []

    try:
        # On transforme le json en objet Pydantic
        json_obj = JSON_Normalise.model_validate_json(jsonn)
        
        # Lecture du texte avec spacy et normalisation
        paragraphes = texte_actu.split('\n\n')
        
        # On nettoie et on garde uniquement les paragraphes qui ont un peu de substance
        blocs_propres = [enlever_accents(p.lower().strip()) for p in paragraphes if len(p.strip()) > 10]
        texte_complet = enlever_accents(texte_actu.lower()) 

        # Dictionnaire pour mapper les ID aux noms
        id_to_name = {}

        for element in json_obj.entities:
            source_name = element.name
            id_to_name[element.id] = source_name 
            
            mots_entite = enlever_accents(source_name.lower()).split()
            for mot in mots_entite:
                if len(mot) > 2:
                    if mot not in texte_complet:
                        liste_critique.append(f"- L'entité '{source_name}' (ID: {element.id}) est introuvable dans le texte. Supprime-la.")
                        break

        liens_existants = set()
        
        for lien in json_obj.Link:
            source_id = lien.source
            cible_id = lien.target

            liens_existants.add(tuple(sorted([source_id, cible_id])))

            if source_id not in id_to_name or cible_id not in id_to_name:
                liste_critique.append(f"- Le lien utilise un ID inconnu ('{source_id}' ou '{cible_id}'). Vérifie les ID déclarés.")
                continue

            source_name = id_to_name[source_id]
            cible_name = id_to_name[cible_id]
            lien_valide = False

            for bloc in blocs_propres:
                if est_present(source_name, bloc) and est_present(cible_name, bloc):
                    lien_valide = True
                    break
            
            if not lien_valide:
                liste_critique.append(f"- Le lien entre '{source_name}' et '{cible_name}' semble inventé (ils ne sont jamais dans la même phrase). Supprime-le.")

        suggestions_liens = set()
        
        for bloc in blocs_propres:
            entites_dans_bloc = [e for e in json_obj.entities if est_present(e.name, bloc)]
            
            if len(entites_dans_bloc) >= 2:
                for e1, e2 in itertools.combinations(entites_dans_bloc, 2):
                    paire_id = tuple(sorted([e1.id, e2.id]))
                    if paire_id not in liens_existants:
                        suggestions_liens.add(tuple(sorted([e1.name, e2.name])))

        for nom1, nom2 in suggestions_liens:
            liste_suggestions.append(f"- Suggestion : '{nom1}' et '{nom2}' apparaissent ensemble. S'il y a une relation, ajoute-la. Sinon, renvoie simplement le JSON.")

    except ValidationError as e:
        liste_critique.append(f"- Erreur fatale de structure JSON (Pydantic) : {e}. Corrige impérativement le format.")
    except Exception as e:
        liste_critique.append(f"- Erreur inattendue lors de la vérification : {str(e)}")

    if tentatives_fatales >= 5:
        print(f"Vérification {index}: Vérification non concluante, vérification suivante.")
        return {
            "messages": [AIMessage(content="VALIDE (Forcé par Kill Switch)")],
            "index": index + 1,
            "tentatives_suggestions": 0,
            "tentatives_fatales": 0
        }

    if len(liste_critique) > 0:
        critique_finale = "ERREURS CRITIQUES (À CORRIGER OBLIGATOIREMENT) :\n" + "\n".join(liste_critique)
        if liste_suggestions:
            critique_finale += "\n\nSUGGESTIONS D'OUBLIS :\n" + "\n".join(liste_suggestions)
            
        print(f"Vérification {index}: Vérification non réussie, nouvel essai (tentative {tentatives_fatales + 1}/5)")
        return {
            "messages": [AIMessage(content=critique_finale)],
            "tentatives_fatales": tentatives_fatales + 1 
        }
        
    elif len(liste_suggestions) > 0:
        if tentatives_actuelles >= 3:
            print(f"Vérification {index}: SUCCÈS.")
            return {
                "messages": [AIMessage(content="VALIDE")],
                "Json_f": state.get("Json_f", []) + [state["messages"][-1]],
                "index": index + 1,
                "tentatives_suggestions": 0,
                "tentatives_fatales": 0
            }
        else:
            critique_finale = "Le JSON est syntaxiquement parfait. Cependant, voici des suggestions :\n" + "\n".join(liste_suggestions)
            print(f"Vérification {index}: Amélioration possible, suggestion donnée au LLM, (Tentative {tentatives_actuelles + 1}/3)")
            return {
                "messages": [AIMessage(content=critique_finale)],
                "tentatives_suggestions": tentatives_actuelles + 1
            }
            
    else:
        print(f"Vérification {index}: SUCCÈS.")
        return {
            "messages": [AIMessage(content="VALIDE")],
            "Json_f": state.get("Json_f", []) + [state["messages"][-1]],
            "index": index + 1,
            "tentatives_suggestions": 0,
            "tentatives_fatales": 0 
        }
    
def Cond(state: GraphState):
    derniere_rep = state["messages"][-1].content
    index = state.get("index", 0)
    pages = state["pages"]

    if "VALIDE" in derniere_rep:
   
        if index >= len(pages): 
            return END
        else:
            return "generateur"
    else:
      
        return "generateur"

   
constructeur= StateGraph(GraphState)

constructeur.add_node("generateur",generateur)

constructeur.add_node("verificateur",Verificateur)
constructeur.add_edge(START,"generateur")

constructeur.add_edge("generateur","verificateur")

constructeur.add_conditional_edges(
   "verificateur",
   Cond,{
      END:END,
      "generateur": "generateur"
   }

)
react_graph=constructeur.compile()


def fusionner_jsons(liste_jsons: List[JSON_Normalise]) -> JSON_Normalise:
    template= """
    Context: I have a dataset consisting of multiple JSON objects that have been crudely concatenated into a python list. This data contains overlapping entities, redundant entries, and fragmented relational mappings that need to be unified into a single, coherent structure.

    Role: You are a Senior Data Engineer specializing in JSON Restructuring and Schema Alignment. Your goal is to ensure data integrity while merging disparate nodes into a unified master document.

    Action:

    1. Analyze the provided concatenated text to identify all individual JSON objects.

    2. Merge all objects into a single, root JSON structure that strictly follows the original schema (Entity/Person/Link).

    3. De-duplicate any identical entities or relationships found across different fragments.

    4. Consolidate attributes: if the same entity appears in multiple fragments with different aliases or links, combine them into a single master entry for that entity.

    5. Format the final output with a standard 4-space indentation for maximum readability.

    Format: Return ONLY the final, prettified JSON code. Do not include any introductory remarks, markdown code blocks (unless specified), or post-processing explanations.

    Target: Produce a single, valid, and perfectly indented JSON document ready for production environment import.

    Target Data: {liste}
    """
    prompt_f=ChatPromptTemplate.from_template(template)
    llm_s=model.with_structured_output(JSON_Normalise)
    chain_f=prompt_f | llm_s
    resultat=chain_f.invoke({"liste": liste_jsons})
    return resultat

import json
from math import log2, floor
from pyvis.network import Network
import webbrowser
import os 

TAILLE_MIN = 10
TAILLE_MAX = 80
def occurence(f_md: str, data: dict):
    """ 
    Fonction qui recherche le nombre de fois que les sommets apparaissent dans le texte.
    Entrées : 
    =========
    f_md : (str) nom du fichier dans lequel on va faire des recherches.
    data : (dict) dictionnaire contenant tous les sommets trouvés par le LLM.
    Retour :
    ========
    (list) une liste mise à jour avec le nombre d'occurrences.
    """ 

    # 1. On lit tout le texte d'un coup et on le met en minuscules
    with open(f_md, "r", encoding="utf-8") as f:
        texte_entier = f.read().lower() 

    texte_entier_sans_accent = enlever_accents(texte_entier)

    liste_mot_occu = []
   
    # 2. On parcourt les entités
    for entity in data["entities"]:
        nom_original = entity["name"].lower() # met tout en minuscule 
    
        # On compte combien de fois apparaît le nom de base
        nbr = texte_entier_sans_accent.count(nom_original)

        # On rajoute avec le nom inversé 
        morceaux = nom_original.split(" ") # coupe au niveau de l'espace 
        
        if len(morceaux) >= 2:
            prenom = morceaux[0]
            nom = morceaux[1]
            nom_inverse = nom + " " + prenom 
                
            nbr += texte_entier_sans_accent.count(nom_inverse)

        # On ajoute le compte des alias
        liste_alias = entity.get("Alias", [])
        for un_alias in liste_alias:
            nbr += texte_entier_sans_accent.count(un_alias.lower()) 
                
        # 3. On sauvegarde le résultat
        liste_mot_occu.append((entity["name"], nbr))
    
    return liste_mot_occu

def echelle(liste_occurrences: list): 
    """
    Fonction qui affecte une échelle pour chaque sommet en fonction de la règle de Sturges.    Entrées : 
    =========
    liste_occurrences : (list) une liste contenant les sommets et leur nombre d'occurrences.
    Retour :
    ========
    (list) Une liste contenant chaque sommet et son niveau.    
    """

    # 1. Calcul de la règle de Sturges 
    nb_paliers = 1 + floor(log2(len(liste_occurrences))) 
    maxi_p = max(occ for mot, occ in liste_occurrences)
    mini_p  = min(occ for mot, occ in liste_occurrences)
    liste_nettoyee = [ ] # sans le max et min 
    for couple in liste_occurrences: 
        if couple != maxi_p and couple != mini_p: 
            liste_nettoyee.append(couple)

    # evite la division par 0  
    if len(liste_nettoyee) == 0 : 
        liste_nettoyee = liste_occurrences 


    maxi = max(occ for _, occ in liste_nettoyee)
    mini = min(occ for _, occ in liste_nettoyee)
    amplitude = (maxi - mini) / nb_paliers 
    # on evite la division par 0 
    if amplitude == 0 : 
        amplitude = 1 
    
    # 2. La répartition dans les paliers
    sommets_repartis = []
    
    for mot, occ in liste_occurrences:
        if occ >= maxi_p:
            numero_palier = nb_paliers + 2
        elif occ <= mini_p:
            numero_palier = 1
        else:
            numero_palier = int(floor((occ - mini) / amplitude)) + 1
        sommets_repartis.append((mot, numero_palier))
    return sommets_repartis

def affichage_graphe(f_json: str, f_md: str):
    """
   
    Fonction qui génère et affiche un graphe interactif à partir des données extraites.
    
    Entrées : 
    =========
    f_json : (str) Fichier de sortie du LLM contenant les sommets et les liens.
    f_md : (str) Fichier d'entrée du LLM utilisé pour calculer les occurrences.
    
    Sortie :
    ========
    (Fichier HTML) Génère et ouvre un fichier HTML (basic.html) contenant le graphe interactif avec une légende.
    """



    couleur = [
        "lightblue", "lightgreen", "lightpink","khaki","plum", "lightsalmon", 
        "paleturquoise", "silver", "thistle","greenyellow", "powderblue", 
        "rosybrown", "palegoldenrod","aquamarine", "violet", "peachpuff",
        "lightsteelblue","darkseagreen", "wheat","orchid", "lightcyan", 
        "burlywood", "hotpink","yellowgreen", "cornflowerblue", "lightcoral", 
        "mediumaquamarine", "tan", "lavender","gainsboro"
    ]

    with open(f_json, "r", encoding="utf-8") as f:
        contenu = f.read()
        data = json.loads(contenu)
        
    color_ed = ["limegreen","yellowgreen","gold","orange","red"]
    nom_sommet = []
    g = Network(height="100vh", width="100%")
    color = {}
    noms = {}  
    i, a = 0, 0  
    
    test = occurence(f_md, data)
    amplitude = echelle(test) 
    dico_amplitude = dict(amplitude)

    # 1. Création des sommets
    for entity in data["entities"]:
        tp = entity["type"]
        nom_entite = entity["name"]
        
        if tp not in color:
            # Le modulo permet de recommencer au début de la liste si on a plus de 30 types
            color[tp] = couleur[i % len(couleur)]
            i += 1

        palier_actuel = dico_amplitude.get(nom_entite, 1)
        noms[entity["id"]] = (a, entity["name"], color[tp], palier_actuel)
        nom_sommet.append(entity["name"])
        a += 1

    # (Le compteur de liens est calculé mais pas utilisé pour le moment, je le laisse)
    compteur = {}
    for lien in data["Link"]:
        compteur[lien["source"]] = compteur.get(lien["source"], 0) + 1
        compteur[lien["target"]] = compteur.get(lien["target"], 0) + 1  

    # 2. Ajout des noeuds au graphe
    nb_paliers_total = max(dico_amplitude.values())  # à calculer avant la boucle

    for _, (id_num, nom, col, ampl) in noms.items():
        taille_seil = TAILLE_MIN + ((ampl - 1) / (nb_paliers_total - 1)) * (TAILLE_MAX - TAILLE_MIN) ** (ampl / nb_paliers_total)
        nb_caracteres = len(nom)
        largeur_texte = nb_caracteres * 13 * 0.6 
        taille_char = (largeur_texte / 2) + 5
        if taille_seil < taille_char :
            forme = "dot"
        else : 
            forme = "circle"
        g.add_node(id_num, label=nom, shape=forme, color=col, size=taille_seil, font={"size": 14})
    # 3. Ajout des liens au graphe
    for lien in data["Link"]:
        id_source = lien["source"]
        id_target = lien["target"]
        if id_source not in noms or id_target not in noms:
            print(f"Lien ignoré : La source '{id_source}' ou la cible '{id_target}' n'existe pas dans les entités.")
            continue # On passe au lien suivant sans faire planter le programme
        source_id = noms[id_source][0]
        target_id = noms[id_target][0]
        index_gravite = min(lien["gravite"] - 1, len(color_ed) - 1)
        g.add_edge(source_id, target_id, label=lien["relation"], color=color_ed[index_gravite], arrows="from")

    # 4. Options pour éviter les chevauchements
    # 4. Options pour bien séparer les groupes (parties connexes)
    g.set_options("""
    {
    "interaction": {
        "navigationButtons": true,
        "keyboard": true
                  }, 
    "physics": {
        "enabled": true,
        "solver": "repulsion",
        "repulsion": {
            "centralGravity": 0.0,
            "springLength": 250,
            "springConstant": 0.05,
            "nodeDistance": 250,
            "damping": 0.09
        },
        "stabilization": {
            "enabled": true,
            "iterations": 300
        }
    },
    "nodes": {
    },
    "edges": {
        "font": {"size": 10, "align": "middle"},
        "smooth": {"type": "continuous"}

    }
    }
    """)

    g.save_graph("basic.html") 

    # 5. GÉNÉRATION DE LA LÉGENDE

    # Style des sommets
    html_sommets = "<h4>Types d'Entités</h4><ul style='list-style: none; padding-left: 0;'>"
    html_sommets += "<style>.circleCustom { display: inline-block; height: 12px; width: 12px; border-radius: 50%; }</style>"

    for som, coul in color.items():
        html_sommets += f"""
        <!-- Le 'span' permet d'écrire à côté du cercle -->
        <li style="display: flex; align-items: center; gap: 8px; margin-bottom: 5px;"> 
            <div class="circleCustom" style="background-color: {coul};"></div>
            <span>{som}</span>
        </li>"""
    html_sommets += "</ul>"

    # Style des liens
    html_liens = "<h4>Gravité des liens</h4><ul style='list-style: none; padding-left: 0;'>"

    # Le style de notre rectangle : 50px de long, 8px de haut (bien épais)
    html_liens += "<style>.line-legend { width: 40px; height: 8px; border-radius: 4px; }</style>"

    for idx, col_l in enumerate(color_ed):
        html_liens += f"""
        <li style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <div class="line-legend" style="background-color: {col_l};"></div>
            <span>gravité {idx + 1}</span>
        </li>"""
    html_liens += "</ul>"


    # Style des tailles
    html_tailles = """
    <h4>Taille des sommets</h4>
      
    <div style="display: flex; align-items: flex-end; justify-content: space-between; margin-bottom: 5px; padding: 0 5px;">
        <div style="height: 8px; width: 8px; background-color: #666; border-radius: 50%;"></div>
        <div style="height: 14px; width: 14px; background-color: #666; border-radius: 50%;"></div>
        <div style="height: 22px; width: 22px; background-color: #666; border-radius: 50%;"></div>
    </div>
    
    <div style="display: flex; justify-content: space-between; font-size: 11px; color: #444;">
        <span>Peu</span>
        <span>Beaucoup</span>
    </div>
    """

    # Assemblage de la légende
    html_legende = f"""
    <div style="position: absolute; top: 20px; left: 20px; background-color: rgba(255, 255, 255, 0.9); 
                border: 2px solid #444; border-radius: 10px; padding: 15px; z-index: 1000; 
                font-family: sans-serif; box-shadow: 5px 5px 15px rgba(0,0,0,0.2); max-width: 220px;">
        <h3 style="margin-top: 0; text-align: center; border-bottom: 1px solid #999;">Légende</h3>
        {html_sommets}
        <hr>
        {html_liens}
        <hr>
        {html_tailles}
    </div>
    """
    html_content = g.generate_html()
    return html_content.replace("</body>", html_legende + "\n</body>")
    

import streamlit.components.v1 as components
st.set_page_config(
    page_title="Visualisation de Graphe",
    layout="wide"
)

st.title("Visualisation du Graphe de Connaissances")
st.markdown("Uploadez un fichier texte ou Markdown pour extraire automatiquement les entités et leurs relations.")

fichier_upload=st.file_uploader("Choisissez un fichier au format .md",type=["md"])
exec=True
if fichier_upload is not None:
    message=st.empty()
    message.info(f"Fichier chargé : {fichier_upload.name}")
    chemin_temp = f"temp_{fichier_upload.name}"
    with open(chemin_temp, "wb") as fi:
        fi.write(fichier_upload.getbuffer())
    
    if st.button("Générer le Graphe de Connaissances"):
        message.empty()       
        message.info("Début de génération de graphe.")
        texte_pages=text_splitter(chemin_temp)
        resultat_f=react_graph.invoke({
                       "pages": texte_pages,
                       "index": 1,
                       "messages": [],
                       "Json_f": []
                    })
        json_valides=[
            JSON_Normalise.model_validate_json(msg.content) 
            for msg in resultat_f["Json_f"]
        ]
        texte_final=fusionner_jsons(json_valides)
        with open(fichier_upload.name.split(".")[0]+".json", "w", encoding="utf-8") as f:
             f.write(texte_final.model_dump_json(indent=4))   
        st.success("Graphe de Connaissances généré avec succès !")
        st.subheader("Visualisation du Graphe")
        code_html=affichage_graphe(fichier_upload.name.split(".")[0]+".json",fichier_upload.name)
        with open(fichier_upload.name.split(".")[0]+".html", 'w', encoding='utf-8') as f:
            f.write(code_html)
        components.html(
            code_html, 
            height=800,      
            scrolling=True
        )
                
