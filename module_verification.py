import random as r
from random import randint, choices, shuffle
import time as t
import copy
import json
import projet16_graphe_src
import spacy 
import unicodedata
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
from pathlib import Path


def enlever_accents(texte):
    """Retire les accents d'une chaîne de caractères."""
    # Sépare la lettre de son accent
    texte_normalise = unicodedata.normalize('NFD', texte)
    # Garde uniquement la lettre de base
    return ''.join(c for c in texte_normalise if unicodedata.category(c) != 'Mn')

def generation_graphe(n : int):  # doit etre discuter avec cyril les ligne 19 , 58,59, 86 rajoute de personne sur les diction et de lien 
    """Fonction qui génère une graphe en format dictionnaire (format avant conversion en JSON dans l'extraction)
    Entrées :
    =========
        n : nombre de sommets du graphe généré

    Sortie :
    ========
        graphe g sous forme de dictionnaire généré aléatoirement"""

    prenoms = [("Haïtem","H"), ("Akram","H"), ("Hannah","F"), ("Rémy","H"), ("Cyril","H"), ("Titouan","H"), ("Jacqueline","F"),("Helene","F"),("Lola","F"),("Andrée","F"),("Jacques","H"),("Emy","F"), ("Léa","F"),("Mia","F"),("Eva","F"),("Jade","F")]
    noms = ["Labeye","Bougoffa","Laïb","Maurice-Demourioux","El Harras", "Lefebvre", "Leroux", "Dubois", "Durand","Dupont", "Petit", "Roux", "Leroy", "Bonnet", "Muller", "Morin", "Meyer", "Brun", "Roy", "Vidal", "Aubert", "Dumas", "Rey", "Lopez", "Sanchez"]
    entreprises = ["Amazon", "Apple", "Samsung", "Google", "Walmart","Volkswagen","McKesson","Mercedes","Peugeot","Citroen","HP","Dell"]
    
    rel_pers_pers = [
    ("H","H","père de","fils de"), ("H","F","père de","fille de"), ("F","H","mère de","fils de"), ("F","F","mère de","fille de"),
    ("H","H","frère de","frère de"), ("H","F","frère de","sœur de"), ("F","H","sœur de","frère de"), ("F","F","sœur de","sœur de"),
    ("H","H","supérieur de","subordonné de"), ("H","F","supérieur de","subordonnée de"), ("F","H","supérieure de","subordonné de"), ("F","F","supérieure de","subordonnée de"),
    ("H","H","mentor de","mentoré de"), ("H","F","mentor de","mentorée de"), ("F","H","mentor de","mentoré de"), ("F","F","mentor de","mentorée de"),
    ("H","H","associé de","associé de"), ("H","F","associé de","associée de"), ("F","H","associée de","associé de"), ("F","F","associée de","associée de"),
    ("H","H","collègue de","collègue de"), ("H","F","collègue de","collègue de"), ("F","H","collègue de","collègue de"), ("F","F","collègue de","collègue de"),
    ("H","H","voisin de","voisin de"), ("H","F","voisin de","voisine de"), ("F","H","voisine de","voisin de"), ("F","F","voisine de","voisine de"),
    ("H","H","rival de","rival de"), ("H","F","rival de","rivale de"), ("F","H","rivale de","rival de"), ("F","F","rivale de","rivale de"),
    ("H","H","complice de","complice de"), ("H","F","complice de","complice de"), ("F","H","complice de","complice de"), ("F","F","complice de","complice de"),
    ("H","H","ami de","ami de"), ("H","F","ami de","amie de"), ("F","H","amie de","ami de"), ("F","F","amie de","amie de"), ]

    rel_pers_entreprise = [
    ("H", "directeur de",   "dirigé par"),
    ("F", "directrice de",  "dirigé par"),
    ("H", "actionnaire de", "financé par"),
    ("F", "actionnaire de", "financé par"),
    ("H", "employé chez",   "employeur de"),
    ("F", "employée chez",  "employeur de"),
    ("H", "fondateur de",   "fondé par"),
    ("F", "fondatrice de",  "fondé par"),]

    rel_entreprise_pers = [
    ("H", "dirigé par",    "directeur de"),
    ("F", "dirigée par",    "directrice de"),
    ("H", "financé par",   "actionnaire de"),
    ("F", "financée par",   "actionnaire de"),
    ("H", "employeur de",  "employé chez"),
    ("F", "employeur de",  "employée chez"),
    ("H", "fondé par",     "fondateur de"),
    ("F", "fondée par",     "fondatrice de"), ]

    rel_entreprise_entreprise = [
    ("actionnaire de", "financé par"),
    ("partenaire de",  "partenaire de"),
    ("filiale de",     "maison mère de"),
    ("concurrent de",  "concurrent de"),
    ("fournisseur de", "client de"),]

    g={"ensemble_liens":[{"entities":[], "links":[]}]}
    entites_DejaVu=[]

    #1/ --- CHOIX DE L'ENTITE DE MANIERE ALEATOIRE ---
    for _ in range(n):
        jamais_vu=False
        #Tant que le choix aléatoire ne donne pas une entité qu'on a jamais vu
        while jamais_vu==False:
            #choix_entite = 0 ou 1
            choix_entite=r.randint(0,1)

            # si choix_entite = 0, alors l'entite est une personne physique (prénom + nom)
            if choix_entite==0:
                prenom_tuple = r.choice(prenoms) 
                prenom_tire = prenom_tuple[0] 
                nom_tire=r.choice(noms)
                if (prenom_tire, nom_tire) not in entites_DejaVu:
                    #la methode lower() permet de mettre tous les caractères d'un string en minuscule
                    id=prenom_tire.lower()+"_"+nom_tire.lower()
                    type="person"
                    name=prenom_tire+" "+nom_tire
                    
                    entites_DejaVu.append((prenom_tire,nom_tire))
                    jamais_vu=True

            # si choix_entite = 1, alors l'entite est une personne morale (entreprise)                
            else :
                entreprise_tire=r.choice(entreprises)
                if entreprise_tire not in entites_DejaVu:
                    id=entreprise_tire.lower()
                    type="company"
                    name=entreprise_tire

                    entites_DejaVu.append(entreprise_tire)
                    jamais_vu=True

        #ajout de l'entité dans le graphe g
        g["ensemble_liens"][0]["entities"].append({"id":id,
                                        "type":type,
                                        "name":name,
                                        "Alias":[]},) 

    #2/ --- GENERATIONS DES LIENS ---
    paires_deja_liees = []
    for _ in range(n // 2): 
        entite_A = r.choice(g["ensemble_liens"][0]["entities"])
        entite_B = r.choice(g["ensemble_liens"][0]["entities"])
        
        paire_test=set([entite_A["id"], entite_B["id"]])


        #On tire l'entité B jusqu'a ce qu'elle soit differente de A
        while entite_A["id"] == entite_B["id"] or paire_test in paires_deja_liees:
            entite_B = r.choice(g["ensemble_liens"][0]["entities"])
            paire_test=set([entite_A["id"], entite_B["id"]])
        
        paires_deja_liees.append(paire_test)

        # On traite les types de relations possibles selon les types des entites
        type_A = entite_A["type"]
        type_B = entite_B["type"]

        if type_A == "person" and type_B == "person":
            prenom_A = entite_A["id"].split("_")[0].capitalize()
            genre_A  = next((genre for p, genre in prenoms if p == prenom_A), "H")
            prenom_B = entite_B["id"].split("_")[0].capitalize()
            genre_B  = next((genre for p, genre in prenoms if p == prenom_B), "H")
            genre_1, genre_2, relation, relation_reciproque = r.choice(rel_pers_pers)
            while genre_1 != genre_A or genre_2 != genre_B:
                genre_1, genre_2, relation, relation_reciproque = r.choice(rel_pers_pers)

        elif type_A == "person" and type_B == "company":
            prenom_A = entite_A["id"].split("_")[0].capitalize()
            genre_A  = next((genre for p, genre in prenoms if p == prenom_A), "H")
            genre_1, relation, relation_reciproque = r.choice(rel_pers_entreprise)
            while genre_1 != genre_A:
                genre_1, relation, relation_reciproque = r.choice(rel_pers_entreprise)

        elif type_A == "company" and type_B == "person":
            prenom_B = entite_B["id"].split("_")[0].capitalize()
            genre_B  = next((genre for p, genre in prenoms if p == prenom_B), "H")
            genre_2, relation, relation_reciproque = r.choice(rel_entreprise_pers)  
            while genre_2 != genre_B:                                                
                genre_2, relation, relation_reciproque = r.choice(rel_entreprise_pers)

        elif type_A == "company" and type_B == "company":
            relation, relation_reciproque = r.choice(rel_entreprise_entreprise)
        
        # On ajoute le lien de A vers B
        g["ensemble_liens"][0]["links"].append({
            "source": entite_A["id"],
            "target": entite_B["id"],
            "relation": relation
        })
        
        # on ajoute le lien reciproque de B vers A
        g["ensemble_liens"][0]["links"].append({
            "source": entite_B["id"],
            "target": entite_A["id"],
            "relation": relation_reciproque
        })
        
    return g
        
#Ouverture de texte_sans_lien.txt qui contient des phrases qui n'ont aucun rapport avec un graphe de relation pour perturber le traitement de l'IA
with open("texte_sans_lien.txt", "r", encoding="utf-8") as texte_sans_lien:
    # On met les phrases sans lien dans une liste
    phrases_alea = texte_sans_lien.readlines()

#Nb de phrases aléatoires
len_phrases_alea=len(phrases_alea)

def generation_morceau_texte_alea():
    """Fonction qui génère un morceau de texte aléatoire qui n'a aucun rapport avec un quelconque graphe de relations.
    Elle choisit aléatoirement entre 0 et 10 phrases consécutives depuis le fichier "texte_sans_lien.txt" pour perturber l'extraction.

    Entrées :
    =========
        Aucune
    
    Sortie :
    ========
        morceau_texte_alea : Chaîne de caractères contenant les phrases aléatoires concaténées. (Retourne une chaîne vide si le tirage est de 0)."""

    morceau_texte_alea=""

    #On choisit entre 0 et 10 lignes du texte sans lien
    choix_taille_texte=r.randint(0,10)
    if choix_taille_texte==0:
        return morceau_texte_alea
    
    #choix de l'index a partir duquel on prend les phrases dans phrases_alea    
    choix_idx=r.randint(0,len_phrases_alea-choix_taille_texte)

    for i in range(choix_idx,choix_idx+choix_taille_texte):
        morceau_texte_alea+=phrases_alea[i]

    return morceau_texte_alea

def generation_texte_markdown(g):
    """Fonction qui génère du texte en markdown à partir d'un graphe de relations
    Entrées :
    =========
        g : graphe de relations
 
    Sortie :
    ========
        Aucune"""
    texte_final = ""
    id_vers_nom = {}
    dejaVu_lien = []
 
    for entite in g["ensemble_liens"][0]["entities"]:
        id_vers_nom[entite["id"]] = entite["name"]
    pos_entreprise1_A_B = ["directeur de", "directrice de", "fondateur de", "fondatrice de", "actionnaire de"]
    pos_entreprise1_B_A = ["dirigé par", "dirigée par", "financé par", "financée par"]
    pos_entreprise2_A_B = ["employeur de"]
    pos_entreprise2_B_A = ["employé chez", "employée chez", "partenaire de"]
    relat_familiale     = ["père de", "mère de", "fils de", "fille de", "frère de", "sœur de"]
    relat_perso         = [
        "supérieur de", "supérieure de", "subordonné de", "subordonnée de",
        "mentor de", "mentoré de", "mentorée de",
        "associé de", "associée de",
        "collègue de",
        "voisin de", "voisine de",
        "rival de", "rivale de",
        "complice de",
        "ami de", "amie de"
    ]
 
    for el in g["ensemble_liens"][0]["links"]:
        id_source = el["source"]
        id_target = el["target"]
 
        ens_source_targ = set([id_source, id_target])
        if ens_source_targ not in dejaVu_lien:
            dejaVu_lien.append(ens_source_targ)
            source   = id_vers_nom[id_source]
            target   = id_vers_nom[id_target]
            relation = el["relation"]
 
            # ── Relation personne → entreprise (A dirige/finance/fonde B) ──
            if relation in pos_entreprise1_A_B:
                phrases_possibles = [
                    f"Sur le plan professionnel, {source} agit en tant que {relation} {target}.",
                    f"Le groupe {target} compte en effet {source} comme {relation.replace(' de', '')}.",
                    f"L'analyse du dossier montre que {source} est {relation} {target}."
                ]
 
            # ── Relation entreprise ← personne (A est dirigé/financé par B) ──
            elif relation in pos_entreprise1_B_A:
                if relation in ("dirigé par", "dirigée par"):
                    titre = "dirigeant"
                else:
                    titre = "investisseur"
                phrases_possibles = [
                    f"Sur le plan professionnel, l'entité {source} est {relation} {target}.",
                    f"Le groupe {source} compte en effet {target} comme {titre}.",
                    f"L'analyse du dossier montre que {source} est {relation} {target}."
                ]
 
            # ── Relation employeur ──
            elif relation in pos_entreprise2_A_B:
                phrases_possibles = [
                    f"Concernant les liens d'affaires, l'entreprise {source} est l'{relation} {target}.",
                    f"Les données confirment que {source} emploie {target}."
                ]
 
            # ── Relation employé/partenaire ──
            elif relation in pos_entreprise2_B_A:
                phrases_possibles = [
                    f"Concernant les liens d'affaires, {source} est actuellement {relation} l'entreprise {target}.",
                    f"Les données confirment que {source} collabore avec {target} ({relation})."
                ]
 
            # ── Relations familiales ──
            elif relation in relat_familiale:
                phrase_famille = f"Les investigations révèlent que {source} et {target} sont de la même famille."
 
                if relation in ("frère de", "sœur de"):
                    phrase_famille += " En effet, ils sont frère et sœur."
 
                elif relation in ("père de", "mère de", "fils de", "fille de"):
                    # On normalise pour toujours avoir le parent en source
                    if relation in ("fils de", "fille de"):
                        source, target = target, source
 
                    suite_phrase = [
                        f" En effet, {source} est le parent de {target}.",
                        f" En effet, {target} est l'enfant de {source}.",
                        f" Nous voyons que {source} est le père ou la mère de {target}.",
                        f" Nous voyons que {target} est le fils ou la fille de {source}.",
                        f" Les documents attestent que {source} détient l'autorité parentale sur {target}."
                    ]
                    phrase_famille += r.choice(suite_phrase)
 
                phrases_possibles = [
                    f"D'un point de vue personnel, nous avons remarqué que {source} est {relation} {target}.",
                    f"Les investigations révèlent que {source} et {target} sont de la même famille.",
                    phrase_famille
                ]
 
            # ── Relations personnelles (pro/social) ──
            elif relation in relat_perso:
                phrases_possibles = [
                    f"Sur le plan personnel, {source} et {target} entretiennent une relation : {source} est {relation} {target}.",
                    f"Les investigations montrent que {source} est {relation} {target}.",
                    f"D'après les documents, {source} et {target} sont liés : {source} est {relation} {target}."
                ]
 
            # ── Toute autre relation non prévue (secours) ──
            else:
                phrases_possibles = [
                    f"{source} et {target} sont liés : {source} est {relation} {target}.",
                    f"Les documents indiquent que {source} est {relation} {target}.",
                    f"Une relation de type '{relation}' a été identifiée entre {source} et {target}."
                ]
 
            phrase_choisie          = r.choice(phrases_possibles)
            morceau_texte_sans_lien = generation_morceau_texte_alea()
            texte_final            += " " + phrase_choisie + " " + morceau_texte_sans_lien
 
    with open("rapport_aleatoire.md", "w", encoding="utf-8") as markdown_final:
        markdown_final.write(texte_final)

def rajout_compte():
    """
    Fonction qui rajoute un conte au fichier déjà existant. 
    Entrée : 
    =========
        Aucune 
    Sortie : 
    ========
        Aucune 
    """
    legende_arthur = [
        "Arthur serait issu d’une relation adultère entre le roi breton Uther Pendragon et la femme de l’un de ses vassaux, Ygern (ou Ygraine).",
        "Arthur naît à Tintagel, en Cornouailles et, en tant que fils illégitime, il est caché par l’enchanteur Merlin qui le confie à un petit noble.",
        "A sa majorité, alors que son père est mort depuis longtemps (tué au combat ? empoisonné en buvant à une source ?), Arthur est reconnu roi légitime des Bretons en parvenant à libérer l’épée Excalibur (ou Caliburnus) du roc dans lequel Uther l’avait plantée.",
        "Certains barons contestent néanmoins sa légitimité, et le roi Arthur passe les premières années de son règne à les combattre.",
        "Il commence à s’entourer de chevaliers, parmi lesquels l’un des plus illustres de ce qui va devenir la Table Ronde, le neveu d'arthur, Gauvain.",
        "Allié de Léodagant, le roi Arthur va jusqu’à combattre en Gaule et, victorieux, il épouse Guenièvre, fille de Léodagant.",
        "Commence une période de prospérité, jusqu’à ce que le chevalier Lancelot du Lac (apparu tardivement dans la légende arthurienne) arrive à la cour d’Arthur, à Camelot.",
        "Très vite ami proche du roi, le jeune homme s’éprend de la reine Guenièvre, avec laquelle débute une relation d’adultère annonciatrice de la décadence du royaume.",
        "Arthur lui-même succombe à l’adultère, séduit par l’enchanteresse Camille.",
        "Les aventures de Lancelot permettent toutefois une réconciliation entre les deux amis (notamment quand le chevalier défait le félon Méléagant), qui se rendent en Gaule pour punir Claudas, usurpateur du père de Lancelot, qui surtout a fait prisonnière Guenièvre.",
        "Victorieux et la reine libérée, commence la Quête du Graal à laquelle cependant ne participent pas Lancelot et Arthur, bien trop impurs face à Gauvain, Perceval et autres Galaad…",
        "Le drame n’est cependant jamais très loin à la cour d’Arthur.",
        "Et c’est encore la relation entre Guenièvre et Lancelot qui aggrave les choses ; le roi trahi est recueilli par sa demi-soeur Morgane.",
        "Les versions divergent alors : ce serait elle, qui serait la mère du fils d’Arthur, Mordred.",
        "Quoiqu’il en soit, Arthur se rend bien coupable d’un inceste, et engendre celui qui va causer sa perte.",
        "De retour à Camelot, il doit défendre l’honneur de sa reine, accusée de tentative d’empoisonnement par Mador de la Porte.",
        "Un chevalier déguisé (en fait Lancelot) se fait le champion de Guenièvre, et l’honneur de cette dernière est sauf.",
        "Un temps seulement puisqu’elle retombe dans le pêché avec son preux chevalier…",
        "C’en est trop pour Arthur, qui condamne sa femme au bûcher ; mais elle est une nouvelle fois sauvée par Lancelot !",
        "Le combat entre les deux amis se finit par la victoire de Lancelot, qui épargne le roi pour une énième réconciliation.",
        "Lancelot disparaît, et Arthur – qui a pardonné aussi à sa reine – décide alors de reprendre ses conquêtes en Gaule, et confie son royaume au fils de Morgane, Mordred.",
        "Mauvaise idée : son neveu/fils prend goût au pouvoir, et surtout à Guenièvre, qu’il finit même par molester !",
        "Arthur revient en Bretagne et affronte Mordred à la bataille de Salisbury (ou de Camlann) ; père et fils s’entretuent, et la plupart des chevaliers de la Table Ronde trépassent également !",
        "Avant de mourir, le roi ordonne à l’un des survivants, Girflet, de jeter Excalibur dans un lac tout proche : une main féminine (celle de la Dame du Lac, qui a élevé Lancelot) récupère l’épée.",
        "Entretemps, Morgane a emporté le corps d’Arthur sur l’île d’Avalon."
    ]
    with open("rapport_aleatoire.md", "r", encoding="utf-8") as f: 
        contenu = f.read().splitlines()
    text_final = []
    i_arthur = 0
    i_contenu = 0 
    while i_arthur < len(legende_arthur):
        choix = choices([0, 1], weights=[70, 30])[0]
        if choix == 0 and i_contenu < len(contenu):
            text_final.append(contenu[i_contenu])
            i_contenu += 1 
        else: 
            text_final.append(legende_arthur[i_arthur])
            i_arthur += 1 
    while i_contenu < len(contenu): 
        text_final.append(contenu[i_contenu]) 
        i_contenu += 1 
    with open("rapport_aleatoire.md", "w", encoding="utf-8") as markdown_final:
        markdown_final.write("\n".join(text_final))
    return "rapport_aleatoire.md"

def partie_stat(g, f_json: str):

    def normaliser(texte): 
        return enlever_accents(texte.lower().strip())

    # SOMMETS
    list_arthure = [
        {"id": "Arthur_Pendragonn",   "type": "person", "name": "Arthur_Pendragon Pendragon",   "Alias": ["Arthur_Pendragon", "Le roi Arthur_Pendragon", "Roi légitime des Bretons", "Le roi trahi", "roi Arthur"]},
        {"id": "Uther_Pendragon",    "type": "person", "name": "Uther Pendragon",    "Alias": ["Uther", "Le roi breton Uther Pendragon", "Son père", "le père d Arthur_Pendragon"]},
        {"id": "Ygern",              "type": "person", "name": "Ygern",              "Alias": ["La femme de l un des vassaux"]},
        {"id": "Merlin",             "type": "person", "name": "Merlin",             "Alias": ["L enchanteur Merlin"]},
        {"id": "Petit_noble",        "type": "person", "name": "Petit noble",        "Alias": []},
        {"id": "Barons",             "type": "person", "name": "Barons",             "Alias": []},
        {"id": "Gauvain",            "type": "person", "name": "Gauvain",            "Alias": ["L un des plus illustres"]},
        {"id": "Léodagant",          "type": "person", "name": "Léodagant",          "Alias": ["Le père de Guenièvre"]},
        {"id": "Guenièvre",          "type": "person", "name": "Guenièvre",          "Alias": ["reine Guenièvre", "reine", "Sa femme"]},
        {"id": "Lancelot_du_Lac",    "type": "person", "name": "Lancelot du Lac",    "Alias": ["Le chevalier Lancelot du Lac", "Le jeune homme", "Un chevalier déguisé", "Le champion de Guenièvre", "preux chevalier", "lancelot"]},
        {"id": "Camille",            "type": "person", "name": "Camille",            "Alias": ["L enchanteresse Camille"]},
        {"id": "Méléagant",          "type": "person", "name": "Méléagant",          "Alias": ["Le félon Méléagant"]},
        {"id": "Claudas",            "type": "person", "name": "Claudas",            "Alias": ["Usurpateur du père de Lancelot"]},
        {"id": "Père_de_Lancelot",   "type": "person", "name": "Père de Lancelot",   "Alias": []},
        {"id": "Perceval",           "type": "person", "name": "Perceval",           "Alias": []},
        {"id": "Galaad",             "type": "person", "name": "Galaad",             "Alias": []},
        {"id": "Morgane",            "type": "person", "name": "Morgane",            "Alias": []},
        {"id": "Morgause",           "type": "person", "name": "Morgause",           "Alias": []},
        {"id": "Mordred",            "type": "person", "name": "Mordred",            "Alias": []},
        {"id": "Mador_de_la_Porte",  "type": "person", "name": "Mador de la Porte",  "Alias": []},
        {"id": "Girflet",            "type": "person", "name": "Girflet",            "Alias": []},
        {"id": "Dame_du_Lac",        "type": "person", "name": "Dame du Lac",        "Alias": []},
    ]

    list_depart_dicts = list(g["ensemble_liens"][0]["entities"]) + list_arthure

    list_depart = [normaliser(e["name"]) for e in list_depart_dicts]  
    with open(f_json, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    list_arriver = [normaliser(e["name"]) for e in data["entities"]]  
    list_alias = set() 

    set_depart  = {e for e in list_depart}  
    set_arriver = {e for e in list_arriver}  

    manquants = set_depart  - set_arriver
    inventes  = set_arriver - set_depart

    for e in list_depart_dicts:
        nom_principal = normaliser(e["name"])

        for alias in e.get("Alias", []):
            if len(alias) != 0:
                alias_norm = normaliser(alias)

                if alias_norm in inventes:
                    inventes.remove(alias_norm)
                    if nom_principal in manquants:
                        manquants.remove(nom_principal)
                    list_alias.add((alias_norm, nom_principal))  

    len_d           = max(len(list_depart), 1)
    stat_faux_pos_s = round(len(manquants) / len_d * 100, 3)
    stat_faux_neg_s = round(len(inventes)  / len_d * 100, 3)

    # LIENS
    relations_arthur = [
        {"source": "Uther_Pendragon",    "target": "Ygern",            "relation": "amant de"},
        {"source": "Ygern",              "target": "Uther_Pendragon",  "relation": "amant de"},
        {"source": "Uther_Pendragon",    "target": "Arthur_Pendragon", "relation": "père de"},
        {"source": "Arthur_Pendragon",   "target": "Uther_Pendragon",  "relation": "fils de"},
        {"source": "Ygern",              "target": "Arthur_Pendragon", "relation": "mère de"},
        {"source": "Arthur_Pendragon",   "target": "Ygern",            "relation": "fils de"},
        {"source": "Léodagant",          "target": "Guenièvre",        "relation": "père de"},
        {"source": "Guenièvre",          "target": "Léodagant",        "relation": "fille de"},
        {"source": "Arthur_Pendragon",   "target": "Guenièvre",        "relation": "époux de"},
        {"source": "Guenièvre",          "target": "Arthur_Pendragon", "relation": "épouse de"},
        {"source": "Arthur_Pendragon",   "target": "Gauvain",          "relation": "oncle de"},
        {"source": "Gauvain",            "target": "Arthur_Pendragon", "relation": "neveu de"},
        {"source": "Arthur_Pendragon",   "target": "Morgane",          "relation": "demi-frère de"},
        {"source": "Morgane",            "target": "Arthur_Pendragon", "relation": "demi-sœur de"},
        {"source": "Arthur_Pendragon",   "target": "Morgause",         "relation": "demi-frère de"},
        {"source": "Morgause",           "target": "Arthur_Pendragon", "relation": "demi-sœur de"},
        {"source": "Arthur_Pendragon",   "target": "Mordred",          "relation": "père de"},
        {"source": "Mordred",            "target": "Arthur_Pendragon", "relation": "fils de"},
        {"source": "Arthur_Pendragon",   "target": "Mordred",          "relation": "oncle de"},
        {"source": "Mordred",            "target": "Arthur_Pendragon", "relation": "neveu de"},
        {"source": "Morgane",            "target": "Mordred",          "relation": "mère de"},
        {"source": "Dame du Lac",        "target": "Lancelot du Lac",  "relation": "mère adoptive de"},
        {"source": "Lancelot du Lac",    "target": "Dame du Lac",      "relation": "fils adoptif de"},
        {"source": "Lancelot du Lac",    "target": "Guenièvre",        "relation": "amant de"},
        {"source": "Guenièvre",          "target": "Lancelot du Lac",  "relation": "maîtresse de"},
        {"source": "Arthur_Pendragon",   "target": "Camille",          "relation": "amant de"},
        {"source": "Camille",            "target": "Arthur_Pendragon", "relation": "maîtresse de"},
        {"source": "Mordred",            "target": "Guenièvre",        "relation": "désire"},
        {"source": "Mordred",            "target": "Guenièvre",        "relation": "moleste"},
        {"source": "Merlin",             "target": "Arthur_Pendragon", "relation": "cache"},
        {"source": "Arthur_Pendragon",   "target": "Merlin",           "relation": "est caché par"},
        {"source": "Merlin",             "target": "Petit noble",      "relation": "confie à"},
        {"source": "Petit noble",        "target": "Merlin",           "relation": "se voit confier"},
        {"source": "Petit noble",        "target": "Arthur_Pendragon", "relation": "élève"},
        {"source": "Arthur_Pendragon",   "target": "Petit noble",      "relation": "a été élevé par"},
        {"source": "Arthur_Pendragon",   "target": "Léodagant",        "relation": "allié de"},
        {"source": "Léodagant",          "target": "Arthur_Pendragon", "relation": "allié de"},
        {"source": "Arthur_Pendragon",   "target": "Lancelot du Lac",  "relation": "ami de"},
        {"source": "Lancelot du Lac",    "target": "Arthur_Pendragon", "relation": "ami de"},
        {"source": "Lancelot du Lac",    "target": "Guenièvre",        "relation": "champion de"},
        {"source": "Lancelot du Lac",    "target": "Guenièvre",        "relation": "sauveur de"},
        {"source": "Girflet",            "target": "Arthur_Pendragon", "relation": "survivant fidèle de"},
        {"source": "Arthur_Pendragon",   "target": "Girflet",          "relation": "a pour survivant fidèle"},
        {"source": "Barons",             "target": "Arthur_Pendragon", "relation": "contestent"},
        {"source": "Arthur_Pendragon",   "target": "Barons",           "relation": "est contesté par"},
        {"source": "Lancelot du Lac",    "target": "Méléagant",        "relation": "défait"},
        {"source": "Méléagant",          "target": "Lancelot du Lac",  "relation": "vaincu par"},
        {"source": "Claudas",            "target": "Père de Lancelot", "relation": "usurpateur de"},
        {"source": "Père de Lancelot",   "target": "Claudas",          "relation": "victime de l'usurpateur"},
        {"source": "Claudas",            "target": "Guenièvre",        "relation": "geôlier de"},
        {"source": "Guenièvre",          "target": "Claudas",          "relation": "est prisonnière de"},
        {"source": "Arthur_Pendragon",   "target": "Claudas",          "relation": "punit"},
        {"source": "Claudas",            "target": "Arthur_Pendragon", "relation": "est puni par"},
        {"source": "Lancelot du Lac",    "target": "Claudas",          "relation": "punit"},
        {"source": "Claudas",            "target": "Lancelot du Lac",  "relation": "est puni par"},
        {"source": "Mador de la Porte",  "target": "Guenièvre",        "relation": "accuse"},
        {"source": "Guenièvre",          "target": "Mador de la Porte","relation": "est accusée par"},
        {"source": "Arthur_Pendragon",   "target": "Guenièvre",        "relation": "condamne au bûcher"},
        {"source": "Guenièvre",          "target": "Arthur_Pendragon", "relation": "est condamnée au bûcher par"},
        {"source": "Lancelot du Lac",    "target": "Arthur_Pendragon", "relation": "combat"},
        {"source": "Arthur_Pendragon",   "target": "Lancelot du Lac",  "relation": "combat"},
        {"source": "Mordred",            "target": "Arthur_Pendragon", "relation": "trahit"},
        {"source": "Arthur_Pendragon",   "target": "Mordred",          "relation": "est trahi par"},
        {"source": "Arthur_Pendragon",   "target": "Mordred",          "relation": "affronte"},
        {"source": "Mordred",            "target": "Arthur_Pendragon", "relation": "affronte"},
        {"source": "Mordred",            "target": "Arthur_Pendragon", "relation": "tue"},
    ]

    # relation manquante
    list_dep_lien      = list(g["ensemble_liens"][0]["links"]) + relations_arthur
    list_arriver_dicts = list(data["Link"])

    liens_reperes = []
    nbre_resultat = 0 

    for dep in list_dep_lien:
        source_dep = normaliser(dep["source"])
        target_dep = normaliser(dep["target"])

        for alias_norm, nom_principal in list_alias:
            if source_dep == alias_norm:
                source_dep = nom_principal
            if target_dep == alias_norm:
                target_dep = nom_principal

        for arr in list_arriver_dicts:
            if normaliser(arr["source"]) == source_dep and normaliser(arr["target"]) == target_dep:
                liens_reperes.append({
                    "version_depart": dep,
                    "version_arrivee": arr
                })

    for duo in liens_reperes:
        relation_depart = duo["version_depart"]["relation"]
        relation_llm    = duo["version_arrivee"]["relation"]

        embeddings          = model.encode([relation_depart, relation_llm])
        resultat_similarite = util.cos_sim(embeddings[0], embeddings[1]).item()

        if resultat_similarite >= 0.75:
            nbre_resultat += 1

    # relation inventée
    liens_reperes     = []
    nbre_resultat_inv = 0 

    for arr in list_arriver_dicts:  
        source_arr = normaliser(arr["source"])
        target_arr = normaliser(arr["target"])

        for alias_norm, nom_principal in list_alias:
            if source_arr == alias_norm:
                source_arr = nom_principal
            if target_arr == alias_norm:
                target_arr = nom_principal

        for dep in list_dep_lien: 
            if normaliser(dep["source"]) == source_arr and normaliser(dep["target"]) == target_arr:
                liens_reperes.append({
                    "version_depart": dep,
                    "version_arrivee": arr
                })

    for duo in liens_reperes:
        relation_depart = duo["version_depart"]["relation"]
        relation_llm    = duo["version_arrivee"]["relation"]

        embeddings          = model.encode([relation_depart, relation_llm])
        resultat_similarite = util.cos_sim(embeddings[0], embeddings[1]).item()

        if resultat_similarite >= 0.75:
            nbre_resultat_inv += 1

    len_d           = max(len(list_dep_lien), 1)
    stat_faux_pos_l = round((len(list_dep_lien)/2 - nbre_resultat)     / len_d * 100, 3)  
    stat_faux_neg_l = round((len(list_arriver_dicts)/2 - nbre_resultat_inv) / len_d * 100, 3)  

    malus_total = stat_faux_pos_s + stat_faux_neg_s + stat_faux_pos_l + stat_faux_neg_l
    stat_graphe = round(max(0, 100 - (malus_total / 4)), 3)

    print(f"Analyse des statistiques")
    print(f"      ENTITÉS (Sommets) :")
    print(f"      - Inventés   (Faux Positifs) : {stat_faux_pos_s} %")
    print(f"      - Oubliés    (Faux Négatifs) : {stat_faux_neg_s} %")
    print()
    print(f"      RELATIONS (Liens) :")
    print(f"      - Inventées  (Faux Positifs) : {stat_faux_pos_l} %")
    print(f"      - Oubliées   (Faux Négatifs) : {stat_faux_neg_l} %")
    print()
    print(f"      SCORE GLOBAL :")
    print(f"      - Précision globale du graphe : {stat_graphe} %")

    return stat_faux_pos_s, stat_faux_neg_s, stat_faux_pos_l, stat_faux_neg_l, stat_graphe

def sauvegarde_historique(stat):
    """Stocke les statistiques dans le fichier CSV."""
    (faux_pos_s, faux_neg_s, faux_pos_l, faux_neg_l, stat_graphe) = stat
    with open("historique_tests.csv", "a", encoding="utf-8") as f:
        f.write(f"{faux_pos_s},{faux_neg_s},{faux_pos_l},{faux_neg_l},{stat_graphe}\n")

def moy_test(nom_fichier="historique_tests.csv"):
    """Calcule et affiche les moyennes depuis le fichier CSV."""
    col_faux_pos_s, col_faux_neg_s, col_faux_pos_l, col_faux_neg_l, col_graphe = [], [], [], [], []

    with open(nom_fichier, "r", encoding="utf-8") as fichier:
        for ligne in fichier:
            elements = ligne.strip().split(",")
            if len(elements) < 5:   # ← ignore les lignes vides/corrompues
                continue
            col_faux_pos_s.append(float(elements[0]))
            col_faux_neg_s.append(float(elements[1]))
            col_faux_pos_l.append(float(elements[2]))
            col_faux_neg_l.append(float(elements[3]))
            col_graphe.append(float(elements[4]))

    nb_tests = len(col_faux_pos_s)
    if nb_tests == 0:
        print("Aucun test enregistré.")
        return

    moy_faux_pos_s  = round(sum(col_faux_pos_s)  / nb_tests, 3)
    moy_faux_neg_s  = round(sum(col_faux_neg_s)  / nb_tests, 3)   # ← était col_faux_pos_s par erreur
    moy_faux_pos_l  = round(sum(col_faux_pos_l)  / nb_tests, 3)
    moy_faux_neg_l  = round(sum(col_faux_neg_l)  / nb_tests, 3)
    moy_graphe      = round(sum(col_graphe)       / nb_tests, 3)

    print(f"""
Analyse des statistiques ({nb_tests} test(s))
    ENTITÉS (Sommets) :
    - Inventés  (Faux Positifs) : {moy_faux_pos_s} %
    - Oubliés   (Faux Négatifs) : {moy_faux_neg_s} %

    RELATIONS (Liens) :
    - Inventées (Faux Positifs) : {moy_faux_pos_l} %
    - Oubliées  (Faux Négatifs) : {moy_faux_neg_l} %

    SCORE GLOBAL :
    - Précision globale du graphe : {moy_graphe} %
    """)

def module_verification(n : int ,n_boucle: int ):  
    """
    Fonction qui lance une série de tests automatisés et affiche le bilan moyen.
    
    Entrées : 
    =========
        n : (int) paramètre pour la génération du graphe (ex: nombre de sommets).
        n_boucle : (int) nombre de tests à effectuer dans la boucle. 
    
    Sorties : 
    =========
        Aucune
    """
    for i in range (0,n_boucle) : 
        Path("rapport_aleatoire.md").unlink(missing_ok=True)

        print(f"Génération du graphe de test numéro {i+1}")
        g = generation_graphe(n)
        generation_texte_markdown(g)
        rajout_compte()
        # appel du llm 
        texte_pages = projet16_graphe_src.text_splitter("rapport_aleatoire.md")
        resultat = projet16_graphe_src.react_graph.invoke({
            "messages": [],
            "pages": texte_pages,
            "index": 0,
            "Json_f": [],
            "tentatives_suggestions": 0,
            "tentatives_fatales": 0
        })
        json_valides = [
            projet16_graphe_src.JSON_Normalise.model_validate_json(msg.content)
            for msg in resultat["Json_f"]
        ]
        texte_final = projet16_graphe_src.fusionner_jsons(json_valides)
        nom_base = "rapport_aleatoire.md".split(".")[0]
        with open(nom_base + ".json", "w", encoding="utf-8") as f:
            f.write(texte_final.model_dump_json(indent=4))
        
        # 3. On calcule les statistiques avec le JSON tout juste généré
        stat_faux_pos_s, stat_faux_neg_s, stat_faux_pos_l, stat_faux_neg_l, stat_graphe= partie_stat(g, "rapport_aleatoire.json")
        sauvegarde_historique((stat_faux_pos_s, stat_faux_neg_s, stat_faux_pos_l, stat_faux_neg_l, stat_graphe))  
    moy_test("historique_tests.csv")


if __name__ == "__main__":
    n = input("Entrez le nombre de sommets du graphe de départ :")
    n_boucle = input("Entrez le nombre de fois que le module de vérification doit effectuer le test : ")
    module_verification(int(n),int(n_boucle))
    
