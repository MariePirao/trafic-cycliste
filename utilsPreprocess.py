import pandas as pd
# Pour éviter d'avoir les messages warning
import warnings
warnings.filterwarnings('ignore')
    
'''FONCTIONS DE PREPROCESSING DU FICHIER CYCLISTE'''

def rename(df):
    # Dictionnaire pour le remplacement des valeurs dans la colonne 'nom_compteur'
    replace_dict = {
        'Face au 48 quai de la marne NE-SO': 'Face au 48 quai de la marne Face au 48 quai de la marne Vélos NE-SO',
        'Face au 48 quai de la marne SO-NE': 'Face au 48 quai de la marne Face au 48 quai de la marne Vélos SO-NE',
        'Pont des Invalides N-S': 'Pont des Invalides (couloir bus) N-S',
        '27 quai de la Tournelle NO-SE': '27 quai de la Tournelle 27 quai de la Tournelle Vélos NO-SE',
        '27 quai de la Tournelle SE-NO': '27 quai de la Tournelle 27 quai de la Tournelle Vélos SE-NO',
        'Quai des Tuileries NO-SE': 'Quai des Tuileries Quai des Tuileries Vélos NO-SE',
        'Quai des Tuileries SE-NO': 'Quai des Tuileries Quai des Tuileries Vélos SE-NO',
        '10 avenue de la Grande Armée 10 avenue de la Grande Armée [Bike IN]': '10 avenue de la Grande Armée SE-NO'
    }
    
    # Remplacer les valeurs dans 'nom_compteur'
    df['nom_compteur'] = df['nom_compteur'].replace(replace_dict)

    # Création  d'un dictionnaire pour remplacer les NaNs en fonction du nom_compteur :
    replacement_dict = {
    'Face au 48 quai de la marne Face au 48 quai de la marne Vélos NE-SO': {'nom_site': 'Face au 48 quai de la marne',
                                                                            'Coordonnées géographiques':'48.89128,2.38606',
                                                                            'Lien vers photo du site de comptage':'https://filer.eco-counter-tools.com/file/09/7b81e8cbd56b562fe76b93caf48d40c87cb73c3ec384291e73be46917b919009/Y2H18086318_20240516111325.jpg'},
    'Face au 48 quai de la marne Face au 48 quai de la marne Vélos SO-NE': {'nom_site': 'Face au 48 quai de la marne',
                                                                           'Coordonnées géographiques':'48.89128,2.38606',
                                                                           'Lien vers photo du site de comptage':'https://filer.eco-counter-tools.com/file/09/7b81e8cbd56b562fe76b93caf48d40c87cb73c3ec384291e73be46917b919009/Y2H18086318_20240516111325.jpg'},
    'Pont des Invalides (couloir bus) N-S': {'nom_site': 'Pont des Invalides (couloir bus)',
                                             'Coordonnées géographiques':'48.86281,2.31037',
                                             'Lien vers photo du site de comptage':'https://filer.eco-counter-tools.com/file/09/3bbfdc91a8bc53cb0b3b2933a2be0ec48c2af827b3f20a8454e46bc30d6c1009/15730380158370.jpg'},
    '27 quai de la Tournelle 27 quai de la Tournelle Vélos NO-SE' : {'nom_site': '27 quai de la Tournelle',
                                                                     'Coordonnées géographiques':'48.85013,2.35423',
                                                                     'Lien vers photo du site de comptage':'https://filer.eco-counter-tools.com/file/2c/fff4646e401ec8bc0a8ad926fa18aa5898e333e683f8d31b82b4092053a36c2c/Y2H19070383_20220803104828.jpg'},
    '27 quai de la Tournelle 27 quai de la Tournelle Vélos SE-NO' : {'nom_site': '27 quai de la Tournelle',
                                                                     'Coordonnées géographiques':'48.85013,2.35423',
                                                                     'Lien vers photo du site de comptage':'https://filer.eco-counter-tools.com/file/2c/fff4646e401ec8bc0a8ad926fa18aa5898e333e683f8d31b82b4092053a36c2c/Y2H19070383_20220803104828.jpg'},
    'Quai des Tuileries Quai des Tuileries Vélos NO-SE' : {'nom_site': 'Quai des Tuileries',
                                                           'Coordonnées géographiques':'48.8635,2.32239',
                                                           'Lien vers photo du site de comptage':'https://filer.eco-counter-tools.com/file/04/eda81070d90f5b3d5bac0eba49b237364d146fdc13aeedcf90391692e6734b04/Y2H19070337_20240611123828.jpg'},
    'Quai des Tuileries Quai des Tuileries Vélos SE-NO' : {'nom_site': 'Quai des Tuileries',
                                                           'Coordonnées géographiques':'48.8635,2.32239',
                                                           'Lien vers photo du site de comptage':'https://filer.eco-counter-tools.com/file/04/eda81070d90f5b3d5bac0eba49b237364d146fdc13aeedcf90391692e6734b04/Y2H19070337_20240611123828.jpg'},
    }
    # Remplacement des NaNs en fonction de 'nom_compteur'
   
    for compteur, replacement in replacement_dict.items():
        for col, value in replacement.items():
            df.loc[df['nom_compteur'] == compteur, col] = df.loc[df['nom_compteur'] == compteur, col].fillna(value)

    return df


def addline(df):

    #liste de toutes les heures/jour entre deux dates en retirant le passage a l'heure d'été
    date_range = pd.date_range("2024-01-01 04:00:00", "2025-01-29 23:00:00", freq="h")
    date_range = date_range[date_range != pd.Timestamp("2024-03-31 02:00:00")]
    d_date_range = pd.DataFrame(date_range, columns=["date_heure_comptage"]) 
    #on créé un dataframe avec toutes les dates/heures pour tous les compteurs
    all_identifiants = pd.DataFrame(df["nom_compteur"].unique(), columns=["nom_compteur"]) 
    # Faire le produit cartésien des identifiants et des dates/heures
    full_df = d_date_range.merge(all_identifiants, how="cross")

    # on merge avec def pour créer toutes les nouvelles lignes
    merged_df = pd.merge(full_df, df[["nom_compteur", "date_heure_comptage"]], on=["nom_compteur", "date_heure_comptage"], how="left", indicator=True)

    #on recupépère les lignes de merged_df qui n'existent pas dans full df
    missing_hours_df = merged_df[merged_df['_merge'] == 'left_only']
    missing_hours_df = missing_hours_df.drop(columns=['_merge'])

    #on ajoute des colonne dans ce missing_hours_df pour permettre les gourpby  
    missing_hours_df["heure"] = missing_hours_df["date_heure_comptage"].dt.hour

    missing_hours_df["num_mois"] = missing_hours_df["date_heure_comptage"].dt.month
    missing_hours_df["num_jour_semaine"] = missing_hours_df["date_heure_comptage"].dt.dayofweek 
    
    groupbycol = ["num_jour_semaine", "num_mois", "heure", "nom_compteur"]
    agg_dict = {col: 'first' for col in df.columns if col not in groupbycol}
    agg_dict['comptage_horaire'] = 'mean'
    df_saveMeanCompteur = df.groupby(groupbycol).agg(agg_dict).reset_index()
    df_saveMeanCompteur.drop(columns=["date_heure_comptage"], inplace=True)   
    merged_df2 = pd.merge(missing_hours_df, df_saveMeanCompteur, on=groupbycol, how="left")
    merged_df2.dropna(subset=["comptage_horaire"], inplace=True) 

    df_final = pd.concat([merged_df2, df], ignore_index=True)

    return df_final


def preprocess_cyclisme(df):

    df_work = df.copy()
    #drop des colonnes non utilisé
    df_work.drop(columns=["Identifiant du compteur","Identifiant du site de comptage","Date d'installation du site de comptage","Identifiant technique compteur"], inplace=True)
    df_work.drop(columns=["url_sites", "id_photo_1", "type_dimage","ID Photos","test_lien_vers_photos_du_site_de_comptage_"], inplace=True)
    
    df_work.rename(columns={"Nom du compteur": "nom_compteur", "Nom du site de comptage": "nom_site",
                            "Comptage horaire": "comptage_horaire","Date et heure de comptage": "date_heure_comptage",
                            "mois_annee_comptage":"mois_année"}, inplace=True) 
    # traitement de la date en format datetime
    df_work["date_heure_comptage"] = df_work["date_heure_comptage"].str[:18]
    df_work["date_heure_comptage"] = pd.to_datetime(df_work["date_heure_comptage"])

     # suppression du doublon de lignes
    df_work = df_work.drop(df_work[(df_work['nom_compteur'].str.contains('27 quai de la Tournelle 27 quai de la Tournelle')) 
                    & (df_work["date_heure_comptage"] == '2024-11-12 23:00:00') & (df_work['comptage_horaire'] == 0.0)].index)

    #ajout d'une photo manquante
    df_work.loc[df_work["nom_compteur"] == "35 boulevard de Ménilmontant NO-SE", ["Lien vers photo du site de comptage"]] = "https://drive.google.com/file/d/1GfPWIbU_Luv7tvOCJAk4AtciLjMEj8GA/view?usp=drive_link"

    #complétude des données manquantes et remplacement de certains noms
    df_work = rename(df_work)

    #ajout de données pour gérer les mois/nom de jour et année /heure etc..
    df_work["année"] = df_work["date_heure_comptage"].dt.year     # personne ne l'utilise pour le moment
    df_work["num_mois"] = df_work["date_heure_comptage"].dt.month  
    df_work["num_jour_mois"] = df_work["date_heure_comptage"].dt.day
    df_work['num_jour_semaine'] = df_work["date_heure_comptage"].dt.dayofweek 
    df_work["heure"] = df_work["date_heure_comptage"].dt.hour
    df_work["date"] = df_work["date_heure_comptage"].dt.date
    df_work["date"] = pd.to_datetime(df_work["date"])

    df_work = addline(df_work)

    df_work[['latitude', 'longitude']] = df_work['Coordonnées géographiques'].str.split(',', expand=True)
    df_work["latitude"] = pd.to_numeric(df_work["latitude"], errors="coerce")
    df_work["longitude"] = pd.to_numeric(df_work["longitude"], errors="coerce")
    df_work['latitude'] = df_work['latitude'].astype(float)
    df_work['longitude'] = df_work['longitude'].astype(float)
    #drop des colonnes non utilisé
    df_work.drop(columns=["Coordonnées géographiques"], inplace=True)

    #voir decision de droper ces 3 compteurs car semble erroné ou trop incomplet
    df_work = df_work[df_work['année'].isin([2024, 2025])]
    
    df_work.sort_values(by=["date_heure_comptage","nom_compteur"], ascending=True, inplace=True)
    df_work.reset_index(drop=True, inplace=True)

    return df_work


'''FONCTION DE PREPROCESSING DU FICHIER METEO'''


def preprocess_meteo(df):

    """Prétraitement des données : suppression des valeurs manquantes et sélection des colonnes numériques."""
    df_m = df.drop(columns='wind_speed_100m (km/h)', axis = 1)
    df_m = df_m.rename(columns={"is_day ()": "fait_jour","precipitation (mm)": "precipitation_mm","wind_speed_10m (km/h)": "wind_speed","temperature_2m (°C)": "temperature_2m"})

    # Convertir la colonne 'heure_gmt' en type datetime
    df_m['time'] = pd.to_datetime(df['time'])
    df_m['time'] = df_m['time'].dt.tz_localize('GMT')
    df_m['time'] = df_m['time'].dt.tz_convert('Europe/Paris')

    # tronquer les decalages horaire
    df_m["time"] = df_m["time"].astype("str").str[:18]
    df_m["time"] = pd.to_datetime(df_m["time"])

    # categoriser les données numériques
    df_m['precipitation'] = pd.cut(x=df_m['precipitation_mm'], bins=[0.0, 0.5, 3.5, 16.0],
                                   labels=['Pas de pluie/bruine', 'Pluie modérée', 'Fortes averses'], right=False)
    df_m['wind'] = pd.cut(x=df_m['wind_speed'], bins=[0.0, 5, 19, 38, 43.0],
                          labels=['Pas de vent', 'Vent modérée', 'vent', 'grand vent'], right=False)
    df_m['temperature'] = pd.cut(x=df_m['temperature_2m'], bins=[-10, 0, 10, 20, 30, 36],
                                 labels=['Gel','Froid', 'Tempéré', 'Chaud', 'Très chaud'], right=False)
    
    return df_m

'''FONCTIONS DE PREPROCESSING DU FICHIER JOUR FERIE'''

def preprocess_vacancesferie(df_v, df_jf):

    """Prétraitement des données : fichier des jours fériés"""
    # creation d'une colonne en datatime
    #df_jf['time'] = pd.to_datetime(df_jf['date'], utc=True).dt.tz_convert(None)
    df_jf['date'] = pd.to_datetime(df_jf['date'])
    #on ne selectionne que les dates qui correspondent a notre jeu de données
    df_jf = df_jf[(df_jf['date'] >= '2024-01-01') & (df_jf['date'] <= '2025-03-31')]
    # Supprimer la colonne 'zone'
    df_jf = df_jf.drop(columns=['zone'])

    # creation d'une colonne en datatime
    df_v['date'] = pd.to_datetime(df_v['date'])
    #suppression des lignes hors période
    df_filtered = df_v[(df_v['date'] >= '2024-01-01') & (df_v['date'] <= '2025-03-31')]

    #creation d'un dataframe commun jour férier et congé
    df_jv = df_filtered
    new_rows = []
    for _, element in df_jf.iterrows():
        if element['date'] not in df_jv['date'].values:
            new_rows.append({
                'date': element['date'],
                'vacances_zone_a': True,
                'vacances_zone_b': True,
                'vacances_zone_c': True,
                'nom_vacances': element['nom_jour_ferie'],
                'time': element['time']
                })
        else:
            df_jv.loc[df_jv['date'] == element['date'], ['vacances_zone_a', 'vacances_zone_b', 'vacances_zone_c']] = True
            df_jv.loc[df_jv['date'] == element['date'], 'nom_vacances'] = (df_jv.loc[df_jv['date'] == element['date'], 'nom_vacances'].fillna('') + " / " + element['nom_jour_ferie'])
    # Ajouter les nouvelles lignes en une seule fois avec pd.concat()
    if new_rows:
        df_jv = pd.concat([df_jv, pd.DataFrame(new_rows)], ignore_index=True)
    
    df_jv = df_jv.rename(columns = {'nom_vacances':'nom_conge'})

    replace_dict = {True: 1, False: 0}
    df_jv['vacances_zone_a'] = df_jv['vacances_zone_a'].replace(replace_dict).infer_objects(copy=False)
    df_jv['vacances_zone_b'] = df_jv['vacances_zone_b'].replace(replace_dict).infer_objects(copy=False)
    df_jv['vacances_zone_c'] = df_jv['vacances_zone_c'].replace(replace_dict).infer_objects(copy=False)

    return df_jv

'''FONCTIONS DE PREPROCESSING DU FICHIER PHOTO'''

def preprocess_photo(df):
    """Suppression colonnes inutiles"""
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df.drop(columns=['2sens'], inplace=True)
    df.drop(columns=['Separe'], inplace=True)
    return df

'''FONCTIONS DE CORRECTIONS SUR FICHIER CYCLISME '''

def correctionDataviz(df_merged_cleaned):
    """Dernier traitement sur le dataframe avant la modélisation"""
    df_work = df_merged_cleaned.copy()
    #selon la recommandation concernant le graph sur les vacances nous allons utiliser qu'une colonne vacances
    df_work['vacances'] = df_work[['vacances_zone_a', 'vacances_zone_b', 'vacances_zone_c']].apply(lambda x: 1 if x.max() == 1 else 0, axis=1)
    # Créer une nouvelle colonne 'weekend' où 1 correspond au weekend (samedi ou dimanche)
    df_work['weekend'] = df_work['num_jour_semaine'].apply(lambda x: 1 if x >= 5 else 0)

    return df_work

def corriger_comptage(dfenter):

    df_result = dfenter.copy()
    
    #On en profites pour retirer ces compteurs qui sont érronés suite au graph
    df_result = df_result[(df_result["nom_compteur"] != "10 avenue de la Grande Armée 10 avenue de la Grande Armée [Bike OUT]")
            & (df_result["nom_compteur"] != "44 avenue des Champs Elysées SE-NO") 
            & (df_result["nom_compteur"] != "106 avenue Denfert Rochereau NE-SO")]

    # On passe à la correstion de la valeur abherrante.
    # Définition des dates de la période à corriger et de la période de référence
    periode_reference_debut = '2024-12-29 01:00'
    periode_reference_fin = '2024-12-30 06:00'
    periode_a_corriger_debut = '2025-01-05 01:00'
    periode_a_corriger_fin = '2025-01-06 06:00'

    # Période de référence
    df_reference = df_result.loc[(df_result['date_heure_comptage'] >= periode_reference_debut) & 
                          (df_result['date_heure_comptage'] <= periode_reference_fin) &
                          (df_result['nom_compteur'] == "Quai d'Orsay O-E")]

    # Période à corriger
    df_incorrecte = df_result.loc[(df_result['date_heure_comptage'] >= periode_a_corriger_debut) & 
                           (df_result['date_heure_comptage'] <= periode_a_corriger_fin) &
                           (df_result['nom_compteur'] == "Quai d'Orsay O-E")]

    # Créer un dictionnaire des valeurs de comptage pour la période de référence en fonction des heures
    reference_dict = df_reference.set_index(df_reference['heure'])['comptage_horaire'].to_dict()

    # Boucle pour appliquer les valeurs de comptage de la période de référence à la période incorrecte
    for idx, row in df_incorrecte.iterrows():
        heure = row['date_heure_comptage'].hour  #
        if heure in reference_dict:
            df_result.at[idx, 'comptage_horaire'] = reference_dict[heure]  # Appliquer la valeur de comptage correspondante

    #on corrige maintenant les compteurs à 0
    df_result = correctionCompteur0(df_result)

    return df_result


def correctionCompteur0(df):
    df_result = df.copy()

    #compteur avec les preriode a 0
    df_result0NotNeutralise = searchCompteur0(df_result)
    df_result = remplacerParMoyenne(df_result, df_result0NotNeutralise)

    return df_result

def remplacerParMoyenne(df, df_0):

    groupbytout = ["num_jour_semaine", "heure"]
    moyenneGen = df.groupby(groupbytout)['comptage_horaire'].mean().reset_index()
    moyenneGen = moyenneGen.rename(columns={"comptage_horaire": "Comptage moyen H/J"})

    df_0 = pd.merge(df_0, moyenneGen, on=groupbytout, how="left")

    #meme démarche mais pour les compteur a corriger
    groupbycolproportion = ["num_jour_semaine", "heure", "nom_compteur"]
    moyenneGenCompteur = df.groupby(groupbycolproportion)['comptage_horaire'].mean().reset_index()
    moyenneGenCompteur = moyenneGenCompteur.rename(columns={"comptage_horaire": "Comptage moyen compteur H/J"})
    
    df_0 = pd.merge(df_0, moyenneGenCompteur, on=groupbycolproportion, how="left")
    
    moyenneGenHeure = df.groupby("date_heure_comptage")['comptage_horaire'].mean().reset_index()
    moyenneGenHeure = moyenneGenHeure.rename(columns={"comptage_horaire": "Comptage moyen par date/heure"})

    df_0 = pd.merge(df_0, moyenneGenHeure, on="date_heure_comptage", how="left")
    
    df_0["compteur_final"] = round((df_0["Comptage moyen compteur H/J"]/df_0["Comptage moyen H/J"])*df_0["Comptage moyen par date/heure"])
    df_0.drop(columns=["Comptage moyen compteur H/J", "Comptage moyen H/J", "Comptage moyen par date/heure","num_jour_semaine", "heure"], inplace=True)

    df_result = pd.merge(df, df_0, on=["date_heure_comptage", "nom_compteur"], how="left")
    df_result["compteur_final"] = df_result["compteur_final"].fillna(df_result["comptage_horaire"])
    df_result["comptage_horaire"] = df_result["compteur_final"]
    df_result = df_result.drop(columns=["compteur_final"])

    return df_result

def searchCompteur0 (df):

    ''' Dans cette méthode on va chercher les itération de 10 lignes sur le meme compteur qui présente un compteur à 0 sans raison de neutralisation'''
    
    df_work = df.copy()
    #on va chercher 10 itérations d'affilées. IL faut donc s'assurer que le df est trié par com de compteur puis date_heure_comptage
    df_work.sort_values(by=["nom_compteur","date_heure_comptage",], ascending=True, inplace=True)
    df_work.reset_index(drop=True, inplace=True)

    df_work["comptage_horaire"]  = df_work["comptage_horaire"].apply(lambda x: 0 if x<1 else x)

    df_work["zero_count"] = ((df_work["comptage_horaire"] == 0) & (df_work["neutralise"] == 0)).astype(int)
    
    df_result = []
    iteration = 0
    addLigne = {}
    previousCompteur = ''
    
    for index, ligne in df_work.iterrows():
        if (ligne["zero_count"] == 0) or (ligne["nom_compteur"] != previousCompteur):
            if addLigne :
                df_result.append(addLigne)
            iteration = 0
            addLigne = {}
        else :
            iteration += 1
            if (iteration == 1):
                dateStart = ligne["date_heure_comptage"]
            if (iteration >= 10):
                dateEnd = ligne["date_heure_comptage"]
                addLigne = {"nom_compteur": ligne["nom_compteur"], "dateStart": dateStart, "dateEnd": dateEnd}
        previousCompteur =ligne["nom_compteur"]
                

    # transformer en dataframe
    df_result = pd.DataFrame(df_result)
    # Trier les résultats par "Nom du compteur", "Jour", et "Heure"
    df_result_sorted = df_result.sort_values(by=["nom_compteur", "dateStart", "dateEnd"])

    # Créer une liste pour stocker les résultats
    expanded_data = []
    # Pour chaque ligne du dataframe d'origine, générer toutes les heures entre dateStart et dateEnd
    for _, row in df_result_sorted.iterrows():
        hours_range = pd.date_range(start=row['dateStart'], end=row['dateEnd'], freq='H')  # Crée un range horaire
        for hour in hours_range:
            datetime = hour.to_pydatetime()
            heure = datetime.hour
            numJour = datetime.weekday()
            expanded_data.append([row['nom_compteur'], hour,heure,numJour])
    # Créer un nouveau dataframe avec les résultats
    expanded_df = pd.DataFrame(expanded_data, columns=['nom_compteur', 'date_heure_comptage', "heure","num_jour_semaine"])

    return expanded_df