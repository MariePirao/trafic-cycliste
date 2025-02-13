import streamlit as st
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
    

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
    'Face au 48 quai de la marne Face au 48 quai de la marne Vélos NE-SO': {#'Identifiant du compteur': '100047542-103047542',
                                                                            #'Identifiant du site de comptage':100047542.0,
                                                                            'nom_site': 'Face au 48 quai de la marne',
                                                                            #"Date d'installation du site de comptage":'2018-11-29',
                                                                            'Coordonnées géographiques':'48.89128,2.38606',
                                                                            #'Identifiant technique compteur' : 'Y2H18086321',
                                                                            'Lien vers photo du site de comptage':'https://filer.eco-counter-tools.com/file/09/7b81e8cbd56b562fe76b93caf48d40c87cb73c3ec384291e73be46917b919009/Y2H18086318_20240516111325.jpg'},
    'Face au 48 quai de la marne Face au 48 quai de la marne Vélos SO-NE': {#'Identifiant du compteur': '100047542-104047542',
                                                                            #'Identifiant du site de comptage':100047542.0,
                                                                           'nom_site': 'Face au 48 quai de la marne',
                                                                           #"Date d'installation du site de comptage":'2018-11-29',
                                                                           'Coordonnées géographiques':'48.89128,2.38606',
                                                                           #'Identifiant technique compteur' : 'Y2H18086321',
                                                                            'Lien vers photo du site de comptage':'https://filer.eco-counter-tools.com/file/09/7b81e8cbd56b562fe76b93caf48d40c87cb73c3ec384291e73be46917b919009/Y2H18086318_20240516111325.jpg'},
    'Pont des Invalides (couloir bus) N-S': {#'Identifiant du compteur': '100056223-101056223',
                                            #'Identifiant du site de comptage':100056223.0,
                                             'nom_site': 'Pont des Invalides (couloir bus)',
                                             #"Date d'installation du site de comptage":'2019-11-07',
                                             'Coordonnées géographiques':'48.86281,2.31037',
                                             #'Identifiant technique compteur' : 'X2H24052142',
                                             'Lien vers photo du site de comptage':'https://filer.eco-counter-tools.com/file/09/3bbfdc91a8bc53cb0b3b2933a2be0ec48c2af827b3f20a8454e46bc30d6c1009/15730380158370.jpg'},
    '27 quai de la Tournelle 27 quai de la Tournelle Vélos NO-SE' : {#'Identifiant du compteur': '100056336-104056336',
                                                                    #'Identifiant du site de comptage':100056336.0,
                                                                     'nom_site': '27 quai de la Tournelle',
                                                                     #"Date d'installation du site de comptage":'2019-11-14',
                                                                     'Coordonnées géographiques':'48.85013,2.35423',
                                                                     #'Identifiant technique compteur' : 'Y2H19070383',
                                                                     'Lien vers photo du site de comptage':'https://filer.eco-counter-tools.com/file/2c/fff4646e401ec8bc0a8ad926fa18aa5898e333e683f8d31b82b4092053a36c2c/Y2H19070383_20220803104828.jpg'},
    '27 quai de la Tournelle 27 quai de la Tournelle Vélos SE-NO' : {#'Identifiant du compteur': '100056336-103056336',
                                                                    #'Identifiant du site de comptage':100056336.0,
                                                                     'nom_site': '27 quai de la Tournelle',
                                                                     #"Date d'installation du site de comptage":'2019-11-14',
                                                                     'Coordonnées géographiques':'48.85013,2.35423',
                                                                     #'Identifiant technique compteur' : 'Y2H19070383',
                                                                     'Lien vers photo du site de comptage':'https://filer.eco-counter-tools.com/file/2c/fff4646e401ec8bc0a8ad926fa18aa5898e333e683f8d31b82b4092053a36c2c/Y2H19070383_20220803104828.jpg'},
    'Quai des Tuileries Quai des Tuileries Vélos NO-SE' : {#'Identifiant du compteur': '100056035-353266462',
                                                            #'Identifiant du site de comptage':100056035.0,
                                                           'nom_site': 'Quai des Tuileries',
                                                           #"Date d'installation du site de comptage":'2021-05-18',
                                                           'Coordonnées géographiques':'48.8635,2.32239',
                                                           #'Identifiant technique compteur' : 'Y2H24102626',
                                                           'Lien vers photo du site de comptage':'https://filer.eco-counter-tools.com/file/04/eda81070d90f5b3d5bac0eba49b237364d146fdc13aeedcf90391692e6734b04/Y2H19070337_20240611123828.jpg'},
    'Quai des Tuileries Quai des Tuileries Vélos SE-NO' : {#'Identifiant du compteur': '100056035-353266460',
                                                            #'Identifiant du site de comptage':100056035.0,
                                                           'nom_site': 'Quai des Tuileries',
                                                           #"Date d'installation du site de comptage":'2021-05-18',
                                                           'Coordonnées géographiques':'48.8635,2.32239',
                                                           #'Identifiant technique compteur' : 'Y2H24102626',
                                                           'Lien vers photo du site de comptage':'https://filer.eco-counter-tools.com/file/04/eda81070d90f5b3d5bac0eba49b237364d146fdc13aeedcf90391692e6734b04/Y2H19070337_20240611123828.jpg'},
    }
    # Remplacement des NaNs en fonction de 'nom_compteur'
   
    #for i in [“Identifiant du compteur”,“Identifiant du site de comptage”,“nom_site”,“Date d’installation du site de comptage”,“Coordonnées géographiques”,“Identifiant technique compteur”,“Lien vers photo du site de comptage”]:
    #    df[i] = df.apply(lambda x: replacement_dict[x[‘nom_compteur’]][i]
    #                     if pd.isna(x[i]) and x[‘nom_compteur’] in replacement_dict
    #                    else x[i], axis=1)
    for compteur, replacement in replacement_dict.items():
        for col, value in replacement.items():
            df.loc[df['nom_compteur'] == compteur, col] = df.loc[df['nom_compteur'] == compteur, col].fillna(value)
   

    return df


def addline(df):

    #creation d'un dataframe présentant toutes les combinaison dat/heure § identifiant compteurs
        #liste de toutes les heures/jour entre deux dates en retirant le passage a l'heure d'été
    date_range = pd.date_range("2024-01-01 04:00:00", "2025-01-29 23:00:00", freq="h")
    date_range = date_range[date_range != pd.Timestamp("2024-03-31 02:00:00")]
    d_date_range = pd.DataFrame(date_range, columns=["date_heure_comptage"])     ###nouveau
        #on créé un dataframe avec toutes les dates/heures pour tous les compteurs
    all_identifiants = pd.DataFrame(df["nom_compteur"].unique(), columns=["nom_compteur"])     ###nouveau
        # Faire le produit cartésien des identifiants et des dates/heures
    full_df = d_date_range.merge(all_identifiants, how="cross")
        #full_df.rename(columns={"0_x": "date_heure_comptage", "0_y": "nom_compteur"}, inplace=True)    devient inutile

    # on merge avec def pour créer toutes les nouvelles lignes
    merged_df = pd.merge(full_df, df[["nom_compteur", "date_heure_comptage"]], on=["nom_compteur", "date_heure_comptage"], how="left", indicator=True)

    #on recupépère les lignes de merged_df qui n'existent pas dans full df
    missing_hours_df = merged_df[merged_df['_merge'] == 'left_only']
    missing_hours_df = missing_hours_df.drop(columns=['_merge'])

    #on ajoute des colonne dans ce missing_hours_df pour permettre les gourpby  
    missing_hours_df["heure"] = missing_hours_df["date_heure_comptage"].dt.hour
    #missing_hours_df["Mois"] = missing_hours_df["date_heure_comptage"].dt.month_name()
    #missing_hours_df["Jour"] = missing_hours_df["date_heure_comptage"].dt.day_name()

    missing_hours_df["num_mois"] = missing_hours_df["date_heure_comptage"].dt.month
    missing_hours_df["num_jour_semaine"] = missing_hours_df["date_heure_comptage"].dt.dayofweek 
    #
    groupbycol = ["num_jour_semaine", "num_mois", "heure", "nom_compteur"]
    agg_dict = {col: 'first' for col in df.columns if col not in groupbycol}
    agg_dict['comptage_horaire'] = 'mean'
    df_saveMeanCompteur = df.groupby(groupbycol).agg(agg_dict).reset_index()
    df_saveMeanCompteur.drop(columns=["date_heure_comptage"], inplace=True)   ###nouveau
    merged_df2 = pd.merge(missing_hours_df, df_saveMeanCompteur, on=groupbycol, how="left")
    # je pense qu'il faut supprimer les lignes où on a pas pu calculer de moyenne mensuelle sur le jour
    merged_df2.dropna(subset=["comptage_horaire"], inplace=True) 

    df_final = pd.concat([merged_df2, df], ignore_index=True)

    return df_final


@st.cache_data
def preprocess_cyclisme(df):

    #drop des colonnes non utilisé
    df.drop(columns=["Identifiant du compteur","Identifiant du site de comptage","Date d'installation du site de comptage","Identifiant technique compteur"], inplace=True)
    df.drop(columns=["url_sites", "id_photo_1", "type_dimage","ID Photos","test_lien_vers_photos_du_site_de_comptage_"], inplace=True)
    
    df.rename(columns={"Nom du compteur": "nom_compteur", 
                       "Nom du site de comptage": "nom_site",
                       "Comptage horaire": "comptage_horaire",
                       "Date et heure de comptage": "date_heure_comptage",
                       "mois_annee_comptage":"mois_année"
                       }, inplace=True) 
    # traitement de la date en format datetime
    df["date_heure_comptage"] = df["date_heure_comptage"].str[:18]
    df["date_heure_comptage"] = pd.to_datetime(df["date_heure_comptage"])

     # traitement de la date en format datetime
    df = df.drop(df[(df['nom_compteur'].str.contains('27 quai de la Tournelle 27 quai de la Tournelle')) 
                    & (df["date_heure_comptage"] == '2024-11-12 23:00:00') & (df['comptage_horaire'] == 0.0)].index)

    #ajout d'une photo
    df.loc[df["nom_compteur"] == "35 boulevard de Ménilmontant NO-SE", ["Lien vers photo du site de comptage"]] = "https://drive.google.com/file/d/1GfPWIbU_Luv7tvOCJAk4AtciLjMEj8GA/view?usp=drive_link"

    #complétude des données manquantes et remplacement de certains noms
    df = rename(df)

    #ajout de données pour gérer les mois/nom de jour et année /heure etc..
    df["année"] = df["date_heure_comptage"].dt.year     # personne ne l'utilise pour le moment
    df["num_mois"] = df["date_heure_comptage"].dt.month    # personne ne l'utilise pour le moment
    df["num_jour_mois"] = df["date_heure_comptage"].dt.day
    df['num_jour_semaine'] = df["date_heure_comptage"].dt.dayofweek #lundi = 0 

    df["heure"] = df["date_heure_comptage"].dt.hour
    #df["Mois"] = df["date_heure_comptage"].dt.month_name()
    #df["Jour"] = df["date_heure_comptage"].dt.day_name()

    df = addline(df)


    df[['latitude', 'longitude']] = df['Coordonnées géographiques'].str.split(',', expand=True)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    #drop des colonnes non utilisé
    df.drop(columns=["Coordonnées géographiques"], inplace=True)

    df = df[df['année'].isin([2024, 2025]) & (df["nom_compteur"] != "10 avenue de la Grande Armée 10 avenue de la Grande Armée [Bike OUT]")
            & (df["nom_compteur"] != "44 avenue des Champs Elysées SE-NO") ]
    
    df.sort_values(by=["date_heure_comptage","nom_compteur"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    #df = correctionCompteur0(df)

    return df

def preprocess_meteo(df):


    """Prétraitement des données : suppression des valeurs manquantes et sélection des colonnes numériques."""
    df_m = df.drop(columns='wind_speed_100m (km/h)', axis = 1)
    df_m = df.drop(columns='wind_speed_100m (km/h)', axis = 1)

    # Convertir la colonne 'heure_gmt' en type datetime
    df_m['time'] = pd.to_datetime(df['time'])

    # Localiser en GMT (UTC dans ce cas)
    df_m['time'] = df_m['time'].dt.tz_localize('GMT')

    # Convertir l'heure GMT en heure de Paris (CET ou CEST en fonction de la date)
    df_m['time'] = df_m['time'].dt.tz_convert('Europe/Paris')

    # stronquer les decalages horaire
    df_m["time"] = df_m["time"].astype("str").str[:18]
    df_m["time"] = pd.to_datetime(df_m["time"])


    df_m['precipitation'] = pd.cut(x=df_m['precipitation (mm)'], bins=[0.0, 0.5, 3.5, 16.0],
                                   labels=['Pas de pluie/bruine', 'Pluie modérée', 'Fortes averses'], right=False)
    df_m['wind'] = pd.cut(x=df_m['wind_speed_10m (km/h)'], bins=[0.0, 5, 19, 38, 43.0],
                          labels=['Pas de vent', 'Vent modérée', 'vent', 'grand vent'], right=False)
    df_m['temperature'] = pd.cut(x=df_m['temperature_2m (°C)'], bins=[-10, 0, 10, 20, 30, 36],
                                 labels=['Gel','Froid', 'Tempéré', 'Chaud', 'Très chaud'], right=False)
    return df_m

def preprocess_vacancesferie(df_v, df_jf):
    """Prétraitement des données : fichier des jours fériés"""
    # creation d'une colonne en datatime
    #df_jf['time'] = pd.to_datetime(df_jf['date'], utc=True).dt.tz_convert(None)
    df_jf['time'] = pd.to_datetime(df_jf['date'])
    #on ne selectionne que les dates qui correspondent a notre jeu de données
    df_jf = df_jf[(df_jf['time'] >= '2024-01-01') & (df_jf['time'] <= '2025-02-01')]
    # Supprimer la colonne 'zone'
    df_jf = df_jf.drop(columns=['zone'])

    # On ajoute les samedi et dimanche
    #all_dates = pd.date_range(start=df_jf['time'].min(), end=df_jf['time'].max() )
    #weekends = all_dates[all_dates.weekday.isin([5, 6])]  # 5 = samedi, 6 = dimanche
    #weekends_df = pd.DataFrame({
    #    'date': weekends.strftime('%Y-%m-%d'),
    #    'annee': weekends.year,
    #    'nom_jour_ferie': weekends.day_name(),  # 'Samedi' ou 'Dimanche'
    #    'time': weekends})
    #on ne créé pas de doublon si la date est déjà un JF on n'ajoute pas samedi et dimanche
    #weekends_df = weekends_df[~weekends_df['date'].isin(df_jf['date'].astype(str))]

    #   3. Ajouter les week-ends au DataFrame original
    #df_jf = pd.concat([df_jf, weekends_df], ignore_index=True)
    #df_jf = df_jf.sort_values(by='time').reset_index(drop=True)


    # creation d'une colonne en datatime
    #df_v['time'] = pd.to_datetime(df_v['date'], utc=True).dt.tz_convert(None)
    df_v['time'] = pd.to_datetime(df_v['date'])
    #suppression des lignes hors période
    #df_v = df_v.dropna(axis = 0, how = 'all', subset = ['nom_vacances'])
    df_filtered = df_v[(df_v['time'] >= '2024-01-01') & (df_v['time'] <= '2025-02-01')]


    #creation d'un dataframe commun jour férier et congé
    df_jv = df_filtered
    new_rows = []
    for _, element in df_jf.iterrows():
        if element['time'] not in df_jv['time'].values:
            new_rows.append({
                'date': element['date'],
                'vacances_zone_a': True,
                'vacances_zone_b': True,
                'vacances_zone_c': True,
                'nom_vacances': element['nom_jour_ferie'],
                'time': element['time']
                })
        else:
            df_jv.loc[df_jv['time'] == element['time'], ['vacances_zone_a', 'vacances_zone_b', 'vacances_zone_c']] = True
            #df_jv.loc[df_jv['time'] == element['time'], 'nom_vacances'] = (df_jv.loc[df_jv['time'] == element['time'], 'nom_vacances'] + " / " + element['nom_jour_ferie'])
            df_jv.loc[df_jv['time'] == element['time'], 'nom_vacances'] = (df_jv.loc[df_jv['time'] == element['time'], 'nom_vacances'].fillna('') + " / " + element['nom_jour_ferie'])
    # Ajouter les nouvelles lignes en une seule fois avec pd.concat()
    if new_rows:
        df_jv = pd.concat([df_jv, pd.DataFrame(new_rows)], ignore_index=True)
    
    df_jv = df_jv.rename(columns = {'nom_vacances':'nom_conge'})

    replace_dict = {True: 1, False: 0}
    df_jv['vacances_zone_a'] = df_jv['vacances_zone_a'].replace(replace_dict).infer_objects(copy=False)
    df_jv['vacances_zone_b'] = df_jv['vacances_zone_b'].replace(replace_dict).infer_objects(copy=False)
    df_jv['vacances_zone_c'] = df_jv['vacances_zone_c'].replace(replace_dict).infer_objects(copy=False)


    return df_jv

def preprocess_photo(df):
    """Prétraitement des données : suppression des valeurs manquantes et sélection des colonnes numériques."""
    df.drop(columns=['Unnamed: 0'], inplace=True)
    return df

def corriger_comptage(dfenter):
# Définir les dates de la période à corriger et de la période de référence
    df_result = dfenter.copy()
    periode_reference_debut = '2024-12-29 01:00'
    periode_reference_fin = '2024-12-30 06:00'
    periode_a_corriger_debut = '2025-01-05 01:00'
    periode_a_corriger_fin = '2025-01-06 06:00'

    # Filtrer les données de la période de référence (29/12/2024 à 01h - 30/12/2024 à 06h)
    df_reference = df_result.loc[(df_result['date_heure_comptage'] >= periode_reference_debut) & 
                          (df_result['date_heure_comptage'] <= periode_reference_fin) &
                          (df_result['nom_compteur'] == "Quai d'Orsay O-E")]

    # Filtrer les données de la période incorrecte (05/01/2025 à 01h - 06/01/2025 à 06h)
    df_incorrecte = df_result.loc[(df_result['date_heure_comptage'] >= periode_a_corriger_debut) & 
                           (df_result['date_heure_comptage'] <= periode_a_corriger_fin) &
                           (df_result['nom_compteur'] == "Quai d'Orsay O-E")]

    # Créer un dictionnaire des valeurs de comptage pour la période de référence en fonction des heures
    reference_dict = df_reference.set_index(df_reference['heure'])['comptage_horaire'].to_dict()

    # Boucle pour appliquer les valeurs de comptage de la période de référence à la période incorrecte
    for idx, row in df_incorrecte.iterrows():
        heure = row['date_heure_comptage'].hour  # Récupérer l'heure de l'enregistrement
        if heure in reference_dict:
            df_result.at[idx, 'comptage_horaire'] = reference_dict[heure]  # Appliquer la valeur de comptage correspondante

    df_result = df_result[df_result["num_mois"] != '7']

    return df_result


def correctionCompteur0(df):
    df_result = df.copy()

    #add colomne neutralisé
    df_result = neutralise(df_result)

    df_result.sort_values(by=["date_heure_comptage","nom_compteur"], ascending=True, inplace=True)
    df_result.reset_index(drop=True, inplace=True)

    df_result0NotNeutralise = searchCompteur0
    #list des compteur avec les preriode a 0
    df_result = remplacerParMoyenne(df_result0NotNeutralise, dat_start, date_end)

    return df_result

def remplacerParMoyenne(df, df_0, start_date, end_date):

    #df2 = df.sample(100)
    #df4 = df2
    #df2["Comptage horaire"] = 0
    #df6 = pd.concat([df2,df],axis=0)
    #df6 = df.drop_duplicates(keep="first")

    groupbytout = ["Num_Jour_Semaine", "Heure"]
    moyenneGen = df.groupby(groupbytout)['Comptage horaire'].mean().reset_index()
    moyenneGen = moyenneGen.rename(columns={"Comptage horaire": "Comptage horaire général moyen"})

    df = pd.merge(df, moyenneGen, on=groupbytout, how="left")

    #meme démarche mais pour les compteur a corriger
    groupbycolproportion = ["Num_Jour_Semaine", "Heure", "Nom du compteur"]
    moyenneGenCompteur = df_0.groupby(groupbycolproportion)['Comptage horaire'].mean().reset_index()
    moyenneGenCompteur = moyenneGenCompteur.rename(columns={"Comptage horaire": "Comptage horaire général compteur"})
    df4 = pd.merge(df4, moyenneGenCompteur, on=groupbycolproportion, how="left")
    
    groupbycolheure = ["Numero_Jour_Mois", "Numero_Mois", "Heure"] # reprendre simplement l'heure
    moyenneGenHeure = df6.groupby(groupbycolheure)['Comptage horaire'].mean().reset_index()
    moyenneGenHeure = moyenneGenHeure.rename(columns={"Comptage horaire": "Comptage horaire général heure"})
    df4 = pd.merge(df4, moyenneGenHeure, on=groupbycolheure, how="left")
    df.head()
    df4["Proportion"] = (df4["Comptage horaire général compteur"]/df4["Comptage horaire général moyen"])*df4["Comptage horaire général heure"]
    df4["diff_proportion"] = (df4["Comptage horaire"] - df4["Proportion"]).abs()
    df5 = df4[["Identifiant du compteur","Nom du compteur","Comptage horaire","Date et heure de comptage","Proportion"]]
    df5
    return df_result

def searchCompteur0 (df):
    df["zero_count"] = (df["Comptage horaire"] == 0).astype(int)

    # Fenêtre mobile de 20 heures sur chaque compteur
    df["rolling_sum"] = df.groupby("Nom du compteur")["zero_count"].rolling(window=24, min_periods=24).sum().reset_index(level=0, drop=True)

    # Filtrer les lignes où il y a 24 heures consécutives de 0
    df_filtered = df[df["rolling_sum"] == 24]

    # Extraire le jour et l'heure de la colonne "Date et heure de comptage"
    df_filtered["Jour"] = df_filtered["Date et heure de comptage"].dt.date
    df_filtered['Heure'] = df_filtered['Date et heure de comptage'].dt.hour

    # Sélectionner les colonnes pertinentes
    df_result = df_filtered[["Nom du compteur", "Jour"]].drop_duplicates()

    # Trier les résultats par "Nom du compteur", "Jour", et "Heure"
    df_result_sorted = df_result.sort_values(by=["Nom du compteur", "Jour"])

    # Afficher les résultats
    print(df_result_sorted.info())
    #print(f"Unique compteur names: {df_result_sorted['Nom du compteur'].unique()}")
    #print(df_result_sorted)