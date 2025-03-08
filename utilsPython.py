import glob
import numpy as np
import pandas as pd
# Pour éviter d'avoir les messages warning
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from config import Config
import requests
import warnings
import os
warnings.filterwarnings('ignore')


def load_data(file_path, separator, numheader):
    return pd.read_csv(file_path, sep=separator, header=numheader)

def create_data(df, file_path):
    df.to_csv(file_path + 'dataframeFinal.csv')

def create_data3J(df, extension):
    csv = df.to_csv(extension+'_Save3J.csv')


def informationDF(df):
    '''Fonction qui permet de faire un affichage propre dans la page streamlit des info du dataframe
    en reepectant le type de la variable'''
    int_columns = df.select_dtypes(include=['int64']).columns
    info_dict = {
      'Valeurs non-null': df.notnull().sum(),
      'Dtype': df.dtypes,
      'Valeur unique': [df[col].nunique(dropna=False) for col in df.columns],
      'Min': [df[col].min() if col in int_columns else '' for col in df.columns],
      'Max': [df[col].max() if col in int_columns else '' for col in df.columns],
      'Médiane': [df[col].median() if col in int_columns else '' for col in df.columns],
      'Moyenne': [df[col].mean() if col in int_columns else '' for col in df.columns]
    }

    info_df = pd.DataFrame(info_dict)
    info_dict_aff = info_df.applymap(str)
    return info_dict_aff

def searchUnique(df, attribut):
    return df[attribut].unique()

def removeJoblibFile():
    # chercher tous les fichiers .joblib
    files_to_delete = glob.glob("*.joblib")
    # Supprimer les fichiers
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Fichier supprimé : {file}")
        except Exception as e:
            print(f"Erreur lors de la suppression de {file}: {e}")

def merge(df_cleaned,df_m,df_jv, df_p, df_ir):

    df_result = df_cleaned.copy()
    df_result['time']=df_result["date_heure_comptage"]

    #merge des données météo dans df
    df_result = df_result.merge(df_m, on='time', how='left')

    #merge des données vacances/férié dans df
    df_result = df_result.merge(df_jv, on='date', how='left')

    df_result['vacances_zone_a'] = df_result['vacances_zone_a'].fillna(0)
    df_result['vacances_zone_b'] = df_result['vacances_zone_b'].fillna(0)
    df_result['vacances_zone_c'] = df_result['vacances_zone_c'].fillna(0)

    df_result.loc[pd.isna(df_result['nom_conge']) & (df_result['vacances_zone_a'] == 0) & (df_result['vacances_zone_b'] == 0) & (df_result['vacances_zone_c'] == 0), 'nom_conge'] = '-'

    df_result['Lien vers photo du site de comptage'] = df_result['Lien vers photo du site de comptage'].fillna("")
    df_result = df_result.merge(df_p, on='Lien vers photo du site de comptage', how='left')

    #droper la colonne Lien vers photo du site de comptage qui ne nous servira plus
    df_result.drop(columns=["Lien vers photo du site de comptage"], inplace=True)

    # maintenant on merge le fichier des travaux/JO pour créer une colonne neutralisé
    df_ir['date'] = pd.to_datetime(df_ir['date'])
    df_ir['date'] = df_ir['date'].dt.date
    df_ir["date"] = pd.to_datetime(df_ir["date"])

    # j'ajoute une colonne rue dans df_result si rue est contenu dans N
    df_result['rue'] = df_result['nom_compteur'].apply(
        lambda x: next((rue for rue in df_ir['rue'].unique() if rue in x), None))
    
    # Faire un merge sur la colonne 'date' et 'rue' 
    merged_df = df_result.merge(df_ir[['rue', 'date']], on=['rue', 'date'], how='left', indicator=True)
    
    # Mettre à jour la colonne 'neutralise si une correspondance est trouvée on met 1 sinon 0
    df_result['neutralise'] = (merged_df['_merge'] == 'both').astype(int)
    df_result['neutralise'] = df_result['neutralise'].fillna(0)

    df_result.drop(columns=["rue"], inplace=True)

    df_result.sort_values(by=["date_heure_comptage","nom_compteur"], ascending=True, inplace=True)
    df_result.reset_index(drop=True, inplace=True)

    return df_result

def translate_month(date_string):
    month_translation = {
    'janvier': 'January', 'février': 'February', 'mars': 'March', 'avril': 'April', 'mai': 'May',
    'juin': 'June', 'juillet': 'July', 'août': 'August', 'septembre': 'September', 'octobre': 'October',
    'novembre': 'November', 'décembre': 'December'}
    for fr_month, en_month in month_translation.items():
        date_string = date_string.replace(fr_month, en_month)
    return date_string

def meteoSearch():
        """
        Permet de faire du webScraping sur le site de la météo pour récupérer les informations pour la prédiction
        """
        url_meteo = Config.URL_METEO
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}
        res = requests.get(url_meteo, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        rows_to_extract = { 1: "dates", 2: "heures",4: "temperatures",5: "précipitation",7: "vent"}
        data = {}
        for row_num, var_name in rows_to_extract.items():
            tr = soup.select(f'tbody tr:nth-of-type({row_num})')
            if tr:
                tds = tr[0].find_all('td')
                data[var_name] = [td.get_text(strip=True) for td in tds]
        date_index = 0
        dates_expanded = []
        for heure in data["heures"]:
            if heure == "00h":
                date_index += 1
            dates_expanded.append(data["dates"][date_index])
        meteo = pd.DataFrame({"date": dates_expanded,"heure": data["heures"],"temperature_2m": data["temperatures"],"precipitation_mm": data["précipitation"],"wind_speed": data["vent"]})
        meteo = meteo[meteo["temperature_2m"] != ""]
        meteo['temperature_2m'] = meteo['temperature_2m'].str.replace('°', '')
        meteo['precipitation_mm'] = meteo['precipitation_mm'].str.replace(' mm', '')
        meteo['wind_speed'] = meteo['wind_speed'].str.extract('(\\d+)')
        meteo['date'] = meteo['date'] + ' 2025'
        meteo['heure'] = meteo['heure'].str.replace("h", "", regex=False)
        meteo['Date_Heure'] = meteo['date'] + ' ' + meteo['heure'] + ":00"
        meteo['Date_Heure'] = meteo['Date_Heure'].apply(lambda x: ' '.join([x.split()[1].zfill(2), x.split()[2], x.split()[3]]) + ' ' + x.split()[4])
        meteo['Date_Heure'] = meteo['Date_Heure'].apply(translate_month)
        meteo['Date_Heure'] = pd.to_datetime(meteo['Date_Heure'], format='%d %B %Y %H:%M', errors='coerce')
        meteo['time_diff'] = meteo['Date_Heure'].diff()
        meteo = meteo[(meteo['time_diff'] == pd.Timedelta(hours=1)) | (meteo['time_diff'].isna())]
        meteo = meteo.drop(columns=["date", "heure","time_diff"])

        return meteo

def get_fait_jour(date):
    """
    Détermine si la date et l'heure donné  correspond à un moment de jour ou de nuit  
    en fonction de l'heure et de la saison (été ou hiver) en France.
    """
    mois = date.month
    heure = date.hour
    
    if mois >= 4 and mois <= 10: 
        lever_soleil = 6 
        coucher_soleil = 21 
    else: 
        # En hiver, le soleil se lève vers 8h00 et se couche vers 17h00
        lever_soleil = 7  
        coucher_soleil = 18  

    if lever_soleil <= heure < coucher_soleil:
        return 1  
    else:
        return 0 
    
def encode_cyclic(df, columns):
    '''
    permet d'encoder les données temporelle en cos/sin
    '''
    period = {'heure': 24, 'num_jour_semaine': 7, 'num_mois': 12, 'heure_num_jour_semaine': 168}
    for col in columns:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period[col])
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period[col])
    return df

def encode_cyclic_prophet(df, column, columnCalc):
    '''
    permet d'encoder les données temporelle en cos/sin pour le modèle prophet
    '''
    period = {'num_jour_semaine': 7, 'num_mois': 12}
    df[f'{column}_sin'] = np.sin(2 * np.pi * columnCalc / period[column])
    df[f'{column}_cos'] = np.cos(2 * np.pi * columnCalc / period[column])
    return df

def createDataframe(meteo, df_work, df_work_vac):
    '''
    permet de créer un dataframe pour les prédictions a 3J avec touts les compteurs, toutes les datea/heures
    et toutes les prédictions des données permettant d'aider la prédiction
    '''
    current_time = datetime.now()
    #différence d'heure
    next_hour = (current_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    # fin de la période J_3
    date_range = pd.date_range(next_hour, current_time + timedelta(days=3), freq="h")
    d_date_range = pd.DataFrame(date_range, columns=["date_heure_comptage"])

    #recupération des compteurs et des informations statiques
    all_identifiants = df_work[['nom_compteur', 'latitude', 'longitude',"1Sens","Partage"]].drop_duplicates(subset=['nom_compteur'])

    # référence croisé pour avoir toutes les heurs pendant 3 J sur tous les compteurs
    full_df = d_date_range.merge(all_identifiants, how="cross")

    full_df["heure"] = full_df["date_heure_comptage"].dt.hour
    full_df["num_mois"] = full_df["date_heure_comptage"].dt.month
    full_df["num_jour_semaine"] = full_df["date_heure_comptage"].dt.dayofweek
    full_df["année"] = full_df["date_heure_comptage"].dt.year
    full_df["neutralise"] = 0
    full_df['jour_mois_annee'] = full_df['date_heure_comptage'].dt.strftime('%Y-%m-%d')
    full_df['jour_mois_annee'] = pd.to_datetime(full_df['jour_mois_annee'])

    full_df["fait_jour"] = full_df["date_heure_comptage"].apply(get_fait_jour)


    df_work_vac = df_work_vac[["date","vacances_zone_c"]]
    df_work_vac['date'] = pd.to_datetime(df_work_vac['date'])
    df_work_vac = df_work_vac.rename(columns={"vacances_zone_c": "vacances"})
    df_work_vac['vacances'] = df_work_vac['vacances'].astype(int)

    full_df = pd.merge(full_df, meteo, left_on='date_heure_comptage', right_on='Date_Heure', how='left')
    full_df = pd.merge(full_df, df_work_vac, left_on='jour_mois_annee', right_on='date', how='left')
    full_df = full_df.drop(columns=["Date_Heure", "date","jour_mois_annee"])
    full_df["weekend"] = full_df["num_jour_semaine"].apply(lambda x: 1 if x >= 5 else 0)

    return full_df