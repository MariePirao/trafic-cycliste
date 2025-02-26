import glob
import pandas as pd
import streamlit as st
# Pour éviter d'avoir les messages warning
import warnings
import os
warnings.filterwarnings('ignore')


@st.cache_data
def load_data(file_path, separator, numheader):
    return pd.read_csv(file_path, sep=separator, header=numheader)

def create_data(df, file_path):
    df.to_csv(file_path + 'dataframeFinal.csv')

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


