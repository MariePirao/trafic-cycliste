import time
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder
import streamlit as st
from joblib import dump, load
# Pour éviter d'avoir les messages warning
import warnings
import os
warnings.filterwarnings('ignore')

#dump(clf_gs, 'md.joblib') librairie save des modèles entrainés
#loaded_model = load('md.joblib')

#loaded_model.predict(X_test_scaled)


@st.cache_data
def load_data(file_path, separator, numheader):
    """Charge le fichier CSV et retourne un DataFrame."""
    return pd.read_csv(file_path, sep=separator, header=numheader)

    
def modelisation(df):
    return 0

def prediction(classifier,X_train, y_train ):
   return 0

def scores(clf, choice, X_train, X_test, y_train, y_test):
    return 0


def merge(df_cleaned,df_m,df_jv, df_p, df_ir):

    df_result = df_cleaned.copy()
    df_result['time']=df_result["date_heure_comptage"]
    #merge des données météo dans df
    df_result = df_result.merge(df_m, on='time', how='left')

    #merge des données vacances/férié dans df
    #df_jv['time'] = pd.to_datetime(df_jv['time'])
                                   #, utc=True)  # Convertit en UTC proprement
    #df_jv['time'] = df_jv['time'].dt.tz_convert('Europe/Paris')  # Convertit en heure locale
    #df_jv['date'] = df_jv['time'].dt.date

    #df['time'] = pd.to_datetime(df['time'])
                                #, utc=True)  # Convertit en UTC proprement
    #df['time'] = df['time'].dt.tz_convert('Europe/Paris')  # C onvertit en heure locale
    #df['date'] = df['time'].dt.date

    # Faire la jointure sur la colonne "date" 
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


def modelisation(df):

    return 0, 0, 0, 0 

def prediction(classifier,X_train, y_train ):

    return 0

def scores(clf, choice, X_train, X_test, y_train, y_test):
    return 0