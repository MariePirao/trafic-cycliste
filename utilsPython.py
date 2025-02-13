#import streamlit as st
import time
import pandas as pd
#import numpy as np
#import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import confusion_matrix
import streamlit as st


@st.cache_data
def load_data(file_path, separator, numheader):
    """Charge le fichier CSV et retourne un DataFrame."""
    return pd.read_csv(file_path, sep=separator, header=numheader)
    
def modelisation(df):

  df_work =df.copy()

  
  #X = df_work.drop(['Identifiant du compteur', 'Identifiant du site de comptage', 'Nom du site de comptage','comptage_horaire','Lien vers photo du site de comptage',
  #             'Identifiant technique compteur','nom_conge',"Date d'installation du site de comptage",'temperature_2m (°C)','precipitation (mm)','wind_speed_10m (km/h)'], axis=1)
  y = df_work['comptage_horaire']
  
  X_cat = df_work[['vacances_zone_c','nom_compteur','Mois', 'Jour','precipitation', 'wind', 'temperature']]
  X_num = df_work[['heure','année','1Sens', '2sens', 'Partage', 'Separe','vacances_zone_c']]
  X_cat = X_cat.copy()
  X_num = X_num.copy()
  for col in X_cat.columns:
     X_cat[col] = X_cat[col].astype(str).fillna(X_cat[col].mode()[0])
  for col in X_num.columns:
     X_num[col] = X_num[col].fillna(X_num[col].median())

  X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
  X = pd.concat([X_cat_scaled, X_num], axis = 1)
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
  scaler = StandardScaler()
  X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
  X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

  return X_train, X_test, y_train, y_test

@st.cache_data
def prediction(classifier,X_train, y_train ):


    start_time = time.time()
    print("Je rentre dans le méthode prediction Heure de début :", time.ctime(start_time))
    
    if classifier == 'Random Forest Regressor':
      clf = RandomForestRegressor()
      param_grid = {'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10]}
    elif classifier == 'LinearRegression':
      clf = LinearRegression()
      param_grid = {'fit_intercept': [True, False],
                    'n_jobs': [None, -1],
                    'positive': [True, False]}
    elif classifier == 'DecisionTreeRegressor':
      clf = DecisionTreeRegressor()
      param_grid = {'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'max_features': ['auto', 'sqrt', 'log2', None]}
    grid_search = RandomizedSearchCV(clf, param_grid, cv=3, n_iter=4, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Afficher les meilleurs paramètres trouvés
    print("Meilleurs paramètres trouvés : ", grid_search.best_params_)
    print("Je sors dans le méthode prediction")

    return grid_search

def scores(clf, choice, X_train, X_test, y_train, y_test):
  if choice == 'score (R²)':
     return clf.score(X_train, y_train),clf.score(X_test,y_test)
  elif choice == 'metrique MAE':
      y_predTest = clf.predict(X_test)
      y_predTrain = clf.predict(X_train)
      return mean_absolute_error(y_train, y_predTrain), mean_absolute_error(y_test, y_predTest),


def merge(df_cleaned,df_m,df_jv, df_p):

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

    # Faire la jointure sur la colonne "date" au lieu de "time"
    df_result = df_result.merge(df_jv, on='time', how='left')

    df_result['vacances_zone_a'] = df_result['vacances_zone_a'].fillna(0)
    df_result['vacances_zone_b'] = df_result['vacances_zone_b'].fillna(0)
    df_result['vacances_zone_c'] = df_result['vacances_zone_c'].fillna(0)

    df_result.loc[pd.isna(df_result['nom_conge']) & (df_result['vacances_zone_a'] == 0) & (df_result['vacances_zone_b'] == 0) & (df_result['vacances_zone_c'] == 0), 'nom_conge'] = '-'

    df_result['Lien vers photo du site de comptage'] = df_result['Lien vers photo du site de comptage'].fillna("")
    df_result = df_result.merge(df_p, on='Lien vers photo du site de comptage', how='left')

    #droper la colonne Lien vers photo du site de comptage qui ne nous servira plus
    df_result.drop(columns=["Lien vers photo du site de comptage"], inplace=True)

    return df_result