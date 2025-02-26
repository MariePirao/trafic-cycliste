import time
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import streamlit as st
import utilsPython as utils # type: ignore
from joblib import dump, load
# Pour éviter d'avoir les messages warning
import warnings
import os
warnings.filterwarnings('ignore')

def encode_cyclic(df, columns):
    period = {'heure': 24, 'num_jour_semaine': 7, 'num_mois': 12, 'heure_num_jour_semaine': 168}
    for col in columns:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period[col])
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period[col])
    return df


def modelisation(df, compteur):

  print("MOdelisation Avant :")

  df_work =df.copy()

  df_work.sort_values(by=["date_heure_comptage","nom_compteur"], ascending=True, inplace=True)

  if(compteur != 'All'):
    utils.removeJoblibFile()
    df_work =df_work[df_work['nom_compteur'] == compteur]
  
  #df_work['weekend_heure_interaction'] = df_work['weekend'] * df_work['heure']
  #df_work['heure_num_jour_semaine'] = df_work['heure'] * df_work['num_jour_semaine']   #18/19 et 11/14

  y = df_work['comptage_horaire']

  X_cat = df_work[['nom_compteur','num_jour_semaine', 
                   'num_mois', 'fait_jour', 
                   #'weekend',                       perte
                   'vacances', 'neutralise', 
                   'precipitation', 
                   'wind',
                   'temperature', 
                   'Partage', 
                   '1Sens', 
                   'latitude', 
                   #'longitude',                       perte
                   #'weekend_heure_interaction',       perte
                   'heure',
                   'année',
                   #'heure_num_jour_semaine'
                   ]]
  X_num = df_work[[
                    #"temperature_2m", "precipitation_mm", "wind_speed"
                    #'heure'
  #                 #,'heure_num_jour_semaine'
                  ]]
  

  #on reforme X pour le donner a split
  X = pd.concat([X_cat, X_num], axis = 1)
  
  X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False,random_state=123)

  #Nicolas
      #categorical_features_ord = ["vacances","heure","fait_jour", "neutralise","Partage","nom_compteur","latitude","longitude","année"]
      #numerical_features = ["temperature_2m", "precipitation_mm", "wind_speed"] --> encoder temperature et precipitation 
      #dejanencode = ['num_mois_cos', 'num_mois_sin',"num_jour_semaine_cos", "num_jour_semaine_sin"] --> ok ca j'ajotue après
      #preprocessor = ColumnTransformer(
      #transformers=[
      #        ('ordinal', OrdinalEncoder(), categorical_features_ord),
      #        ('standard', StandardScaler(), numerical_features),
      #        ('passthrough', 'passthrough', dejanencode)])

  #Aurelie/Ingrid mise à l'échelle
  #scaler = MinMaxScaler()
  #X_train0[X_num.columns]= scaler.fit_transform(X_train0[X_num.columns])
  #X_test0[X_num.columns]= scaler.transform(X_test0[X_num.columns])


  # Transformation de l'heure en variables sinus et cosinus (APRES le split sinon fuite de données) en dropant moins bon
  X_train0 = encode_cyclic(X_train0, ['heure', 
                                      #'heure_num_jour_semaine', 
                                      'num_mois', 'num_jour_semaine'])
  X_test0 = encode_cyclic(X_test0, ['heure', 
                                    #'heure_num_jour_semaine', 
                                    'num_mois', 'num_jour_semaine'])
  X_train0 = X_train0.drop(columns=[
                                    #'heure_num_jour_semaine', 
                                    'num_mois', 
                                    #'num_jour_semaine'           perte
                                    ])
  X_test0 = X_test0.drop(columns=[
                                   #'heure_num_jour_semaine', 
                                    'num_mois', 
                                    #'num_jour_semaine'          perte
                                    ])

  #en dropant moins bon
  encoder = OrdinalEncoder(categories=[['Gel', 'Froid', 'Tempéré', 'Chaud', 'Très chaud']])
  X_train0['temperature_encoded'] = encoder.fit_transform(X_train0[['temperature']])
  X_test0['temperature_encoded'] = encoder.transform(X_test0[['temperature']])
  encoderP = OrdinalEncoder(categories=[['Pas de pluie/bruine', 'Pluie modérée', 'Fortes averses']])
  X_train0['precipitation_encoded'] = encoderP.fit_transform(X_train0[['precipitation']])
  X_test0['precipitation_encoded'] = encoderP.transform(X_test0[['precipitation']])
  encoderW = OrdinalEncoder(categories=[['Pas de vent', 'Vent modérée', 'vent', 'grand vent']])
  X_train0['wind_encoded'] = encoderW.fit_transform(X_train0[['wind']])
  X_test0['wind_encoded'] = encoderW.transform(X_test0[['wind']])

  X_cat1 = df_work[[
                    'nom_compteur',
                    'num_jour_semaine', 
                    #'num_mois', 
                    'fait_jour', 
                    #'weekend', 
                    'vacances', 'neutralise', 
                    'precipitation', 
                    'wind',
                    'temperature', 
                    'Partage', 
                    '1Sens', 
                    'latitude', 
                    #'weekend_heure_interaction',
                    'heure',
                    #'heure_num_jour_semaine'
                    'année',
                    #'longitude',
                    ]]

  X_train_cat = pd.get_dummies(X_train0[X_cat1.columns], drop_first=True)
  X_test_cat = pd.get_dummies(X_test0[X_cat1.columns], drop_first=True)

  # Aligner les colonnes de X_test avec celles de X_train (ajout de colonnes manquantes dans X_test)
  X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='left', axis=1, fill_value=0)

    # Combine les variables catégorielles transformées et les variables numériques dans X_train et X_test
  X_train = pd.concat([X_train_cat, X_train0[['heure_sin', 'heure_cos', 
                                            #'heure_num_jour_semaine_sin', 'heure_num_jour_semaine_cos', 
                                              'num_mois_sin', 'num_mois_cos','num_jour_semaine_cos','num_jour_semaine_sin','precipitation_encoded','temperature_encoded','wind_encoded']]], axis=1)
  X_test = pd.concat([X_test_cat, X_test0[['heure_sin', 'heure_cos', 
                                           #'heure_num_jour_semaine_sin', 'heure_num_jour_semaine_cos', 
                                           'num_mois_sin', 'num_mois_cos','num_jour_semaine_cos','num_jour_semaine_sin','precipitation_encoded','temperature_encoded','wind_encoded']]], axis=1)
                          
  return X_train, X_test, y_train, y_test


def prediction(classifier,X_train, y_train):
        
    if classifier == 'Random Forest Regressor':
      if os.path.exists('model_RFR.joblib'):
        # Si le fichier existe, charger le modèle sauvegardé
         clf = load('model_RFR.joblib')
      else:
        clf = RandomForestRegressor(n_estimators = 100, min_samples_split = 10, max_depth = None, random_state=123)
        clf.fit(X_train, y_train)
        dump(clf, 'model_RFR.joblib')
    #  clf = RandomForestRegressor()
    #  param_grid = {'n_estimators': [50, 100, 200],
    #                'max_depth': [5, 10, None],
    #                'min_samples_split': [2, 5, 10]} #Meilleurs paramètres trouvés :  {'n_estimators': 100, 'min_samples_split': 10, 'max_depth': None}
    elif classifier == 'DecisionTreeRegressor':
      if os.path.exists('model_DTR.joblib'):
         # Si le fichier existe, charger le modèle sauvegardé
         clf = load('model_DTR.joblib')
      else:
        clf = DecisionTreeRegressor(random_state=42, min_samples_split = 5, max_features = None, max_depth=15)
        clf.fit(X_train, y_train)
        dump(clf, 'model_DTR.joblib')
    elif classifier == 'GradientBoostingRegressor':
      if os.path.exists('model_GBR.joblib'):
         # Si le fichier existe, charger le modèle sauvegardé
         clf = load('model_GBR.joblib')
      else:
        clf = GradientBoostingRegressor(n_estimators = 200, max_depth = 5, learning_rate = 0.1, random_state=123)
        clf.fit(X_train, y_train)
        dump(clf, 'model_GBR.joblib')
    elif classifier == 'BaggingRegressor':
      if os.path.exists('model_BR.joblib'):
         # Si le fichier existe, charger le modèle sauvegardé
         clf = load('model_BR.joblib')
      else:
        clf = BaggingRegressor(n_estimators = 100, max_samples = 0.7, max_features = 0.7, random_state=123)
        clf.fit(X_train, y_train)
        dump(clf, 'model_BR.joblib')
    elif classifier == 'AdaBoostRegressor':
      if os.path.exists('model_ABR.joblib'):
         # Si le fichier existe, charger le modèle sauvegardé
         clf = load('model_ABR.joblib')
      else:
        clf = AdaBoostRegressor(n_estimators = 50, loss = 'square', learning_rate = 0.01,random_state=123)
        clf.fit(X_train, y_train)
        dump(clf, 'model_ABR.joblib')
    #elif classifier == 'XGBRegressor':
    #  if os.path.exists('model_XGB.joblib'):
    #     # Si le fichier existe, charger le modèle sauvegardé
    #     clf = load('model_XGB.joblib')
    #  else:
    #    clf = XGBRegressor(n_estimators=727,max_depth=13,random_state=42,learning_rate=0.06548873215567408,
    #                       objective='reg:squarederror',colsample_bytree=0.9704143531548255,gamma=2.5660403767871864,
    #                       reg_alpha=2.245326956161123,reg_lambda=2.9798175822945447,subsample=0.9342721748104903)
    #    clf.fit(X_train, y_train)
    #    dump(clf, 'model_XGB.joblib')
    #elif classifier == 'StackingRegressor':
    #  if os.path.exists('model_SR.joblib'):
    #     # Si le fichier existe, charger le modèle sauvegardé
    #     clf = load('model_SR.joblib')
    #  else:
    #    model_XGB = XGBRegressor(n_estimators = 100, max_depth = 9, learning_rate = 0.1, random_state=123)
    #    model_bag = BaggingRegressor(n_estimators = 100, max_samples = 0.7, max_features = 0.7, random_state=123)
    #    # Modèle final (métamodèle) : XGBRegressor
    #    meta_model = LinearRegression(n_jobs = None, fit_intercept = False)
    #    # Stack des modèles (RandomForest et BaggingRegressor) avec XGBRegressor comme métamodèle
    #    clf = StackingRegressor(estimators=[('XGB', model_XGB), ('bagging', model_bag)],
    #                             final_estimator=meta_model)
    #    # Entraînement du modèle empilé
    #    clf.fit(X_train, y_train)
    #    dump(clf, 'model_SR.joblib')

    return clf

def scores(clf, choice, X_train, X_test, y_train, y_test):
  if choice == 'score (R²)':
     return clf.score(X_train, y_train),clf.score(X_test,y_test)
  elif choice == 'metrique MAE':
      y_predTest = clf.predict(X_test)
      y_predTrain = clf.predict(X_train)
      print(mean_absolute_error(y_train, y_predTrain), mean_absolute_error(y_test, y_predTest))
      return mean_absolute_error(y_train, y_predTrain), mean_absolute_error(y_test, y_predTest)

#start_time = time.time()
#print("Je rentre dans le méthode prediction Heure de début :", time.ctime(start_time))
