from random import choices
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from prophet import Prophet
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor as xgb
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


def modelisation1(df, compteur):

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

  # Transformation de l'heure en variables sinus et cosinus (APRES le split sinon fuite de données) en dropant moins bon
  X_train0 = encode_cyclic(X_train0, ['heure', 'num_mois', 'num_jour_semaine'])
  X_test0 = encode_cyclic(X_test0, ['heure', 'num_mois', 'num_jour_semaine'])
  X_train0 = X_train0.drop(columns=['num_mois'])
  X_test0 = X_test0.drop(columns=['num_mois'])

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

  X_cat1 = df_work[['nom_compteur','num_jour_semaine', 'fait_jour', 
                    #'num_mois', 'weekend', 
                    'vacances', 'neutralise', 'precipitation', 'wind','temperature', 'Partage', '1Sens', 'latitude', 'heure','année',
                    #'weekend_heure_interaction','heure_num_jour_semaine''longitude',
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


def encode_cyclic(df, columns):
    for col in columns:
        period = {'heure': 24, 'num_jour_semaine': 7, 'num_mois': 12}
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period[col])
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period[col])
    return df

def modelisation(df, classifier):

    print("Modelisation pour XGB et Stacking :")

    df_work = df.copy()

    y = df_work["comptage_horaire"]
    features = ['nom_compteur', 'heure', 'num_mois', 'num_jour_semaine', 'latitude', 'longitude',
    'temperature_2m', 'precipitation_mm', 'wind_speed', 
    'fait_jour', 'neutralise', 'vacances', "Partage", "année"]
    cyclic_columns = ['heure', 'num_jour_semaine', 'num_mois']
    df_work = encode_cyclic(df_work, cyclic_columns)
    features_extended = features + [f'{col}_sin' for col in cyclic_columns] + [f'{col}_cos' for col in cyclic_columns]
    X = df_work[features_extended]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    categorical_features_ord = ["vacances","heure","fait_jour", "neutralise","Partage","nom_compteur",
                                "latitude","longitude", "année"]
    numerical_features = ["temperature_2m", "precipitation_mm", "wind_speed"]
    dejanencode = ['num_mois_cos', 'num_mois_sin',"num_jour_semaine_cos", "num_jour_semaine_sin"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features_ord), # j'ai du ajouter ca car pas d'encodage pour 2025 dans le jeu de 80%
            ('standard', StandardScaler(), numerical_features),
            ('passthrough', 'passthrough', dejanencode)])
    
    if classifier == 'XGBRegressor':

        start_time = time.time()
        print("debut de l'entrainement de XGB :", time.ctime(start_time))
        # Instancier le modèle XGB
        xgb_params = {'n_estimators': 982,'max_depth': 15,'learning_rate': 0.010839297840803218,'colsample_bytree': 0.9763448531819173,
        'subsample': 0.8004427030847461,'gamma': 2.1532845545117967,'reg_alpha': 6.080159216998929,'reg_lambda': 8.274555339627717,
        'random_state': 42}
        xgb_model = xgb.XGBRegressor(**xgb_params)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', xgb_model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        start_time = time.time()


    elif classifier == 'StackingRegressor':
        start_time = time.time()
        print("debut de l'entrainement de StackingRegressor :", time.ctime(start_time))
        model_XGB = xgb.XGBRegressor(**xgb_params)
        model_bag = BaggingRegressor(n_estimators=100,max_samples=0.7,max_features=0.7,random_state=123)
        meta_model = LinearRegression(fit_intercept=False,copy_X=False)
        stack_model = StackingRegressor(estimators=[('XGB', model_XGB), ('bagging', model_bag)],final_estimator=meta_model)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', stack_model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

    print("fin de l'entrainement :", time.ctime(start_time))

    return y_pred, y_test

    #start_time = time.time()
    #print("debut de l'entrainement de RF :", time.ctime(start_time))
    # Instancier le modèle RandomForest
    #rf = RandomForestRegressor(n_estimators=200,max_depth=None,
    #        min_samples_split=16,min_samples_leaf=10,max_features=None,bootstrap=True,random_state=42)
    # Créer le pipeline
    #pipeline = Pipeline(steps=[
    #    ('preprocessor', preprocessor),
    #    ('regressor', rf)])
    #pipeline.fit(X_train, y_train)
    #y_pred = pipeline.predict(X_test)
    #maeRF = mean_absolute_error(y_test, y_pred)
    #start_time = time.time()
    #print("fin de l'entrainement de RF :", time.ctime(start_time))
    #print(f"Mean Absolute Error (MAE) to random Forest : {maeRF}")
    #bagging_model = BaggingRegressor(n_estimators = 100, max_samples = 0.7, max_features = 0.7, 
    #                                 bootstrap=False,random_state=42)
    #pipeline = Pipeline(steps=[
    #    ('preprocessor', preprocessor),  
    #    ('regressor', bagging_model)])
    #start_time = time.time()
    #print("Début de l'entraînement de BaggingRegressor :", time.ctime(start_time))
    #pipeline.fit(X_train, y_train)
    #y_pred_bagging =  pipeline.predict(X_test)
    #maeBagging = mean_absolute_error(y_test, y_pred_bagging)
    #end_time = time.time()
    #print("Fin de l'entraînement de BaggingRegressor :", time.ctime(end_time))
    #print(f"Mean Absolute Error (MAE) avec BaggingRegressor : {maeBagging}")

#Les focntions suivantes sont pour les modelisation temporelle

def get_temperature_for_date(date, df):
    """
    Retourne la temperature en prenant le mode de la précipitation pour la même heure 
    au meme mois en 2024.
    """
    df_train = df[df['ds'] < date]  
    df_train_same_hour = df_train[df_train['ds'].dt.hour == date.hour]  
    return df_train_same_hour['temperature_2m'].mean()

def get_precipitation_for_date(date, df):
    """
    Retourne la précipitation en prenant le mode de la précipitation pour la même heure 
    au meme mois en 2024.
    """
    df_train = df[df['ds'] < date] 
    df_train_same_hour = df_train[df_train['ds'].dt.hour == date.hour]  
    return df_train_same_hour['precipitation_mm'].mean()


def get_wind_for_date(date, df):
    """
    Retourne la vitesse du wind en prenant le mode de la vitesse du wind pour la même heure 
    au mois passé en 2024.
    """
    df_train = df[df['ds'] < date]  
    df_train_same_hour = df_train[df_train['ds'].dt.hour == date.hour]  
    return df_train_same_hour['wind_speed'].mean()

def get_vacances_for_date(date, df):
    """
    Retourne 1 si la date est une date de vacances en 2024, sinon 0.
    """
    df_2024 = df[(df['ds'].dt.year == 2024) & (df['ds'].dt.month == date.month) & (df['ds'].dt.day == date.day)]
    return 1 if not df_2024.empty and df_2024['vacances'].iloc[0] == 1 else 0

def get_neutralise_for_date(date, df):
    """
    Génère un mode aléatoire pour la présence de neutralise en fonction des proportions 
    présentes dans l'ensemble d'entraînement.
    """
    neutralise_proportion = df['neutralise'].mean()  
    return choices([0, 1], [1 - neutralise_proportion, neutralise_proportion])[0]


def get_fait_jour(date):
    """
    Détermine si la date correspond à un jour en fonction de l'heure et de la saison
    (été ou hiver) en France.
    """
    mois = date.month
    heure = date.hour
    
    if mois >= 4 and mois <= 10: 
        lever_soleil = 6 
        coucher_soleil = 21 
    else: 
        # En hiver, le soleil se lève vers 8h00 et se couche vers 17h00
        lever_soleil = 8  
        coucher_soleil = 17  

    if lever_soleil <= heure < coucher_soleil:
        return 1  
    else:
        return 0 


def modelisation_per_compteur(df, compteur):
    """
    Fonction pour entraîner un modèle Prophet pour un compteur spécifique.
    """
    # Filtrer les données du compteur
    df_compteur = df[df['nom_compteur'] == compteur]
    df_compteur['ds'] = pd.to_datetime(df_compteur["date_heure_comptage"])
    df_compteur['y'] = df_compteur['comptage_horaire']

    train_size = int(len(df_compteur) * 0.8)
    train_data = df_compteur[:train_size]
    test_data = df_compteur[train_size:]

    #train_data['heure_sin'] = np.sin(2 * np.pi * train_data['ds'].dt.hour / 24)
    #train_data['heure_cos'] = np.cos(2 * np.pi * train_data['ds'].dt.hour / 24)
    #test_data['heure_sin'] = np.sin(2 * np.pi * test_data['ds'].dt.hour / 24)
    #test_data['heure_cos'] = np.cos(2 * np.pi * test_data['ds'].dt.hour / 24)
    
    train_data['num_mois_sin'] = np.sin(2 * np.pi * train_data['ds'].dt.month / 12)
    train_data['num_mois_cos'] = np.cos(2 * np.pi * train_data['ds'].dt.month / 12)
    test_data['num_mois_sin'] = np.sin(2 * np.pi * test_data['ds'].dt.month / 12)
    test_data['num_mois_cos'] = np.cos(2 * np.pi * test_data['ds'].dt.month / 12)
    
    train_data['num_jour_semaine_sin'] = np.sin(2 * np.pi * train_data['ds'].dt.weekday / 7)
    train_data['num_jour_semaine_cos'] = np.cos(2 * np.pi * train_data['ds'].dt.weekday / 7)
    test_data['num_jour_semaine_sin'] = np.sin(2 * np.pi * test_data['ds'].dt.weekday / 7)
    test_data['num_jour_semaine_cos'] = np.cos(2 * np.pi * test_data['ds'].dt.weekday / 7)

    #param_fixed = {  
    #'seasonality_mode': 'multiplicative',
    #'daily_seasonality': True,
    #'weekly_seasonality': True,
    #'yearly_seasonality': False}
    #param_grid = {  
    #'changepoint_prior_scale': [0.001, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0], 
    #'seasonality_prior_scale': [0.01, 0.1, 0.5, 1.0, 10.0, 100.0],
    #'changepoint_range':  [0.05, 0.1, 0.2, 0.5, 0.75, 0.8, 0.95],
    #'holidays_prior_scale': [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0]
    #}

    
    # Generate all combinations of parameters
    #all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    #rmses = []  # Store the RMSEs for each params here
    # Use cross validation to evaluate all parameters
    #for params in all_params:
    #    model_cv = Prophet(**param_fixed, **params)  # Fit model with given params
    #    model_cv.add_regressor('temperature_2m')
    #    model_cv.add_regressor('precipitation_mm')
    #    model_cv.add_regressor('wind_speed')
    #    model_cv.add_regressor('vacances')
    #    #model.add_regressor('neutralise')
    #    model_cv.add_regressor('weekend')
    #    model_cv.add_regressor('heure_sin')
    #    model_cv.add_regressor('heure_cos')
    #    model_cv.add_regressor('num_mois_sin')
    #    model_cv.add_regressor('num_mois_cos')
    #    model_cv.add_regressor('num_jour_semaine_sin')
    #    model_cv.add_regressor('num_jour_semaine_cos')
    #    model_cv.fit(train_data)
    #    df_cv = cross_validation(model_cv, initial='250 days', period='7 days', horizon = '7 days', parallel="threads")
    #    df_p = performance_metrics(df_cv, rolling_window=1)
    #    rmses.append(df_p['rmse'].values[0])
    #tuning_results = pd.DataFrame(all_params)
    #tuning_results['rmse'] = rmses
    #print(tuning_results) 
    #best_params = all_params[np.argmin(rmses)]
    #best_score = tuning_results['rmse'].min()
    #print(best_params)
    #print(best_score) {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 100.0, 'changepoint_range': 0.1, 'holidays_prior_scale': 0.1}

    # Créer le modèle Prophet
    model = Prophet( 
                    changepoint_prior_scale=0.5,   # La flexibilité de la tendance      cross validation = 0.001  moi : 0.5
                    seasonality_prior_scale=1.0,   # La flexibilité des composantes saisonnières  cross validation = 100.0   moi : 1.0
                    holidays_prior_scale=10.0,      # La flexibilité des effets des jours fériés cross validation = 0.1  moi : 10.0
                    seasonality_mode='multiplicative',    # Mode de saisonnalité ('additive' ou 'multiplicative')
                    yearly_seasonality=False,        # Inclusion de la saisonnalité annuelle
                    weekly_seasonality=True,        # on observe une saisonnalité sur la semainbe
                    daily_seasonality=True,      # on observe une saisonnalité sur les jours
                    changepoint_range=0.80        #, 0.75, 0.95]      cross validation = 0.1  moi 0.80 
    )

    # Ajouter les régressions externes
    #model.add_regressor('temperature_2m')
    model.add_regressor('precipitation_mm')
    model.add_regressor('wind_speed')
    model.add_regressor('vacances')
    #model.add_regressor('neutralise')
    #model.add_regressor('année')
    model.add_regressor('fait_jour')
    model.add_regressor('weekend')
    model.add_regressor('heure')
    #model.add_regressor('heure_sin')
    #model.add_regressor('heure_cos')
    model.add_regressor('num_mois_sin')
    model.add_regressor('num_mois_cos')
    model.add_regressor('num_mois')
    #model.add_regressor('num_jour_semaine')
    model.add_regressor('num_jour_semaine_sin')
    model.add_regressor('num_jour_semaine_cos')

    # Entraîner le modèle
    model.fit(train_data)

    return model, train_data, test_data

def predict_and_evaluate(model, df, test_data):
    """
    Effectue les prédictions pour un compteur donné model : model entrainé sur le compteur, df dataframe filtré que le compteur
    """
    future = test_data[['ds']].copy()
    
    # Ajouter les colonnes de régressions externes pour les dates futures
    #future['temperature_2m'] = future['ds'].apply(lambda x: get_temperature_for_date(x, df))
    future['precipitation_mm'] = future['ds'].apply(lambda x: get_precipitation_for_date(x, df))
    future['wind_speed'] = future['ds'].apply(lambda x: get_wind_for_date(x, df))
    future['vacances'] = future['ds'].apply(lambda x: get_vacances_for_date(x, df))
    #future['neutralise'] = future['ds'].apply(lambda x: get_neutralise_for_date(x, df))
    future['weekend'] = future['ds'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
    #future['année'] = future['ds'].dt.year
    future['fait_jour'] = future['ds'].apply(get_fait_jour)
    future['heure'] = future['ds'].dt.hour
    future['num_mois'] = future['ds'].dt.month
    #future['heure_sin'] = np.sin(2 * np.pi * future['ds'].dt.hour / 24)
    #future['heure_cos'] = np.cos(2 * np.pi * future['ds'].dt.hour / 24)
    future['num_mois_sin'] = np.sin(2 * np.pi * future['ds'].dt.month / 12)
    future['num_mois_cos'] = np.cos(2 * np.pi * future['ds'].dt.month / 12)
    #future['num_jour_semaine'] = future['ds'].dt.dayofweek 
    future['num_jour_semaine_sin'] = np.sin(2 * np.pi * future['ds'].dt.weekday / 7)
    future['num_jour_semaine_cos'] = np.cos(2 * np.pi * future['ds'].dt.weekday / 7)


    # Prédictions
    forecast = model.predict(future)
    # Prédictions pour la période de test
    test_predictions = forecast[forecast['ds'].isin(test_data['ds'])]

    # Calcul du score (Mean Absolute Error)
    mae = mean_absolute_error(test_data['y'], test_predictions['yhat'])

    return test_data, test_predictions, mae

#@st.cache(allow_output_mutation=True, hash_funcs={dict: id, list: id})
def modelisationT(df):
    
    df_work = df.copy()
    df_work.sort_values(by=["date_heure_comptage","nom_compteur"], ascending=True, inplace=True)

    df_work = df_work[['nom_compteur',    # label 100 valeurs différentes
                  'num_jour_semaine', # de 0 à 6
                   'num_mois',   #de 0 a 11
                   'fait_jour',  #0,1
                   'weekend',    #0,1
                   'vacances', 
                   'neutralise',  #0,1 et  #0,1
                   'temperature_2m', 'precipitation_mm', 'wind_speed', 
                   #'Partage',        #0,1
                   #'1Sens',          #0,1
                   #'latitude',       #int format latitude
                   'heure',         #heure datatime
                   'année',        #année datatime
                   "date_heure_comptage",
                   "comptage_horaire"
                   ]]
    
    models = {}
    compteurs = df_work['nom_compteur'].unique()
    for compteur in compteurs:
        model, train_data, test_data = modelisation_per_compteur(df_work, compteur)
        models[compteur] = {
            'model': model,
            'train_data': train_data,
            'test_data': test_data
        }
    return models
