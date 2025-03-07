from random import choices
import streamlit as st
import numpy as np
import pandas as pd
from config import Config
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, StackingRegressor
from prophet import Prophet
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import utilsPython as utils 
from joblib import dump, load
# Pour éviter d'avoir les messages warning
import warnings
import os
warnings.filterwarnings('ignore')


'''TOUTES LES METHODES D ENTRAINEMENT DE MODELES'''

@st.cache_data
def modelisationRFBase(df_merged_cleaned_final):  
    '''Modelisation pour RF de base pour trouver les importance feature
    '''

    df_work = df_merged_cleaned_final[['comptage_horaire', 'heure', 'nom_compteur', 
                                       'num_jour_semaine', 'num_mois', 'fait_jour', 'weekend', 'vacances', 
                       'neutralise', 'precipitation', 'temperature', 'Partage', '1Sens', 'latitude']].copy()
    df_work['precipitation'] = df_work['precipitation'].replace(['Pas de pluie/bruine', 'Pluie modérée', 'Fortes averses'],[0,1,2])
    df_work['temperature'] = df_work['temperature'].replace(['Gel', 'Froid', 'Tempéré', 'Chaud', 'Très chaud'],[0,1,2,3,4])
    df_work = pd.get_dummies(df_work, columns=['nom_compteur']).astype(int)
    feats = df_work.drop('comptage_horaire', axis=1) # On retire la cible et on garde les variables explicatives
    target = df_work['comptage_horaire'] 

    X_train, X_test, y_train, y_test = train_test_split(feats, target,test_size=0.25,random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if os.path.exists('model_RFRBA.joblib'):
        # Si le fichier existe, charger le modèle sauvegardé
         model3 = load('model_RFRBA.joblib')
    else: #14min
        model3 = RandomForestRegressor(random_state=42)
        model3.fit(X_train_scaled, y_train)
        dump(model3, 'model_RFRBA.joblib')
     
    return model3,X_train,feats


@st.cache_data
def modelisationRFR(df):
  ''' Modelisation pour le modèle RFR
    '''
  df_work =df.copy()
  df_work.sort_values(by=["date_heure_comptage","nom_compteur"], ascending=True, inplace=True)

  y = df_work['comptage_horaire']

  X_cat = df_work[['nom_compteur','num_jour_semaine', 'num_mois', 'fait_jour', 
                   'vacances', 'neutralise', 'precipitation', 'wind','temperature', 'Partage', '1Sens', 'latitude', 'heure','année']]
  X_num = df_work[[]]
  
  #on reforme X pour le donner a split
  X = pd.concat([X_cat, X_num], axis = 1)
  
  X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False,random_state=123)

  # Transformation de l'heure en variables sinus et cosinus (APRES le split sinon fuite de données) en dropant moins bon
  X_train0 = utils.encode_cyclic(X_train0, ['heure', 'num_mois', 'num_jour_semaine'])
  X_test0 = utils.encode_cyclic(X_test0, ['heure', 'num_mois', 'num_jour_semaine'])
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
                    'vacances', 'neutralise', 'precipitation', 'wind','temperature', 'Partage', '1Sens', 'latitude', 'heure','année']]

  X_train_cat = pd.get_dummies(X_train0[X_cat1.columns], drop_first=True)
  X_test_cat = pd.get_dummies(X_test0[X_cat1.columns], drop_first=True)

  # Aligner les colonnes de X_test avec celles de X_train (ajout de colonnes manquantes dans X_test)
  X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='left', axis=1, fill_value=0)

    # Combine les variables catégorielles transformées et les variables numériques dans X_train et X_test
  X_train = pd.concat([X_train_cat, X_train0[['heure_sin', 'heure_cos', 
                                              'num_mois_sin', 'num_mois_cos','num_jour_semaine_cos','num_jour_semaine_sin','precipitation_encoded','temperature_encoded','wind_encoded']]], axis=1)
  X_test = pd.concat([X_test_cat, X_test0[['heure_sin', 'heure_cos', 
                                           'num_mois_sin', 'num_mois_cos','num_jour_semaine_cos','num_jour_semaine_sin','precipitation_encoded','temperature_encoded','wind_encoded']]], axis=1)

  if os.path.exists('model_RFR.joblib'):
    # Si le fichier existe, charger le modèle sauvegardé
    clf = load('model_RFR.joblib')
  else:
    clf = RandomForestRegressor(n_estimators = 100, min_samples_split = 10, max_depth = None, random_state=123)
    clf.fit(X_train, y_train)
    dump(clf, 'model_RFR.joblib')

  return clf, X_train, X_test, y_train, y_test 


def modelisation(df, classifier):
    ''' Modelisation pour le modèle XGB et Stacking
    '''

    df_work = df.copy()

    y = df_work["comptage_horaire"]
    features = ['nom_compteur', 'heure', 'num_mois', 'num_jour_semaine', 'latitude', 'longitude','temperature_2m', 'precipitation_mm', 'wind_speed', 
    'fait_jour', 'neutralise', 'vacances', "Partage", "année"]
    cyclic_columns = ['heure', 'num_jour_semaine', 'num_mois']
    df_work = utils.encode_cyclic(df_work, cyclic_columns)
    features_extended = features + [f'{col}_sin' for col in cyclic_columns] + [f'{col}_cos' for col in cyclic_columns]
    X = df_work[features_extended]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    categorical_features_ord = ["vacances","heure","fait_jour", "neutralise","Partage","nom_compteur", "latitude","longitude", "année"]
    numerical_features = ["temperature_2m", "precipitation_mm", "wind_speed"]
    dejanencode = ['num_mois_cos', 'num_mois_sin',"num_jour_semaine_cos", "num_jour_semaine_sin"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features_ord), # j'ai du ajouter ca car pas d'encodage pour 2025 dans le jeu de 80%
            ('standard', StandardScaler(), numerical_features),
            ('passthrough', 'passthrough', dejanencode)])
    # Instancier le modèle XGB
    xgb_params = {'n_estimators': 982,'max_depth': 15,'learning_rate': 0.010839297840803218,'colsample_bytree': 0.9763448531819173,
        'subsample': 0.8004427030847461,'gamma': 2.1532845545117967,'reg_alpha': 6.080159216998929,'reg_lambda': 8.274555339627717,
        'random_state': 42}
    
    if classifier == 'XGBRegressor':

        if os.path.exists('model_XGB.joblib'):
            # Si le fichier existe, charger le modèle sauvegardé
            pipeline = load('model_XGB.joblib')
        else:
            xgb_model = XGBRegressor(**xgb_params)
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', xgb_model)])
            pipeline.fit(X_train, y_train)
            dump(pipeline, 'model_XGB.joblib') 
        #xgb_cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        #print(f'Validation croisée XGBoost : MAE moyen = {-xgb_cv_scores.mean():.4f} (±{xgb_cv_scores.std():.4f})')

    elif classifier == 'StackingRegressor':
        if os.path.exists('model_ST.joblib'):
            # Si le fichier existe, charger le modèle sauvegardé
            pipeline = load('model_ST.joblib')
        else:
            model_XGB = XGBRegressor(**xgb_params)
            model_bag = BaggingRegressor(n_estimators=100,max_samples=0.7,max_features=0.7,random_state=123)
            meta_model = LinearRegression(fit_intercept=False,copy_X=False)
            stack_model = StackingRegressor(estimators=[('XGB', model_XGB), ('bagging', model_bag)],final_estimator=meta_model)
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', stack_model)])
            pipeline.fit(X_train, y_train)
            dump(pipeline, 'model_ST.joblib') 
        #stack_cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        #print(f'Validation croisée Stacking Regressor : MAE moyen = {-stack_cv_scores.mean():.4f} (±{stack_cv_scores.std():.4f})')

    return pipeline, X_train, X_test, y_train, y_test 


'''TOUTES LES FONCTIONS POUR MODELE TEMPOREL'''

def modelisation_per_compteur(df, compteur):
    """
    Fonction qui entraîne un modèle Prophet pour un compteur spécifique.
    """
    # Filtrer les données du compteur
    df_compteur = df[df['nom_compteur'] == compteur]

    df_compteur['ds'] = pd.to_datetime(df_compteur["date_heure_comptage"])
    df_compteur['y'] = df_compteur['comptage_horaire']

    train_size = int(len(df_compteur) * 0.8)
    train_data = df_compteur[:train_size]
    test_data = df_compteur[train_size:]

    train_data = utils.encode_cyclic_prophet(train_data,"num_mois",train_data['ds'].dt.month)
    test_data = utils.encode_cyclic_prophet(test_data,"num_mois",train_data['ds'].dt.month)
    train_data = utils.encode_cyclic_prophet(train_data,"num_jour_semaine",train_data['ds'].dt.weekday)
    test_data = utils.encode_cyclic_prophet(test_data,"num_jour_semaine",test_data['ds'].dt.weekday)

    # Créer le modèle Prophet
    model = Prophet( changepoint_prior_scale=0.5,   # cross validation = 0.001  moi : 0.5
                    seasonality_prior_scale=1.0,   # cross validation = 100.0   moi : 1.0
                    holidays_prior_scale=10.0,      #  cross validation = 0.1  moi : 10.0
                    seasonality_mode='multiplicative', 
                    yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True,      
                    changepoint_range=0.80        # cross validation = 0.1  moi 0.80 
    )
    model.add_country_holidays(country_name='FR')

    # Ajouter les régressions externes
    model.add_regressor('precipitation_mm')
    model.add_regressor('wind_speed')
    model.add_regressor('vacances')
    model.add_regressor('fait_jour')
    model.add_regressor('weekend')
    model.add_regressor('heure')
    model.add_regressor('num_mois_sin')
    model.add_regressor('num_mois_cos')
    model.add_regressor('num_mois')
    model.add_regressor('num_jour_semaine_sin')
    model.add_regressor('num_jour_semaine_cos')

    # Entraîner le modèle
    model.fit(train_data)

    return model, train_data, test_data

def predict_and_evaluate(model, test_data):
    """
    Effectue les prédictions et l'évaluation pour un compteur donné avec model entrainé
    """
    #on recupère que les dates de la partie test pour faire la prédiction
    future = test_data[['ds','precipitation_mm','wind_speed','vacances','weekend','fait_jour','heure','num_mois']].copy() 
    future = utils.encode_cyclic_prophet(future,"num_mois",future['ds'].dt.month)
    future = utils.encode_cyclic_prophet(future,"num_jour_semaine",future['ds'].dt.weekday)

    # Prédictions
    forecast = model.predict(future)
    test_predictions = forecast[forecast['ds'].isin(test_data['ds'])]
 
    # Calcul du score (Mean Absolute Error)
    mae = mean_absolute_error(test_data['y'], test_predictions['yhat'])

    return test_data, test_predictions, mae

def modelisationProphet(df,compteurs):
    
    df_work = df.copy()
    df_work.sort_values(by=["date_heure_comptage","nom_compteur"], ascending=True, inplace=True)

    df_work = df_work[['nom_compteur','num_jour_semaine', 'num_mois', 'fait_jour', 'weekend', 'vacances', 'neutralise',
                   'temperature_2m', 'precipitation_mm', 'wind_speed', 'heure', 'année', "date_heure_comptage","comptage_horaire"]]
    
    models = {}
    #compteurs = df_work['nom_compteur'].unique()
    for compteur in compteurs:
        model, train_data, test_data = modelisation_per_compteur(df_work, compteur)
        models[compteur] = {
            'model': model,
            'train_data': train_data,
            'test_data': test_data
        }
    return models

'''FONCTION POUR PREDICTION SUR FEVRIER'''

def predictionModel(classifier, modelProphetCompteur, compteur):

    df_fevrier = utils.load_data(Config.FILE_PATH + Config.FILE_FEVRIER, ",",0) 
    df_fevrier.sort_values(by=["date_heure_comptage","nom_compteur"], ascending=True, inplace=True)
    df_fevrier.reset_index(drop=True, inplace=True)

    # 
    if classifier == 'XGBRegressor':
        #récupération du model
        modelXGB = load('model_XGB.joblib')
        features = ['nom_compteur', 'heure', 'num_mois', 'num_jour_semaine', 'latitude', 'longitude',
        'temperature_2m', 'precipitation_mm', 'wind_speed', 'fait_jour', 'neutralise', 'vacances', "Partage", "année"]
        cyclic_columns = ['heure', 'num_jour_semaine', 'num_mois']
        df_fevrier = utils.encode_cyclic(df_fevrier, cyclic_columns)
        features_extended = features + [f'{col}_sin' for col in cyclic_columns] + [f'{col}_cos' for col in cyclic_columns]
        y_pred = modelXGB.predict(df_fevrier[features_extended])
        df_fevrier['predictions_comptage_horaire'] = y_pred

    elif classifier == 'StackingRegressor':
         #récupération du model
        modelST = load('model_ST.joblib')
        features = ['nom_compteur', 'heure', 'num_mois', 'num_jour_semaine', 'latitude', 'longitude',
        'temperature_2m', 'precipitation_mm', 'wind_speed', 'fait_jour', 'neutralise', 'vacances', "Partage", "année"]
        cyclic_columns = ['heure', 'num_jour_semaine', 'num_mois']
        df_fevrier = utils.encode_cyclic(df_fevrier, cyclic_columns)
        features_extended = features + [f'{col}_sin' for col in cyclic_columns] + [f'{col}_cos' for col in cyclic_columns]
        y_pred = modelST.predict(df_fevrier[features_extended])
        df_fevrier['predictions_comptage_horaire'] = y_pred

    elif classifier == 'Random Forest Regressor':
        modelRFR = load('model_RFR.joblib')

        df_fevrier = utils.encode_cyclic(df_fevrier, ['heure', 'num_mois', 'num_jour_semaine'])
        df_fevrier = df_fevrier.drop(columns=['num_mois'])

        #en dropant moins bon
        df_fevrier['precipitation'] = pd.cut(x=df_fevrier['precipitation_mm'], bins=[0.0, 0.5, 3.5, 16.0],
                                   labels=['Pas de pluie/bruine', 'Pluie modérée', 'Fortes averses'], right=False)
        df_fevrier['wind'] = pd.cut(x=df_fevrier['wind_speed'], bins=[0.0, 5, 19, 38, 43.0],
                          labels=['Pas de vent', 'Vent modérée', 'vent', 'grand vent'], right=False)
        df_fevrier['temperature'] = pd.cut(x=df_fevrier['temperature_2m'], bins=[-10, 0, 10, 20, 30, 36],
                                 labels=['Gel','Froid', 'Tempéré', 'Chaud', 'Très chaud'], right=False)
        encoder = OrdinalEncoder(categories=[['Gel', 'Froid', 'Tempéré', 'Chaud', 'Très chaud']])
        df_fevrier['temperature_encoded'] = encoder.fit_transform(df_fevrier[['temperature']])
        encoderP = OrdinalEncoder(categories=[['Pas de pluie/bruine', 'Pluie modérée', 'Fortes averses']])
        df_fevrier['precipitation_encoded'] = encoderP.fit_transform(df_fevrier[['precipitation']])
        encoderW = OrdinalEncoder(categories=[['Pas de vent', 'Vent modérée', 'vent', 'grand vent']])
        df_fevrier['wind_encoded'] = encoderW.fit_transform(df_fevrier[['wind']])

        df_fevrier_cat = pd.get_dummies(df_fevrier[['nom_compteur','num_jour_semaine', 'fait_jour', 
                            'vacances', 'neutralise', 'precipitation', 'wind','temperature', 'Partage', '1Sens', 'latitude', 'heure','année']], drop_first=True)

        # Combine les variables catégorielles transformées et les variables numériques
        df_fevrier_encoded = pd.concat([df_fevrier_cat, df_fevrier[['heure_sin', 'heure_cos', 
                                                    'num_mois_sin', 'num_mois_cos','num_jour_semaine_cos','num_jour_semaine_sin','precipitation_encoded','temperature_encoded','wind_encoded']]], axis=1)
       
        y_pred = modelRFR.predict(df_fevrier_encoded)
        df_fevrier['predictions_comptage_horaire'] = y_pred

    elif classifier == 'Prophet':


        df_fevrier_copy = df_fevrier[df_fevrier['nom_compteur'] == compteur].copy()
        df_fevrier_copy['ds'] = pd.to_datetime(df_fevrier_copy["date_heure_comptage"])
        df_fevrier_copy.sort_values(by=["ds"], ascending=True, inplace=True)
        df_fevrier_copy.reset_index(drop=True, inplace=True)

        # Créer une copie de df_fevrier_copy avec les colonnes spécifiées
        future = df_fevrier_copy[['ds', 'precipitation_mm', 'wind_speed', 'vacances', 'weekend', 'fait_jour', 'heure', 'num_mois']].copy()
        future = utils.encode_cyclic_prophet(future,"num_mois",future['ds'].dt.month)
        future = utils.encode_cyclic_prophet(future,"num_jour_semaine",future['ds'].dt.weekday)

        forecast = modelProphetCompteur['model'].predict(future)
        df_fevrier_copy['predictions_comptage_horaire'] = forecast['yhat']

        df_fevrier= df_fevrier_copy

    return df_fevrier

'''FONCTION POUR PREDICTION A 3JOUR'''

def prediction3JModel(classifier,df, df_vac, modelProphetCompteur, compteur):

    df_work = df.copy()
    df_work_vac = df_vac.copy()
    meteo = utils.meteoSearch()
    df3J = utils.createDataframe(meteo, df_work, df_work_vac)

    if classifier == 'XGBRegressor':
        #récupération du model
        pipeline = load('model_XGB.joblib')
        
        features = ['nom_compteur', 'heure', 'num_mois', 'num_jour_semaine', 'latitude', 'longitude',
                    'temperature_2m', 'precipitation_mm', 'wind_speed', 'fait_jour', 'neutralise', 'vacances', "Partage", "année"]
        cyclic_columns = ['heure', 'num_jour_semaine', 'num_mois']
        df3J = utils.encode_cyclic(df3J, cyclic_columns)
        features_extended = features + [f'{col}_sin' for col in cyclic_columns] + [f'{col}_cos' for col in cyclic_columns]
        y_pred = pipeline.predict(df3J[features_extended])
        df3J['predictions_comptage_horaire'] = y_pred
        utils.create_data3J(df3J, "XGB")   

    elif classifier == 'StackingRegressor':
        classifier = classifier
    elif classifier == 'Random Forest Regressor':
        classifier = classifier
    elif classifier == 'Prophet':  
        df3J_copy = df3J[df3J['nom_compteur'] == compteur].copy()
        df3J_copy['ds'] = pd.to_datetime(df3J_copy["date_heure_comptage"])
        df3J_copy.sort_values(by=["ds"], ascending=True, inplace=True)
        df3J_copy.reset_index(drop=True, inplace=True)

        # Créer une copie de df_fevrier_copy avec les colonnes spécifiées
        future = df3J_copy[['ds', 'precipitation_mm', 'wind_speed', 'vacances', 'weekend', 'fait_jour', 'heure', 'num_mois']].copy()
        future = utils.encode_cyclic_prophet(future,"num_mois",future['ds'].dt.month)
        future = utils.encode_cyclic_prophet(future,"num_jour_semaine",future['ds'].dt.weekday)

        forecast = modelProphetCompteur['model'].predict(future)
        df3J_copy['predictions_comptage_horaire'] = forecast['yhat']



        df3J= df3J_copy 
        utils.create_data3J(df3J, "PROPHET")  

    return df3J



def scores(clf, choice, X_train, X_test, y_train, y_test):
  if choice == 'score (R²)':
     score1, score2 = clf.score(X_train, y_train),clf.score(X_test,y_test) 
  elif choice == 'metrique MAE':
      y_predTest = clf.predict(X_test)
      y_predTrain = clf.predict(X_train)
      score1, score2 =  mean_absolute_error(y_train, y_predTrain), mean_absolute_error(y_test, y_predTest)
  return score1, score2