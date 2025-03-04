from random import choices
import time
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, StackingRegressor
from prophet import Prophet
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import utilsPython as utils # type: ignore
from joblib import dump, load
# Pour éviter d'avoir les messages warning
import warnings
import os

warnings.filterwarnings('ignore')

@st.cache_data
def modelisationRFBase(df_merged_cleaned_final):  
    '''Modelisation pour RF de base pour trouver les importance feature
    '''
    start_time = time.time()
    print("Debut de modelisationBA :", time.ctime(start_time))

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
         print('le fichier existe déjà')
         model3 = load('model_RFRBA.joblib')
    else: #14min
        print('le fichier n existe pas')
        model3 = RandomForestRegressor(random_state=42)
        model3.fit(X_train_scaled, y_train)

        dump(model3, 'model_RFRBA.joblib')

    start_time = time.time()
    print("Debut prediction modelisationBA :", time.ctime(start_time))        
    y_pred3 = model3.predict(X_test_scaled) 
    print("MAEpred", mean_absolute_error(y_test, y_pred3))

    start_time = time.time()
    print("Fin de modelisationBA :", time.ctime(start_time))

    return model3,X_train,feats



def encode_cyclic(df, columns):
    period = {'heure': 24, 'num_jour_semaine': 7, 'num_mois': 12, 'heure_num_jour_semaine': 168}
    for col in columns:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period[col])
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period[col])
    return df

@st.cache_data
def modelisation1(df, compteur , option):
  ''' Modelisation pour les autes modèles <> XGB et STacking
    '''
  start_time = time.time()
  print("Debut de modelisation1 :", time.ctime(start_time))
  print(compteur)

  df_work =df.copy()

  df_work.sort_values(by=["date_heure_comptage","nom_compteur"], ascending=True, inplace=True)

  if(compteur != 'All'):
    utils.removeJoblibFile()
    df_work =df_work[df_work['nom_compteur'] == compteur]
  
  y = df_work['comptage_horaire']

  X_cat = df_work[['nom_compteur','num_jour_semaine', 'num_mois', 'fait_jour', 
                   #'weekend',                       perte
                   'vacances', 'neutralise', 'precipitation', 'wind','temperature', 'Partage', '1Sens', 'latitude', 
                   #'longitude', 'weekend_heure_interaction', 'heure_num_jour_semaine'      perte
                   'heure','année']]
  X_num = df_work[[]]
  

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
                                              'num_mois_sin', 'num_mois_cos','num_jour_semaine_cos','num_jour_semaine_sin','precipitation_encoded','temperature_encoded','wind_encoded']]], axis=1)
  X_test = pd.concat([X_test_cat, X_test0[['heure_sin', 'heure_cos', 
                                           'num_mois_sin', 'num_mois_cos','num_jour_semaine_cos','num_jour_semaine_sin','precipitation_encoded','temperature_encoded','wind_encoded']]], axis=1)

  start_time = time.time()
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
    #clf_cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    #print(f'Validation croisée {classifier} : MAE moyen = {-clf_cv_scores.mean():.4f} (±{clf_cv_scores.std():.4f})')
    
  start_time = time.time()
  print("fin de prediction :", time.ctime(start_time))  
  return clf, X_train, X_test, y_train, y_test 





def scores(clf, choice, X_train, X_test, y_train, y_test):
  if choice == 'score (R²)':
     score1, score2 = clf.score(X_train, y_train),clf.score(X_test,y_test) 
  elif choice == 'metrique MAE':
      y_predTest = clf.predict(X_test)
      y_predTrain = clf.predict(X_train)
      score1, score2 =  mean_absolute_error(y_train, y_predTrain), mean_absolute_error(y_test, y_predTest)
  return score1, score2

#start_time = time.time()
#print("Je rentre dans le méthode prediction Heure de début :", time.ctime(start_time))


def encode_cyclic(df, columns):
    for col in columns:
        period = {'heure': 24, 'num_jour_semaine': 7, 'num_mois': 12}
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period[col])
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period[col])
    return df


def modelisation(df, classifier):

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
    # Instancier le modèle XGB
    xgb_params = {'n_estimators': 982,'max_depth': 15,'learning_rate': 0.010839297840803218,'colsample_bytree': 0.9763448531819173,
        'subsample': 0.8004427030847461,'gamma': 2.1532845545117967,'reg_alpha': 6.080159216998929,'reg_lambda': 8.274555339627717,
        'random_state': 42}
    
    if classifier == 'XGBRegressor':
        start_time = time.time()
        print("debut de l'entrainement de XGB :", time.ctime(start_time))
        if os.path.exists('model_XGB.joblib'):
            # Si le fichier existe, charger le modèle sauvegardé
            print('le fichier existe déjà')
            pipeline = load('model_XGB.joblib')
        else:
            print('le fichier n existe pas')
            xgb_model = XGBRegressor(**xgb_params)
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', xgb_model)])
            pipeline.fit(X_train, y_train)
            dump(pipeline, 'model_XGB.joblib') 
        #xgb_cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        #print(f'Validation croisée XGBoost : MAE moyen = {-xgb_cv_scores.mean():.4f} (±{xgb_cv_scores.std():.4f})')

    elif classifier == 'StackingRegressor':
        start_time = time.time()
        print("debut de l'entrainement de StackingRegressor :", time.ctime(start_time))
        if os.path.exists('model_ST.joblib'):
            # Si le fichier existe, charger le modèle sauvegardé
            print('le fichier existe déjà')
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

    start_time = time.time()
    print("fin de l'entrainement :", time.ctime(start_time))

    return pipeline, X_train, X_test, y_train, y_test 
    #return y_pred, y_test


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





def modelisation_per_compteur(df, compteur):
    """
    Fonction qui entraîne un modèle Prophet pour un compteur spécifique.
    """
    print(compteur)
    # Filtrer les données du compteur
    df_compteur = df[df['nom_compteur'] == compteur]
    print(df['nom_compteur'].unique())
    print(df_compteur)
    df_compteur['ds'] = pd.to_datetime(df_compteur["date_heure_comptage"])
    df_compteur['y'] = df_compteur['comptage_horaire']

    train_size = int(len(df_compteur) * 0.8)
    train_data = df_compteur[:train_size]
    test_data = df_compteur[train_size:]
    
    train_data['num_mois_sin'] = np.sin(2 * np.pi * train_data['ds'].dt.month / 12)
    train_data['num_mois_cos'] = np.cos(2 * np.pi * train_data['ds'].dt.month / 12)
    test_data['num_mois_sin'] = np.sin(2 * np.pi * test_data['ds'].dt.month / 12)
    test_data['num_mois_cos'] = np.cos(2 * np.pi * test_data['ds'].dt.month / 12)
    
    train_data['num_jour_semaine_sin'] = np.sin(2 * np.pi * train_data['ds'].dt.weekday / 7)
    train_data['num_jour_semaine_cos'] = np.cos(2 * np.pi * train_data['ds'].dt.weekday / 7)
    test_data['num_jour_semaine_sin'] = np.sin(2 * np.pi * test_data['ds'].dt.weekday / 7)
    test_data['num_jour_semaine_cos'] = np.cos(2 * np.pi * test_data['ds'].dt.weekday / 7)

    # Créer le modèle Prophet
    model = Prophet( 
                    changepoint_prior_scale=0.5,   # cross validation = 0.001  moi : 0.5
                    seasonality_prior_scale=1.0,   # cross validation = 100.0   moi : 1.0
                    holidays_prior_scale=10.0,      #  cross validation = 0.1  moi : 10.0
                    seasonality_mode='multiplicative',    # Mode de saisonnalité ('additive' ou 'multiplicative')
                    yearly_seasonality=False,        
                    weekly_seasonality=True,        
                    daily_seasonality=True,      
                    changepoint_range=0.80        # cross validation = 0.1  moi 0.80 
    )

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

    #dump(model, 'model_BR.joblib') il va falloir enregistré le model sur le compteur

    return model, train_data, test_data

def predict_and_evaluate(model, df, test_data):
    """
    Effectue les prédictions pour un compteur donné model : model entrainé sur le compteur, df dataframe filtré que le compteur
    """
    #on recupère que les dates de la partie test pour faire la prédiction
    future = test_data[['ds']].copy() 
    
    # Ajouter les colonnes de régressions externes pour les dates futures
    future['precipitation_mm'] = future['ds'].apply(lambda x: get_precipitation_for_date(x, df))
    future['wind_speed'] = future['ds'].apply(lambda x: get_wind_for_date(x, df))
    future['vacances'] = future['ds'].apply(lambda x: get_vacances_for_date(x, df))
    future['weekend'] = future['ds'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
    future['fait_jour'] = future['ds'].apply(utils.get_fait_jour)
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

@st.cache_data
def modelisationT(df,compteurs):
    
    df_work = df.copy()
    df_work.sort_values(by=["date_heure_comptage","nom_compteur"], ascending=True, inplace=True)

    df_work = df_work[['nom_compteur',    # va nous permettre de selectionner le compteur
                   'num_jour_semaine', 'num_mois', 'fait_jour', 'weekend', 'vacances', 'neutralise',  #0,1 et  #0,1
                   'temperature_2m', 'precipitation_mm', 'wind_speed', 'heure', 'année', "date_heure_comptage",
                   "comptage_horaire"]]
    
    print(df_work['nom_compteur'].unique())
    
    models = {}
    #compteurs = df_work['nom_compteur'].unique()
    for compteur in compteurs:
        print(compteur)
        model, train_data, test_data = modelisation_per_compteur(df_work, compteur)
        models[compteur] = {
            'model': model,
            'train_data': train_data,
            'test_data': test_data
        }
    return models

def predictionModel(classifier, df_fevrier):

    df_fevrier.sort_values(by=["date_heure_comptage","nom_compteur"], ascending=True, inplace=True)
    df_fevrier.reset_index(drop=True, inplace=True)

    # Exemple de logique de prédiction
    # Remplacez cela par votre logique réelle de prédiction
    if classifier == 'XGBRegressor':
        #récupération du model
        pipeline = load('model_XGB.joblib')
        features = ['nom_compteur', 'heure', 'num_mois', 'num_jour_semaine', 'latitude', 'longitude',
        'temperature_2m', 'precipitation_mm', 'wind_speed', 'fait_jour', 'neutralise', 'vacances', "Partage", "année"]
        cyclic_columns = ['heure', 'num_jour_semaine', 'num_mois']
        df_fevrier = encode_cyclic(df_fevrier, cyclic_columns)
        features_extended = features + [f'{col}_sin' for col in cyclic_columns] + [f'{col}_cos' for col in cyclic_columns]
        y_pred = pipeline.predict(df_fevrier[features_extended])
        df_fevrier['predictions_comptage_horaire'] = y_pred

    elif classifier == 'StackingRegressor':
         #récupération du model
        pipeline = load('model_ST.joblib')
        features = ['nom_compteur', 'heure', 'num_mois', 'num_jour_semaine', 'latitude', 'longitude',
        'temperature_2m', 'precipitation_mm', 'wind_speed', 'fait_jour', 'neutralise', 'vacances', "Partage", "année"]
        cyclic_columns = ['heure', 'num_jour_semaine', 'num_mois']
        df_fevrier = encode_cyclic(df_fevrier, cyclic_columns)
        features_extended = features + [f'{col}_sin' for col in cyclic_columns] + [f'{col}_cos' for col in cyclic_columns]
        y_pred = pipeline.predict(df_fevrier[features_extended])
        df_fevrier['predictions_comptage_horaire'] = y_pred

    elif classifier == 'Random Forest Regressor':
        df_fevrier = df_fevrier
    elif classifier == 'Prophet':
        df_fevrier = df_fevrier        
    return df_fevrier

def prediction3JModel(classifier,df, df_vac):

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
        df3J = encode_cyclic(df3J, cyclic_columns)
        features_extended = features + [f'{col}_sin' for col in cyclic_columns] + [f'{col}_cos' for col in cyclic_columns]
        y_pred = pipeline.predict(df3J[features_extended])
        df3J['predictions_comptage_horaire'] = y_pred

    elif classifier == 'StackingRegressor':
        classifier = classifier
    elif classifier == 'Random Forest Regressor':
        classifier = classifier
    elif classifier == 'Prophet':
        classifier = classifier     
    utils.create_data3J(df3J)   
    return df3J