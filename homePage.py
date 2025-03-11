from datetime import datetime, time #as dt_time  # Garde time tel quel
from datetime import timedelta # Importer la classe time sous un alias
import pandas as pd
import streamlit as st
import utilsPython as utils 
import utilsPreprocess as preproc 
import utilsGraph as graph 
import modelisation as modelisation 
from config import Config
import streamlit.components.v1 as components
#from streamlit_folium import st_folium
# Pour éviter d'avoir les messages warning
import warnings
warnings.filterwarnings('ignore')

# Personnalisation de la page streamlit avec CSS
st.markdown(
    """
    <style>
        /* Largeur de la zone centrale */
        .main {
            max-width: 65% !important; 
            margin-left: auto;
            margin-right: auto;
        }

        /* Pour les pages d'index (si besoin) */
        .block-container {
            max-width: 65% !important;  
        }

        /* Réduire la taille des graphiques */
        .stImage {
            width: auto;  /* Réduire la largeur de l'image */
            height: auto;
        }

        /* Ajuster la taille de l'expander header */
        .streamlit-expanderHeader {
            font-size: 16px !important;
        }

        /* Ajuster l'affichage des graphiques matplotlib */
        .stPlot {
            width: auto;  /* Ajuster la largeur du graphique */
            height: auto;
        }
       /* Augmenter la taille des libellés des onglets */
        div[data-baseweb="tab-list"] button[data-testid="stTab"] {
            fontSize: 30px;  /* Ajustez la taille des libellés des onglets */
            padding: 12px 24px;  /* Ajuste le padding des onglets */
        }

        /* Ajuster le fond des onglets actifs */
        div[data-baseweb="tab-list"] button[data-testid="stTab"][aria-selected="true"] {
            fontSize: 30px;  /* Ajustez la taille de l'onglet actif */
            font-weight: bold;  /* Applique le gras sur l'onglet actif */
        }
    </style>
    """, unsafe_allow_html=True
)

#on evite de refaire les diffférents dataframe s'ils sont déjà en session, sinon on les calcule et on les met en session
if 'df_merged_cleaned_final' not in st.session_state: # 60 secondes
    df = utils.load_data(Config.FILE_PATH + Config.FILE, ";",0) 
    df_m = utils.load_data(Config.FILE_PATH + Config.FILE_METEO, ',', 2)	
    df_v = utils.load_data(Config.FILE_PATH + Config.FILE_VAC, ',', 0)
    df_f = utils.load_data(Config.FILE_PATH + Config.FILE_FERIE, ',', 0)
    df_p = utils.load_data(Config.FILE_PATH + Config.FILE_PHOTO, ',', 0)
    df_ir = utils.load_data(Config.FILE_PATH + Config.FILE_TRAVAUX, ';', 0)
    df_cleaned = preproc.preprocess_cyclisme(df)
    df_m_cleaned = preproc.preprocess_meteo(df_m)
    df_vjf_cleaned = preproc.preprocess_vacancesferie(df_v,df_f)
    df_p_cleaned = preproc.preprocess_photo(df_p)
    df_merged = utils.merge(df_cleaned,df_m_cleaned,df_vjf_cleaned,df_p_cleaned,df_ir)
    df_merged_cleaned = preproc.corriger_comptage(df_merged) 
    df_merged_cleaned_final = preproc.correctionDataviz(df_merged_cleaned) 

    st.session_state.df = df
    st.session_state.df_m = df_m 
    st.session_state.df_v = df_v
    st.session_state.df_f = df_f 
    st.session_state.df_p = df_p 
    st.session_state.df_ir = df_ir
    st.session_state.df_cleaned = df_cleaned
    st.session_state.df_m_cleaned = df_m_cleaned
    st.session_state.df_vjf_cleaned = df_vjf_cleaned
    st.session_state.df_p_cleaned = df_p_cleaned
    st.session_state.df_merged = df_merged
    st.session_state.df_merged_cleaned = df_merged_cleaned
    st.session_state.df_merged_cleaned_final = df_merged_cleaned_final
else:  #0 seconde
    df =st.session_state.df
    df_m = st.session_state.df_m
    df_v = st.session_state.df_v
    df_f = st.session_state.df_f
    df_vjf_cleaned = st.session_state.df_vjf_cleaned
    df_p_cleaned = st.session_state.df_p_cleaned
    df_ir= st.session_state.df_ir
    df_merged = st.session_state.df_merged
    df_merged_cleaned = st.session_state.df_merged_cleaned
    df_merged_cleaned_final = st.session_state.df_merged_cleaned_final 


#affichage de l'image
st.image(Config.IMAGE, width=400) 

#titre et sommaire
st.title("Trafic cycliste à PARIS")
st.sidebar.title("Sommaire")
pages=["Présentation du sujet","Exploration", "DataVizualization", "Modélisation","Prédiction"]
page=st.sidebar.radio("Aller vers", pages, key='menu_principal')

#Page 1
if page == pages[0] : #0seconde
  st.write("### Présentation")
  st.markdown(Config.PRESENTATION, unsafe_allow_html=True)

  st.markdown('<p style="margin-bottom: 0px;font-size:12px; font-style:italic;">Bootcamp Analytics Engineer JAN2025', unsafe_allow_html=True)
  st.markdown('<p style="font-size:12px; font-style:italic;">Aurélie Guilhem - Ingrid Plessis - Nicolas Couvez - Marie Pirao', unsafe_allow_html=True)

  st.markdown(f'<p style="margin-bottom: 0px;">Source à exploiter : {Config.FILE}</p>', unsafe_allow_html=True)
  st.markdown('<p style="font-size:12px; font-style:italic;">source : <a href="https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs" target="_blank">https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs</a></p>', unsafe_allow_html=True)
  

  if st.checkbox("Afficher les données externes", key='page1') : # sinon les chekbox de toute l'appli sont liées
    st.markdown(f'<p style="margin-left: 50px;margin-bottom: 0px;">🌤️ Données météorologiques :  {Config.FILE_METEO}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="margin-left: 50px;font-size:12px; font-style:italic;">sources : <a href="https://www.data.gouv.fr/fr/organizations/meteo-france/" target="_blank">https://www.data.gouv.fr/fr/organizations/meteo-france/</a></p>', unsafe_allow_html=True)
    st.markdown('<p style="margin-left: 50px;font-size:12px; font-style:italic;">         <a href="https://www.meteo-paris.com/meteo-8-jours/paris-75000" target="_blank">https://www.meteo-paris.com/meteo-8-jours/paris-75000</a></p>', unsafe_allow_html=True)

    st.markdown(f'<p style="margin-left: 50px;margin-bottom: 0px;">🏖️ Données vacances scolaires :  {Config.FILE_VAC}</p>', unsafe_allow_html=True)
    st.markdown('<p style="margin-left: 50px;font-size:12px; font-style:italic;">source : <a href="https://www.data.gouv.fr/fr/datasets/calendrier-scolaire/" target="_blank">https://www.data.gouv.fr/fr/datasets/calendrier-scolaire/</a></p>', unsafe_allow_html=True)
    
    st.markdown(f'<p style="margin-left: 50px;margin-bottom: 0px;">🎌 Données jours fériés :  {Config.FILE_FERIE}</p>', unsafe_allow_html=True)
    st.markdown('<p style="margin-left: 50px;font-size:12px; font-style:italic;">source : <a href="https://www.data.gouv.fr/fr/datasets/jours-feries-en-france/" target="_blank">https://www.data.gouv.fr/fr/datasets/jours-feries-en-france/</a></p>', unsafe_allow_html=True)
    
    st.markdown(f'<p style="margin-left: 50px;margin-bottom: 0px;">📸 Données détail photo :  {Config.FILE_PHOTO}</p>', unsafe_allow_html=True)
    st.markdown('<p style="margin-left: 50px;font-size:12px; font-style:italic;">source : effectué manuellement', unsafe_allow_html=True)
        
    st.markdown(f'<p style="margin-left: 50px;margin-bottom: 0px;">:🚧 Données détail travaux ou blocage des JO :  {Config.FILE_TRAVAUX}</p>', unsafe_allow_html=True)
    st.markdown('<p style="margin-left: 50px;font-size:12px; font-style:italic;">source : effectué manuellement', unsafe_allow_html=True)

    
#Page 2
if page == pages[1] : #4 seconde

  st.write("### Exploration des données")
  titres_onglets2 = ['Visualisation des données', 'Travail de nettoyage', 'Analyse spécifique de certains compteurs', 'Dataframe final']
  ongletA, ongletB,ongletC, ongletD = st.tabs(titres_onglets2)

# ONglet de présentation des différents dataframes créés 
  with ongletA:  
    st.subheader("Aperçu du jeu de données")
    st.dataframe(df.head(5))  # Affichage de l'apercu du dataframe initial

    st.subheader("Informations sur le jeu de données")
    info_dict_aff = utils.informationDF(df)
    st.dataframe(info_dict_aff)
    st.markdown(Config.OBSERVATION_DF, unsafe_allow_html=True)
    st.markdown(Config.OBSERVATION_DF1, unsafe_allow_html=True)

    if st.checkbox("Afficher les données externes", key='ongletA') :
      st.subheader("Aperçu du jeu de données externes")
      
      st.write("Données météorologiques")
      st.dataframe(df_m.head())
      st.markdown(Config.OBSERVATION_METEO, unsafe_allow_html=True)

      st.write("Données vacances scolaires")
      df_v['date'] = pd.to_datetime(df_v['date'])
      df_v_2024 = df_v[df_v['date'].dt.year == 2024]
      st.dataframe(df_v_2024.head(), hide_index=True)

      st.markdown(Config.OBSERVATION_VAC, unsafe_allow_html=True)

      st.write("Données jours fériés")
      df_f['date'] = pd.to_datetime(df_f['date'])
      df_f_2024 = df_f[df_f['date'].dt.year == 2024]
      df_f_2024['annee'] = df_f_2024['annee'].astype(str)  #streamlit apparemment affiche 2,024 si on ne force pas a str
      st.dataframe(df_f_2024.head(), hide_index=True)
      st.markdown(Config.OBSERVATION_JF, unsafe_allow_html=True)

      st.write("Données détails photo")
      st.dataframe(df_p_cleaned.head(), hide_index=True)
      st.markdown(Config.OBSERVATION_PHOTO, unsafe_allow_html=True)

      st.write("Données blocage rue")
      st.dataframe(df_ir.iloc[128:133].head(), hide_index=True)
      st.markdown(Config.OBSERVATION_TRAVAUX, unsafe_allow_html=True)
      
  with ongletB: 
    #analyse de la dsitribution des compteurs selon les mois/année et conclusion
    st.subheader("Analyse de la répartition des compteurs")
    fig = graph.plot_avg_mensuel(df,"all")
    st.plotly_chart(fig)
    st.markdown(Config.CONCLUSION_REPARTITION, unsafe_allow_html=True)

    #Analyse des lignes manaquantes pour un dataframe complet sur la durée
    st.subheader("Compteurs ayant des lignes manquantes sur 2024/2025")
    fig = graph.nbLigne_compteur(df)
    st.plotly_chart(fig)
    st.markdown(Config.CONCLUSION_NBLIGNE, unsafe_allow_html=True)

    #analyse du compteur de la Grande Armée pour montrer qu'il faudra corriger certains compteurs
    st.subheader("Analyse du compteur : Grande Armée")
    fig = graph.plot_avg_mensuel(df,"GrandeArmee")
    st.plotly_chart(fig)
    st.markdown(Config.CONCLUSION_GA, unsafe_allow_html=True)

    #Analyse des valeurs manquantes
    if st.checkbox("Afficher les NA", key='ongletB') :
      st.subheader("Gestion des valeurs manquantes")
      fig = graph.heatmap_isna(df)
      st.pyplot(fig)
      st.markdown(Config.CONCLUSION_NAN, unsafe_allow_html=True)


  with ongletC: 
    #Analyse spécifique de la période des JO et impact sur les compteurs
    st.subheader("Impact des Jeux Olympiques sur le trafic cycliste")
    fig = graph.px_compteurs_mensuel_JO(df_merged)
    st.plotly_chart(fig)
    st.markdown(Config.EXPLICATIONJO, unsafe_allow_html=True)

    #Analyse spécifique de certains compteurs présentant de grandes interation de compteurs à 0
    st.subheader("Itération de 10 comptages à 0 sur certains compteurs")
    #possibilité choix compteurs
    Compteur = ['10 avenue de la Grande Armée SE-NO','106 avenue Denfert Rochereau NE-SO','135 avenue Daumesnil SE-NO','24 boulevard Jourdan E-O',
                '33 avenue des Champs Elysées NO-SE','38 rue Turbigo','boulevard Richard Lenoir','Pont des Invalides',
                "27 quai de la Tournelle","7 avenue de la Grande Armée NO-SE",
                "Porte des Ternes", "Face au 48 quai de la marne",
                "Totem 73 boulevard de Sébastopol"]
    # Liste déroulante pour choisir un compteur
    selected_compteur = st.selectbox("Sélectionnez un compteur",options=Compteur,index=0) 
    #affichage du graphique
    fig = graph.px_compteurs_quotidien_0(df_merged, selected_compteur)
    st.plotly_chart(fig, use_container_width=True)
    #affichage explication
    explication0 = ''''''
    if (selected_compteur == '24 boulevard Jourdan E-O'):
      explication0 = Config.BOULEVARD_JOURDAN
    elif (selected_compteur == '10 avenue de la Grande Armée SE-NO'):
      explication0 = Config.GRANDE_ARMEE
    elif (selected_compteur == '135 avenue Daumesnil SE-NO'):
      explication0 = Config.DAUMESNIL
    elif (selected_compteur == '33 avenue des Champs Elysées NO-SE'):
      explication0 = Config.CHAMPS_ELYSEE
    elif (selected_compteur == '38 rue Turbigo'):
      explication0 = Config.TURBIGO
    elif (selected_compteur == '106 avenue Denfert Rochereau NE-SO'):
      explication0 = Config.ROCHEREAU
    elif (selected_compteur == '27 quai de la Tournelle'):
      explication0 = Config.TOURNELLE   
    elif selected_compteur in ['boulevard Richard Lenoir','7 avenue de la Grande Armée NO-SE','Pont des Invalides','Porte des Ternes','Face au 48 quai de la marne','Totem 73 boulevard de Sébastopol']:
      explication0 = Config.DEUXSENS 
                
    st.markdown(explication0, unsafe_allow_html=True)

    #Analkyse ciblé sur valeur abhérrante
    st.subheader("Gestion des outliers")
    st.write("Distribution de la variable comptage_horaire des vélos")
    fig = graph.boxplot(df_merged['comptage_horaire'])
    st.pyplot(fig)
    st.markdown(Config.ABERRANTE, unsafe_allow_html=True)

    #Mise en évidencce de la problématique
    fig1 = graph.plot_abherrante(df_merged)
    st.plotly_chart(fig1, key="graph_abherrante_1")
    st.markdown(Config.ABERRANTE1, unsafe_allow_html=True)

    #Proposition de correction et résultat de la correction
    st.markdown(Config.ABERRANTECORRECTION, unsafe_allow_html=True)
    fig2 = graph.plot_abherrante(df_merged_cleaned)
    st.plotly_chart(fig2, key="graph_abherrante_2")

  with ongletD: #2seconde
    #Présentation du dataframe final après correction
    st.subheader(" Dataframe final")
    df_merged_cleaned_final_aff = df_merged_cleaned_final.copy()
    df_merged_cleaned_final_aff['année'] = df_merged_cleaned_final_aff['année'].astype(int).astype(str) #streamlit apparemment affiche 2,024 si on ne force pas a str
    st.dataframe(df_merged_cleaned_final_aff.head(10), hide_index=True)
    st.markdown(Config.DFFINAL, unsafe_allow_html=True)

#Page 3 Dataviz
if page == pages[2] : 

  st.subheader("DataVizualization")
  titres_onglets = ['Univarié','Multivarié']
  onglet1, onglet2 = st.tabs(titres_onglets)
 
  #Analyse avec graphiques univariés
  with onglet1:
    
    st.subheader("Trafic cycliste quotidien à Paris entre 01/2024 et 01/2025")
    fig = graph.journalyCount(df_merged_cleaned_final)
    st.pyplot(fig)
    st.markdown(Config.DATAVIZ1, unsafe_allow_html=True)

    st.subheader("Comptage moyen par heure")
    fig = graph.averageCountByHour(df_merged_cleaned_final)
    st.pyplot(fig)
    st.markdown(Config.DATAVIZ2, unsafe_allow_html=True)

    st.subheader("Comptage moyen par heure weekend VS semaine")
    fig = graph.averageCountByWeek(df_merged_cleaned_final)
    st.pyplot(fig)
    st.markdown(Config.DATAVIZ3, unsafe_allow_html=True)

    st.subheader("Carte interactive des comptages horaires moyens par compteur")
    map_file = graph.generate_folium_map(df_merged_cleaned_final,"carte_bornes_velos.html")
    # Afficher la carte dans Streamlit
    with open(map_file, "r", encoding="utf-8") as f:
      html_code = f.read()
    components.html(html_code, height=600)
    st.markdown(Config.DATAVIZ4, unsafe_allow_html=True)

    st.subheader("Top10 et Flop10 des Bornes selon le passages horaires moyen")
    fig,fig1 = graph.top10Flop10(df_merged_cleaned_final)
    st.pyplot(fig)
    st.pyplot(fig1)
    st.markdown(Config.DATAVIZ5, unsafe_allow_html=True)

  #Analyse avec graphiques multivariés
  with onglet2:

    st.subheader("Impact des spécificités de la piste cyclable")
    period_selector1 = st.selectbox("Sélectionnez la période", options=['semaine', 'week-end'], index=0, key="period_selector1")
    fig = graph.filter_data_photo(period_selector1, df_merged_cleaned_final)
    st.pyplot(fig)
    st.markdown(Config.DATAVIZ6, unsafe_allow_html=True)

    st.subheader("Impact du jour ou de la nuit sur le trafic")
    fig = graph.dayNight(df_merged)
    st.plotly_chart(fig)
    st.markdown(Config.DATAVIZ7, unsafe_allow_html=True)

    st.subheader("Distribution du trafic vélo selon la température")
    fig = graph.boxplotTemperature(df_merged)
    st.pyplot(fig)
    st.markdown(Config.DATAVIZ8, unsafe_allow_html=True)

    st.subheader("Distribution du trafic vélo selon la vitesse du vent")
    fig = graph.boxplotVent(df_merged)
    st.pyplot(fig)
    st.markdown(Config.DATAVIZ9, unsafe_allow_html=True)

    st.subheader("Nombre de passages de vélos selon l'heure et la température")
    period_selector = st.selectbox("Sélectionnez la période", options=['semaine', 'week-end'], index=0, key="period_selector")
    fig = graph.filter_data_temp(period_selector, df_merged_cleaned_final)
    st.plotly_chart(fig)
    st.markdown(Config.DATAVIZ11, unsafe_allow_html=True)

    st.subheader("Moyenne des passages de vélos en Fonction des Conditions Météorologiques")
    fig = graph.go_bar_meteo(df_merged)
    st.plotly_chart(fig)
    st.markdown(Config.DATAVIZ10, unsafe_allow_html=True)

    st.subheader("Relation entre les vacances par zone et trafic cycliste")
    fig = graph.boxplot_vacances1(df_merged_cleaned)
    st.pyplot(fig)
    st.markdown(Config.DATAVIZ12, unsafe_allow_html=True)

    st.subheader("Impact des vacances sur le trafic cycliste")
    fig = graph.sns_scatter_vacances(df_merged_cleaned_final)
    st.pyplot(fig)
    st.markdown(Config.DATAVIZ13, unsafe_allow_html=True)

    st.subheader("Impact de la neutralisation des rues sur trafic cycliste")
    fig = graph.countByHourAndNeutralise(df_merged_cleaned)
    st.pyplot(fig)
    st.markdown(Config.DATAVIZ14, unsafe_allow_html=True)


#Page 4, les modélisation
if page == pages[3] : 

  st.subheader("Modélisation")
  titres_onglets = ["Choix du modèle","Modélisation Regressor", "Modélisation Temporelle"]
  onglet31, onglet32, onglet33  = st.tabs(titres_onglets)
 
  with onglet31:
    
    st.subheader("Diagramme de corrélation entre les variables")
    fig = graph.plot_heatmap(df_merged_cleaned_final)
    st.pyplot(fig)
    st.markdown(Config.MODELISATION_1, unsafe_allow_html=True)

    st.subheader("Analyse des corrélations linaires")
    st.image(Config.TABLEAU_CORR_LINE, width=1000) 
    st.markdown(Config.MODELISATION_2, unsafe_allow_html=True)

    st.subheader("Analyse de la saisonnalité des comptages de vélos")
    fig = graph.plot_moyenne_par_semaine(df_merged_cleaned_final)
    st.pyplot(fig)
    st.markdown(Config.MODELISATION_4, unsafe_allow_html=True)
    
    st.subheader("Importance des variables sur RandomForest")
    model3, X_train, feats = modelisation.modelisationRFBase(df_merged_cleaned_final)
    fig = graph.plot_feature_importances_RF(model3,X_train,feats)
    st.pyplot(fig)  
    st.markdown(Config.MODELISATION_3, unsafe_allow_html=True)

  with onglet32:
    #Cas des modèles regressor
    st.write("### Modèles Regressor")

    choix = ['XGBRegressor','StackingRegressor','Random Forest Regressor']
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)
    #préprocess et entrainement des modèles XGBRegressor et StackingRegressor
    if option in ['StackingRegressor','XGBRegressor']:
      clf, X_train, X_test, y_train, y_test  = modelisation.modelisation(df_merged_cleaned_final, option)
    #préprocess et entrainement spécique pour les modèles Random Forest Regressor'
    else:
      clf, X_train, X_test, y_train, y_test = modelisation.modelisationRFR(df_merged_cleaned_final)

    #possibilité d'afficher différents résultats 
    display = st.radio('Que souhaitez-vous montrer ?', ('metrique MAE','score (R²)', 'Nuage de point de prédiction'))
    if display == 'metrique MAE':
      trainMae,testMae = modelisation.scores(clf, display, X_train, X_test, y_train, y_test)
      st.write("Sur les données d'entrainement :",trainMae)
      st.write("Sur les données de test :",testMae)
    elif display == 'score (R²)':
      trainScore,testScore = modelisation.scores(clf, display,X_train, X_test, y_train, y_test)
      st.write("Sur les données d'entrainement :",trainScore)
      st.write("Sur les données de test :",testScore)
    elif display == 'Nuage de point de prédiction':
      fig = graph.pix_prediction(clf, X_test,y_test)
      st.pyplot(fig)

  with onglet33:
    #Cas du modèle reprophet 
    st.write("### Modèles temporelles")
    # l'entrainement des modèles se fait sur un nombres restreint de compteurs pour eviter la surcharge de la session
    listCompteur2 = ["10 avenue de la Grande Armée SE-NO","16 avenue de la Porte des Ternes E-O","18 quai de l'Hôtel de Ville NO-SE",
                  "147 avenue d'Italie S-N","27 boulevard Davout N-S"]
    
    #si le tableau contenant les modèles entrainés n'est déjà en session on le calcul et on le met en session
    if 'models' not in st.session_state:
      models = modelisation.modelisationProphet(df_merged_cleaned_final, listCompteur2)
      st.session_state.models = models
    else:
      models = st.session_state.models

    # proposition des compteurs 
    compteur = st.selectbox("Choisissez le nom du compteur", listCompteur2)
    st.write(f"Le compteur choisi est : {compteur}")

    # On récupère le model correspondant au compteurs choisi
    model = models[compteur]['model']
    test_data = models[compteur]['test_data']
    train_data = models[compteur]['train_data']

    #on reapplique la prédiction pour calculer la MAE et afficher le graphique
    test_data, test_predictions, mae = modelisation.predict_and_evaluate(model,test_data)
    
    # Afficher le MAE
    st.write(f"Mean Absolute Error (MAE) pour le compteur {compteur}: {mae}")

    # Affichage du graphque
    fig = graph.generate_graph_Tempo(test_data, test_predictions, compteur)
    st.pyplot(fig)


#Page 4 les prédictions futurs
if page == pages[4] : 

  titres_onglets4 = ['Prédiction VS réalité', 'Prédiction à 3 jours', 'Suivi du modèle']
  onglet10, onglet11, onglet12= st.tabs(titres_onglets4)

  with onglet10 :

    st.subheader("Analyse sur Février 2025")
    st.markdown(Config.PREDICTION3J_2, unsafe_allow_html=True)

    #Mise en forme et proposition de date pour le mois de février
    date_debut = datetime(2025, 2, 1,0,0)
    date_limite = datetime(2025, 2, 28, 0, 0).date()

    col1, col2 = st.columns(2)
    with col1:
      date_debut_choisie = st.date_input("Choisissez une date de début", min_value=date_debut, max_value=date_limite - timedelta(days=1),value=date_debut) 
      date_debut_choisie = datetime.combine(date_debut_choisie, datetime.min.time())
    with col2:
      date_fin_choisie = st.date_input("Choisissez une date de fin", min_value=date_debut_choisie + timedelta(days=1), max_value=date_limite, value=date_limite)
      date_fin_choisie = datetime.combine(date_fin_choisie, datetime.min.time())
  
    #Mise en forme et proposition de modèles
    modèles = ['XGBRegressor','StackingRegressor','Random Forest Regressor', 'Prophet']
    modelChoisi = st.selectbox('Choix du modèle', modèles)
    st.write('Le modèle choisi est :', modelChoisi)

    #Mise en forme et proposition de compteurs
    if modelChoisi != 'Prophet':
      listCompteur2 = ['All'] + utils.searchUnique(df_merged_cleaned_final, 'nom_compteur').tolist()
    else:
      listCompteur2 = ["10 avenue de la Grande Armée SE-NO","16 avenue de la Porte des Ternes E-O","18 quai de l'Hôtel de Ville NO-SE",
                    "147 avenue d'Italie S-N","27 boulevard Davout N-S"]
    compteur = st.selectbox("Choisissez le nom du compteur", listCompteur2)
    st.write(f"Le compteur choisi est : {compteur}")

    infoModelCompteur = {}
    #si le modèle est prophet on cherche si les modèles par compteurs sont déjà en session
    if modelChoisi == 'Prophet' :
      if 'models' not in st.session_state:
          models = modelisation.modelisationProphet(df_merged_cleaned_final, listCompteur2)
          st.session_state.models = models
      else:
          models = st.session_state.models
      infoModelCompteur = models[compteur]

    # Bouton de lancement de la prédiction
    if st.button("Lancer la prédiction"):
        # Appeler la méthode de prédiction selon le modèe et le compteurs-
        df_février = modelisation.predictionModel(modelChoisi, infoModelCompteur, compteur)
        # Afficher le graphe de la prédiction vs réalité sur les dates et le compteurs choisi
        st.subheader("Comparaison entre prédiction et réalité du comptage cycliste en Février 2025")
        fig = graph.courbePrediction(df_février, compteur, date_debut_choisie,date_fin_choisie)
        st.pyplot(fig)
        if modelChoisi == 'XGBRegressor' :
              st.markdown(Config.PREDICTION3J_3, unsafe_allow_html=True)

  with onglet11 :
    st.subheader("Prédiction à 3J")
    st.markdown(Config.PREDICTION3J, unsafe_allow_html=True)

    #Proposition du compteur
    listCompteur2 = ['All'] + utils.searchUnique(df_merged_cleaned_final, 'nom_compteur').tolist()
    compteur = st.selectbox("Choisissez le nom du compteur", listCompteur2, key='onglet11_1')
    st.write(f"Le compteur choisi est : {compteur}")

    # lancement de la prédiction dans le futur sur 3J
    df3J = modelisation.prediction3JModel(df_merged_cleaned_final,df_vjf_cleaned)
    st.markdown("Prédictions des 3 prochains jours sur l'ensemble des compteurs")

    #graphique de le prédiction et commmentaire
    fig = graph.courbePrediction3J(df3J, compteur)
    st.pyplot(fig)
    st.markdown(Config.PREDICTION3J_1, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col2:
          st.subheader("Filtres")
          #sliders pour choix de la date
          date_select = st.date_input('Choisissez la date',min_value=df3J['date_heure_comptage'].min().date(),max_value=df3J['date_heure_comptage'].max().date(),
                                      value=df3J['date_heure_comptage'].min().date())
          
          # Selon le jour on recherche la plage horaire dans le dataframe
          df_heureJourSelect= df3J[df3J['date_heure_comptage'].dt.date == date_select]
          # slider pour choisir la date
          heure_select = st.slider('Choisissez l\'heure',int(df_heureJourSelect['heure'].min()),int(df_heureJourSelect['heure'].max()),int(df_heureJourSelect['heure'].min()) if not df_heureJourSelect.empty else int(df_heureJourSelect['heure'].min()))

          #Proposition du compteur
          compteur_select = st.selectbox('Choisissez le compteur',['All'] + list(df3J['nom_compteur'].unique()))

          #Affichage d'un tableau de prédiction : température/vent/précipitation/compteur
          filtered_df, table_df = utils.completeDataframe(df3J,compteur_select, date_select, heure_select)
          st.dataframe(table_df)

    with col1:
          st.subheader("Carte des prévisions")
          map_file = graph.generate_folium_map_prediction(df3J,compteur_select,"carte_bornes_velos.html", date_select, heure_select)
          # Afficher la carte des prédictions
          with open(map_file, "r", encoding="utf-8") as f:
            html_code = f.read()
          components.html(html_code, height=600)
      

  with onglet12 :
    st.subheader("Suivi du modèle selon les prédictions effectuées")
    st.markdown(Config.SUIVI_1, unsafe_allow_html=True)

    #Proposition du choix du compteurs    
    compteur = st.selectbox("Choisissez le nom du compteur", listCompteur2, key='onglet12_1')
    st.write(f"Le compteur choisi est : {compteur}")

    #creation d'un dataframe contenant tous les fichiers présents dans le répertoire prédiction 
    df_list = utils.findListCSV()

    #creation du dataframe a partir du dernier csv tlépchargé sur le site
    df_realité = utils.load_data(Config.FILE_PATH +"lastDataComptage.csv", ",", 0)
    df_realité["Date et heure de comptage"] = pd.to_datetime(df_realité["Date et heure de comptage"])

    #proposition du fichier à vérifier specifiquement
    fichier = st.selectbox("Sélection fichier:", list(df_list.keys()))
    st.write(f"Le fichier choisi est : {fichier}")
    #affichage du graph pour ce fichier
    if st.button('Afficher le graphique'):
      fig = graph.plot_graph(fichier, compteur, df_realité,df_list)
      st.pyplot(fig)

    st.subheader("Suivi de la performance du modèle")
    st.markdown(Config.SUIVI_2, unsafe_allow_html=True)
    #Tableau de suivi des performances selon tous les fichiers de prediction sauvegardés
    df_resultats = modelisation.calculMetriquePrediction(df_realité)
    st.dataframe(df_resultats)
