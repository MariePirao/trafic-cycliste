from datetime import datetime, time #as dt_time  # Garde time tel quel
from datetime import timedelta # Importer la classe time sous un alias
import streamlit as st
import utilsPython as utils 
import utilsPreprocess as preproc 
import utilsGraph as graph 
import modelisation as modelisation 
from config import Config
import streamlit.components.v1 as components
from streamlit_folium import st_folium
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

  st.markdown('<p style="margin-bottom: 0px;font-size:12px; font-style:italic;">Bootcamp Analystics Enginner JAN2025', unsafe_allow_html=True)
  st.markdown('<p style="font-size:12px; font-style:italic;">Aurélie Guilhem - Ingrid Plessis - Nicolas Couvez - Marie Pirao', unsafe_allow_html=True)

  st.markdown(f'<p style="margin-bottom: 0px;">Source à exploiter : {Config.FILE}</p>', unsafe_allow_html=True)
  st.markdown('<p style="font-size:12px; font-style:italic;">source : <a href="https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs" target="_blank">https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs</a></p>', unsafe_allow_html=True)

  if st.checkbox("Afficher les données externes", key='page1') : # sinon les chekbox de toute l'appli sont liées
    st.markdown(f'<p style="margin-left: 50px;margin-bottom: 0px;">🌤️ Données météorologiques :  {Config.FILE_METEO}</p>', unsafe_allow_html=True)
    st.markdown('<p style="margin-left: 50px;font-size:12px; font-style:italic;">source : <a href="https://www.data.gouv.fr/fr/organizations/meteo-france/" target="_blank">https://www.data.gouv.fr/fr/organizations/meteo-france/</a></p>', unsafe_allow_html=True)

    st.markdown(f'<p style="margin-left: 50px;margin-bottom: 0px;">🏖️ Données vacances scolaires :  {Config.FILE_VAC}</p>', unsafe_allow_html=True)
    st.markdown('<p style="margin-left: 50px;font-size:12px; font-style:italic;">source : <a href="https://www.data.gouv.fr/fr/datasets/calendrier-scolaire/" target="_blank">https://www.data.gouv.fr/fr/datasets/calendrier-scolaire/</a></p>', unsafe_allow_html=True)
    
    st.markdown(f'<p style="margin-left: 50px;margin-bottom: 0px;">🎌 Données jours férié :  {Config.FILE_FERIE}</p>', unsafe_allow_html=True)
    st.markdown('<p style="margin-left: 50px;font-size:12px; font-style:italic;">source : <a href="https://www.data.gouv.fr/fr/datasets/jours-feries-en-france/" target="_blank">https://www.data.gouv.fr/fr/datasets/jours-feries-en-france/</a></p>', unsafe_allow_html=True)
    
    st.markdown(f'<p style="margin-left: 50px;margin-bottom: 0px;">📸 Données detail photo :  {Config.FILE_PHOTO}</p>', unsafe_allow_html=True)
    st.markdown('<p style="margin-left: 50px;font-size:12px; font-style:italic;">source : effectué manuellement', unsafe_allow_html=True)
        
    st.markdown(f'<p style="margin-left: 50px;margin-bottom: 0px;">:🚧 Données detail travaux ou bloquage des JO :  {Config.FILE_TRAVAUX}</p>', unsafe_allow_html=True)
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

    if st.checkbox("Afficher les données externes", key='ongletA') :
      st.subheader("Aperçu du jeu de données externes")
      
      st.write("Données météorologiques")
      st.dataframe(df_m.head())
      st.markdown(Config.OBSERVATION_METEO, unsafe_allow_html=True)

      st.write("Données vacances scolaire")
      st.dataframe(df_v.head())
      st.markdown(Config.OBSERVATION_VAC, unsafe_allow_html=True)

      st.write("Données jours férié")
      st.dataframe(df_f.head())
      st.markdown(Config.OBSERVATION_JF, unsafe_allow_html=True)

      st.write("Données detail photo")
      st.dataframe(df_p_cleaned.head())
      st.markdown(Config.OBSERVATION_PHOTO, unsafe_allow_html=True)

      st.write("Données blocage rue")
      st.dataframe(df_ir.head())
      st.markdown(Config.OBSERVATION_TRAVAUX, unsafe_allow_html=True)
      
  with ongletB: 
    #analyse de la dsitribution des compteurs selon les mois/année et conclusion
    st.subheader("Analyse de la répartition des compteurs")
    fig = graph.plot_avg_mensuel(df,"all")
    st.plotly_chart(fig)
    st.markdown(Config.CONCLUSION_REPARTITION, unsafe_allow_html=True)

    #analyse du compteur de la Grande Armée pour montrer qu'il faudra corriger certains compteurs
    st.subheader("Analyse du compteur : Grande Armée")
    fig = graph.plot_avg_mensuel(df,"GrandeArmee")
    st.plotly_chart(fig)
    st.markdown(Config.CONCLUSION_GA, unsafe_allow_html=True)

    #Analyse des lignes manaquantes pour un dataframe complet sur la durée
    st.subheader("Compteurs ayant des lignes manquantes sur 2024/2025")
    fig = graph.nbLigne_compteur(df)
    st.plotly_chart(fig)
    st.markdown(Config.CONCLUSION_NBLIGNE, unsafe_allow_html=True)

    #Analyse des valeurs manquantes
    if st.checkbox("Afficher les NA", key='ongletB') :
      st.subheader("Répartition des données manquantes sur les compteurs")
      fig = graph.heatmap_isna(df)
      st.pyplot(fig)
      st.markdown(Config.CONCLUSION_NAN, unsafe_allow_html=True)


  with ongletC: 
    #Analyse spécifique de la période des JO et impact sur les compteurs
    st.subheader("Impact des JO sur le trafic cycliste")
    fig = graph.px_compteurs_mensuel_JO(df_merged)
    st.plotly_chart(fig)
    st.markdown(Config.EXPLICATIONJO, unsafe_allow_html=True)

    #Analyse spécifique de certains compteurs présentant de grandes interation de compteurs à 0
    st.subheader("Itération de 10 comptage à 0 sur certains compteurs")
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
    elif (selected_compteur == 'boulevard Richard Lenoir'):
      explication0 = Config.LENOIR 
    elif (selected_compteur == 'Pont des Invalides'):
      explication0 = Config.INVALIDES 
    elif (selected_compteur == '27 quai de la Tournelle'):
      explication0 = Config.TOURNELLE        
    elif (selected_compteur == '"7 avenue de la Grande Armée NO-SE'):
      explication0 = Config.GRANDE_ARMEE7               
    elif (selected_compteur == '"Porte des Ternes'):
      explication0 = Config.TERNES 
    elif (selected_compteur == 'Face au 48 quai de la marne'):
      explication0 = Config.MARNE
    elif (selected_compteur == '"Totem 73 boulevard de Sébastopol'):
      explication0 = Config.SEBASTOPOL
    st.markdown(explication0, unsafe_allow_html=True)

    #Analkyse ciblé sur valeur abhérrante
    st.subheader("Distribution de la variable comptage_horaire des vélos")
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
    #fig = graph.boxplot(df_merged_cleaned['comptage_horaire'])
    #st.pyplot(fig)

  with ongletD: #2seconde
    #Présentation du dataframe final après correction
    st.subheader(" Dataframe final")
    st.dataframe(df_merged_cleaned_final.head(5))


#Page 3 Dataviz
if page == pages[2] : 

  st.write("### DataVizualization")
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

    st.subheader("Carte des Bornes de Comptage Vélo")
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

    #Analyse impact spécificités de la piste cyclable
    st.subheader("Impact des spécificités de la piste cyclable")
    period_selector1 = st.selectbox("Sélectionnez la période", options=['semaine', 'week-end'], index=0, key="period_selector1")
    fig = graph.filter_data_photo(period_selector1, df_merged_cleaned_final)
    st.pyplot(fig)
    st.markdown(Config.DATAVIZ6, unsafe_allow_html=True)

    #Analyse détaillé de l'impact des heures où il faut jour ou nuit
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

    #Analyse détaillé de l'impact des 3 conditions météorologiques
    st.subheader("Moyenne des passages de vélos en Fonction des Conditions Météorologiques")
    fig = graph.go_bar_meteo(df_merged)
    st.plotly_chart(fig)
    st.markdown(Config.DATAVIZ10, unsafe_allow_html=True)
    #fig = graph.sns_scatter_meteo(df_merged)
    #st.pyplot(fig)

    #Analyse détaillé de l'impact de la téempérature
    st.subheader("Nombre de passages de vélos selon l'heure et la température")
    period_selector = st.selectbox("Sélectionnez la période", options=['semaine', 'week-end'], index=0, key="period_selector")
    fig = graph.filter_data_temp(period_selector, df_merged_cleaned_final)
    st.plotly_chart(fig)
    st.markdown(Config.DATAVIZ11, unsafe_allow_html=True)

    #Analyse impact spécificités des vacances
    st.subheader("Relation entre les vacances par zone et trafic cycliste")
    fig = graph.boxplot_vacances1(df_merged_cleaned)
    st.pyplot(fig)
    st.markdown(Config.DATAVIZ12, unsafe_allow_html=True)

    st.subheader("Impact des vacances sur trafic cycliste")
    fig = graph.sns_scatter_vacances(df_merged_cleaned_final)
    st.pyplot(fig)
    st.markdown(Config.DATAVIZ13, unsafe_allow_html=True)

    st.subheader("Impact de la neutralisation des rues sur trafic cycliste")
    fig = graph.countByHourAndNeutralise(df_merged_cleaned)
    st.pyplot(fig)
    st.markdown(Config.DATAVIZ14, unsafe_allow_html=True)



#Page 4, les modélisation
if page == pages[3] : 

  st.write("### Modélisation")
  titres_onglets = ["Choix du modèle","Modélisation Regressor", "Modélisation Temporelle"]
  onglet31, onglet32, onglet33  = st.tabs(titres_onglets)
 
  with onglet31:
    #Analyse des corrélation
    st.subheader("Diagramme de correlation entre les variables")
    fig = graph.plot_heatmap(df_merged_cleaned_final)
    st.pyplot(fig)

    #Cas particulier des features importances
    st.subheader("Graphique de l'importance des variables en nous basant sur RandomForest")
    model3, X_train, feats = modelisation.modelisationRFBase(df_merged_cleaned_final)
    fig = graph.plot_feature_importances_RF(model3,X_train,feats)
    st.pyplot(fig)  

  with onglet32:
    #Cas des modèles regressor
    st.write("### Modèles Regressor")

    choix = ['StackingRegressor','XGBRegressor','Random Forest Regressor']
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

  titres_onglets4 = ['Prédiction VS réalité', 'Prédiction à 3 jours']
  onglet10, onglet11= st.tabs(titres_onglets4)

  with onglet10 :

    date_debut = datetime(2025, 2, 1,0,0)
    date_limite = datetime(2025, 2, 28, 0, 0).date()

    # Créer deux colonnes pour les boutons
    col1, col2 = st.columns(2)
    with col1:
      date_debut_choisie = st.date_input("Choisissez une date de début", min_value=date_debut, max_value=date_limite - timedelta(days=1),value=date_debut) 
      date_debut_choisie = datetime.combine(date_debut_choisie, datetime.min.time())
    with col2:
      date_fin_choisie = st.date_input("Choisissez une date de fin", min_value=date_debut_choisie + timedelta(days=1), max_value=date_limite, value=date_limite)
      date_fin_choisie = datetime.combine(date_fin_choisie, datetime.min.time())
  
    modèles = ['XGBRegressor','StackingRegressor','Random Forest Regressor', 'Prophet']
    modelChoisi = st.selectbox('Choix du modèle', modèles)
    st.write('Le modèle choisi est :', modelChoisi)
    if modelChoisi != 'Prophet':
      listCompteur2 = ['All'] + utils.searchUnique(df_merged_cleaned_final, 'nom_compteur').tolist()
    else:
      listCompteur2 = ["10 avenue de la Grande Armée SE-NO","16 avenue de la Porte des Ternes E-O","18 quai de l'Hôtel de Ville NO-SE",
                    "147 avenue d'Italie S-N","27 boulevard Davout N-S"]

    compteur = st.selectbox("Choisissez le nom du compteur", listCompteur2)
    st.write(f"Le compteur choisi est : {compteur}")
    infoModelCompteur = {}
    if modelChoisi == 'Prophet' :
      if compteur != 'All':
        if 'models' not in st.session_state:
          models = modelisation.modelisationProphet(df_merged_cleaned_final, listCompteur2)
          st.session_state.models = models
        else:
          models = st.session_state.models
        infoModelCompteur = models[compteur]
      else:
        st.error("Veuillez choisir un compteur spécifique.")

    # Bouton de lancement de la prédiction
    if st.button("Lancer la prédiction"):
        # Appeler la méthode de prédiction 
        df_février = modelisation.predictionModel(modelChoisi, infoModelCompteur, compteur)
        # Afficher le graphe de la prédiction vs réalité sur les date et le compteurs choisis
        st.subheader("Comparaison entre prédiction et réalité du comptage cycliste en Février 2025")
        fig = graph.courbePrediction(df_février, compteur, date_debut_choisie,date_fin_choisie)
        st.pyplot(fig)

  with onglet11 :
    st.subheader("Prédiction à 3J")
    st.markdown(Config.PREDICTION3J, unsafe_allow_html=True)
    st.image(Config.EXEMPLE, width=1000)

    st.subheader("Lancez la prédiction ")
    modèles = ['XGBRegressor','Prophet']
    modelChoisi = st.selectbox('Choix du modèle', modèles, key='onglet11')
    if modelChoisi == 'XGBRegressor':
      listCompteur2 = ['All'] + utils.searchUnique(df_merged_cleaned_final, 'nom_compteur').tolist()
    else:
      listCompteur2 = ["10 avenue de la Grande Armée SE-NO","16 avenue de la Porte des Ternes E-O","18 quai de l'Hôtel de Ville NO-SE",
                    "147 avenue d'Italie S-N","27 boulevard Davout N-S"]
    compteur = st.selectbox("Choisissez le nom du compteur", listCompteur2, key='onglet11_1')
    st.write(f"Le compteur choisi est : {compteur}")

    # Bouton de lancement de la prédiction
    if st.button("Lancer la prédiction", key='onglet11_2'):
        # Appeler la méthode de prédiction en passant la date et l'heure choisie
        df3J = modelisation.prediction3JModel(modelChoisi, df_merged_cleaned_final,df_vjf_cleaned, infoModelCompteur, compteur)
        # Afficher les résultats sous le premier bouton
        st.markdown("Prédictions des 3 prochains jours sur l'ensemble des compteurs")
        fig = graph.courbePrediction3J(df3J, compteur)
        st.pyplot(fig)
