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

#on evite de refaire les diffférents dataframe s'ils sont déjà en session
if 'df_merged_cleaned_final' not in st.session_state: # 60 secondes
    # Si pas encore dans session_state, faire les calculs
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
    # Sinon, récupérer les données depuis session_state
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

#ce qui s'affiche si l 'option 1 de pages est sélectionné
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

    
#ce qui s'affiche si l 'option 1 de pages est sélectionné
if page == pages[1] : #4 seconde

  st.write("### Exploration des données")
  titres_onglets2 = ['Visualisation des données', 'Travail de nettoyage', 'Analyse spécifique de certains compteurs', 'DataframeFinal']
  ongletA, ongletB,ongletC, ongletD = st.tabs(titres_onglets2)

# ONglet de présentation des différents dataframes créés 
  with ongletA:  
    st.subheader("Aperçu du jeu de données")
    st.dataframe(df.head(5))  # Affichage de l'apercu du dataframe initial

    st.subheader("Informations sur le jeu de donnée")
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
      
  # Ajouter du contenu à chaque onglet
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
    st.subheader("Impact des JO sur le trafic cycliste")
    fig = graph.px_compteurs_mensuel_JO(df_merged)
    st.plotly_chart(fig)
    st.markdown(Config.EXPLICATIONJO, unsafe_allow_html=True)

    st.subheader("Moyenne journalière à 0 sur le trafic cycliste")
    Compteur = ['10 avenue de la Grande Armée SE-NO','106 avenue Denfert Rochereau NE-SO','135 avenue Daumesnil SE-NO','24 boulevard Jourdan E-O',
                '33 avenue des Champs Elysées NO-SE','38 rue Turbigo','boulevard Richard Lenoir','Pont des Invalides',
                "27 quai de la Tournelle","7 avenue de la Grande Armée NO-SE",
                "Porte des Ternes", "Face au 48 quai de la marne",
                "Totem 73 boulevard de Sébastopol"]
    # Liste déroulante pour choisir un compteur
    selected_compteur = st.selectbox("Sélectionnez un compteur",options=Compteur,index=0)  # Par défaut, sélectionner le premier compteur
    fig = graph.px_compteurs_quotidien_0(df_merged, selected_compteur)
    st.plotly_chart(fig, use_container_width=True)
    explication0 = ''''''
    if (selected_compteur == '24 boulevard Jourdan E-O'):
      explication0 = '''Des travaux ont eu lieu sur le boulevard Jourdan à Paris entre le 3 octobre et le 18 décembre 2024. 
      Ces travaux comprenaient la réalisation d'un quai bus déporté, l'installation de séparateurs et de balises, ainsi que des opérations de marquage et de reprise de trottoir. '''
    elif (selected_compteur == '10 avenue de la Grande Armée SE-NO'):
      explication0 = '''Des restrictions de circulation ont été mises en place à Paris en préparation des festivités du Nouvel An, notamment le 31 décembre 2024. 
      Un arrêté municipal a interdit la circulation de tout véhicule à partir du 31 décembre 2024 à 16h00 jusqu'au 1er janvier 2025 à 04h00 dans les 8ᵉ, 16ᵉ et 17ᵉ arrondissements de Paris, 
      incluant des zones comme l'avenue de la Grande Armée. '''
    elif (selected_compteur == '135 avenue Daumesnil SE-NO'):
      explication0 = '''Des travaux ont eu lieu sur l'avenue Daumesnil à Paris entre le 29 janvier et le 15 mars 2024, affectant la circulation des vélos. 
      Ces travaux comprenaient la remise en état des bandes stabilisées et la création d'accroches vélos sur les trottoirs, avec des impacts principalement sur le trottoir 
      et un cheminement piéton protégé tout au long du chantier.'''
    elif (selected_compteur == '33 avenue des Champs Elysées NO-SE'):
      explication0 = '''?????'''
    elif (selected_compteur == '38 rue Turbigo'):
      explication0 = '''?????'''
    elif (selected_compteur == '106 avenue Denfert Rochereau NE-SO'):
      explication0 = '''La borne semble hors service, nous décidons de la retirer de notre analyse'''
    elif (selected_compteur == 'boulevard Richard Lenoir'):
      explication0 = '''Il y a un compteur dans chaque sens. Puisque la borne a compté correctement dans un sens. Nous sommes en droit de penser que la borne à était inopérante à ce moment là'''
    elif (selected_compteur == 'Pont des Invalides'):
      explication0 = '''Il y a un compteur dans chaque sens. Puisque la borne a compté correctement dans un sens. Nous sommes en droit de penser que la borne à était inopérante à ce moment là'''
    elif (selected_compteur == '27 quai de la Tournelle'):
      explication0 = '''La réouverture de la cathédrale Notre-Dame de Paris a eu lieu le 8 décembre 2024, avec des cérémonies officielles le 7 décembre. 
      Ces événements ont entraîné des restrictions de circulation dans un large périmètre autour de la cathédrale, notamment sur les quais hauts, incluant les pistes cyclables'''          
    elif (selected_compteur == '"7 avenue de la Grande Armée NO-SE'):
      explication0 = '''?????'''                
    elif (selected_compteur == '"Porte des Ternes'):
      explication0 = '''Il y a un compteur dans chaque sens. Puisque la borne a compté correctement dans un sens. Nous sommes en droit de penser que la borne à était inopérante à ce moment là'''
    elif (selected_compteur == 'Face au 48 quai de la marne'):
      explication0 = '''?????'''
    elif (selected_compteur == '"Totem 73 boulevard de Sébastopol'):
      explication0 = '''Il y a un compteur dans chaque sens. Puisque la borne a compté correctement dans un sens. Nous sommes en droit de penser que la borne à était inopérante à ce moment là'''
    st.markdown(explication0, unsafe_allow_html=True)

    #Analkyse ciblé sur valeur abhérrante
    st.subheader("Distribution de la variable comptage_horaire des vélos")
    fig = graph.boxplot(df_merged['comptage_horaire'])
    st.pyplot(fig)
    abherrante0 = '''On voit très nettement une valeur extrême ou même abherrante. 
    Ce qui nous incite a analyser ce compteur ce jour-là'''
    st.markdown(abherrante0, unsafe_allow_html=True)

    fig1 = graph.plot_abherrante(df_merged)
    st.plotly_chart(fig1, key="graph_abherrante_1")
    abherrante1 = '''Nous pouvons déduire que la borne ne fonctionnait pas du 05/01/2024 à 01H au 06/05/2025 à 6H.
    La valeur de 3070 est donc une valeur aberrante. Nous choisirons pour ce jour de prendre les mêmes valeurs que le dimanche précédent.'''
    st.markdown(abherrante1, unsafe_allow_html=True)

    correct = '''Proposition de correction : Nous allons reprendre les données d'un autre jour sur le meme compteur en respectant le jour de la smeaine et les heures : 
    periode utilisée du 2024-12-29 01:00 au 2024-12-30 06:00
    periode à corriger du 2025-01-05 01:00 au 2025-01-06 06:00'''

    st.markdown(correct, unsafe_allow_html=True)
    fig2 = graph.plot_abherrante(df_merged_cleaned)
    st.plotly_chart(fig2, key="graph_abherrante_2")
    fig = graph.boxplot(df_merged_cleaned['comptage_horaire'])
    st.pyplot(fig)

  with ongletD: #2seconde
    st.subheader(" Dataframe final")
    st.dataframe(df_merged_cleaned_final.head(5))


#ce qui s'affiche si l 'option 2 de pages est sélectionné
if page == pages[2] : 

  st.write("### DataVizualization")
  titres_onglets = ['Univarié','Multivarié']
  onglet1, onglet2 = st.tabs(titres_onglets)
 
  with onglet1:
    st.subheader("Carte des Bornes de Comptage Vélo")
    map_file = graph.generate_folium_map(df_merged,"carte_bornes_velos.html")
    # Afficher la carte dans Streamlit
    with open(map_file, "r", encoding="utf-8") as f:
      html_code = f.read()
    components.html(html_code, height=600)

    st.subheader("Trafic cycliste quotidien à Paris entre 01/2024 et 01/2025")
    fig = graph.journalyCount(df_merged)
    st.pyplot(fig)
    st.subheader("Top10 et Flop10 des Bornes selon le passages horaires moyen")
    fig,fig1 = graph.top10Flop10(df_merged)
    st.pyplot(fig)
    st.pyplot(fig1)

  with onglet2:
    st.write("Nombre de passages de vélos selon l'heure et la température")
    period_selector = st.selectbox("Sélectionnez la période", options=['semaine', 'week-end'], index=0, key="period_selector")

    # Générer et afficher le graphique en fonction de la sélection
    fig = graph.filter_data_temp(period_selector, df_merged_cleaned_final)
    st.plotly_chart(fig)

    st.subheader("Moyenne des passages de vélos en Fonction des Conditions Météorologiques")
    fig = graph.go_bar_meteo(df_merged)
    st.plotly_chart(fig)
    fig = graph.sns_scatter_meteo(df_merged)
    st.pyplot(fig)
    st.subheader("Analyse de l'impact du jour ou de la nuit sur le nombre de passages")
    fig = graph.dayNight(df_merged)
    st.plotly_chart(fig)
    st.subheader("Distribution du trafic vélo selon la température")
    fig = graph.boxplotTemperature(df_merged)
    st.pyplot(fig)
    st.subheader("Distribution du trafic vélo selon la vitesse du vent")
    fig = graph.boxplotVent(df_merged)
    st.pyplot(fig)

    st.subheader("Comparaison des comptages horaires sur la période choisie")
    period_selector1 = st.selectbox("Sélectionnez la période", options=['semaine', 'week-end'], index=0, key="period_selector1")
    fig = graph.filter_data_photo(period_selector1, df_merged_cleaned_final)
    st.pyplot(fig)
 
    st.subheader("Relation entre les vacances et le nombre de passages")
    fig = graph.boxplot_vacances1(df_merged_cleaned)
    st.pyplot(fig)
    conclusionVacances = '''Vu le résultat de ce graph,nous pouvons clairement affirmer que la zone n'a pas d'influence.
    nous allons donc merger les 3 colonnes pour la suite de l'analyse '''
    st.markdown(conclusionVacances, unsafe_allow_html=True)


#ce qui s'affiche si l 'option 2 de pages est sélectionné
if page == pages[3] : 

  st.write("### Modélisation")
  titres_onglets = ["Choix du modèle","Modélisation Regressor", "Modélisation Temporelle"]
  onglet31, onglet32, onglet33  = st.tabs(titres_onglets)
 
  with onglet31:
    st.subheader("Diagramme de correlation entre les variables")
    fig = graph.plot_heatmap(df_merged_cleaned_final)
    st.pyplot(fig)

    st.subheader("Graphique de l'importance des variables en nous basant sur RandomForest")
    model3, X_train, feats = modelisation.modelisationRFBase(df_merged_cleaned_final)
    fig = graph.plot_feature_importances_RF(model3,X_train,feats)
    st.pyplot(fig)  

  with onglet32:
    #utilisation de modèles sur dataframe classique
    st.write("### Modèles Regressor")

    #listCompteur = ['All'] + utils.searchUnique(df_merged_cleaned_final, 'nom_compteur').tolist()
    #nom_compteur_selectionne = st.selectbox('Sélectionnez un nom de compteur', options=listCompteur)
    #st.write('Le compteur choisi est :', nom_compteur_selectionne )
    nom_compteur_selectionne = 'All'

    choix = ['StackingRegressor','XGBRegressor','Random Forest Regressor']
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)
    #préprocess spécique pour les modèles XGBRegressor et StackingRegressor
    if option in ['StackingRegressor','XGBRegressor']:
      clf, X_train, X_test, y_train, y_test  = modelisation.modelisation(df_merged_cleaned_final, option)
    #préprocess spécique pour les modèles Random Forest Regressor','BaggingRegressor', 'DecisionTreeRegressor
    else:
      clf, X_train, X_test, y_train, y_test = modelisation.modelisationRFR(df_merged_cleaned_final,nom_compteur_selectionne, option)

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
    #prise en compte des specificités des données temporelle
    st.write("### Modèles temporelles")

    listCompteur2 = ["10 avenue de la Grande Armée SE-NO","16 avenue de la Porte des Ternes E-O","18 quai de l'Hôtel de Ville NO-SE",
                  "147 avenue d'Italie S-N","27 boulevard Davout N-S"]
    models = modelisation.modelisationT(df_merged_cleaned_final, listCompteur2)
    compteur = st.selectbox("Choisissez le nom du compteur", listCompteur2)
    st.write(f"Le compteur choisi est : {compteur}")

    # Extraire les données, former et évaluer le modèle
    model = models[compteur]['model']
    test_data = models[compteur]['test_data']
    train_data = models[compteur]['train_data']
  
    test_data, test_predictions, mae = modelisation.predict_and_evaluate(model, train_data,test_data)
    
    # Afficher le MAE
    st.write(f"Mean Absolute Error (MAE) pour le compteur {compteur}: {mae}")

    # Générer et afficher le graphique
    fig = graph.generate_graph_Tempo(test_data, test_predictions, compteur)
    st.pyplot(fig)


  #ce qui s'affiche si l 'option 2 de pages est sélectionné
if page == pages[4] : 

  titres_onglets4 = ['Prédiction VS réalité', 'Prédiction à 3 jours']
  onglet10, onglet11= st.tabs(titres_onglets4)

  with onglet10 :

    date_debut = datetime(2025, 2, 1,0,0)
    date_limite = datetime(2025, 2, 28, 0, 0).date()

    # Créer deux colonnes pour les boutons
    col1, col2 = st.columns(2)
    with col1:
      date_debut_choisie = st.date_input("Choisissez une date de début", min_value=date_debut, max_value=date_limite - timedelta(days=1)) 
      date_debut_choisie = datetime.combine(date_debut_choisie, datetime.min.time())
    with col2:
      date_fin_choisie = st.date_input("Choisissez une date de fin", min_value=date_debut_choisie + timedelta(days=1), max_value=date_limite)
      date_fin_choisie = datetime.combine(date_fin_choisie, datetime.min.time())
  
    modèles = ['XGBRegressor','StackingRegressor','Random Forest Regressor', 'Prophet']
    modelChoisi = st.selectbox('Choix du modèle', modèles)
    st.write('Le modèle choisi est :', modelChoisi)

    listCompteur2 = ['All',"10 avenue de la Grande Armée SE-NO","16 avenue de la Porte des Ternes E-O","18 quai de l'Hôtel de Ville NO-SE",
                    "147 avenue d'Italie S-N","27 boulevard Davout N-S"]
    #models = modelisation.modelisationT(df_merged_cleaned_final, listCompteur2)
    compteur = st.selectbox("Choisissez le nom du compteur", listCompteur2)
    st.write(f"Le compteur choisi est : {compteur}")
    #if modelChoisi == 'Prophet':
    #  infoModelCompteur = models[compteur]
    #else:
    #  infoModelCompteur = {}

    # Créer un bouton "Lancer la prédiction" dans la première colonne
    if st.button("Lancer la prédiction"):
        # Appeler la méthode de prédiction en passant la date et l'heure choisie
        df_février = modelisation.predictionModel(modelChoisi)
        # Afficher les résultats sous le premier bouton
        st.subheader("Comparaison entre prédiction et réalité du comptage cycliste en Février 2025")
        fig = graph.courbePrediction(df_février, compteur, date_debut_choisie,date_fin_choisie)
        st.pyplot(fig)

  with onglet11 :
    st.subheader("Prédiction à 3J")
    st.markdown(Config.PREDICTION3J, unsafe_allow_html=True)
    st.image(Config.EXEMPLE, width=800)

    st.subheader("Lancez une prédiction !")
    #modèles = ['XGBRegressor','StackingRegressor','Random Forest Regressor', 'Prophet']
    #modelChoisi = st.selectbox('Choix du modèle', modèles, key='onglet11')
    listCompteur2 = ['All'] + utils.searchUnique(df_merged_cleaned_final, 'nom_compteur').tolist()
    compteur = st.selectbox("Choisissez le nom du compteur", listCompteur2, key='onglet11_1')
    st.write(f"Le compteur choisi est : {compteur}")

    # Créer un bouton "Lancer la prédiction" dans la première colonne
    if st.button("Lancer la prédiction", key='onglet11_2'):
        # Appeler la méthode de prédiction en passant la date et l'heure choisie
        df3J = modelisation.prediction3JModel(modelChoisi, df_merged_cleaned_final,df_vjf_cleaned)
        # Afficher les résultats sous le premier bouton
        st.markdown("Prédictions des 3 prochains jours sur l'ensemble des compteurs")
        fig = graph.courbePrediction3J(df3J, compteur)
        st.pyplot(fig)
