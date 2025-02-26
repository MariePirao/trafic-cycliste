from sklearn.metrics import mean_absolute_error
import streamlit as st
import utilsPython as utils # type: ignore
import utilsPreprocess as preproc # type: ignore
import utilsGraph as graph # type: ignore
import modelisation as modelisation # type: ignore
import streamlit.components.v1 as components
#import folium
from streamlit_folium import st_folium
import pandas as pd
# Pour éviter d'avoir les messages warning
import warnings
warnings.filterwarnings('ignore')

# Personnalisationde la largeur de l'affichage
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
    </style>
    """, unsafe_allow_html=True
)

file_path = "data/"
uploaded_file = "comptage-velo-donnees-compteurs.csv"
uploaded_file_meteo = "meteo-Paris-2024.csv"
uploaded_file_vac = "vacances-scolaire.csv"
uploaded_file_ferie = "jours_feries.csv"
uploaded_file_photo = "detail_photo.csv"
uploaded_file_travaux = "detail_impact_rue.csv"


#on evite de refaire les diffférents dataframe s'ils sont déjà en session
if 'df_merged_cleaned_final' not in st.session_state: # si on a déjà caculer ce df c'est que tout a déjà été calculé et mis en session
    # Si pas encore dans session_state, faire les calculs
    # permet de charger le df avec le fichier compteurs
    df = utils.load_data(file_path + uploaded_file, ";",0) 
    df_m = utils.load_data(file_path + uploaded_file_meteo, ',', 2)	
    df_v = utils.load_data(file_path + uploaded_file_vac, ',', 0)
    df_f = utils.load_data(file_path + uploaded_file_ferie, ',', 0)
    df_p = utils.load_data(file_path + uploaded_file_photo, ',', 0)
    df_ir = utils.load_data(file_path + uploaded_file_travaux, ';', 0)
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
else:
    # Sinon, récupérer les données depuis session_state
    df =st.session_state.df
    df_m = st.session_state.df_m
    df_v = st.session_state.df_v
    df_f = st.session_state.df_f
    df_p_cleaned = st.session_state.df_p_cleaned
    df_ir= st.session_state.df_ir
    df_merged = st.session_state.df_merged
    df_merged_cleaned = st.session_state.df_merged_cleaned
    df_merged_cleaned_final = st.session_state.df_merged_cleaned_final


#utils.create_data(df_merged_cleaned_final, file_path)

image_path = "image_Top.jpg"  
st.image(image_path, width=400) 

st.title("Trafic cycliste à PARIS")
st.sidebar.title("Sommaire")
pages=["Présentation du sujet","Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages, key='menu_principal')

#ce qui s'affiche si l 'option 1 de pages est sélectionné
if page == pages[0] : 
  st.write("### Présentation")

  multi = '''La Ville de Paris déploie depuis plusieurs années des compteurs à vélo permanents pour évaluer le développement de la pratique cycliste.
  Ce projet a pour objectif d’effectuer une analyse des données récoltées par ces compteurs vélo afin de visualiser les horaires et les zones d’affluences.
  Ceci aura pour but de fournir des outils à la mairie de Paris afin qu’elle juge les améliorations à apporter sur les différents endroits cyclables de la ville. 
  '''
  st.markdown(multi, unsafe_allow_html=True)

  st.markdown('<p style="margin-bottom: 0px;font-size:12px; font-style:italic;">Bootcamp Analystics Enginner JAN2025', unsafe_allow_html=True)
  st.markdown('<p style="font-size:12px; font-style:italic;">Aurélie Guilhem - Ingrid Plessis - Nicolas Couvez - Marie Pirao', unsafe_allow_html=True)

  st.markdown(f'<p style="margin-bottom: 0px;">Source à exploiter : {uploaded_file}</p>', unsafe_allow_html=True)
  st.markdown('<p style="font-size:12px; font-style:italic;">source : <a href="https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs" target="_blank">https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs</a></p>', unsafe_allow_html=True)

  if st.checkbox("Afficher les données externes", key='page1') : # sinon les chekbox de toute l'appli sont liées
    st.markdown(f'<p style="margin-left: 50px;margin-bottom: 0px;">🌤️ Données météorologiques :  {uploaded_file_meteo}</p>', unsafe_allow_html=True)
    st.markdown('<p style="margin-left: 50px;font-size:12px; font-style:italic;">source : <a href="https://www.data.gouv.fr/fr/organizations/meteo-france/" target="_blank">https://www.data.gouv.fr/fr/organizations/meteo-france/</a></p>', unsafe_allow_html=True)

    st.markdown(f'<p style="margin-left: 50px;margin-bottom: 0px;">🏖️ Données vacances scolaires :  {uploaded_file_vac}</p>', unsafe_allow_html=True)
    st.markdown('<p style="margin-left: 50px;font-size:12px; font-style:italic;">source : <a href="https://www.data.gouv.fr/fr/datasets/calendrier-scolaire/" target="_blank">https://www.data.gouv.fr/fr/datasets/calendrier-scolaire/</a></p>', unsafe_allow_html=True)
    
    st.markdown(f'<p style="margin-left: 50px;margin-bottom: 0px;">🎌 Données jours férié :  {uploaded_file_ferie}</p>', unsafe_allow_html=True)
    st.markdown('<p style="margin-left: 50px;font-size:12px; font-style:italic;">source : <a href="https://www.data.gouv.fr/fr/datasets/jours-feries-en-france/" target="_blank">https://www.data.gouv.fr/fr/datasets/jours-feries-en-france/</a></p>', unsafe_allow_html=True)
    
    st.markdown(f'<p style="margin-left: 50px;margin-bottom: 0px;">📸 Données detail photo :  {uploaded_file_photo}</p>', unsafe_allow_html=True)
    st.markdown('<p style="margin-left: 50px;font-size:12px; font-style:italic;">source : effectué manuellement', unsafe_allow_html=True)
        
    st.markdown(f'<p style="margin-left: 50px;margin-bottom: 0px;">:🚧 Données detail travaux ou bloquage des JO :  {uploaded_file_travaux}</p>', unsafe_allow_html=True)
    st.markdown('<p style="margin-left: 50px;font-size:12px; font-style:italic;">source : effectué manuellement', unsafe_allow_html=True)
    
#ce qui s'affiche si l 'option 1 de pages est sélectionné
if page == pages[1] : 
  st.write("### Exploration des données")
  titres_onglets2 = ['Visualisation des données', 'Travail de nettoyage', 'Analyse spécifique de certains compteurs', 'DataframeFinal']
  ongletA, ongletB,ongletC, ongletD = st.tabs(titres_onglets2)

# Ajouter du contenu à chaque onglet
  with ongletA:
    st.subheader("Aperçu du jeu de donnée")
    st.dataframe(df.head(5))

    st.subheader("Informations sur le jeu de donnée")
    # Extraire les informations sur chaque colonne (describe() que pour colonne int)
    int_columns = df.select_dtypes(include=['int64']).columns
    info_dict = {
      'Valeurs non-null': df.notnull().sum(),
      'Dtype': df.dtypes,
      'Valeur unique': [df[col].nunique(dropna=False) for col in df.columns],
      'Min': [df[col].min() if col in int_columns else '' for col in df.columns],
      'Max': [df[col].max() if col in int_columns else '' for col in df.columns],
      'Médiane': [df[col].median() if col in int_columns else '' for col in df.columns],
      'Moyenne': [df[col].mean() if col in int_columns else '' for col in df.columns]
    }
 
    info_df = pd.DataFrame(info_dict)
    st.dataframe(info_df)
    observation = '''
    La variable 'Nom du compteur' est toujours indiquée car le nombre de valeurs non nulles correspond à la taille du df : 913738<br>
    La variable 'Identifiant du site de comptage' est parfois nulle mais est dans un mauvais type. Nous verrons si cette donnée est interessante pour la suite de l'analyse<br>
    La variable 'Comptage horaire' est toujorus renseignée. Mais nous devrons étudier la valeur max car elle être très éloignée de la médiane.<br>
    La variable 'Date et heure comptage' est toujours indiquée mais devrait être dans un type Datetime(FR).<br>
    Les variables id_photo_1 et type_dimage n'ont chcune qu'une valeur et nan. Nous les supprimerons du df pour notre analyse.<br> '''
    st.markdown(observation, unsafe_allow_html=True)
    if st.checkbox("Afficher les données externes", key='ongletA') :
      st.subheader("Aperçu du jeu de donnée externes")
      st.write("Données météorologiques")
      st.dataframe(df_m.head())
      observation = '''
      Tout d'abord on peut s'apercevoir que la date est (comme l'indique le fichier source, en format GMT. Nous devrons la convertir en heure francaise pour rapprocher ces enregitrements de nos données de base<br>
      Pour une meilleur lecture nous pourrons categoriser les données precipitation, temperature_2m et wind_speed_10m.<br>
      Nous supprimerons la donnée concernant le vent a 100m de hauteur, qui ne peut pas avoir d'impact sur le trafic des vélo<br>
      is_Day() indique s'il fait jour ou nuit (exemple à 19H du soir)<br>'''
      st.markdown(observation, unsafe_allow_html=True)
      st.write("Données vacances scolaire")
      st.dataframe(df_v.head())
      observation = '''
      ????'''
      st.markdown(observation, unsafe_allow_html=True)
      st.write("Données jours férié")
      st.dataframe(df_f.head())
      observation = '''
      ????'''
      st.markdown(observation, unsafe_allow_html=True)
      st.write("Données detail photo")
      st.dataframe(df_p_cleaned.head())
      observation = '''
      ????'''
      st.markdown(observation, unsafe_allow_html=True)
      st.write("Données blocage rue")
      st.dataframe(df_ir.head())
      observation = '''
      ????'''
      st.markdown(observation, unsafe_allow_html=True)

  # Ajouter du contenu à chaque onglet
  with ongletB:
    #analyse de la dsitribution des compteurs selon les mois année
    st.subheader("Analyse de la répartition des compteurs")
    fig = graph.plot_avg_mensuel(df,"all")
    st.plotly_chart(fig)
    conclusion = '''Nous observons que la totalité des relevé de compteurs concerne l'année 2024/2025 avec une exception pour un relevé sur 2022.<br>
    Nous décidons d'exclure l'énnée 2022'''
    st.markdown(conclusion, unsafe_allow_html=True)
    #analyse du compteur de la Grande Armée
    st.subheader("Analyse du compteur : Grande Armée")
    fig = graph.plot_avg_mensuel(df,"GrandeArmee")
    st.plotly_chart(fig)
    conclusion = '''Nous pouvons déduire que la borne "7 avenue de la Grande Armée NO-SE" est correcte. <br>Par contre la borne 10 avenue de la Grande Armée [Bike IN]
    à l'air d'avoir remplacé la borne 10 avenue de la Grande Armée. Nous prenons la décision de liées les deux bornes. <br>Pour la borne 10 avenue de la Grande Armée [Bike OUT], 
    n'ayant remonté qu'un valeur, nous decidons de supprimer ce compteur'''
    st.markdown(conclusion, unsafe_allow_html=True)
    st.subheader("Compteurs ayant des lignes manquantes sur 2024/2025")
    fig = graph.nbLigne_compteur(df)
    st.plotly_chart(fig)
    conclusion = '''Si nous avions un jeu de donées complet, nous aurions 9475 lignes par compteurs. Le graphique ci-dessus nous montre que c'est a peu de ligne près le cas, 
    mais pas du tout vrai pour 7 compteurs. Mais si nous regardons les noms d'un peu plus près nous pouvons voir que ce sont tous des compteurs dont le nom est très proche d'un autre compteur
    à la même adresse.  Nous décidons de renommer :<br>
    Face au 48 quai de la marne NE-SO/SO-NE' en 'Face au 48 quai de la marne Face au 48 quai de la marne Vélos NE-SO/SO-NE<br>
    Pont des Invalides N-S en Pont des Invalides (couloir bus) N-S<br>
    27 quai de la Tournelle NO-SE/SE-NO en 27 quai de la Tournelle 27 quai de la Tournelle Vélos NO-SE/SE-NO<br>
    Quai des Tuileries NO-SE/SE-NO en Quai des Tuileries Quai des Tuileries Vélos NO-SE/SE-NO<br>
    Pour les autres compteurs nous allons proposer des alogorithmes pour ajouter les données.
      '''
    st.markdown(conclusion, unsafe_allow_html=True)
    if st.checkbox("Afficher les NA", key='ongletB') :
      st.subheader("Répartition des données manquantes sur les compteurs")
      fig = graph.heatmap_isna(df)
      st.pyplot(fig)
      conclusion = '''Nous observons que lorsqu'il manque un donnée sur une ligne du dataframe, génréralement il manque les 12 colonnes qui concerne le compteur.<br>
      En analysant les compteurs concernés, nous pouvons compléter les données manquantes à l'aide d'une ligne du compteur sur laquelle les données concernées sont indiquées.'''
      st.markdown(conclusion, unsafe_allow_html=True)


  with ongletC:
    st.markdown("### Impact des JO sur le trafic cycliste")
    fig = graph.px_compteurs_mensuel_JO(df_merged)
    st.plotly_chart(fig)
    explicationJO = '''Si nous faisons un focus sur les compteurs (Cours la Reine, quai des tuileries et pont de la concorde). Nous pouvons clairement voir l'impact des Jeux olympique sur les bornes.
    En effet, le Cours de la Reine a été fermé à la circulation à partir du 26 avril 2024, entre les ponts des Invalides et Alexandre-III, pour permettre l'installation des infrastructures nécessaires aux Jeux. 
    Le pont Alexandre III à été totalement fermé a partir du 17 mai.
    À partir du 27 juin 2024, le Parc Rives de Seine, du pont Louis-Philippe au tunnel des Tuileries (quai bas – rive droite), a été fermé au grand public. 
    Cette fermeture a été en vigueur jusqu'au 1ᵉʳ août 2024, avec des réouvertures partielles en soirée et les week-ends. 
    Bien que le Pont de la Concorde soit resté accessible depuis le Cours la Reine dans le sens nord-sud, et pour la desserte locale dans le sens sud-nord, 
    les fermetures et restrictions de circulation aux abords immédiats ont pu entraîner une diminution significative du trafic sur le pont. 
    '''
    st.markdown(explicationJO, unsafe_allow_html=True)

    st.markdown("### Moyenne journalière à 0 sur le trafic cycliste")
    Compteur = ['10 avenue de la Grande Armée SE-NO','106 avenue Denfert Rochereau NE-SO','135 avenue Daumesnil SE-NO','24 boulevard Jourdan E-O',
                '33 avenue des Champs Elysées NO-SE','38 rue Turbigo','boulevard Richard Lenoir','Pont des Invalides',
                "27 quai de la Tournelle","7 avenue de la Grande Armée NO-SE",
                "Porte des Ternes", "Face au 48 quai de la marne",
                #"Quai d'Orsay",
                "Totem 73 boulevard de Sébastopol"]
                #"Totem 85 quai d'Austerlitz" ]
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
      explication0 = '''?????'''
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
    st.markdown("### Distribution de la variable comptage_horaire des vélos")
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
  

  with ongletD:
    st.markdown("### Dataframe final")
    st.dataframe(df_merged_cleaned_final.head(5))

#ce qui s'affiche si l 'option 2 de pages est sélectionné
if page == pages[2] : 
  st.write("### DataVizualization")
  titres_onglets = ['Univarié','Multivarié']
  onglet1, onglet2 = st.tabs(titres_onglets)
 
  with onglet1:
    st.markdown("### Carte des Bornes de Comptage Vélo")
    map_file = graph.generate_folium_map(df_merged,"carte_bornes_velos.html")
    # Afficher la carte dans Streamlit
    with open(map_file, "r", encoding="utf-8") as f:
      html_code = f.read()
    components.html(html_code, height=600)

    st.markdown("### Trafic cycliste quotidien à Paris entre 01/2024 et 01/2025")
    fig = graph.journalyCount(df_merged)
    st.pyplot(fig)
    st.markdown("### Top10 et Flop10 des Bornes selon le passages horaires moyen")
    fig,fig1 = graph.top10Flop10(df_merged)
    st.pyplot(fig)
    st.pyplot(fig1)

  with onglet2:
    #st.header('Multivarié')
    st.write('Visualisation multivarié des données')
    st.write("Analyse de l'impact de la météo sur le nombre de passages")
    fig = graph.go_bar_meteo(df_merged)
    st.plotly_chart(fig)
    fig = graph.sns_scatter_meteo(df_merged)
    st.pyplot(fig)
    st.write("Analyse de l'impact du jour ou de la nuit sur le nombre de passages")
    fig = graph.dayNight(df_merged)
    st.plotly_chart(fig)
    st.write('Diagramme de correlation entre les variables')
    fig = graph.plot_heatmap(df_merged_cleaned_final)
    st.pyplot(fig)
    st.markdown("### Distribution du trafic vélo selon la température")
    fig = graph.boxplotTemperature(df_merged)
    st.pyplot(fig)
    st.markdown("### Distribution du trafic vélo selon la vitesse du vent")
    fig = graph.boxplotVent(df_merged)
    st.pyplot(fig)
 
    st.markdown("###Diagramme de correlation entre les vacances et le nombre de passages")
    fig = graph.boxplot_vacances1(df_merged_cleaned)
    st.pyplot(fig)
    conclusionVacances = '''Vu le résultat de ce graph,nous pouvons clairement affirmer que la zone n'a pas d'influence.
    nous allons donc merger les 3 colonnes pour la suite de l'analyse '''
    st.markdown(conclusionVacances, unsafe_allow_html=True)


  

#ce qui s'affiche si l 'option 2 de pages est sélectionné
if page == pages[3] : 

  sous_menus = ["Modélisation Regressor", "Modélisation Temporelle"]
  sous_menu = st.sidebar.radio("Choisissez un type de modélisation", sous_menus, key='sous_menu')

  if sous_menu == sous_menus[0]:

    st.write("### Modèles Regressor")

    listCompteur = ['All'] + utils.searchUnique(df_merged_cleaned_final, 'nom_compteur').tolist()
    nom_compteur_selectionne = st.selectbox('Sélectionnez un nom de compteur', options=listCompteur)
    st.write('Le compteur choisi est :', nom_compteur_selectionne )

    choix = ['XGBRegressor', 'DecisionTreeRegressor','Random Forest Regressor','GradientBoostingRegressor','BaggingRegressor','StackingRegressor', 'AdaBoostRegressor']
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)
    if option in ['XGBRegressor', 'StackingRegressor']:
      y_pred, y_test = utils.modelisation(df_merged_cleaned_final, option)
      st.write("Sur les données de test :",mean_absolute_error(y_test, y_test))
    else:
      X_train, X_test, y_train, y_test = modelisation.modelisation(df_merged_cleaned_final,nom_compteur_selectionne)
      clf = modelisation.prediction(option,X_train, y_train)
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
    
  if sous_menu == sous_menus[1]:

    st.write("### Modèles temporelles")

    models = utils.modelisationT(df_merged_cleaned_final)

    # Sélection du compteur par l'utilisateur
    listCompteur2 = utils.searchUnique(df_merged_cleaned_final, 'nom_compteur').tolist()
    compteur_a_predire = st.selectbox('Sélectionnez un nom de compteur', options=listCompteur2)
    st.write('Le compteur choisi est :', compteur_a_predire )

    # Extraire les données, former et évaluer le modèle
    model = models[compteur_a_predire]['model']
    test_data = models[compteur_a_predire]['test_data']
    train_data = models[compteur_a_predire]['train_data']
  
    test_data, test_predictions, mae = utils.predict_and_evaluate(model, train_data,test_data)
    
    # Afficher le MAE
    st.write(f"Mean Absolute Error (MAE) pour le compteur {compteur_a_predire}: {mae}")

    # Générer et afficher le graphique
    fig = graph.generate_graph_Tempo(test_data, test_predictions, compteur_a_predire)
    st.pyplot(fig)