class Config:
    FILE_PATH = "data/"
    FILE_PATH_PREDIC = "prediction/"
    FILE = "comptage-velo-donnees-compteurs.csv"
    FILE_METEO = "meteo-Paris-2024.csv"
    FILE_VAC = "vacances-scolaire.csv"
    FILE_FERIE = "jours_feries.csv"
    FILE_PHOTO = "detail_photo.csv"
    FILE_TRAVAUX = "detail_impact_rue.csv"
    FILE_FEVRIER = "fevrier.csv"
    IMAGE = "image_Top.jpg"  
    EXEMPLE = "exemplePrediction3J.png"
    AXE_AMELIORATION = "axes_d_amélioration.png"
    CONCLUSION = "conclusion.png"
    URL_METEO = "https://www.meteo-paris.com/meteo-8-jours/paris-75000"
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}
    PRESENTATION = '''<div style="text-align: justify;">Alors que le vélo a désormais surpassé la voiture dans la capitale et que la fréquentation des aménagements cyclables a bondi depuis la période Covid,
    la Ville de Paris poursuit ses aménagements avec la création de plus d’une vingtaine de nouvelles pistes en 2024.<br>
    Ces constats ont pu être effectués grâce notamment au déploiement par la Ville de Paris de près de soixante-dix capteurs de vélos permanents depuis 2020,   
    qui collectent en temps réel le nombre de cyclistes passant par chaque site de comptage, à chaque heure de la journée. Ces données précises permettent ainsi de visualiser 
    l’évolution de la pratique cycliste à Paris, et sont également analysées dans le but d’améliorer les connaissances sur l’usage des infrastructures de la capitale.<br>
    Notre projet a pour objectif d’effectuer une analyse des données récoltées par ces compteurs de vélos sur la période du 1er janvier 2024 au 29 janvier 2025, 
    afin de visualiser dans un premier temps les horaires et les zones d’affluence, mais également afin de déterminer quels paramètres peuvent influencer l’intensité du trafic cycliste.
    In fine, il s’agira de proposer un modèle de prédiction du trafic cycliste, en s’appuyant sur les variables les plus explicatives identifiées lors de notre analyse. <br>
    L’objectif est de pouvoir anticiper la fréquentation des pistes cyclables en fonction de différents facteurs (jour de la semaine, heure, météo, localisation, etc.),   
    afin d’apporter une aide à la décision pour l’aménagement urbain, la gestion des mobilités ou encore la planification d’événements dans la ville.</div>'''

    OBSERVATION_DF = '''
    Les données sont mises à jour quotidiennement et remontent sur 13 mois glissants. Nous avons récupéré toutes les données du 1er janvier 2024 au 29 janvier 2025.<br>
    Le jeu de données initial se compose de 16 colonnes et 913 738 lignes. Les types de données incluent des chaînes de caractères, des entiers et des nombres décimaux.<br>
    Les variables peuvent être regroupées par thématiques :<br>
    - le compteur et ses moyens d’identification<br>
    - les liens renvoyant vers la/les photo(s) associée(s) au compteur <br>
    - les données temporelles<br>'''
    OBSERVATION_DF1 = '''Le dataframe ne contient pas de doublons.<br>
    En revanche le dataframe contient près de 7% de valeurs manquantes au global. Toutefois, nous constatons que les variables « Nom du compteur » et « Comptage   horaire »   
    notamment,   ne   contiennent   aucune   valeur manquante.<br>
    Enfin, la variable cible est le “Comptage horaire”, c’est-à-dire le nombre de vélos/heure/site. C’est la seule variable numérique continue.'''
    OBSERVATION_METEO = '''
    Les données météorologiques (précipitations, température et vitesse du vent) feront l’objet d’une catégorisation pour une meilleure lecture, et la
    variable “time” sera convertie en heure française (car au format GMT dans le fichier source).<br>
    La variable relative à la vitesse du vent à 100m de hauteur sera supprimée, puisque sans impact sur le trafic cycliste.<br>
    La variable « is_day » indique s’il fait jour ou nuit (exemple : le soir à 19h).<br>'''
    OBSERVATION_VAC = '''
      Nous avons intégré les périodes de vacances scolaires en fonction des zones A, B ou C.'''
    OBSERVATION_JF = '''
      Au même titre que les vacances scolaires, il nous a semblé intéressant d’intégrer les jours fériés à notre étude.'''
    OBSERVATION_PHOTO = '''
      Nous avons exploité les informations contenues dans les photos présentes dans le dataset initial, et récupéré notamment des indications concernant la
    structure de la voie cyclable (si elle est dans un seul sens ou non, si elle est partagée avec la chaussée ou non)'''
    OBSERVATION_TRAVAUX = '''
      Cette dernière information est relative à la nature du blocage de la rue, le cas échéant (travaux, événement spécifique ponctuel).'''
    
    ABERRANTE = '''Le   boxplot   ci-dessus   nous   a   permis   de   mettre   en   évidence   une   valeur extrême voire aberrante avec un comptage de plus de 3000 vélos.<br>
    Ce qui nous incite à analyser ce compteur ce jour-là.'''
    ABERRANTE1 = '''Une analyse du compteur concerné (Quai d’Orsay O-E) révèle que la borne ne fonctionnait pas du 05/01/2024 à 01h au 06/01/2025 à 6h. La valeur de
    3070 est donc une valeur aberrante. Nous choisirons pour ce jour de prendre les mêmes valeurs que le dimanche précédent en respectant les heures : 
    période utilisée du 2024-12-29 01:00 au 2024-12-30 06:00.'''
    ABERRANTECORRECTION = ''''''
    
    DFFINAL= '''Après nettoyage et ajout de nouvelles variables, notre dataframe contient à présent 30 variables pour un total de 900 222 entrées.'''

    CONCLUSION_REPARTITION = '''Nous observons que notre jeu de données contient quelques (5) valeurs pour l’année 2022;<br>
    nous supprimons ces entrées afin de n’avoir que des données sur 2024/2025'''
    CONCLUSION_GA = '''Nous prenons par ailleurs la décision de lier les deux bornes nommées ‘7 avenue de la Grande Armée’ et ‘10 avenue de la Grande Armée [Bike IN]' et
    de renommer cette dernière, car elle semble prendre le relais de la borne précédemment identifiée au 7.<br>
    Nous supprimons le compteur ‘10 avenue de la Grande Armée [Bike OUT]’, qui n’a remonté qu’une seule valeur sur la périod.'''
    CONCLUSION_NBLIGNE = '''Si nous avions un jeu de données complet, nous devrions avoir environ 9500 entrées par compteur (soit 24 relevés quotidiens sur 13 mois), ce qui ne
    semble pas être le cas au vu du graphique ci-dessus.
    Pour 7 d’entre eux, nous constatons qu’ils ne sont pas identifiés de manière unique au niveau de la variable ‘Nom du compteur’.<br>
    Ainsi, ‘27 quai de la Tournelle NO-SE' et '27 quai de la Tournelle 27 quai de la Tournelle Vélos NO-SE' désignent le même compteur.<br>
    Nous décidons de les renommer afin d'harmoniser notre dataset sur ces 7 compteurs.<br>
    Le dataframe fera l’objet d’une complétion avec l’ajout des valeurs manquantes pour chaque compteur, basées sur la moyenne de ce compteur
    sur les mêmes jours/heures/mois.'''
    CONCLUSION_NAN = '''Ce graphe met en évidence la liaison entre certaines variables de notre dataframe : en l’occurrence, il apparait que ces variables (toutes relatives à
    l’identification des compteurs) présentent des NaNs pour les mêmes entrées.<br>
    Nous retrouvons les informations via les adresses des sites de comptage et remplaçons ainsi toutes les valeurs manquantes.'''
    
    EXPLICATIONJO = '''Nous avons choisi de mettre en évidence sur ce graphique l’impact des Jeux Olympiques  sur certaines bornes de comptage. 
    On voit ainsi très clairement que le nombre de vélos a fortement chuté, voire tombé à zéro sur la période.
    C’est notamment le cas pour les deux compteurs situés « Cours de la Reine », le Cours ayant été fermé à la circulation à partir du 26 avril 2024
    entre les ponts des Invalides et Alexandre-III, afin de permettre l'installation des infrastructures nécessaires aux Jeux.<br>
    Outre l’impact des Jeux Olympiques, nous avons constaté que certains compteurs présentaient des relevés à zéro sur une période plus ou moins
    longue. <br>
    Si un relevé à 0 n’est pas aberrant en soi (période nocturne, zones peu passagères), il est en revanche anormal de constater un relevé à zéro
    sur une période de plusieurs heures consécutives (nous avons considéré un seuil de 10 heures).<br>
    Ci-après les compteurs concernés :
    '''
    PREDICTION3J = '''Après avoir vérifié le bon fonctionnement de notre modèle sur les données de février 2025, nous pouvons désormais le mettre en application pour des prévisions futures. <br>  
    Pour simplifier notre démarche, nous nous sommes concentrés sur les trois prochains jours.
    Les données météorologiques sont collectées directement depuis le site : 
    <a href="https://www.meteo-paris.com/meteo-8-jours/paris-75000" target="_blank">https://www.meteo-paris.com/meteo-8-jours/paris-75000</a> '''

    PREDICTION3J_1 = '''Ci-dessous, un affichage interactif permettant de visualiser les prévisions des comptages cyclistes en temps réel.Cette fonctionnalité est une étape concrète dans la mise en application de notre modèle.<br>
    Cet écran pourrait être utilisée, dans un cadre réel, pour des applications visant à fournir aux cyclistes, urbanistes ou autres une prévision du trafic cycliste à court terme dans différentes zones de la ville.'''

    PREDICTION3J_2 = '''Suite aux tests de nos modèles avec notre jeu de données initial, nous avons souhaité vérifier la fiabilité de nos modèles 
    en les testant avec des données réelles sur une période non connue du modèle.'''

    PREDICTION3J_3 = '''Concernant le modèle XGBoost, nous obtenons les résultats suivants :<br>
        - MAE : 14,79<br>
        - MAPE: 19,2 %<br>
        - RMSE : 27,49<br>
    Les valeurs des métriques MAE et MAPE sont plutôt satisfaisantes et proches des résultats obtenus lors des tests sur le jeu de données initial. <br>
    En revanche, la RMSE est un peu plus élevée, ce qui indique des écarts plus importants à certains moments. <br>
    En analysant le graphique, on remarque que les erreurs surviennent principalement pendant les périodes de forte affluence.'''

    SUIVI_1 = '''Finalement, nous avons souhaité faire une phase de monitoring, c'est-à-dire vérifier la performance du modèle une fois en application.<br>
    Vous pouvez, afficher le graphique à courbes des comptages réels et prédits sur les anciennes prédictions.'''

    SUIVI_2 = '''Dans l'optique de mettre notre modèle en application, nous avons décidé d'automatiser le suivi de ses métriques. <br>
                Cela nous permet d'analyser ses performances en temps réel et, en cas de dégradation, de revoir le modèle.'''

    BOULEVARD_JOURDAN = '''Des travaux ont eu lieu sur le boulevard Jourdan à Paris entre le 3 octobre et le 18 décembre 2024. 
      Ces travaux comprenaient la réalisation d'un quai bus déporté, l'installation de séparateurs et de balises, ainsi que des opérations de marquage et de reprise de trottoir. '''
    GRANDE_ARMEE= '''Des restrictions de circulation ont été mises en place à Paris en préparation des festivités du Nouvel An, notamment le 31 décembre 2024. 
      Un arrêté municipal a interdit la circulation de tout véhicule à partir du 31 décembre 2024 à 16h00 jusqu'au 1er janvier 2025 à 04h00 dans les 8ᵉ, 16ᵉ et 17ᵉ arrondissements de Paris, 
      incluant des zones comme l'avenue de la Grande Armée. '''
    DAUMESNIL = '''Des travaux ont eu lieu sur l'avenue Daumesnil à Paris entre le 29 janvier et le 15 mars 2024, affectant la circulation des vélos. 
      Ces travaux comprenaient la remise en état des bandes stabilisées et la création d'accroches vélos sur les trottoirs, avec des impacts principalement sur le trottoir 
      et un cheminement piéton protégé tout au long du chantier.'''
    CHAMPS_ELYSEE = '''Nous n’avons pas identifié de travaux de voirie ou d’événement ponctuel ayant eu pour conséquence des restrictions de circulation ; <br>
    nous en déduisons que la borne était inopérante à ce moment-là.'''
    TURBIGO = '''Nous n’avons pas identifié de travaux de voirie ou d’événement ponctuel ayant eu pour conséquence des restrictions de circulation ; <br>
    nous en déduisons que la borne était inopérante à ce moment-là.'''
    ROCHEREAU = '''Le compteur semble avoir cessé de fonctionner courant janvier 2024, donc quasiment sur toute la période de notre étude ; 
    nous avons décidé  de l’exclure de nos données.'''
    TOURNELLE = '''La réouverture de la cathédrale Notre-Dame de Paris a eu lieu le 8 décembre 2024, avec des cérémonies officielles le 7 décembre. 
    Ces événements ont entraîné des restrictions de circulation dans un large périmètre autour de la cathédrale, notamment sur les quais hauts, incluant les pistes cyclables.'''          
    DEUXSENS = '''Compte tenu de la présence d’un compteur vélo dans chaque sens, et de l’absence d’incohérence de comptage sur l’autre borne, nous en déduisons que la borne était inopérante à ce moment-là.'''

    DATAVIZ1 = '''Sur cette courbe, on visualise une phase montante des comptages moyens entre janvier et juillet, puis une phase montante entre septembre et décembre. <br>
    On remarque également une baisse importante du trafic vélo sur les vacances d'été (surtout août) et les vacances de Noël.'''

    DATAVIZ2 = '''Sur les comptages moyens /heure, on visualise deux pics entre 7h et 9h le matin et 17h et 19h le soir, liés aux horaires d'entrée et sortie du travail.'''

    DATAVIZ3 = '''Pour rendre la tendance plus nette, nous avons décidé de discriminer les données en fonction du weekend et de la semaine. <br>
    Nous visualisons une tendance très différente entre les deux variables.'''

    DATAVIZ4 = '''Les comptages horaires moyens par site nous permettent de visualiser les trajets préférentiels : dans le centre, le long des quais de Seine et rive droite essentiellement. <br>
    Les contrastes sont importants avec les zones les moins fréquentés, principalement en périphérie et autour des quartiers résidentiels.'''

    DATAVIZ5 = '''Les plus gros compteurs sont dans le top 10 dans les deux sens : Sebastopol, Rivoli et Magenta.<br>
    De la même manière, on retrouve dans les flops certains compteurs dans les deux sens : quai d'Issy, porte de Charenton. Ils sont essentiellement excentrés.'''
    DATAVIZ6 = '''Les pistes séparées et à deux sens accueillent le plus de vélos. <br>
    Nous ne savons pas si c'est parce qu'elles ont été placées à des endroits déjà très fréquentés ou bien si les utilisateurs les préfèrent aux autre types de voies cyclables.'''
    DATAVIZ7 = '''Les comptages vélo sont moins importants aux mêmes horaires de nuit que de jour. Il est difficile de savoir si c'est réellement l'effet de la lumière ou si c'est l'effet de la saison. <br>
    Nous savons que les comptages sont plus faibles en hiver, quand les journées sont plus courtes.'''
    DATAVIZ8 = '''Les comptages vélos augmentent en fonction de la température jusqu'à 30°C, puis ils ont tendance à diminuer.'''
    DATAVIZ9 = '''Les comptages vélos sont stables jusqu'à une vitesse du vent de 40 kms/h puis diminuent fortement.'''
    DATAVIZ10 = '''Pour visualiser plus clairement l'impact de la météo sur les comptages, nous avons catégorisé les variables.<br>
    On retrouve les mêmes tendances que plus haut :<br>
    - précipitations : effet direct sur le comptage entre pluie vs pas de pluie<br>
    - vent :  effet direct sur le comptage entre grand vent vs. les autres types de vent<br>
    - température : une tendance ascendante sur les comptages en même temps que la hausse de température jusqu'à "chaud", puis une baisse ensuite'''
    DATAVIZ11 = '''Le trafic moyen augmente en même temps que la température. <br>
    On retrouve les basses températures (plus claires) en bas du graphe et les hautes températures (plus foncées) en haut. <br>
    On retrouve cette tendance le weekend et la semaine.'''
    DATAVIZ12 = '''En ajoutant la variable vacances par zone et on ne remarque pas de réel impact sur le comptage. '''
    DATAVIZ13 = '''En regroupant toutes les zones pour la variable vacances, en attribuant 1 (vacances dans au moins une zone) et 0 (pas de vacances dans aucune zone). <br>
    Cette catégorisation nous permet de voir qu'il semble y avoir une corrélation entre les vacances et la variable cible.'''
    DATAVIZ14 = '''Nous avons ajouté la variable "neutralise" qui nous permet d'expliquer certains comptages très bas. <br>
    Les comptages sont influencés par les travaux, les événements (comme par exmple les JO) qui ont lieu sur la zone du compteur.'''

    MODELISATION_1='''Nous observons une corrélation moyenne entre la variable cible et les données telles que l’ heure, fait_jour et latitude. <br>
    Il existe également une anticorrélation avec les variables num_jour_semaine et weekend.'''
    COMPARAISON_MODELES= "comparaison_modèles_Regressor.png"
    MODELISATION_2='''Le XGBRegressor et le StackingRegressor montrent les meilleures performances en termes de score R² et de MAE, avec des scores très proches.<br> 
    En termes de stabilité, les trois modèles présentent des écarts-types faibles, mais le RandomForestRegressor, bien qu'un peu plus stable, a des erreurs plus élevées. <br> 
    Globalement, le XGBRegressor semble offrir la meilleure combinaison de précision et de stabilité, avec des résultats cohérents à travers les itérations de validation croisée.
    '''
    MODELISATION_3='''Pour le choix des variables nous étions partis sur la totalité des données qui, pour nous, pouvaient avoir un impact sur la prédiction, 
    en nous inspirant du diagramme de corrélation.<br> 
    Puis nous avons remarqué que le choix des variables était dépendant du modèle lui-même et donc devait être ajusté à partir des résultats de prédictions.'''
    
    MODELISATION_4='''En observant la courbe des moyennes de comptages par jour dans la partie exploration de données où ici sur la moyenne hebdomadaire, on a identifié une certaine saisonnalité<br>
    La répartition des comptages est très dépendante des vacances.'''