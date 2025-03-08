class Config:
    FILE_PATH = "data/"
    FILE = "comptage-velo-donnees-compteurs.csv"
    FILE_METEO = "meteo-Paris-2024.csv"
    FILE_VAC = "vacances-scolaire.csv"
    FILE_FERIE = "jours_feries.csv"
    FILE_PHOTO = "detail_photo.csv"
    FILE_TRAVAUX = "detail_impact_rue.csv"
    FILE_FEVRIER = "fevrier.csv"
    IMAGE = "image_Top.jpg"  
    EXEMPLE = "exemplePrediction3J.png"
    URL_METEO = "https://www.meteo-paris.com/meteo-8-jours/paris-75000"
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}
    PRESENTATION = '''La Ville de Paris déploie depuis plusieurs années des compteurs à vélo permanents pour évaluer le développement de la pratique cycliste.
    Ce projet a pour objectif d’effectuer une analyse des données récoltées par ces compteurs vélo afin de visualiser les horaires et les zones d’affluences.
    Ceci aura pour but de fournir des outils à la mairie de Paris afin qu’elle juge les améliorations à apporter sur les différents endroits cyclables de la ville. '''

    OBSERVATION_DF = '''
    La variable 'Nom du compteur' est toujours indiquée car le nombre de valeurs non nulles correspond à la taille du df : 913738<br>
    La variable 'Identifiant du site de comptage' est parfois nulle mais est dans un mauvais type. Nous verrons si cette donnée est interessante pour la suite de l'analyse<br>
    La variable 'Comptage horaire' est toujorus renseignée. Mais nous devrons étudier la valeur max car elle être très éloignée de la médiane.<br>
    La variable 'Date et heure comptage' est toujours indiquée mais devrait être dans un type Datetime(FR).<br>
    Les variables id_photo_1 et type_dimage n'ont chcune qu'une valeur et nan. Nous les supprimerons du df pour notre analyse.<br> '''
    OBSERVATION_METEO = '''
    Tout d'abord on peut s'apercevoir que la date est (comme l'indique le fichier source, en format GMT. Nous devrons la convertir en heure francaise pour rapprocher ces enregitrements de nos données de base<br>
    Pour une meilleur lecture nous pourrons categoriser les données precipitation, temperature_2m et wind_speed_10m.<br>
    Nous supprimerons la donnée concernant le vent a 100m de hauteur, qui ne peut pas avoir d'impact sur le trafic des vélo<br>
    is_Day() indique s'il fait jour ou nuit (exemple à 19H du soir)<br>'''
    OBSERVATION_VAC = '''
      ????'''
    OBSERVATION_JF = '''
      ????'''
    OBSERVATION_PHOTO = '''
      ????'''
    OBSERVATION_TRAVAUX = '''
      ????'''
    
    ABERRANTE = '''On voit très nettement une valeur extrême ou même abherrante. 
    Ce qui nous incite a analyser ce compteur ce jour-là'''
    ABERRANTE1 = '''Nous pouvons déduire que la borne ne fonctionnait pas du 05/01/2024 à 01H au 06/05/2025 à 6H.
    La valeur de 3070 est donc une valeur aberrante. Nous choisirons pour ce jour de prendre les mêmes valeurs que le dimanche précédent.'''
    ABERRANTECORRECTION = '''Proposition de correction : Nous allons reprendre les données d'un autre jour sur le meme compteur en respectant le jour de la smeaine et les heures : 
    periode utilisée du 2024-12-29 01:00 au 2024-12-30 06:00
    periode à corriger du 2025-01-05 01:00 au 2025-01-06 06:00'''
    
    CONCLUSION_REPARTITION = '''Nous observons que la totalité des relevé de compteurs concerne l'année 2024/2025 avec une exception pour un relevé sur 2022.<br>
    Nous décidons d'exclure l'énnée 2022'''
    CONCLUSION_GA = '''Nous pouvons déduire que la borne "7 avenue de la Grande Armée NO-SE" est correcte. <br>Par contre la borne 10 avenue de la Grande Armée [Bike IN]
    à l'air d'avoir remplacé la borne 10 avenue de la Grande Armée. Nous prenons la décision de liées les deux bornes. <br>Pour la borne 10 avenue de la Grande Armée [Bike OUT], 
    n'ayant remonté qu'un valeur, nous decidons de supprimer ce compteur'''
    CONCLUSION_NBLIGNE = '''Si nous avions un jeu de donées complet, nous aurions 9475 lignes par compteurs. Le graphique ci-dessus nous montre que c'est a peu de ligne près le cas, 
    mais pas du tout vrai pour 7 compteurs. Mais si nous regardons les noms d'un peu plus près nous pouvons voir que ce sont tous des compteurs dont le nom est très proche d'un autre compteur
    à la même adresse.  Nous décidons de renommer :<br>
    Face au 48 quai de la marne NE-SO/SO-NE' en 'Face au 48 quai de la marne Face au 48 quai de la marne Vélos NE-SO/SO-NE<br>
    Pont des Invalides N-S en Pont des Invalides (couloir bus) N-S<br>
    27 quai de la Tournelle NO-SE/SE-NO en 27 quai de la Tournelle 27 quai de la Tournelle Vélos NO-SE/SE-NO<br>
    Quai des Tuileries NO-SE/SE-NO en Quai des Tuileries Quai des Tuileries Vélos NO-SE/SE-NO<br>
    Pour les autres compteurs nous allons proposer des alogorithmes pour ajouter les lignes maquantes.'''
    CONCLUSION_NAN = '''Nous observons que lorsqu'il manque un donnée sur une ligne du dataframe, génréralement il manque les 12 colonnes qui concerne le compteur.<br>
      En analysant les compteurs concernés, nous pouvons compléter les données manquantes à l'aide d'une ligne du compteur sur laquelle les données concernées sont indiquées.'''
    
    EXPLICATIONJO = '''Si nous faisons un focus sur les compteurs (Cours la Reine, quai des tuileries et pont de la concorde). Nous pouvons clairement voir l'impact des Jeux olympique sur les bornes.
    En effet, le Cours de la Reine a été fermé à la circulation à partir du 26 avril 2024, entre les ponts des Invalides et Alexandre-III, pour permettre l'installation des infrastructures nécessaires aux Jeux. 
    Le pont Alexandre III à été totalement fermé a partir du 17 mai.
    À partir du 27 juin 2024, le Parc Rives de Seine, du pont Louis-Philippe au tunnel des Tuileries (quai bas – rive droite), a été fermé au grand public. 
    Cette fermeture a été en vigueur jusqu'au 1ᵉʳ août 2024, avec des réouvertures partielles en soirée et les week-ends. 
    Bien que le Pont de la Concorde soit resté accessible depuis le Cours la Reine dans le sens nord-sud, et pour la desserte locale dans le sens sud-nord, 
    les fermetures et restrictions de circulation aux abords immédiats ont pu entraîner une diminution significative du trafic sur le pont. 
    '''
    PREDICTION3J = '''Texte de Nicolas'''

    BOULEVARD_JOURDAN = '''Des travaux ont eu lieu sur le boulevard Jourdan à Paris entre le 3 octobre et le 18 décembre 2024. 
      Ces travaux comprenaient la réalisation d'un quai bus déporté, l'installation de séparateurs et de balises, ainsi que des opérations de marquage et de reprise de trottoir. '''
    GRANDE_ARMEE= '''Des restrictions de circulation ont été mises en place à Paris en préparation des festivités du Nouvel An, notamment le 31 décembre 2024. 
      Un arrêté municipal a interdit la circulation de tout véhicule à partir du 31 décembre 2024 à 16h00 jusqu'au 1er janvier 2025 à 04h00 dans les 8ᵉ, 16ᵉ et 17ᵉ arrondissements de Paris, 
      incluant des zones comme l'avenue de la Grande Armée. '''
    DAUMESNIL = '''Des travaux ont eu lieu sur l'avenue Daumesnil à Paris entre le 29 janvier et le 15 mars 2024, affectant la circulation des vélos. 
      Ces travaux comprenaient la remise en état des bandes stabilisées et la création d'accroches vélos sur les trottoirs, avec des impacts principalement sur le trottoir 
      et un cheminement piéton protégé tout au long du chantier.'''
    CHAMPS_ELYSEE = '''?????'''
    TURBIGO = '''?????'''
    ROCHEREAU = '''La borne semble hors service, nous décidons de la retirer de notre analyse'''
    LENOIR = '''Il y a un compteur dans chaque sens. Puisque la borne a compté correctement dans un sens. Nous sommes en droit de penser que la borne à était inopérante à ce moment là'''
    INVALIDES = '''Il y a un compteur dans chaque sens. Puisque la borne a compté correctement dans un sens. Nous sommes en droit de penser que la borne à était inopérante à ce moment là'''
    TOURNELLE = '''La réouverture de la cathédrale Notre-Dame de Paris a eu lieu le 8 décembre 2024, avec des cérémonies officielles le 7 décembre. 
      Ces événements ont entraîné des restrictions de circulation dans un large périmètre autour de la cathédrale, notamment sur les quais hauts, incluant les pistes cyclables'''          
    GRANDE_ARMEE7 = '''?????'''                
    TERNES = '''Il y a un compteur dans chaque sens. Puisque la borne a compté correctement dans un sens. Nous sommes en droit de penser que la borne à était inopérante à ce moment là'''
    MARNE = '''?????'''
    SEBASTOPOL = '''Il y a un compteur dans chaque sens. Puisque la borne a compté correctement dans un sens. Nous sommes en droit de penser que la borne à était inopérante à ce moment là'''



    DATAVIZ1 = '''Dataviz 1'''

    DATAVIZ2 = '''Dataviz 2'''

    DATAVIZ3 = '''Dataviz 3'''

    DATAVIZ4 = '''Dataviz 4'''

    DATAVIZ5 = '''Dataviz 5'''
    DATAVIZ6 = '''Dataviz 6'''
    DATAVIZ7 = '''Dataviz 7'''
    DATAVIZ8 = '''Dataviz 8'''
    DATAVIZ9 = '''Dataviz 9'''
    DATAVIZ10 = '''Dataviz 10'''
    DATAVIZ11 = '''Dataviz 11'''
    DATAVIZ12 = '''Dataviz 12 Vu le résultat de ce graph,nous pouvons clairement affirmer que la zone n'a pas d'influence.
    nous allons donc merger les 3 colonnes pour la suite de l'analyse '''
    DATAVIZ13 = '''Dataviz 13'''
    DATAVIZ14 = '''Dataviz 14'''