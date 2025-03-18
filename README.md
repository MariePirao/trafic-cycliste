# trafic-cycliste
Projet de Data Analyst/Data scientist (Analyse Trafic cycliste sur Paris)

    <br>Alors que le vélo a désormais surpassé la voiture dans la capitale et que la fréquentation des aménagements cyclables a bondi depuis la période Covid,
la Ville de Paris poursuit ses aménagements avec la création de plus d’une vingtaine de nouvelles pistes en 2024.
Ces constats ont pu être effectués grâce notamment au déploiement par la Ville de Paris de près de soixante-dix capteurs de vélos permanents depuis 2020,   
qui collectent en temps réel le nombre de cyclistes passant par chaque site de comptage, à chaque heure de la journée. Ces données précises permettent ainsi de visualiser 
l’évolution de la pratique cycliste à Paris, et sont également analysées dans le but d’améliorer les connaissances sur l’usage des infrastructures de la capitale.<br>
    Notre projet a pour objectif d’effectuer une analyse des données récoltées par ces compteurs de vélos sur la période du 1er janvier 2024 au 29 janvier 2025, 
afin de visualiser dans un premier temps les horaires et les zones d’affluence, mais également afin de déterminer quels paramètres peuvent influencer l’intensité du trafic cycliste.
In fine, il s’agira de proposer un modèle de prédiction du trafic cycliste, en s’appuyant sur les variables les plus explicatives identifiées lors de notre analyse. <br>
    L’objectif est de pouvoir anticiper la fréquentation des pistes cyclables en fonction de différents facteurs (jour de la semaine, heure, météo, localisation, etc.),   
afin d’apporter une aide à la décision pour l’aménagement urbain, la gestion des mobilités ou encore la planification d’événements dans la ville.

PREREQUIS : 
Creation d'un repertoire Projets en local
Sous vscode ouvrir ce répertoire puis faire un git clone : git clone  (--> cela creera un répertoire trafic-cycliste dans votre répertoire Projets

Dans votre répertoire local aller dans Projets/trafic-cycliste
Télécharger du drive le fichier "Archive_Finale.zip" et 
- deposer dans data/ les fichiers présents dans l'archive dans data (fichiers *.csv)
- deposer dans prediction/ les fichiers présents dans l'archive dans prediction (fichiers prediction*.csv)
- deposer à la racine les fichiers présents dans l'archive dans la racine (fichiers .jobllib)

TEST DE L'APPLICATION:
Pour lancer l'application , dDans le terminal de VSCODE tapez : streamlit run homePage.py