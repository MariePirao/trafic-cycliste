# 🚲 Trafic Cycliste - Prédiction du Trafic Vélo à Paris

## Contexte
Ce projet a pour objectif d'analyser ces données de comptage sur la période du **1er janvier 2024 au 29 janvier 2025**, en identifiant les horaires et zones d’affluence et en explorant les facteurs influençant l'intensité du trafic cycliste. Nous visons à développer un modèle de prédiction du trafic en fonction de plusieurs variables, telles que l'heure, le jour de la semaine, la météo, et la localisation.

Le vélo a désormais surpassé la voiture dans la capitale, et la fréquentation des aménagements cyclables a considérablement augmenté depuis la période Covid. La Ville de Paris continue de développer son infrastructure cyclable avec la création de nouvelles pistes en 2024. Pour suivre l'évolution de la pratique cycliste, près de 70 capteurs permanents ont été installés depuis 2020, collectant en temps réel le nombre de cyclistes par site de comptage et par heure.

L’objectif final est d’anticiper la fréquentation des pistes cyclables et d’aider à la gestion des infrastructures et des mobilités urbaines.

## Installation via Docker et lancement de l'application

1. **Clonez ce projet sur votre machine locale :**

   Ouvrez votre terminal et clonez le repository dans un répertoire de votre choix :

   ```bash
   git clone https://github.com/MariePirao/trafic-cycliste.git
   cd trafic-cycliste

2. **Construisez l’image Docker:**

   ```bash
   docker build -t trafic-cycliste .

3. **📎 Recupération des fichiers nécessaire sur votre machine locale :**

   👉 Important : certains fichiers de données (comptage cycliste, météo, etc.) ne sont pas inclus dans le dépôt.
   📁 Données sources  → [Télécharger via google drive](https://drive.google.com/file/d/1io9GVvzC9bkwmpEznzBBSIogoiitdLVd/view?usp=drive_link)
   Sans ces fichiers, l'application ne pourra pas fonctionner. 

5. **Lancer le conteneur :** 

    Cela ouvrira l'application dans votre navigateur, où vous pourrez : 
      - interagir avec les différentes visualisation
      - voir les prédiction du trafic cycliste et faire un suivi de ces prédidictions


   ```bash
   docker run -p 8501:8501 trafic-cycliste
