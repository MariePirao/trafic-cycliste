# 🚲 Trafic Cycliste - Prédiction du Trafic Vélo à Paris

## Contexte
Ce projet a pour objectif d'analyser ces données de comptage sur la période du **1er janvier 2024 au 29 janvier 2025**, en identifiant les horaires et zones d’affluence et en explorant les facteurs influençant l'intensité du trafic cycliste. Nous visons à développer un modèle de prédiction du trafic en fonction de plusieurs variables, telles que l'heure, le jour de la semaine, la météo, et la localisation.

Le vélo a désormais surpassé la voiture dans la capitale, et la fréquentation des aménagements cyclables a considérablement augmenté depuis la période Covid. La Ville de Paris continue de développer son infrastructure cyclable avec la création de nouvelles pistes en 2024. Pour suivre l'évolution de la pratique cycliste, près de 70 capteurs permanents ont été installés depuis 2020, collectant en temps réel le nombre de cyclistes par site de comptage et par heure.

L’objectif final est d’anticiper la fréquentation des pistes cyclables et d’aider à la gestion des infrastructures et des mobilités urbaines.

## Prérequis

Avant de pouvoir exécuter l'application, assurez-vous d'avoir les éléments suivants installés :

- **Python 3.x** ou supérieur
- **Streamlit** pour l'interface interactive (Installation via `pip install streamlit`)
- **Bibliothèques Python nécessaires :**
  - Pandas
  - NumPy
  - Scikit-learn
  - XGBoost
  - Matplotlib
  - Seaborn
  - BeautifulSoup

## Installation

1. **Clonez ce projet sur votre machine locale :**

   Ouvrez votre terminal et clonez le repository dans un répertoire de votre choix :

   ```bash
   git clone https://github.com/MariePirao/trafic-cycliste.git

2. **Recupération des fichiers nécessaire sur votre machine locale :**  Sur demande

## Lancer l'application
Une fois l'installation terminée, vous pouvez démarrer l'application Streamlit pour visualiser les résultats de la prédiction :

1. **Dans le terminal, naviguez dans le répertoire du projet.**

1. **Lancez l'application Streamlit :**  streamlit run homePage.py
Cela ouvrira l'application dans votre navigateur, où vous pourrez voir les résultats du trafic cycliste prédit et interagir avec les visualisations.
