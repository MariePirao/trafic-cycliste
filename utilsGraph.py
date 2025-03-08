import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.subplots as sp
import plotly.graph_objects as go
import folium
import plotly.express as px
from sklearn.calibration import LabelEncoder
import matplotlib.dates as mdates
# Pour éviter d'avoir les messages warning
import warnings
warnings.filterwarnings('ignore')

    
def plot_heatmap(df):
    '''
    Affichage de la matri de corrélation
    '''

    df_work = df.copy()

    #encoder = OrdinalEncoder()
    le = LabelEncoder()
    df_work['nom_compteur'] = le.fit_transform(df_work[['nom_compteur']])

    columns_to_consider = ['comptage_horaire','heure','num_jour_semaine','num_mois','fait_jour', 
                           'weekend','vacances','neutralise',
                           'Partage','1Sens','latitude','longitude',
                           "temperature_2m", "precipitation_mm", "wind_speed",'nom_compteur']

    # Calcul de la matrice de corrélation sur les données selectionnes
    corr_matrix = df_work[columns_to_consider].corr()

    fig, ax = plt.subplots(figsize=(12, 8))  # Ajuste la taille du graphique

    # Tracer la heatmap avec ajustement de la taille de la police
    sns.heatmap(corr_matrix.corr(), ax=ax, annot=True, cmap='coolwarm', fmt=".2f",annot_kws={'size': 10})
    plt.xticks(fontsize=10) 
    plt.yticks(fontsize=10)

    #corr_with_target = corr_matrix['comptage_horaire']
    #print(corr_with_target)

    return fig  # Retourne la figure pour Streamlit

def heatmap_isna(df):
    """Affiche une heatmap de isna()."""
    fig = plt.figure()
    sns.heatmap(df.isna(), cmap="viridis", cbar=False)
    return fig  # Retourne la figure pour Streamlit

def px_compteurs_quotidien_0(df, selected_compteur):
    #copy du df
    df_filtered = df.copy()

    # Définir les périodes imposées pour chaque compteur
    period_dict = {
        "10 avenue de la Grande Armée SE-NO": ("2024-12-27", "2025-01-08"),
        "7 avenue de la Grande Armée NO-SE" : ("2024-03-25", "2024-04-10"),
        "106 avenue Denfert Rochereau NE-SO": ("2024-01-01", "2024-06-15"),
        "135 avenue Daumesnil SE-NO": ("2024-01-21", "2024-02-17"),
        "24 boulevard Jourdan E-O": ("2024-09-15", "2025-02-15"),
        "33 avenue des Champs Elysées NO-SE": ("2024-01-01", "2024-02-20"),
        "38 rue Turbigo NE-SO": ("2024-12-23", "2025-01-10"),
        "38 rue Turbigo SO-NE" : ("2024-12-29","2025-01-08"),
        "Face au 48 quai de la marne Face au 48 quai de la marne Vélos NE-SO" : ("2024-05-02", "2024-05-12"),
        "Face au 48 quai de la marne Face au 48 quai de la marne Vélos SO-NE" : ("2024-05-02", "2024-05-12"),
        "72 boulevard Richard Lenoir S-N": ("2024-01-05", "2024-02-03"),
        "77 boulevard Richard Lenoir N-S": ("2024-01-05", "2024-02-03"),
        "Pont des Invalides (couloir bus) N-S": ("2024-10-28", "2024-11-15"),
        "Pont des Invalides S-N": ("2024-10-28", "2024-11-15"),
        "27 quai de la Tournelle 27 quai de la Tournelle Vélos NO-SE" : ("2024-12-02","2024-12-15"),
        "27 quai de la Tournelle 27 quai de la Tournelle Vélos SE-NO" : ("2024-12-02","2024-12-15"),
        "Face au 16 avenue de la  Porte des Ternes O-E" : ("2024-10-02","2024-11-08"),
        "16 avenue de la Porte des Ternes E-O" : ("2024-10-02","2024-11-08"),
        "Totem 73 boulevard de Sébastopol N-S": ("2024-03-10","2024-04-17"),
        "Totem 73 boulevard de Sébastopol S-N": ("2024-03-10","2024-04-17"),
        }

    if selected_compteur in period_dict:
        selected_compteurs = [selected_compteur]  
    else:
        selected_compteurs = [compteur for compteur in period_dict.keys() if selected_compteur in compteur]

    start_date, end_date = period_dict[selected_compteurs[0]]
    if (selected_compteur == "10 avenue de la Grande Armée SE-NO"):
        selected_compteurs.append('7 avenue de la Grande Armée NO-SE')
    elif (selected_compteur == "7 avenue de la Grande Armée NO-SE"):
        selected_compteurs.append('10 avenue de la Grande Armée SE-NO')
    # Filtrer les données en fonction de la période et du compteur
    df_filtered = df_filtered[(df_filtered["nom_compteur"].isin(selected_compteurs)) & 
                     (df_filtered["date_heure_comptage"] >= start_date) & 
                     (df_filtered["date_heure_comptage"] <= end_date)]
    
    # Extraction des informations de date pour la moyenne journalière
    df_filtered = df_filtered.copy()
    df_filtered.loc[:, "Jour"] = df_filtered["date_heure_comptage"].dt.to_period("D")

    # Calcul des moyennes journalières (en groupant par "nom_compteur" et "Jour")
    df_daily = df_filtered.groupby(["nom_compteur", "Jour"])["comptage_horaire"].mean().reset_index()

    # Conversion du format période en datetime pour l'affichage
    df_daily["Jour"] = df_daily["Jour"].astype(str)

    # Création du graphique interactif
    fig = px.line(df_daily, 
                  x="Jour", 
                  y="comptage_horaire", 
                  color="nom_compteur", 
                  width=1000, height=600,
                  labels={'comptage_horaire': 'Comptage Horaire', 'Jour': 'Jour'})

    # Afficher le graphique dans Streamlit
    return fig

def px_compteurs_mensuel_JO(df):

    Compteur = ['Totem Cours la Reine E-O','Totem Cours la Reine O-E','Quai des Tuileries Quai des Tuileries Vélos NO-SE',
                'Quai des Tuileries Quai des Tuileries Vélos SE-NO','Pont de la Concorde N-S','Pont de la Concorde S-N']

    df_filtered = df[df["nom_compteur"].isin(Compteur)]

    # Extraction des informations de date
    df_filtered = df_filtered.copy()
    df_filtered.loc[:,"Mois"] = df_filtered["date_heure_comptage"].dt.to_period("M")
     

    # Calcul des moyennes mensuelles
    df_monthly = df_filtered.groupby(["nom_compteur", "Mois"])["comptage_horaire"].mean().reset_index()

    # Conversion du format période en datetime pour l'affichage
    df_monthly["Mois"] = df_monthly["Mois"].astype(str)
    
    # Création du graphique interactif
    fig = px.line(df_monthly, 
              x="Mois", 
              y="comptage_horaire", 
              color="nom_compteur", 
              width=100, height=600)
    return fig



def go_bar_meteo(df):
    # Calculer la moyenne des comptages par catégorie qualitative
    df_precipitation = df.groupby('precipitation', as_index=False, observed=False)['comptage_horaire'].mean()
    df_wind = df.groupby('wind', as_index=False, observed=False)['comptage_horaire'].mean()
    df_temperature = df.groupby('temperature', as_index=False, observed=False)['comptage_horaire'].mean()

    temperature_order = ['Gel','Froid', 'Tempéré', 'Chaud', 'Très chaud']
    wind_order = ['Pas de vent', 'Vent modérée', 'vent', 'grand vent']
    precipitation_order = ['Pas de pluie/bruine', 'Pluie modérée', 'Fortes averses']

    # Créer une figure avec plusieurs sous-graphiques
    fig = sp.make_subplots(rows = 1, cols = 3, subplot_titles=[" Précipitations", "Vent", "Température"])

    # Ajouter les barres pour chaque catégorie
    fig.add_trace(go.Bar(x=df_precipitation['precipitation'], y=df_precipitation['comptage_horaire'], marker_color='blue', name='precipitation'), row=1, col=1)
    fig.add_trace(go.Bar(x=df_wind['wind'], y=df_wind['comptage_horaire'], marker_color='grey', name='wind'),row=1, col=2)
    fig.add_trace(go.Bar(x=df_temperature['temperature'], y=df_temperature['comptage_horaire'], marker_color='red', name='temperature'), row=1, col=3)

    # Mettre à jour la mise en page et titre
    fig.update_layout(height=400, width=1100,
                      xaxis1=dict(categoryorder='array', categoryarray=precipitation_order),  # Définit l'ordre des catégories pour 'precipitation'
                      xaxis2=dict(categoryorder='array', categoryarray=wind_order),  # Définit l'ordre des catégories pour 'wind'
                      xaxis3=dict(categoryorder='array', categoryarray=temperature_order)  # Définit l'ordre des catégories pour 'temperature'
                      )
    return fig

def sns_scatter_meteo(df):
    df3 = df[df["nom_compteur"] == "Totem 73 boulevard de Sébastopol S-N"]
    df2 = df3[df3['fait_jour'] == 1][['comptage_horaire', 'temperature_2m', 'wind_speed', 'precipitation']].dropna()
    df2 = df2[df2['comptage_horaire'] >= 500]

    fig = plt.figure(figsize=(20, 10))
    #scatter
    sns.scatterplot(data=df2, x='temperature_2m', y='wind_speed',  hue='comptage_horaire', size='comptage_horaire', sizes=(50, 200),
                    palette='coolwarm', style='precipitation', markers=['o', 's', 'D'], alpha=0.8)
    plt.title("Nombre de passages de vélos selon la météo")
    plt.xlabel("Température (°C)")
    plt.ylabel("Vitesse du vent (km/h)")
    plt.legend()
    return fig

def sns_scatter_vacances(df):

    fig = plt.figure(figsize=(6, 6))
    fig = sns.catplot(y="comptage_horaire", x="vacances", kind="box", data = df)
    plt.xlabel('vacances toutes zones')
    plt.ylim(0, 300)
    return fig

def top10Flop10(df):
 
    df_grouped = df.groupby('nom_compteur')['comptage_horaire'].mean().reset_index()
    df_sorted = df_grouped.sort_values(by='comptage_horaire', ascending=False)

    top_10 = df_sorted.head(10)
    fig = plt.figure(figsize=(16, 10))
    bar = sns.barplot(data=top_10, x='comptage_horaire', y='nom_compteur', palette='Blues_r', hue='nom_compteur')
    plt.xlabel('comptage_horaire moyen')
    plt.ylabel('nom_compteur')
    plt.ylabel('Top 10 des compteurs de vélo')

    flop_10 = df_sorted.tail(10)
    fig1 = plt.figure(figsize=(16, 10))
    bar = sns.barplot(data=flop_10, x='comptage_horaire', y='nom_compteur', palette='Reds_r', hue='nom_compteur')
    plt.xlabel('comptage_horaire moyen')
    plt.ylabel('nom_compteur')
    plt.ylabel('Flop 10 des compteurs de vélo')
    return fig, fig1

def dayNight(df):  

    # Filtrer les données en fonction de 'is_day'
    all_hours = pd.DataFrame({'heure': range(24)})
    df_day = df.loc[df['fait_jour'] == 1] 
    df_night = df.loc[df['fait_jour'] == 0] 

     # Calculer la moyenne des passages des vélos par heure pour le jour
    df_day_avg = df_day.groupby('heure')['comptage_horaire'].mean()
    df_day_avg = df_day_avg.reset_index()
    df_day_avg = all_hours.merge(df_day_avg, on='heure', how='left')
    df_day_avg['comptage_horaire'] = df_day_avg['comptage_horaire'].fillna(0)  

    # Calculer la moyenne des passages des vélos par heure pour la nuit
    df_night_avg = df_night.groupby('heure')['comptage_horaire'].mean()
    df_night_avg = df_night_avg.reset_index()
    df_night_avg = all_hours.merge(df_night_avg, on='heure', how='left')
    df_night_avg['comptage_horaire'] = df_night_avg['comptage_horaire'].fillna(0)
    
    fig = go.Figure()
    # Ajouter la courbe pour les passages de vélos pendant le jour
    fig.add_trace(go.Scatter(x=df_day_avg['heure'], y=df_day_avg['comptage_horaire'], mode='lines', name='Jour', line=dict(color='purple')))
    # Ajouter la courbe pour les passages de vélos pendant la nuit
    fig.add_trace(go.Scatter(x=df_night_avg['heure'], y=df_night_avg['comptage_horaire'], mode='lines', name='Nuit', line=dict(color='blue')))
    # Mettre à jour les options du graphique
    fig.update_layout(
        xaxis_title="Heure",
        yaxis_title="Nombre Moyen de Passages",
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),  # Afficher toutes les heures
        width=800,
        height=600)
    return fig


# Graphique représentant le trafic cycliste quotidien à Paris :
def journalyCount(df):   
    #df['time'] = pd.to_datetime(df['date_heure_comptage'], utc=True).dt.tz_convert(None)
    df_daily = df.groupby(df['date_heure_comptage'].dt.date)['comptage_horaire'].mean().reset_index()
    df_daily.columns = ['date', 'comptage_moyen']

    plt.figure(figsize=(18, 6))
    plt.plot(df_daily['date'], df_daily['comptage_moyen'], linestyle='-', color = 'blue')
    plt.xlabel("Date")
    plt.ylabel("Nombre moyen de vélos par heure et par site")
    plt.xticks(rotation=45)
    plt.grid(True)  # Ajout du quadrillage avec ax
    return plt


def averageCountByHour(df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="heure", y="comptage_horaire", data = df)
    plt.xlabel("Comptage horaire")
    return plt


def countByHourAndNeutralise(df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="heure", y="comptage_horaire", hue="neutralise", data=df)
    plt.xlabel("Comptage horaire")
    return plt


def averageCountByWeek(df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="heure", y="comptage_horaire", hue="weekend", data=df)
    plt.xlabel("Comptage horaire")
    return plt

# Distribution de la variable comptage_horaire et identification des outliers :
def boxplot(column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x = column, color = 'blue')
    plt.xlabel("Comptage horaire")
    return plt

# cartographie
def generate_folium_map(df,map_filename):

    df_work = df.copy()
    df_work['weekend'] = df_work['num_jour_semaine'].apply(lambda x: 1 if x >= 5 else 0)
    df_comptage_bornes = df_work.groupby(["latitude","longitude"])["comptage_horaire"].sum().reset_index()
       
    # Supprimer les lignes avec valeurs manquantes
    df_comptage_bornes = df_comptage_bornes.dropna(subset=["latitude", "longitude", "comptage_horaire"])

    # Créer une carte centrée sur Paris
    m = folium.Map(location=[48.8566, 2.3522], zoom_start=12)

    # Ajouter des cercles proportionnels avec une taille plus petite
    for _, row in df_comptage_bornes.iterrows():
        compteur = df_work.loc[(df_work["latitude"] == row["latitude"]) & (df_work["longitude"] == row["longitude"]), "nom_compteur"]
        value = row["comptage_horaire"]
        popup_text = f"{compteur.iloc[0]} : {value} "
        #popup_text = f"Borne: {value}"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(row["comptage_horaire"] / 100000, 2),  # Divise par un nombre plus grand pour éviter des cercles géants
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.5,
            popup=popup_text
        ).add_to(m)
  
    # Sauvegarder la carte
    m.save(map_filename)
    return map_filename


def boxplotTemperature(df):
    df3 = df[df["nom_compteur"]=="Totem 73 boulevard de Sébastopol S-N"]
    df2 = df3[['comptage_horaire', 'date_heure_comptage', 'temperature_2m']].copy()
    bins = [-20, 0, 5, 10, 15, 20, 25, 30, 35, 60]
    labels = ["<0°C", "0-5°C", "5-10°C", "10-15°C", "15-20°C", "20-25°C", "25-30°C", "30-35°C", ">35°C"]
    df2.loc[:, 'Température Catégorie'] = pd.cut(df2['temperature_2m'], bins=bins, labels=labels)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df2, x='Température Catégorie', y='comptage_horaire', palette='coolwarm', hue='Température Catégorie')
    plt.ylim(-10, 1500)
    plt.xlabel("Température (°C)")
    plt.ylabel("Nombre de vélos comptés")

    return plt

def boxplotVent(df):
    # Code boxplot Trafic vélo selon la vitesse du vent
    df3 = df[df["nom_compteur"]=="Totem 73 boulevard de Sébastopol S-N"]
    df2 = df3[['comptage_horaire', 'date_heure_comptage', 'wind_speed']].copy()
    #df2['date_heure_comptage'] = pd.to_datetime(df2['date_heure_comptage'], errors='coerce')
    bins = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    labels = ["0-5 km/h", "5-10 km/h", "10-15 km/h", "15-20 km/h", "20-25 km/h","25-30 km/h", "30-40 km/h", "40-50 km/h"]

    df2.loc[:, 'Vent Catégorie'] = pd.cut(df2['wind_speed'], bins=bins, labels=labels)

    plt.figure(figsize=(10,6))
    sns.boxplot(data=df2, x='Vent Catégorie', y='comptage_horaire', palette='coolwarm', hue='Vent Catégorie')
    plt.ylim(-10, 1500)
    plt.xlabel("Vitesse du vent (km/h)")
    plt.ylabel("Nombre de vélos comptés")   

    return plt

def plot_abherrante(df):
    
    # Filtrer les données pour 'Quai d’Orsay O-E' et la plage de dates spécifiée
    df_filtered = df.loc[(df['nom_compteur'] == "Quai d'Orsay O-E") & \
                         (df['date_heure_comptage'] >= '2025-01-04 12:00') & \
                         (df['date_heure_comptage'] <= '2025-01-06 12:00')]

    # Créer le graphique 
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered['date_heure_comptage'],
        y=df_filtered['comptage_horaire'],
        mode='lines',  # Afficher la courbe avec des marqueurs
        name='comptage_horaire',
        marker=dict(color='blue'),
        text=df_filtered.apply(lambda row: f"Date: {row['date_heure_comptage']}<br>Compteur: {row['nom_compteur']}<br>Comptage: {row['comptage_horaire']}", axis=1),
        hoverinfo='text'  # Afficher les informations du 'text' lors du survol
    )) 
    # Mettre à jour le layout du graphique
    fig.update_layout(
        title="comptage_horaire pour 'Quai d'Orsay O-E'",
        xaxis_title="Date et Heure",
        yaxis_title="comptage_horaire",
        xaxis=dict(tickformat="%Y-%m-%d %H:%M"),  # Format de l'axe X (Date et Heure)
        width=800,
        height=600,
        hovermode='closest'  # Permet d'afficher le tooltip pour la donnée la plus proche de la souris
    )

    return fig

def pix_prediction(clf, X_test,y_test):
    y_pred_test = clf.predict(X_test)
 
    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred_test, y_test, color='#4529de')
    plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color='#26dbe0')
    plt.title("\nNuage de points sur la prédiction des passages vélo \n", fontsize=10)
    plt.xlabel("prediction",rotation=0, labelpad=15, fontsize=10)
    plt.ylabel("vrai valeur", rotation=90, labelpad=15, fontsize=10)
    return plt


def boxplot_vacances1(df):
    
    # Créer des subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Boxplot pour zone vacances A
    sns.violinplot(x='vacances_zone_a', y='comptage_horaire', data=df, ax=axes[0], palette='Blues', hue='vacances_zone_a', legend=False)
    axes[0].set_title('Zone A')
    axes[0].set_xlabel('Vacances Zone A')
    axes[0].set_ylabel('comptage_horaire')

    # Boxplot pour zone vacances B
    sns.violinplot(x='vacances_zone_b', y='comptage_horaire', data=df, ax=axes[1], palette='Greens', hue='vacances_zone_b', legend=False)
    axes[1].set_title('Zone B')
    axes[1].set_xlabel('Vacances Zone B')
    axes[1].set_ylabel('comptage_horaire')

    # Boxplot pour zone vacances C
    sns.violinplot(x='vacances_zone_c', y='comptage_horaire', data=df, ax=axes[2], palette='Reds', hue='vacances_zone_c', legend=False)
    axes[2].set_title('Zone C')
    axes[2].set_xlabel('Vacances Zone B')
    axes[2].set_ylabel('comptage_horaire')

    return plt


def boxplot_vacances3(df):
    df.loc[:,'Vacances_zone_a'] = df['vacances_zone_a'].map({0: 'Non', 1: 'Oui'})
    df.loc[:,'Vacances_zone_b'] = df['vacances_zone_b'].map({0: 'Non', 1: 'Oui'})
    df.loc[:,'Vacances_zone_c'] = df['vacances_zone_c'].map({0: 'Non', 1: 'Oui'})

    # Calculer la moyenne du comptage_horaire pour chaque zone (vacances oui ou non)
    df_a = df.groupby(['Vacances_zone_a'])['comptage_horaire'].mean().reset_index()
    df_b = df.groupby(['Vacances_zone_b'])['comptage_horaire'].mean().reset_index()
    df_c = df.groupby(['Vacances_zone_c'])['comptage_horaire'].mean().reset_index()

    # Créer des subplots pour afficher les trois graphiques
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Graphique pour la zone A
    sns.barplot(data=df_a, x='Vacances_zone_a', y='comptage_horaire', ax=axes[0], palette='Blues', hue='Vacances_zone_a', legend=False)
    axes[0].set_title('Zone A')
    axes[0].set_xlabel('Vacances')
    axes[0].set_ylabel('comptage_horaire moyen')

    # Graphique pour la zone B
    sns.barplot(data=df_b, x='Vacances_zone_b', y='comptage_horaire', ax=axes[1], palette='Greens', hue='Vacances_zone_b', legend=False)
    axes[1].set_title('Zone B')
    axes[1].set_xlabel('Vacances')
    axes[1].set_ylabel('comptage_horaire moyen')

    # Graphique pour la zone C
    sns.barplot(data=df_c, x='Vacances_zone_c', y='comptage_horaire', ax=axes[2], palette='Reds', hue='Vacances_zone_c', legend=False)
    axes[2].set_title('Zone C')
    axes[2].set_xlabel('Vacances')
    axes[2].set_ylabel('comptage_horaire moyen')

    # Ajouter un titre global
    plt.suptitle("comptage_horaire moyen en fonction des vacances (Zone A, B, C)", fontsize=16)

    # Ajuster la mise en page
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Pour éviter que le titre chevauche les graphiques
    return fig


def plot_avg_mensuel(df, rue):

    df_work = df.copy()

    compteurs_specifiques = []

    if rue=="GrandeArmee":
        compteurs_specifiques = [
            "10 avenue de la Grande Armée SE-NO", 
            "7 avenue de la Grande Armée NO-SE",
            "10 avenue de la Grande Armée 10 avenue de la Grande Armée [Bike IN]",
            "10 avenue de la Grande Armée 10 avenue de la Grande Armée [Bike OUT]"]
        
    elif rue=="all":
        compteurs_specifiques = df_work["Nom du compteur"].unique()
        
    df_filtered = df_work[df_work["Nom du compteur"].isin(compteurs_specifiques)]
    
    # on cherche le mois avec la données 
    df_filtered["Date et heure de comptage"] = df_filtered["Date et heure de comptage"].str[:18]
    df_filtered["Date et heure de comptage"] = pd.to_datetime(df_filtered["Date et heure de comptage"])

    df_filtered['Mois'] = df_filtered['Date et heure de comptage'].dt.to_period('M').astype(str)
    
    # Calculer la moyenne des passages par mois et par compteur
    df_monthly_avg = df_filtered.groupby(['Mois', 'Nom du compteur'])['Comptage horaire'].mean().reset_index()
    
    # Création du graphique
    fig = px.bar(df_monthly_avg, 
                 x="Mois", 
                 y="Comptage horaire", 
                 color="Nom du compteur", 
                 barmode="group", 
                 labels={'Comptage horaire': 'Passages moyens de vélos', 'Mois': 'Mois'},
                 height=600, width=800)
    
    return fig

def nbLigne_compteur(df):

    df_work = df.copy()

    df_work["Date et heure de comptage"] = df_work["Date et heure de comptage"].str[:18]
    df_work["Date et heure de comptage"] = pd.to_datetime(df_work["Date et heure de comptage"])

    # On filtre sur les dares concernées par l'analyse
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2025-02-01')
    df_filtered = df_work[(df_work['Date et heure de comptage'] >= start_date) & (df_work['Date et heure de comptage'] <= end_date)]

    #compter déjà traité
    df_filtered = df_filtered[(df_filtered['Nom du compteur'] !="7 avenue de la Grande Armée NO-SE") & (df_filtered['Nom du compteur'] !="10 avenue de la Grande Armée 10 avenue de la Grande Armée [Bike IN]") 
                      & (df_filtered['Nom du compteur'] !="10 avenue de la Grande Armée") & (df_filtered['Nom du compteur'] !="10 avenue de la Grande Armée 10 avenue de la Grande Armée [Bike UOT]") ]

    # On compte le nombre de lignes pour chaque compteur
    df_counts = df_filtered['Nom du compteur'].value_counts().reset_index()

    df_counts.columns = ['Nom du compteur', 'Nombre de Lignes']
    df_counts_filtered = df_counts[df_counts['Nombre de Lignes'] < 9475]

    fig = px.bar(df_counts_filtered, 
                 x='Nom du compteur', 
                 y='Nombre de Lignes',
                 labels={'Nom du compteur': 'compteur', 'Nombre de Lignes': 'Nombre de lignes'},
                 height=400)    
    return fig

def generate_graph_Tempo(test_data, test_predictions, compteur):
    """
    Génère un graphique comparant les données réelles et les prédictions.
    """
    test_data['ds'] = pd.to_datetime(test_data['ds'])
    test_predictions['ds'] = pd.to_datetime(test_predictions['ds'])

    start_date = pd.to_datetime('2025-01-25')
    end_date = pd.to_datetime('2025-01-31')

    # Filtrer pour janvier 2015
    period_test_data = test_data[(test_data['ds'] >= start_date) & (test_data['ds'] <= end_date)]
    period_test_predictions = test_predictions[(test_predictions['ds'] >= start_date) & (test_predictions['ds'] <= end_date)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(period_test_data['ds'], period_test_data['y'], label='Vraies données', color='blue')
    ax.plot(period_test_predictions['ds'], period_test_predictions['yhat'], label='Prédictions', color='red', linestyle='--')
    ax.fill_between(period_test_predictions['ds'], period_test_predictions['yhat_lower'], 
                period_test_predictions['yhat_upper'], color='orange', alpha=0.15, label='Intervalle de confiance (80%)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Comptage horaire')
    ax.set_title(f'Prédictions vs Réalité pour le compteur {compteur}')
    ax.legend()

    return fig


def plot_feature_importances_RF(model3,X_train,feats):

    importances = model3.feature_importances_
    # definition du seuil d'importance et selection des features
    threshold = 0.015
    selected_features = feats.columns[importances > threshold]
    X_train_selected = X_train[selected_features]

    selected_importances = importances[feats.columns.isin(selected_features)]

    # Créer la figure pour les importances des variables
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.bar(X_train_selected.columns, selected_importances)
    ax.set_ylabel("Feature")
    ax.tick_params(axis='x', rotation=90) 
    plt.tight_layout()  

    return fig


def filter_data_temp(period,df):
    '''Ce graph montre les moyennes de passages de vélos selon la températures et les heures'''

    df_copy= df.copy()

    df_work = df_copy.groupby(["date_heure_comptage"], as_index=False).agg({"comptage_horaire": "mean","num_jour_semaine": "first", "temperature_2m": "first","heure": "first"})

    # Filtrage des données selon la période
    if period == "semaine":
        filtered_df = df_work[df_work["num_jour_semaine"].isin([0, 1, 2, 3, 4])]
    else:
        filtered_df = df_work[df_work["num_jour_semaine"].isin([5, 6])]

    # Créer un graphique avec Plotly Express
    fig = px.scatter(filtered_df, x="heure", y='comptage_horaire', color='temperature_2m',color_continuous_scale='ylorrd')
    fig.update_layout(width=2000, height=700)
    
    return fig

def filter_data_photo(period,df):
    '''Ce graph montre les moyennes de passages de vélos selon les détails de la piste cyclable'''

    df_work= df.copy()

    # Filtrage des données selon la période
    if period == "semaine":
        df_work = df_work[df_work["num_jour_semaine"].isin([0, 1, 2, 3, 4])]
    else:
        df_work = df_work[df_work["num_jour_semaine"].isin([5, 6])]

    df_moyenne_partage = df_work[df_work['Partage'] == 1].groupby('heure')['comptage_horaire'].mean()
    df_moyenne_separe = df_work[df_work['Partage'] == 0].groupby('heure')['comptage_horaire'].mean()
    df_moyenne_1sens = df_work[df_work['1Sens'] == 1].groupby('heure')['comptage_horaire'].mean()
    df_moyenne_2sens = df_work[df_work['1Sens'] == 0].groupby('heure')['comptage_horaire'].mean()

    # Tracer les 4 courbes
    plt.figure(figsize=(10, 6))
    plt.plot(df_moyenne_partage.index, df_moyenne_partage, label="Piste partagé", color="blue", alpha=0.7)
    plt.plot(df_moyenne_separe.index, df_moyenne_separe, label="Piste séparée", color="red", alpha=0.7)
    plt.plot(df_moyenne_1sens.index, df_moyenne_1sens, label="Piste à 1 sens", color="green", alpha=0.7)
    plt.plot(df_moyenne_2sens.index, df_moyenne_2sens, label="Piste à deux sens", color="orange", alpha=0.7)
    
    # Ajouter des labels et une légende
    plt.xlabel("Heure de la journée", fontsize=12)
    plt.ylabel("Comptage horaire moyen", fontsize=12)
    plt.legend()

    return plt



def courbePrediction(df_fevrier, compteur, date_debut_choisie,date_fin_choisie):

   
    df_fev_avec_pred = df_fevrier.copy()

    df_fev_avec_pred["date_heure_comptage"] = pd.to_datetime(df_fev_avec_pred["date_heure_comptage"])
    df_fev_avec_pred = df_fev_avec_pred[(df_fev_avec_pred["date_heure_comptage"] >= date_debut_choisie) &
                                        (df_fev_avec_pred["date_heure_comptage"] <= date_fin_choisie)]
    

    
    fig, ax = plt.subplots(figsize=(10, 5))

    if compteur == 'All':
        df_moyenne = df_fev_avec_pred.groupby("date_heure_comptage")[["comptage_horaire", "predictions_comptage_horaire"]].mean().reset_index()
        ax.plot(df_moyenne["date_heure_comptage"], df_moyenne["comptage_horaire"], label="comptage réel", color="blue", alpha=0.7)
        ax.plot(df_moyenne["date_heure_comptage"], df_moyenne["predictions_comptage_horaire"], label="Prédiction", color="red", alpha=0.7)

    else:
        #copy et filtrage du dataframe qui continent les prédictions et les résultats réels
        df_fev_avec_pred = df_fev_avec_pred[df_fevrier["nom_compteur"] == compteur]
        ax.plot(df_fev_avec_pred["date_heure_comptage"], df_fev_avec_pred["comptage_horaire"], label="comptage réel", color="blue", alpha=0.7)
        ax.plot(df_fev_avec_pred["date_heure_comptage"], df_fev_avec_pred["predictions_comptage_horaire"], label="Prédiction", color="red", alpha=0.7)
       
    ax.set_xlabel("date_heure_comptage", fontsize=12)
    ax.set_ylabel("Comptage horaire / Prédiction", fontsize=12)
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    #plt.tight_layout()

    return fig

def courbePrediction3J(df3J, compteur):

    plt.figure(figsize=(25, 6))

    if compteur == 'All':
        df_moyenne_semaine = df3J.groupby("date_heure_comptage")["predictions_comptage_horaire"].mean().reset_index()
        plt.plot(df_moyenne_semaine["date_heure_comptage"], df_moyenne_semaine["predictions_comptage_horaire"], label="Prédiction", color="red", alpha=0.7)
        plt.xlim(df_moyenne_semaine["date_heure_comptage"].min(), df_moyenne_semaine["date_heure_comptage"].max())
    else:
        df3JCompteur = df3J[df3J["nom_compteur"] == compteur]
        plt.plot(df3JCompteur["date_heure_comptage"], df3JCompteur["predictions_comptage_horaire"], label="Prédiction", color="red", alpha=0.7)
        plt.xlim(df3JCompteur["date_heure_comptage"].min(), df3JCompteur["date_heure_comptage"].max())
    plt.xlabel("Date et heure de comptage", fontsize=12)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y %Hh00'))
    plt.ylabel("Comptage horaire / Prédiction", fontsize=12)
    plt.legend()
    plt.grid(True)
    return plt