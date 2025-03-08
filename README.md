# trafic-cycliste
Premier projet de Data Analyst/Data scientist

PREREQUIS : 
creation d'un compte git (il faut a ce moement là me donner votre email de connexion pour que je vous invite sur le projet)
creation d'un repertoire Porjets/workspaces
installation vscode qui va installer un client git lié a Github

LIAISON DE VSCODE AVEC LE PROJET STREAMLIT ET CREATION D'UNE BRANCHE :
Dans Vscode ouvrir le répertoire créé en local (open folder dans le menu 'File tout en haut)

faire un git clone  (--> cela creera un répertoire trafic-cycliste)

cliquer en bas a gauche de Vscode pour detacher le main, sinon vos commit iront directemment dans Git Main, ce qui n'est pas un travail collaboratif
creer une branche avec votre prénom par exemple.

Dans le repertoire de votre poste git a donc créé un répertoire trafic-cycliste mais aussi trafic-cycliste/data.
En effet ce n'est pas une bonne pratique de remonter dans got les fichiers de données (csv). Mais pour avoir un code commun, j'ai au minimum créer le répertoire.
Vous devez donc déposer les csv utiles au code dans ce répertoire en local.



TEST DE L'APPLICATION:
Tester l'application : Dans le terminal de VSCODE tapez : streamlit run homePage.py
C'est la commande qui lance la page web. c'est un processus donc dans votre terminal vous n'avez plus la main. Pour récupérer la main et arrêter le processus/page web,
faites ctr C attention il faut bien avoir cliquer sur le terminal et pas être encore en modif sur un fichier.
(N.B : quand vous faites des modif de code, vous n'etes pas obligé de stopper et relancer streamlit. La page web detecte une modification sauvegardé (Cmd C),
vous avez juste a cliquer sur rerun et la page Web prendra vos modif en compte)

TEST DU COMMIT : 
Faites un test de commit (quand vous allez commiter vous allez commité dans votre branche (pas d'impact sur les collaborateur))

Dans Vscode allez dans le menu source control tout a gauche et cliquer sur commit & push
Vous allez avoir certainement une erreur il faut dans le terminal de Vscode entrée ces deux commandes successivement :

Pour le commit de votre beanch
git config --global user.email "emailGit@toto.com"
git config --global user.name "usernameGit"


Sur votre PC dans le repertoire que vous aurez crééer/trafic-cycliste/data il faut copier les csv utiles au projet

Pour les fichiers joblib vous pouvez les laisser dans le répertoire courant.