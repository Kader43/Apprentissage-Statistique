#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')



#%%
###############################################################################
#               Iris Dataset
###############################################################################
# Chargement du jeu de données Iris depuis scikit-learn
iris = datasets.load_iris()
# On récupère toutes les variables (longueur/largeur des sépales et pétales)
X = iris.data
# Normalisation des données 
X = scaler.fit_transform(X)
# On récupère les étiquettes (les classes : 0 = setosa, 1 = versicolor, 2 = virginica)
y = iris.target
# On ne garde que les classes 1 et 2 (versicolor et virginica) pour faire un problème binaire
X = X[y != 0, :2]  # on ne garde que les 2 premières features
y = y[y != 0]

# Séparation des données en un ensemble d’entraînement et un ensemble de test
# Ici : 25 % des données pour le test, 75 % pour l’entraînement
# Le paramètre random_state=42 permet d’avoir toujours la même séparation (reproductible)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


###############################################################################
# fit the model with linear vs polynomial kernel
###############################################################################

#%%
# Q1 Linear kernel


# Définition des hyperparamètres à tester
# Ici : on fixe le noyau à 'linéaire' et on fait varier le paramètre C 
# sur une échelle logarithmique entre 10^-3 et 10^3 avec 200 valeurs
parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 200))}

# Création d’un classificateur SVM de base
clf_basic = SVC()

# Recherche des meilleurs hyperparamètres par validation croisée
# GridSearchCV va tester toutes les valeurs possibles de C et garder celle qui donne les meilleures performances
# n_jobs=-1 permet d’utiliser tous les cœurs du processeur pour accélérer la recherche
clf_linear = GridSearchCV(clf_basic, parameters, n_jobs=-1)

# Entraînement du modèle sur les données d’entraînement
clf_linear.fit(X_train, y_train)

# Évaluation des performances finales (score de généralisation)
# On calcule le score sur l’ensemble d’apprentissage et sur l’ensemble de test
print('Generalization score for linear kernel: %s, %s' %
      (clf_linear.score(X_train, y_train),
       clf_linear.score(X_test, y_test)))


#%%

#Q2 polynomial kernel
# Valeurs possibles du paramètre C, choisies sur une échelle logarithmique entre 10^-3 et 10^3 (5 valeurs)
Cs = list(np.logspace(-3, 3, 5))

# Valeurs possibles du paramètre gamma (contrôle l'influence d’un point d’entraînement)
# Ici, gamma = 10
gammas = 10. ** np.arange(1, 2)
# Valeurs possibles du degré du polynôme (contrôle la complexité du noyau polynomial)
degrees = np.r_[1, 2, 3]
# fit the model and select the best set of hyperparameters
parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}
clf_poly = GridSearchCV(SVC(), parameters, n_jobs=-1)
clf_poly.fit(X_train, y_train)

print(clf_poly.best_params_)
print('Generalization score for polynomial kernel: %s, %s' %
      (clf_poly.score(X_train, y_train),
       clf_poly.score(X_test, y_test)))


#%%
# Définition de fonctions utilitaires pour visualiser les frontières de décision
# à l’aide de la fonction 'frontiere' du fichier svm_source.py

# f_linear : utilise le classificateur SVM linéaire entraîné (clf_linear)
# Elle prend un point 'xx' (sous forme de vecteur), 
# le redimensionne en (1, -1) pour respecter le format attendu par scikit-learn,
# puis renvoie la classe prédite (0 ou 1).

def f_linear(xx):
    return clf_linear.predict(xx.reshape(1, -1))

# f_poly : utilise le classificateur SVM polynomial entraîné (clf_poly)
# Fonction identique à f_linear, mais appliquée au modèle polynomial.
# Elle sert à tracer la frontière de séparation obtenue avec un noyau polynomial.

def f_poly(xx):
    return clf_poly.predict(xx.reshape(1, -1))
#%%

# Active le mode interactif de matplotlib (les figures se mettent à jour automatiquement)
plt.ion()

# Crée une nouvelle figure avec une taille de 15x5
plt.figure(figsize=(15, 5))

# Premier sous-graphe (1 ligne, 3 colonnes, 1er plot)
plt.subplot(131)
# Affiche les données (nuage de points) à deux dimensions
plot_2d(X, y)
plt.title("iris dataset") # Titre du premier plot

# Deuxième sous-graphe (toujours 1 ligne, 3 colonnes, mais 2e plot)
plt.subplot(132)    
# Trace la frontière de décision du SVM linéaire sur les données
frontiere(f_linear, X, y)
plt.title("linear kernel") # Titre du deuxième graphe

#%%
# Crée une figure de taille 15x5 (si déjà défini avant, continue sur la même)
plt.figure(figsize=(15, 5))

# Troisième sous-graphe (1 ligne, 3 colonnes, 3e plot)
plt.subplot(133)

# Trace la frontière de décision du SVM polynomial sur les données
frontiere(f_poly, X, y)

# Donne un titre au graphe
plt.title("polynomial kernel")

# Ajuste automatiquement l'espacement entre les sous-graphes 
# pour éviter que les titres ou axes se chevauchent
plt.tight_layout()

# Redessine la figure (utile si on est en mode interactif avec plt.ion())
plt.draw()

#%%
###############################################################################
#               SVM GUI
###############################################################################

# please open a terminal and run python svm_gui.py
# Then, play with the applet : generate various datasets and observe the
# different classifiers you can obtain by varying the kernel


#%%
###############################################################################
#               Face Recognition Task
###############################################################################
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

####################################################################
# Télécharge (si nécessaire) et charge la base de données "Labeled Faces in the Wild" (LFW)
# - min_faces_per_person=70 : ne garde que les personnes avec au moins 70 photos
# - resize=0.4 : réduit la taille des images à 40 % pour accélérer le traitement
# - color=True : charge les images en couleur (3 canaux RGB)
# - funneled=False : n'utilise pas la version "alignée/funneled" du dataset
# - slice_=None : garde les images entières (pas de découpage)
# - download_if_missing=True : télécharge le dataset s'il n'est pas encore présent en local
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
# data_home='.'  # (optionnel) pour spécifier un dossier de téléchargement

# Récupération des images en tableau numpy
images = lfw_people.images

# On extrait les dimensions du tableau :
# n_samples = nombre d'images
# h = hauteur de chaque image
# w = largeur de chaque image
# n_colors = nombre de canaux (3 si RGB, 1 si niveaux de gris)
n_samples, h, w, n_colors = images.shape

# Les étiquettes cibles correspondent à l'identité de chaque personne
# target_names contient la liste des noms des personnes présentes dans le dataset
target_names = lfw_people.target_names.tolist()

####################################################################
# Choix de deux personnes à distinguer (classification binaire)
names = ['Tony Blair', 'Colin Powell']
# Exemple alternatif : ['Donald Rumsfeld', 'Colin Powell']
# Création d’un masque booléen pour sélectionner uniquement les images de la 1ère personne
idx0 = (lfw_people.target == target_names.index(names[0]))
# Création d’un masque booléen pour sélectionner uniquement les images de la 2ème personne
idx1 = (lfw_people.target == target_names.index(names[1]))
# On empile toutes les images des deux personnes sélectionnées
images = np.r_[images[idx0], images[idx1]]
# On met à jour le nombre total d’images
n_samples = images.shape[0]

# Création du vecteur des labels (y)
# - 0 pour toutes les images de la première personne
# - 1 pour toutes les images de la deuxième personne
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

# Visualisation : on affiche un échantillon de 12 images parmi celles sélectionnées

plot_gallery(images, np.arange(12))
plt.show()

#%%
####################################################################
# Extract features

# Ici, on extrait les caractéristiques en utilisant seulement l'intensité lumineuse (niveaux de gris).
# np.mean(images, axis=3) calcule la moyenne des valeurs des 3 canaux (R, G, B) → image en niveaux de gris.
# reshape(n_samples, -1) aplatit chaque image en un vecteur 1D de pixels.
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# Variante : utiliser directement les couleurs (3 canaux)
# Dans ce cas, chaque image est aplatie en un vecteur 1D contenant toutes les valeurs RGB.
# Cela donne 3 fois plus de features qu’en niveaux de gris.
# X = images.copy().reshape(n_samples, -1)

# Centrage : on soustrait la moyenne (par colonne, i.e. par pixel)
X -= np.mean(X, axis=0)
# Réduction : on divise par l’écart-type (par colonne)
# Résultat : chaque feature a une moyenne nulle et un écart-type de 1.
X /= np.std(X, axis=0)

#%%
####################################################################
# Split data into a half training and half test set
# X_train, X_test, y_train, y_test, images_train, images_test = \
#    train_test_split(X, y, images, test_size=0.5, random_state=0)
# X_train, X_test, y_train, y_test = \
#    train_test_split(X, y, tee=0.5, random_state=0)

# Fixe la graine aléatoire pour que la séparation soit toujours la même (reproductibilité)
np.random.seed(42)
# Crée une permutation aléatoire des indices des échantillons (de 0 à n_samples-1)
indices = np.random.permutation(X.shape[0])
# Découpe des indices en deux moitiés :
# - première moitié → ensemble d'apprentissage (train)
# - deuxième moitié → ensemble de test
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
# Construction des ensembles train/test à partir des indices
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
# Idem pour les images originales (utile pour visualiser quelques exemples)
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]

###############################st_siz#####################################
# Quantitative evaluation of the model quality on the test set

#%%
# Q4
#On démarre un chronomètre pour mesurer le temps d’entraînement.
print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
t0 = time()

# Liste de valeurs possibles pour C (de 10^-5 à 10^5)
Cs = 10. ** np.arange(-5, 6)
scores = []

# Pour chaque valeur de C, on entraîne un SVM linéaire
for C in Cs:
    clf = SVC(kernel='linear', C=C)  # SVM avec noyau linéaire
    clf.fit(X_train, y_train)         # apprentissage sur l’ensemble train
    score = clf.score(X_train, y_train)   # précision sur l’apprentissage
    scores.append(score)           # on stocke le score

# On identifie la valeur de C qui donne la meilleure précision
ind = np.argmax(scores)
print("Best C: {}".format(Cs[ind]))

# Visualisation de la courbe des scores d’apprentissage en fonction de C
plt.figure()
plt.plot(Cs, scores)
plt.xlabel("Parametres de regularisation C")
plt.ylabel("Scores d'apprentissage")
plt.xscale("log")      # échelle logarithmique pour mieux voir la variation de C
plt.tight_layout()
plt.show()
print("Best score: {}".format(np.max(scores)))

print("Predicting the people names on the testing set")
t0 = time()   # redémarre le chronomètre pour la prédiction


errors = []

for C in Cs:
    clf = SVC(kernel="linear", C=C)
    clf.fit(X_train, y_train)
    
    score = clf.score(X_train, y_train)  # ou sur test
    error = 1 - score   # erreur = 1 - précision
    errors.append(error)

# Courbe des erreurs en fonction de C
plt.figure()
plt.plot(Cs, errors, marker="o")
plt.xscale("log")
plt.xlabel("Paramètre de régularisation C")
plt.ylabel("Erreur de prédiction")
plt.title("Erreur en fonction de C (SVM linéaire)")
plt.tight_layout()
plt.show()


# predict labels for the X_test images with the best classifier
clf =  SVC(kernel='linear', C=Cs[ind])  # on reprend le meilleur C trouvé
clf.fit(X_train, y_train)   # réapprentissage complet
y_pred = clf.predict(X_test)   # prédictions sur l’ensemble test

print("done in %0.3fs" % (time() - t0))  # temps de calcul
# Comparaison avec le niveau aléatoire (= prédire la classe majoritaire)
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))
# Précision finale du modèle sur le test
print("Accuracy : %s" % clf.score(X_test, y_test))


####################################################################
# Qualitative evaluation of the predictions using matplotlib
# Création des titres pour chaque image prédite
# Pour chaque image de l’ensemble de test, on compare la prédiction (y_pred[i])
# avec la vraie étiquette (y_test[i]) et on génère un titre explicite.
# La fonction 'title' est probablement définie ailleurs : elle sert à afficher
# le nom de la personne prédite et indiquer si c’est correct ou incorrect.
prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

# Affichage d’une galerie d’images de test avec leurs prédictions
# Chaque image est annotée avec le titre généré ci-dessus
plot_gallery(images_test, prediction_titles)
# Affiche la figure à l’écran
plt.show()

####################################################################
# Look at the coefficients
# Crée une nouvelle figure matplotlib
plt.figure()
# Visualise les coefficients appris par le classificateur SVM linéaire
# - clf.coef_ contient les poids associés à chaque pixel (après l'apprentissage)
# - np.reshape(..., (h, w)) remet ces coefficients sous forme d'image 2D
#   (mêmes dimensions que les images originales, hauteur h et largeur w)
# Résultat : on obtient une "carte de chaleur" qui montre quels pixels 
# sont les plus discriminants pour séparer les deux classes
plt.imshow(np.reshape(clf.coef_, (h, w)))
# Affiche la figure
plt.show()


#%%
# Q5
# Fonction qui exécute un SVM avec validation croisée simple (train/test split aléatoire)
def run_svm_cv(_X, _y):
    # Mélange aléatoire des indices des échantillons
    _indices = np.random.permutation(_X.shape[0])
    # Séparation en 2 moitiés : train (50%) et test (50%)
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]
    # Définition des hyperparamètres à tester : noyau linéaire + grille de C (de 10^-3 à 10^3)
    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    # Création d’un SVM de base
    _svr = svm.SVC()
    # Recherche du meilleur hyperparamètre via GridSearchCV
    _clf_linear = GridSearchCV(_svr, _parameters)
    # Entraînement sur l’ensemble d’apprentissage
    _clf_linear.fit(_X_train, _y_train)
    # Affiche les scores (train et test) obtenus avec le meilleur C
    print('Generalization score for linear kernel: %s, %s \n' %
          (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)))

print("Score sans variable de nuisance")
# Exécution du SVM sur les données d’origine
run_svm_cv(X, y)

print("Score avec variable de nuisance")
n_features = X.shape[1]
# Ajout de variables de nuisance (bruit gaussien indépendant)
sigma = 1
noise = sigma * np.random.randn(n_samples,300, )  # bruit de dimension 300
# On concatène les vraies variables avec ce bruit
X_noisy = np.concatenate((X, noise), axis=1)
# Mélange des échantillons pour casser tout ordre
X_noisy = X_noisy[np.random.permutation(X.shape[0])]
# Réexécution du SVM sur les données bruitées
run_svm_cv(X_noisy, y)

#%%
# Q6
print("Score apres reduction de dimension")

# Nombre de composantes principales à conserver
# Ici on garde 100 composantes, mais on peut jouer avec ce paramètre pour voir l’impact
n_components = 100  # jouer avec ce parametre

# Application d’une ACP (PCA) :
# - n_components = 100 → réduction de la dimensionnalité à 100 variables principales
# - svd_solver='randomized' → méthode rapide pour de grandes matrices
# - whiten=True → normalise les composantes pour avoir une variance unitaire 
#   (utile pour stabiliser les algorithmes en aval, comme le SVM)
pca = PCA(n_components=n_components,svd_solver='randomized',whiten=True)
# Transformation des données bruitées (X_noisy) en un nouvel espace de dimension réduite
X_pca = pca.fit_transform(X_noisy)
# Réentraînement et évaluation du SVM linéaire sur ces données réduites
run_svm_cv(X_pca, y)

# %%
