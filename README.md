
# Apprentissage Statistique – Rapport & Codes

Ce dépôt contient le rapport du projet **Support Vector Machines (SVM)** ainsi que les codes Python utilisés pour les expérimentations.

---

## 📂 Contenu du dépôt
- `tpSVM.tex` → fichier principal du rapport LaTeX  
- `rapport.cls` → fichier de classe LaTeX personnalisé  
- `logos/` → dossier contenant les images et logos utilisés dans le rapport  
- `*.py` → scripts Python pour les expériences (SVM, PCA, etc.)  

---

## 🖋️ Rapport LaTeX

### Compilation locale
Pour compiler le rapport en PDF avec **VS Code** ou en terminal :

1. Installer une distribution **LaTeX complète** :
   - **Ubuntu/Debian** :  
     ```bash
     sudo apt install texlive-full
     ```
   - **Windows** : installer [MiKTeX](https://miktex.org/download)  
   - **MacOS** : installer [MacTeX](https://tug.org/mactex/)  

2. Compiler avec `latexmk` :  
   ```bash
   latexmk -pdf tpSVM.tex


Dépendances

Les scripts Python nécessitent les packages suivants :

---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time
import warnings
---
Installation

Créer un environnement virtuel et installer les dépendances :
---
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
---
pip install numpy matplotlib scikit-learn

