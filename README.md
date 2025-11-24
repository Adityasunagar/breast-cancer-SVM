# breast-cancer-SVM

Support Vector Machines (SVM) Classification – Breast Cancer Dataset

This project demonstrates Support Vector Machines (SVM) for binary classification using the Breast Cancer Wisconsin Diagnostic Dataset.
The workflow includes preprocessing, training Linear & RBF SVM, hyperparameter tuning, cross-validation, and decision boundary visualization.

1. Open in Codespaces
  --Click Code → Create Codespace on main.

2. Install dependencies
  --Run in terminal:
  -- "pip install -r requirements.txt "

3. Run the notebook (optional)
  --jupyter notebook --ip 0.0.0.0 --no-browser

**Tools and Libraries
  --Python
  --Pandas
  --NumPy
  --Matplotlib
  --Scikit-learn
  --GitHub Codespaces
  --Import Libraries

  """import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
      from sklearn.preprocessing import StandardScaler
      from sklearn.svm import SVC
      from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
      from matplotlib.colors import ListedColormap """

**Dataset Description
  The dataset used is the Breast Cancer Wisconsin Diagnostic Dataset.

    Target column (y):
    diagnosis
    M = Malignant
    B = Benign

  Converted to:
    
    1 = Malignant
    0 = Benign

Feature columns (X):

    All columns except:
   -- id
   -- diagnosis


The dataset includes 30 numeric features such as:

    radius_mean
    texture_mean
    perimeter_mean
    area_worst
    concavity_se
    symmetry_worst
    and more.

Project Workflow
1. Load and Prepare Dataset

    Drop id

    Convert diagnosis to numeric codes

    Standardize features using StandardScaler

2. Split Dataset

    Use 80% training and 20% testing.

3. Train SVM Models

    Train two classifiers:

    Linear SVM: SVC(kernel='linear')

    RBF SVM: SVC(kernel='rbf')

4. Model Evaluation

    Metrics used:

    Accuracy

    Confusion Matrix

    Classification Report

5. Hyperparameter Tuning

    Tune:

      C (margin hardness)

      gamma (RBF kernel spread)

      Using GridSearchCV with 5-fold cross-validation.

6. Decision Boundary Visualization

      Using only 2 features:
      radius_mean
      texture_mean
      Visualization performed using meshgrid + contour plot.
      Results Summary
      Linear SVM works well for linearly separable data.
      RBF SVM captures complex non-linear boundaries and often gives higher accuracy.
      Best model found by GridSearchCV typically uses:
      medium C values
      gamma tuned to dataset size
      Feature scaling significantly improves performance.

    <img width="689" height="547" alt="97f4f253-41f8-4159-9e32-67680a1ff0c7" src="https://github.com/user-attachments/assets/85c371ba-49a9-4f5d-b451-10e57521ef58" />


**Project Structure
      svm-breast-cancer/
  │
  ├── README.md
  ├── breast-cancer.csv
  ├── 2d boundary.png
  ├── breast_cancer.ipynb
  └── requirements.txt

requirements.txt

--pandas
--numpy
--matplotlib
--scikit-learn
--jupyter
