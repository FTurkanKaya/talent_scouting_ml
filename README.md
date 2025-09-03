# Talent Scouting met Machine Learning

Dit project gebruikt machine learning technieken om het potentieel van voetballers te classificeren. Het doel is om op basis van de scores die scouts aan spelers geven, hun potentieel label (average, highlighted) te voorspellen.  

---

## Dataset

Gebruikte datasets:

- `scoutium_attributes.csv` : Eigenschappen van spelers en scores gegeven door scouts.
- `scoutium_potential_labels.csv` : Potentieel label van spelers (average, highlighted, below_average).

Verwerkingsstappen:

1. CSV-bestanden inlezen en samenvoegen.
2. Doelmannen (`position_id = 1`) en het `below_average` label verwijderen.
3. Pivot table maken zodat elke rij overeenkomt met één speler.
4. `potential_label` omzetten naar numerieke waarden (Label Encoding).

---

## Data Voorbewerking

- Ontbrekende waarden zijn ingevuld met de mediaan.
- Numerieke kolommen (`num_cols`) zijn geschaald met StandardScaler.

---

## Modellen

Verschillende machine learning algoritmes zijn gebruikt om het potentieel van spelers te voorspellen:

- Logistic Regression (LR)
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Decision Tree (CART)
- Random Forest (RF)
- AdaBoost
- Gradient Boosting (GBM)
- XGBoost
- LightGBM

---

## Prestaties en Hyperparameter Optimalisatie

ROC AUC scores voor de modellen vóór en na hyperparameter optimalisatie:
########## KNN ##########
roc_auc (Before): 0.7599
roc_auc (After): 0.7599
KNN best params: {'n_neighbors': 5}

########## CART ##########
roc_auc (Before): 0.6833
roc_auc (After): 0.7236
CART best params: {'max_depth': 1, 'min_samples_split': 2}

########## RF ##########
roc_auc (Before): 0.8867
roc_auc (After): 0.8968
RF best params: {'max_depth': None, 'max_features': 5, 'min_samples_split': 20, 'n_estimators': 200}

########## XGBoost ##########
roc_auc (Before): 0.8308
roc_auc (After): 0.8648
XGBoost best params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 200}

########## LightGBM ##########
roc_auc (Before): 0.8502
roc_auc (After): 0.8389
LightGBM best params: {'learning_rate': 0.01, 'n_estimators': 300}

## Resultaten

- Random Forest en XGBoost hebben de hoogste ROC AUC scores behaald.
- Hyperparameter optimalisatie verbeterde de prestaties van sommige modellen.
- Met deze modellen kan het potentieel van spelers voorspeld worden.

---
## Visualisatie

Feature importance kan worden weergegeven met een staafdiagram.

ROC, Precision en Recall metrics kunnen worden gebruikt om modelprestaties te evalueren.

<img width="1000" height="600" alt="feature_importance" src="https://github.com/user-attachments/assets/00eb5ff5-a214-407d-805a-89d54dfedca6" />
