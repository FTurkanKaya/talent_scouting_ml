import os
import numpy as np
import pandas as pd
import warnings

from win32comext.adsi.demos.scp import verbose

warnings.filterwarnings("ignore")

import joblib
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import os
# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# =========================
# Adım 1–2: CSV'leri oku ve merge et
# =========================
base_path = os.path.join(os.getcwd(), "..", "HW_05")
file_attributes = "scoutium_attributes.csv"
file_potential_labels = "scoutium_potential_labels.csv"

def load_csv(path):
    return pd.read_csv(path, sep=";")

df_attributes = load_csv(os.path.join(base_path, file_attributes))
df_potential = load_csv(os.path.join(base_path, file_potential_labels))

df = pd.merge(
    df_attributes,
    df_potential,
    on=["task_response_id", "match_id", "evaluator_id", "player_id"],
    how="inner"
)

# =========================
# Adım 3: Kaleci (position_id = 1) verilerini çıkar
# =========================
df = df[df["position_id"] != 1]

# =========================
# Adım 4: below_average sınıfını çıkar
# (geriye 'average' ve 'highlighted' kalacak)
# =========================
df = df[df["potential_label"] != "below_average"]

# =========================
# Adım 5: Pivot tablo
# index: player_id, position_id, potential_label
# columns: attribute_id
# values: attribute_value (mean)
# =========================
df_pivot = pd.pivot_table(
    df,
    index=["player_id", "position_id", "potential_label"],
    columns="attribute_id",
    values="attribute_value",
    aggfunc="mean"
).reset_index()

# Sütun adlarını string yap (attribute_id'ler için garanti)
df_pivot.columns = df_pivot.columns.map(str)

# =========================
# Adım 6: potential_label -> sayısal (LabelEncoder)
# =========================
le = LabelEncoder()
df_pivot["potential_label_encoded"] = le.fit_transform(df_pivot["potential_label"])

print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
# Beklenen: {'average': 0, 'highlighted': 1}


# =========================
# Adım 7: Sayısal kolonlar (num_cols)
# - player_id gibi kimlikler sayısal olsa da model için isteğe bağlıdır.
# - Talep: tüm sayısal kolonları num_cols'a koyup ölçekleyelim.
# - potential_label_encoded hedef, onu num_cols'a dahil ETME.
# =========================
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df_pivot, cat_th=5, car_th=20)

num_cols = [c for c in num_cols if c != "potential_label_encoded"]  # player_id ve position_id dahildir

# =========================
# Adım 8: Impute (median) + StandardScaler
# =========================

imputer = SimpleImputer(strategy="median")
df_pivot[num_cols] = imputer.fit_transform(df_pivot[num_cols])

scaler = StandardScaler()
df_pivot[num_cols] = scaler.fit_transform(df_pivot[num_cols])

# =========================
# Adım 9: Modelleme (temel modeller + metrikler)
# =========================
X = df_pivot[num_cols]
y = df_pivot["potential_label_encoded"]

def base_models(X, y, scoring=("accuracy", "f1", "precision", "recall", "roc_auc")):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier(verbose = -1)),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"\n{name}")
        for metric in scoring:
            score_name = f"test_{metric}"
            print(f"   {metric}: {cv_results[score_name].mean():.4f}")

base_models(X, y)


######################################################
# 4. Automated Hyperparameter Optimization
######################################################


knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}



classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(verbose = -1), lightgbm_params)]


def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)
"""
Hyperparameter Optimization....
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
"""

# =========================
# Adım 10: Özellik önemleri (feature importance)
# Ağaç tabanlı bir model (RandomForest) i
# =========================
rf = RandomForestClassifier(random_state=42).fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

import matplotlib
matplotlib.use("Qt5Agg")
top_n = 20  # ilk 20 önemli özellik
plt.figure(figsize=(10, 6))
importances.head(top_n).iloc[::-1].plot(kind="barh")
plt.title(f"RandomForest Feature Importances (Top {top_n})")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()