import pandas as pd
import time, pickle

from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


import mlflow
import mlflow.lightgbm
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, \
    recall_score, f1_score, classification_report

z = ZipFile("./data_train.zip")
data_train = pd.read_csv(z.open('data_train.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
#data_train.drop('SK_ID_CURR', axis=1, inplace=True)
#print(data_train.shape)

TARGET = pd.read_csv('./TARGET.csv', index_col='SK_ID_CURR')
#print(TARGET.shape)

X_train, X_test, y_train, y_test = train_test_split(data_train.values, TARGET.values, test_size=0.3, random_state=42)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)




def train_models(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    output = {
        'AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        'Accuracy': accuracy_score(y_test, model.predict(X_test)),
        'Precision': precision_score(y_test, model.predict(X_test)),
        'Recall': recall_score(y_test, model.predict(X_test)),
        'F1': f1_score(y_test, model.predict(X_test))
    }
    return output


def train_lgbm_with_mlflow(params):
    # Démarrer une expérience MLflow
    with mlflow.start_run():
        # Entraîner et évaluer le modèle
        model = LGBMClassifier(**params)
        results = train_models(model, X_train, X_test, y_train, y_test)

        # Enregistrer les métriques dans MLflow
        for metric_name, metric_value in results.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.lightgbm.log_model(model, "model")

# Paramètres
params = {
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "max_depth": 5
}

#train_lgbm_with_mlflow(params)

param_grid = {
    'num_leaves': [31, 50, 75, 100],
    'min_data_in_leaf': [10, 20, 30, 40],
    'max_depth': [4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    # ajoutez d'autres paramètres si nécessaire
}


grid_model = LGBMClassifier()

grid = GridSearchCV(
    estimator=grid_model,
    param_grid=param_grid,
    cv=3,  # nombre de folds pour la validation croisée
    scoring='roc_auc',  # vous pouvez choisir d'autres métriques comme 'accuracy'
    verbose=1,
    n_jobs=-1  # utilise tous les cœurs disponibles
)


grid_result = grid.fit(X_train, y_train.ravel())
best_params = grid_result.best_params_
print("Meilleurs paramètres:", best_params)

train_lgbm_with_mlflow(best_params)


#mlflow ui