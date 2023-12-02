from fastapi import FastAPI, HTTPException
import uvicorn
import pandas as pd
import pickle

app = FastAPI()
sample = pd.read_csv('X_sample.csv', index_col='SK_ID_CURR', encoding ='utf-8')
data = pd.read_csv('default_risk.csv', index_col='SK_ID_CURR', encoding ='utf-8')

with open('LGBMClassifier.pkl', 'rb') as f:
    clf = pickle.load(f)

@app.post("/client_info/{id_client}")
async def client_info(id_client: int):
    row_client = data[data.index == int(id_client)]

    if row_client.empty:
        raise HTTPException(status_code=404, detail="Client not found")

    client_age = int(round((row_client["DAYS_BIRTH"] /365).values[0], 0))
    Family_status = row_client["NAME_FAMILY_STATUS"].values[0]
    Number_of_children = row_client["CNT_CHILDREN"].values[0]

    info_dict = {

        "Age du client": int(client_age),
        "Situation familiale": Family_status,
        "Nombre d'enfants": int(Number_of_children)
    }

    return info_dict




@app.post("/predict/{id_client}")
async def predict(id_client: int):
    X = sample.iloc[:, :-1]
    proba = clf.predict_proba(X[X.index == int(id_client)])[:, 1]

    prediction_dict = {
        'id_client': int(id_client),
        'probability': float(proba[0]),
    }

    return prediction_dict

@app.get("/predict/{id_client}")
async def predict(id_client: int):
    X = sample.iloc[:, :-1]
    proba = clf.predict_proba(X[X.index == int(id_client)])[:, 1]

    prediction_dict = {
        'id_client': int(id_client),
        'probability': float(proba[0]),  # Indexez seulement la première dimension
    }

    return prediction_dict

#/predict/196288
#/client_info/196288

#run command
# uvicorn main:app --reload