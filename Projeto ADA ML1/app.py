from fastapi import FastAPI
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import numpy as np
import json, os

ART_DIR = os.path.join(os.path.dirname(__file__), "artefatos")
MODEL_PATH = os.path.join(ART_DIR, "pipeline_modelo_final.joblib")
CONFIG_PATH = os.path.join(ART_DIR, "config_modelo.json")

app = FastAPI(title="ARTESP 2025 — Fatalidade API", version="1.0.0")

# Carrega modelo e config
try:
    modelo = joblib.load(MODEL_PATH)
except Exception as e:
    modelo = None

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
except Exception:
    config = {}

LIMIAR = float(config.get("limiar_operacional", 0.1333))
COLUNAS_ENTRADA = list(config.get("colunas_entrada", []))

@app.get("/health")
def health():
    ok = modelo is not None and isinstance(COLUNAS_ENTRADA, list)
    return {"ok": ok, "limiar": LIMIAR, "n_campos": len(COLUNAS_ENTRADA)}

@app.get("/schema")
def schema():
    return {"limiar_operacional": LIMIAR, "colunas_entrada": COLUNAS_ENTRADA}

def _padroniza_payload(payload):
    if not COLUNAS_ENTRADA:
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
        return pd.DataFrame(payload)

    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
    else:
        df = pd.DataFrame(payload)

    for c in COLUNAS_ENTRADA:
        if c not in df.columns:
            df[c] = None
    df = df[COLUNAS_ENTRADA]
    return df

@app.post("/predict")
def predict(payload: dict):
    if modelo is None:
        return JSONResponse(status_code=500, content={"erro": "Modelo não carregado."})

    if "registros" in payload and isinstance(payload["registros"], list):
        X = _padroniza_payload(payload["registros"])
        probs = modelo.predict_proba(X)[:, 1]
        preds = (probs >= LIMIAR).astype(int)
        return {"predicoes": preds.tolist(), "probabilidades": probs.tolist(), "limiar_usado": LIMIAR, "n": len(preds)}
    else:
        X = _padroniza_payload(payload)
        prob = float(modelo.predict_proba(X)[:, 1][0])
        pred = int(prob >= LIMIAR)
        return {"predicao": pred, "probabilidade": prob, "limiar_usado": LIMIAR}