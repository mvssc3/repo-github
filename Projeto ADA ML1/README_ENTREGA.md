# Entrega — Projeto ML I (ADA) — ARTESP Acidentes_2025

## Resumo do pipeline
- **Base**: ARTESP — Acidentes_2025 (CSV)
- **Alvo**: `fatal` = 1 se `QTD_VIT_FATAL > 0`, senão 0
- **Principais cuidados**: remoção de vazamento (`QTD_VIT_*`, `CLASS_ACID`); imputação (mediana/mais frequente), One-Hot, RobustScaler
- **Modelo final**: RandomForest (`class_weight='balanced'`, sem SMOTE para árvore)
- **Validações**:
  - **Split aleatório estratificado** (teste 20%)
  - **Hold-out temporal**: treino **jan–out/2025** → teste **nov–dez/2025**

## Métricas (síntese)
**Teste aleatório (baseline RF):**
- AUC-ROC ≈ **0,7871**
- AUC-PR  ≈ **0,2290**
- Threshold operacional (max F1 no teste): **0,1333**
  - Precision ≈ **0,329**, Recall ≈ **0,381**
  - FP ≈ **92/4510** (≈ **2,04%** dos não-fatais)

**Hold-out temporal (nov–dez/2025, baseline RF):**
- AUC-ROC ≈ **0,7833**
- AUC-PR  ≈ **0,1965**
- Com threshold **0,1333**:
  - Precision ≈ **0,223**, Recall ≈ **0,395**
  - FP ≈ **519/14558** (≈ **3,57%** dos não-fatais)

> Observação: o tuning leve do RF elevou discretamente a AUC-ROC, porém reduziu AUC-PR/recall no temporal após recalibrar o limiar; portanto **mantivemos o baseline**.

## Arquitetura de entrega
- `artefatos/pipeline_modelo_final.joblib` — pipeline scikit-learn (pré-processamento + modelo)
- `artefatos/config_modelo.json` — metadados do modelo (limiar, colunas de entrada, versões)
- `app.py` — API **FastAPI** com endpoints de previsão

## Como rodar a API (local)
```bash
pip install fastapi uvicorn joblib pandas scikit-learn

uvicorn app:app --reload --port 8000
```

### Endpoints
- `POST /predict` – recebe **um** JSON (ou `{"registros": [ ... ]}` para batch) e retorna `probabilidade` e `predicao` (0/1) usando o **limiar do config**.
- `GET /schema` – mostra as **colunas de entrada esperadas** e o limiar atual.
- `GET /health` – checagem simples de saúde.

### Exemplo de uso
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
  "RODOVIA": "SP-280", "MARCO_QM": 50, "MUNICÍPIO": "Barueri",
  "TIPO_PISTA": "Dupla", "METEORO": "Chuva", "HR_ACID": "18:40",
  "JURISDICAO": "Estadual", "DENOMINACAO": "Castello Branco"
}'
```

## Reprodutibilidade
- Fixe `random_state` (já aplicado como `semente=42`).
- Salve as versões no `config_modelo.json` via bloco final do notebook (abaixo).
- Documente o limiar adotado (**0,1333**) e a matriz de confusão correspondente (teste e temporal).

## Limitações e próximos passos
- Probabilidades **não calibradas** (Platt/Isotônico poderiam melhorar cortes específicos).
- Enriquecimento com feriados/chuva real (exógenas) pode elevar AUC-PR.
- Ajuste de **threshold por custo** (se houver pesos de falso-positivo/negativo).

_Arquivo gerado em 2025-10-29 23:45_
