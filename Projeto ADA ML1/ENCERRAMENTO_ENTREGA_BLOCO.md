**Modelo final adotado:** `RandomForest (class_weight='balanced', sem SMOTE)` — *baseline* mais estável.  
**Limiar operacional:** **0,1333** (ponto de melhor F1 no teste), mantido no *hold-out* temporal.

**Síntese de métricas**  
- **Teste aleatório:** AUC-ROC ≈ 0,787; AUC-PR ≈ 0,229; no limiar 0,1333 → *precisão* ≈ 0,329; *recall* ≈ 0,381.  
- **Hold-out temporal (nov–dez/2025):** AUC-ROC ≈ 0,783; AUC-PR ≈ 0,196; no limiar 0,1333 → *precisão* ≈ 0,223; *recall* ≈ 0,395; *FPR* ≈ 3,57%.

**Justificativa:** O *grid search* elevou discretamente o AUC-ROC, porém reduziu AUC-PR/recall no temporal após recalibrar o limiar, então foi mantido o *baseline*.


```python
# === Salvar artefatos finais (pipeline + config) ===
import os, json, joblib, platform, sklearn, pandas as pd, numpy as np
from datetime import datetime

# 1) Definições finais
modelo_final = melhor_pipe                # baseline RF escolhido
LIMIAR_OPERACIONAL = 0.1333               # ponto de melhor F1 no teste
CAMINHO_ARTEFATOS = "artefatos"
os.makedirs(CAMINHO_ARTEFATOS, exist_ok=True)

# 2) (Opcional) re-treino em todo o conjunto se desejar "modelo de produção"
# modelo_final.fit(pd.concat([X_treino, X_teste]), pd.concat([y_treino, y_teste]))

# 3) Salvar pipeline
caminho_modelo = os.path.join(CAMINHO_ARTEFATOS, "pipeline_modelo_final.joblib")
joblib.dump(modelo_final, caminho_modelo)
print(f"Pipeline salvo em: {caminho_modelo}")

# 4) Colunas de entrada esperadas
colunas_entrada = list(X_treino.columns)

# 5) Capturar versões
versoes = {
    "python": platform.python_version(),
    "scikit_learn": getattr(sklearn, "__version__", "unknown"),
    "pandas": getattr(pd, "__version__", "unknown"),
    "numpy": getattr(np, "__version__", "unknown"),
}

# 6) Config JSON
config = {
    "criacao_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "limiar_operacional": float(LIMIAR_OPERACIONAL),
    "colunas_entrada": colunas_entrada,
    "versoes": versoes,
    "notas": "RandomForest baseline; class_weight=balanced; sem SMOTE; validação temporal nov–dez/2025.",
}
caminho_config = os.path.join(CAMINHO_ARTEFATOS, "config_modelo.json")
with open(caminho_config, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
print(f"Config salvo em: {caminho_config}")

# 7) Checagem rápida
print("Exemplo de campos de entrada (top 10):", colunas_entrada[:10])

```