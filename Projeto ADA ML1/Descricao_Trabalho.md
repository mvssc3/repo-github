Escolher uma base desafiadora, de preferência rotulada (aprendizado. supervisionado) binária. Aceita problemas de regressão
Contar o StoryTelling - qual problema a base traz, o que fizeram para resolver esse problema e os resultados encontrados.

Trabalhar as etapas da pipeline, organizando o notebook por makdowons

1. Import de lib
2. Carregamento da base
3. Análise Exploratória
	- Balancemanto de classes (histograma de classe)
	- Análise de Correlação (heatmap)
	- Distribuição Estatística das Features (histplot, boxplot etc)
	- Verificar dados ausentes, not a number, dados categóricos etc
4. Pré-processamento (Preferência usar a classe Pipeline do scikitlearn com ColumnTransform)
	- Tratamento de dados ausentes, not a number e dados categóricos (one-hot-encoding)
	- Balanceamento de classes
	- Normalizaçao e Padronização (Ajuste de distruibuições e escala de dados)
	- Análise de Componentes Princiapis (Reduzir Dimensionalidade)
5. Treinamento, Validação e Teste (Modelo: Knn, Árvores de Decisão, Random Forest, SVM, XBoost, LightGBM, Bayesiano etc)
	- Split de Dados
	- Treinamento e Validação (Cross-Validation)
	- Teste
6. Otimização de Modelos (Fine Tuning)
	- Ajustar hiperparâmetros de modelos complexos (XBoost, SVM, Random Forest etc) com técnicas como Grid Search, Random Search, Bayesian Search
7. Avaliação de Modelos
	- Gerar Matriz de Confusão e Métricas
    - Para problemas multi classe utilizar estratégias adequadas (One-Vs-One, One-Vs-All, All-Vs-All)
8. Comporativo de Performances
	- Gerar a Curva ROC e calcuar AuC (Apenas para problemas binários)
9. Explicabilidade - Extra 1
	- Rodar técnicas para explicar a decisão do modelo (Shap Value, Lime)
10. Produtização - Extra 2
    - Serializar o modelo no formato json, servir o modelo utilizando um webserver simples em python (FastAPI ou flask) e se possível criar uma instância em um pod conteinerizado (docker), para predizer um exemplo em produção.