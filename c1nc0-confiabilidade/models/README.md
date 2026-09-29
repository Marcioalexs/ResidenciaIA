# Artefatos treinados

Execute `python -m ml.train_models "CAMINHO_DO_DATASET.csv"` na raiz de `c1nc0-confiabilidade`.
O script gera aqui:
- gaussian_nb_pipeline.joblib
- tfidf_vectorizer.joblib
- multinomial_nb.joblib
- model_metadata.json

Os `.joblib` precisam estar versionados para o deployment usar exatamente os modelos treinados.
