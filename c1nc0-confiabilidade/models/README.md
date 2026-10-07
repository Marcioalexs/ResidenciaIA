# Artefatos treinados do C1NC0

Para uma atualização completa do C1NC0 com um novo dataset, use o gerador unificado:

```powershell
python .\scripts\gerar_pacote_treinamento_c1nc0.py `
  --dataset "C:\caminho\dataset_crawler_ptbr_consolidado.csv"
```

Por padrão, se existir a coluna `extracao_elegivel`, somente registros com valor verdadeiro entram no treinamento.

O script atualiza aqui:

- `gaussian_nb_pipeline.joblib`
- `tfidf_vectorizer.joblib`
- `multinomial_nb.joblib`
- `model_metadata.json`
- `dataset_info.json`
- `training_manifest.json`

Também regenera os JSONs B1–B5 em `static/laboratorio/` e cria um ZIP em `dist/` com os arquivos que devem ser versionados.

O dataset CSV completo não deve ser publicado no Vercel.
