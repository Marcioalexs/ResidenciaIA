# Artefatos treinados do C1NC0

A publicação atual usa três artefatos de runtime:

- `gaussian_nb_pipeline.joblib` — Gaussian Naive Bayes com 19 features estruturais/linguísticas;
- `tfidf_vectorizer.joblib` — vetorizador de título + texto;
- `multinomial_nb.joblib` — Multinomial Naive Bayes textual.

Também são gerados:

- `model_metadata.json` — métricas e metodologia do treinamento;
- `dataset_info.json` — composição do dataset usada pela interface;
- `training_manifest.json` — rastreabilidade e hashes dos artefatos.

Para a atualização completa, execute na raiz de `c1nc0-confiabilidade`:

```powershell
python .\scripts\gerar_pacote_treinamento_c1nc0.py `
  --dataset "C:\caminho\dataset_crawler_ptbr_consolidado_revC1NC0.csv"
```

Por padrão, o script:

1. usa somente `extracao_elegivel=True`, quando a coluna existe;
2. normaliza os rótulos para `true` / `fake`;
3. remove repetições da mesma URL;
4. mantém URLs diferentes com conteúdo exatamente igual;
5. agrupa conteúdos iguais para que nunca apareçam simultaneamente em treino e teste;
6. regenera os modelos e os artefatos B1–B5.

O CSV completo não é publicado no Vercel.
