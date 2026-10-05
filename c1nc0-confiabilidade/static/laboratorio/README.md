# Artefatos da Etapa B — C1NC0

Esta pasta recebe os JSONs pré-calculados pelo script:

```powershell
python .\scripts\gerar_artefatos_etapa_b.py --dataset "C:\caminho\03_Dataset_Final_C1NC0-Tabular.csv"
```

O `/laboratorio` no Vercel **não treina modelos**. Ele apenas lê:

- `dataset-summary.json` — B1;
- `model-comparison.json` — B2;
- `confusion-matrix.json` e `error-examples.json` — B3;
- `model-card.json` — B4;
- `clusters.json` — B5;
- `manifest.json` — rastreabilidade da geração.

Os JSONs devem ser gerados localmente com o dataset real e versionados junto com a branch da Etapa B.
