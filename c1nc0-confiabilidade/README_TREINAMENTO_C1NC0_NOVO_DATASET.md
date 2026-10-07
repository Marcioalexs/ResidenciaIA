# C1NC0 — treinamento com novo dataset

## Objetivo

Gerar, em uma única execução, todos os artefatos necessários para atualizar o C1NC0 com o novo dataset consolidado do crawler.

## Arquivos deste pacote

Copie os arquivos mantendo a estrutura de pastas:

- `scripts/gerar_pacote_treinamento_c1nc0.py` — novo gerador unificado;
- `ml/model_service.py` — passa a carregar dinamicamente `models/dataset_info.json`;
- `templates/index.html` — deixa de exibir as origens antigas de forma fixa;
- `models/README.md` — instrução atualizada.

## Preparação

Na raiz de `c1nc0-confiabilidade`:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-offline.txt
```

## Executar com o CSV consolidado

```powershell
python .\scripts\gerar_pacote_treinamento_c1nc0.py `
  --dataset "C:\CAMINHO\dataset_crawler_ptbr_consolidado.csv"
```

Por padrão, quando a coluna `extracao_elegivel` existe, o script usa somente os registros elegíveis.
No dataset consolidado atual, isso significa aproximadamente 33.618 registros de treinamento, com os rótulos normalizados para `true` e `fake`.

## O que o script gera

### Runtime do site (`models/`)

- `gaussian_nb_pipeline.joblib`
- `tfidf_vectorizer.joblib`
- `multinomial_nb.joblib`
- `model_metadata.json`
- `dataset_info.json`
- `training_manifest.json`

A Logistic Regression é usada apenas na auditoria/comparação B2 e não é publicada como modelo de runtime para evitar aumentar o bundle do Vercel.

### Laboratório B1–B5 (`static/laboratorio/`)

- `manifest.json`
- `dataset-summary.json`
- `model-comparison.json`
- `confusion-matrix.json`
- `error-examples.json`
- `model-card.json`
- `clusters.json`

### Pacote final (`dist/`)

O script cria automaticamente:

```text
c1nc0-modelos-novo-dataset-AAAAMMDD-HHMMSS.zip
```

Esse ZIP contém somente os modelos, metadados, JSONs do laboratório e arquivos-fonte necessários para a atualização do C1NC0. O CSV completo não é incluído.

## Validações automáticas

O script interrompe a execução se:

- faltar uma das 19 features esperadas pelo GaussianNB;
- as classes não puderem ser convertidas para `true`/`fake`;
- faltar título ou texto;
- houver valores não reconhecidos na coluna `extracao_elegivel`;
- os modelos serializados não tiverem as classes `true` e `fake`;
- faltar algum JSON obrigatório B1–B5.

## Observação sobre tempo

Com o dataset completo, a etapa de TF-IDF e principalmente a geração B1–B5 podem levar vários minutos. Isso é esperado: todo o processamento pesado ocorre localmente e o Vercel recebe apenas os artefatos prontos.
