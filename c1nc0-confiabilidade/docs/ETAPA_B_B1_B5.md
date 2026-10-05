# C1NC0 — Etapa B / Incrementos B1–B5

Branch alvo: `feature/c1nc0-lab-entenda-ia-v1`

## Arquitetura

A Etapa B foi organizada para evitar treinamento pesado no Vercel:

1. o dataset completo é processado **localmente**;
2. o script `scripts/gerar_artefatos_etapa_b.py` gera JSONs estáticos;
3. o `/laboratorio` apenas lê e apresenta esses JSONs;
4. nenhuma previsão é apresentada como prova de veracidade factual.

## Incrementos

### B1 — Conheça os dados

`dataset-summary.json` contém:

- quantidade de registros e colunas;
- distribuição de classes;
- origem/fonte do dataset quando a coluna é reconhecida;
- fallback para domínio das URLs quando não há coluna explícita de origem;
- ausentes, inclusive tokens textuais como `None`, `null`, `nan` e vazio;
- linhas duplicadas;
- títulos e textos repetidos;
- grupos de conteúdo exato normalizado;
- grupos que contêm rótulos conflitantes;
- cobertura de título, texto, URL e das 19 features estruturais/linguísticas.

### B2 — Compare os modelos

`model-comparison.json` compara, no mesmo split agrupado:

- GaussianNB com 19 features estruturais/linguísticas;
- MultinomialNB com título + texto / TF-IDF;
- Logistic Regression com título + texto / TF-IDF.

Métricas:

- Accuracy;
- Precision da classe de foco (`fake`, quando existente);
- Recall da classe de foco;
- F1 da classe de foco;
- Balanced Accuracy.

O split usa `StratifiedGroupKFold`, mantendo grupos de conteúdo exato inteiros em treino ou teste.

### B3 — Onde os modelos erram?

Arquivos:

- `confusion-matrix.json`;
- `error-examples.json`.

São registrados:

- matrizes de confusão;
- total de falsos positivos;
- total de falsos negativos;
- exemplos reais do conjunto de teste com título, trecho e URL quando disponível.

### B4 — Audite nossa IA

`model-card.json` registra:

- arquivo e SHA-256 do dataset;
- versão das bibliotecas;
- representações;
- estratégia de validação;
- modelos e métricas;
- limitações;
- usos recomendados;
- seção destacada **“O que este modelo NÃO faz”**.

### B5 — Descubra padrões

`clusters.json` contém:

- K-Means sem uso dos rótulos TRUE/FAKE durante o agrupamento;
- `k=2` por padrão, configurável na CLI;
- PCA para projeção 2D;
- amostra visual determinística para manter o JSON leve;
- composição TRUE/FAKE guardada separadamente para ser revelada somente após a ação do usuário.

Quando as 19 features existem, elas são imputadas/padronizadas e usadas no K-Means/PCA. Se não existirem, o gerador usa TF-IDF + TruncatedSVD como representação de entrada e PCA para a visualização 2D.

## Gerar artefatos localmente

A partir de `c1nc0-confiabilidade`:

```powershell
python .\scripts\gerar_artefatos_etapa_b.py `
  --dataset "C:\caminho\03_Dataset_Final_C1NC0-Tabular.csv"
```

Saída padrão:

```text
static/laboratorio/
  manifest.json
  dataset-summary.json
  model-comparison.json
  confusion-matrix.json
  error-examples.json
  model-card.json
  clusters.json
```

Opções úteis:

```powershell
python .\scripts\gerar_artefatos_etapa_b.py `
  --dataset "C:\caminho\dataset.csv" `
  --random-state 42 `
  --test-size 0.20 `
  --max-tfidf-features 100000 `
  --error-examples 6 `
  --clusters 2 `
  --cluster-points 2000
```

## Validação local

```powershell
python -m py_compile app.py services\laboratorio_service.py scripts\gerar_artefatos_etapa_b.py
python app.py
```

Abrir:

```text
http://127.0.0.1:5000/laboratorio
```

Conferir principalmente:

1. B1 mostra classes, origens, duplicatas e ausentes;
2. B2 mostra as cinco métricas e o mesmo split;
3. B3 mostra matrizes e exemplos do conjunto de teste;
4. B4 destaca “O que este modelo NÃO faz”;
5. B5 abre colorido por **grupo**, sem revelar TRUE/FAKE;
6. o botão “Revelar TRUE/FAKE” troca a leitura visual e libera a composição dos grupos.

## Git

Na raiz do repositório:

```powershell
git status
git switch feature/c1nc0-lab-entenda-ia-v1
git pull --ff-only origin feature/c1nc0-lab-entenda-ia-v1
```

Depois de copiar os arquivos desta entrega e gerar os JSONs com o dataset real:

```powershell
git add c1nc0-confiabilidade/app.py
git add c1nc0-confiabilidade/services/laboratorio_service.py
git add c1nc0-confiabilidade/templates/laboratorio.html
git add c1nc0-confiabilidade/scripts/gerar_artefatos_etapa_b.py
git add c1nc0-confiabilidade/static/laboratorio/
git add c1nc0-confiabilidade/docs/ETAPA_B_B1_B5.md

git diff --cached --stat
git status

git commit -m "feat: implement C1NC0 lab stage B1-B5"
git push -u origin feature/c1nc0-lab-entenda-ia-v1
```

## Vercel

A branch `feature/c1nc0-lab-entenda-ia-v1` deve gerar um **Preview Deployment** quando o projeto está conectado ao GitHub. Antes de considerar a Etapa B validada, abrir o preview e testar `/laboratorio`.

O deploy web não precisa do CSV original: precisa apenas dos JSONs gerados e versionados em `static/laboratorio/`.
