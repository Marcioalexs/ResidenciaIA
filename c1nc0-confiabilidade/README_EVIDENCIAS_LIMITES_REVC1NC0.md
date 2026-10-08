# C1NC0 — Evidências e Limites explicados + retreinamento revC1NC0

Esta atualização tem dois objetivos coordenados:

1. tornar **O que observamos** e **O que NÃO sabemos** mais claros para pessoas sem conhecimento técnico;
2. atualizar o treinamento supervisionado com `dataset_crawler_ptbr_consolidado_revC1NC0.csv`.

## 1. O que muda na interface

Os itens de **O que observamos** passam a separar:

- o que foi encontrado na página;
- como aquela informação deve ser interpretada;
- a ressalva de que o indício não prova veracidade.

Exemplos:

- a data ISO é apresentada como `dd/mm/aaaa`;
- a distância em relação ao dia da análise é descrita em linguagem humana;
- links externos são apresentados como caminhos de checagem, não como prova;
- dados estruturados são descritos como informações fornecidas pelo próprio site;
- coerência de títulos é traduzida em alta, moderada ou baixa semelhança.

Os itens de **O que NÃO sabemos** passam a mostrar:

- qual conclusão o C1NC0 não pode tirar;
- como a pessoa pode reduzir aquela incerteza com checagem humana.

## 2. Regra para a data

A data é contextualizada como:

- hoje;
- ontem;
- há poucos dias;
- há aproximadamente alguns meses;
- há mais de um ano;
- ou, se estiver no futuro, como uma informação que merece conferência.

Recência **não é usada como prova de confiabilidade**.

## 3. Preparação do dataset revC1NC0

O script `scripts/gerar_pacote_treinamento_c1nc0.py`:

1. detecta separador e encoding;
2. normaliza rótulos para `true` e `fake`;
3. usa `extracao_elegivel=True` por padrão;
4. remove repetição da mesma URL;
5. mantém URLs diferentes com o mesmo conteúdo;
6. agrupa conteúdo exatamente igual para nunca atravessar treino/teste;
7. treina GaussianNB e MultinomialNB;
8. calcula LogisticRegression somente para comparação do Laboratório;
9. gera `dataset_info.json`, `model_metadata.json` e `training_manifest.json`;
10. regenera B1–B5;
11. cria um ZIP em `dist/`.

Para o `dataset_crawler_ptbr_consolidado_revC1NC0.csv`, os valores de controle esperados são:

- arquivo original: 48.484 registros;
- após `extracao_elegivel=True`: 34.872;
- repetições da mesma URL removidas: 973;
- conjunto final de treinamento: 33.899;
- TRUE: 22.511;
- FAKE: 11.388;
- sobreposição de grupos entre treino/teste: 0.

## 4. Treinamento

A partir de `c1nc0-confiabilidade`:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\scripts\gerar_treinamento_c1nc0.ps1 `
  -Dataset "C:\caminho\dataset_crawler_ptbr_consolidado_revC1NC0.csv"
```

Alternativa direta:

```powershell
python .\scripts\gerar_pacote_treinamento_c1nc0.py `
  --dataset "C:\caminho\dataset_crawler_ptbr_consolidado_revC1NC0.csv"
```

Não use `--incluir-nao-elegiveis` para a publicação padrão.

## 5. Arquivos gerados

### Runtime

- `models/gaussian_nb_pipeline.joblib`
- `models/tfidf_vectorizer.joblib`
- `models/multinomial_nb.joblib`
- `models/model_metadata.json`
- `models/dataset_info.json`
- `models/training_manifest.json`

### Laboratório B1–B5

- `static/laboratorio/manifest.json`
- `static/laboratorio/dataset-summary.json`
- `static/laboratorio/model-comparison.json`
- `static/laboratorio/confusion-matrix.json`
- `static/laboratorio/error-examples.json`
- `static/laboratorio/model-card.json`
- `static/laboratorio/clusters.json`

O CSV completo não é incluído no pacote de publicação e não deve ser enviado ao Vercel.
