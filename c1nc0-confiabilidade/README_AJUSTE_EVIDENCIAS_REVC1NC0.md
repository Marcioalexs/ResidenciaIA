# C1NC0 — Evidências explicadas + dataset revC1NC0

Esta atualização reúne duas evoluções:

1. explicações mais claras para pessoas sem conhecimento de Machine Learning sobre por que cada característica estrutural favoreceu relativamente os grupos `TRUE` ou `FAKE`;
2. novo treinamento com `dataset_crawler_ptbr_consolidado_revC1NC0.csv`.

## Regra de preparação do dataset

- arquivo original: **48.484** registros;
- `extracao_elegivel=True`: **34.872** registros;
- repetições da mesma `url_dataset` removidas: **973**;
- conjunto final usado no treinamento: **33.899** registros;
- classes: **22.511 TRUE** e **11.388 FAKE**;
- URLs diferentes com o mesmo conteúdo são preservadas;
- conteúdos exatamente iguais ficam no mesmo grupo do split e nunca atravessam treino/teste;
- após a deduplicação de URL, há 144 grupos de conteúdo repetido, envolvendo 425 registros e 281 repetições além da primeira ocorrência.

## Interpretação das evidências

`TRUE` e `FAKE` são rótulos dos grupos do dataset, não um veredito factual.

Para cada uma das 19 características, o GaussianNB compara o valor observado com as distribuições aprendidas nos dois grupos. A interface passa a mostrar:

- o que a característica mede;
- o valor observado na notícia;
- qual grupo foi relativamente favorecido;
- as médias de referência do treinamento;
- uma explicação em linguagem simples do motivo da associação;
- o aviso de que a evidência é estatística e não prova de veracidade.

## Métricas do treinamento revC1NC0

Holdout agrupado: 27.119 treino / 6.780 teste; sobreposição de grupos = 0.

- GaussianNB — Accuracy 53,13%; Recall FAKE 93,81%; F1 FAKE 57,35%; Balanced Accuracy 63,18%.
- MultinomialNB — Accuracy 83,42%; Recall FAKE 88,41%; F1 FAKE 78,18%; Balanced Accuracy 84,65%.
- LogisticRegression (somente laboratório/comparação) — Accuracy 92,18%; Recall FAKE 82,22%; F1 FAKE 87,61%; Balanced Accuracy 89,72%.

A LogisticRegression continua sendo usada apenas para comparação no Laboratório B1–B5 e não é carregada pelo runtime da análise de URLs.
