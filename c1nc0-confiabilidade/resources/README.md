# C1NC0 — Resources

Pasta de referências do projeto **Ferramenta de apoio à avaliação da confiabilidade da informação**,
Grupo C1NC0 — Residência em IA / PUC Campinas / Instituto ELDORADO.

Esta estrutura acompanha a versão revisada **v0.9 — 03/09/2026** da metodologia.

## Objetivo

Garantir rastreabilidade entre:

`afirmação/decisão no documento → identificador [R#] → referência → recurso na pasta /resources`

As referências dão suporte a cinco frentes do projeto:

1. **Credibilidade, procedência e pensamento crítico**
2. **EDA — Análise Exploratória de Dados**
3. **Machine Learning**
4. **Imagem / media forensics**
5. **Guiding Constraints**

## Estrutura

```text
resources/
├── README.md
├── resources_manifest.csv
├── 01_credibilidade_e_procedencia/
│   ├── README.md
│   ├── R1_Metzger_2007_Credibility_Web.url
│   ├── R2_Wineburg_McGrew_2019_Lateral_Reading.url
│   ├── R3_McGrew_2024_Teaching_Lateral_Reading.url
│   ├── R4_C2PA_Content_Provenance_Authenticity.url
│   ├── R5_IFLA_How_To_Spot_Fake_News.url
│   └── R11_Stanford_Web_Credibility_Project.url
├── 02_EDA/
│   ├── README.md
│   ├── R6_pandas_User_Guide.url
│   └── R7_Matplotlib_Getting_Started.url
├── 03_Machine_Learning/
│   ├── README.md
│   ├── R8_scikit_learn_Unsupervised_Learning.url
│   └── R9_scikit_learn_Metrics_and_Scoring.url
├── 04_imagem_e_media_forensics/
│   ├── README.md
│   └── R10_NISTIR_8377_Media_Forensics.url
└── 05_Guiding_Constraints/
    ├── README.md
    └── R12_Material_Interno_Guiding_Constraints.md
```

## Convenção

- O identificador **R1...R12** deve ser o mesmo utilizado no documento metodológico.
- Arquivos `.url` são atalhos de Internet e podem ser abertos diretamente no Windows.
- Quando houver licença/permissão para manter uma cópia local de um PDF, artigo ou material, ele pode ser salvo
  na mesma pasta do atalho correspondente sem remover o `.url`.
- **R12** é material interno. Não foi fabricado um link público; a pasta contém apenas o registro e instruções
  para adicionar o arquivo original fornecido pelos mentores.
- Não renumerar referências sem também atualizar as citações `[R#]` no documento.

## Referências

| ID | Área | Referência / uso no C1NC0 |
|---|---|---|
| R1 | Credibilidade | Metzger (2007): modelos de avaliação de credibilidade online. |
| R2 | Pensamento crítico | Wineburg & McGrew (2019): leitura lateral. |
| R3 | Pensamento crítico | McGrew (2024): ensino de leitura lateral. |
| R4 | Procedência / imagem | C2PA: proveniência e histórico de mídia digital. |
| R5 | Literacia informacional | IFLA: fonte, autor, data, apoio e vieses. |
| R6 | EDA | pandas User Guide. |
| R7 | EDA | Matplotlib Getting Started. |
| R8 | ML | scikit-learn: aprendizado não supervisionado / K-Means. |
| R9 | ML | scikit-learn: métricas e avaliação. |
| R10 | Imagem | NISTIR 8377: media forensics. |
| R11 | Credibilidade | Stanford Web Credibility Project. |
| R12 | Guiding Constraints | Material interno fornecido pelos mentores. |

## Boas práticas no GitHub

- Alterações em `/resources` devem entrar no mesmo fluxo de branch/PR do projeto.
- Ao adicionar uma nova fonte, registrar no `resources_manifest.csv`:
  identificador, categoria, referência, finalidade, URL e local de citação.
- Se uma fonte deixar de ser usada no documento, marcar como **não utilizada** no manifest antes de removê-la.
- Materiais internos ou com restrição de redistribuição não devem ser publicados sem autorização.
