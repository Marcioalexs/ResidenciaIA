# C1NC0 — Etapa A — Interface Web V1

Base utilizada: `main` no commit `d957805660a255a74ac758805cf0b2789a200709`.
Branch indicada para a evolução: `feature/c1nc0-etapa-a-interface-web-v1`.

## Objetivo

Atualizar a experiência visual da Etapa A usando `C1NC0_web.zip` como referência de interface, sem substituir a metodologia já implementada no projeto Residência IA.

A nova interface mantém:

- Flask + Jinja;
- coleta da notícia por URL;
- indícios observáveis e limites explícitos;
- roteiro de leitura lateral;
- comparação experimental Gaussian Naive Bayes × Multinomial Naive Bayes;
- Guiding Constraint: nenhum resultado é apresentado como veredito verdadeiro/falso;
- piloto, registro de análises, feedback e Supabase;
- extensão opcional;
- `/resultados-piloto`, `/laboratorio` e `/health`.

## Arquivos alterados

- `c1nc0-confiabilidade/app.py`
- `c1nc0-confiabilidade/ml/article_extractor.py`
- `c1nc0-confiabilidade/templates/index.html`

## Arquivos novos

- `c1nc0-confiabilidade/static/css/c1nc0-interface.css`
- `c1nc0-confiabilidade/static/js/c1nc0-interface.js`

## Principais mudanças

1. Novo header, hero e campo de análise com estética semelhante à referência recebida.
2. Estado visual de carregamento durante a análise.
3. Resumo visual da notícia com imagem, domínio, data, autoria, palavras e parágrafos.
4. Cartões expansíveis para os indícios observados.
5. Seção dedicada aos limites da análise.
6. Checklist de leitura lateral.
7. Comparação dos dois modelos preservada, mas apresentada como informação experimental e não como veredito.
8. Informações técnicas mantidas em seção avançada.
9. Piloto, extensão, WhatsApp de ajuda e formulário de feedback preservados.
10. `article_extractor.py` agora normaliza URLs relativas de imagem/canonical e gera resumo/contagens para a interface.

## Dependências e Vercel

Nenhuma dependência nova foi adicionada. `requirements.txt` e `vercel.json` permanecem válidos.

Não altere nem versione `.env`. As variáveis atuais do Supabase no Vercel devem ser preservadas.
