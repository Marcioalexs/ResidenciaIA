# C1NC0 — Extensão Browser V1

Extensão Manifest V3 para Chrome/Edge que identifica localmente páginas com características jornalísticas e oferece acesso ao Painel C1NC0.

## Privacidade

A detecção é feita no navegador. A URL da página não é enviada automaticamente ao C1NC0. Ela só é encaminhada quando o usuário clica em **Analisar com C1NC0** ou **Analisar página atual**.

## Detecção V1

A heurística considera sinais como JSON-LD Article/NewsArticle, `og:type=article`, elemento `<article>`, data de publicação, autoria, H1 e volume de texto. A sugestão aparece quando a soma atinge o limiar configurado em `content.js`.

## Instalação local

1. Abra `chrome://extensions` ou `edge://extensions`.
2. Ative o modo do desenvolvedor.
3. Escolha **Carregar sem compactação**.
4. Selecione a pasta `c1nc0-extension`.
5. Abra uma página jornalística para testar a sugestão automática.
6. O ícone da extensão também permite analisar manualmente a página atual.

## Aplicação Web

A extensão está configurada para abrir:

`https://residencia-ia-2r6h.vercel.app/`

O backend aceita `url`, `analisar=1` e `origem=extensao` na query string.
