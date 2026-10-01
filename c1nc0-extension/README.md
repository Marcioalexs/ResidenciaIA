# C1NC0 — Extensão opcional do piloto

A extensão **não é obrigatória** para usar o C1NC0. No celular, notebook ou computador,
a pessoa pode copiar o link da notícia, abrir o C1NC0, colar a URL e selecionar **Analisar**.

## Para que serve a extensão

No Chrome/Edge de computador, ela agiliza o fluxo: identifica características jornalísticas
na página e permite abrir a análise C1NC0 com a URL atual. A URL só é enviada quando a
pessoa escolhe analisar.

## Instalação manual do MVP

1. Descompacte `c1nc0-extension-piloto-v1.zip`.
2. Abra `chrome://extensions` ou `edge://extensions`.
3. Ative **Modo do desenvolvedor**.
4. Clique em **Carregar sem compactação**.
5. Selecione a pasta `c1nc0-extension`.
6. Recarregue qualquer aba do C1NC0 que já estava aberta.

Versão: **1.0.2**.

## Detecção no site

A versão 1.0.2 responde a `C1NC0_EXTENSION_PING` e também anuncia
`C1NC0_EXTENSION_PONG` ao carregar. Isso reduz falhas de detecção por ordem de carregamento.

Se o site não detectar uma extensão recém-instalada, recarregue a aba do C1NC0 e use
**Verificar extensão**. Mesmo sem detecção, copiar e colar a URL continua funcionando.
