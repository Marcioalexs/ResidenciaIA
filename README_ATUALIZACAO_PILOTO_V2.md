# C1NC0 — Piloto Feedback V2

Atualização da branch `feature/c1nc0-piloto-feedback-v1`.

## Evoluções

- tutorial/modal automático no primeiro acesso;
- opção "Não mostrar automaticamente novamente neste navegador" via localStorage;
- botão permanente "Como participar do piloto?";
- download da extensão diretamente pela aplicação;
- instruções de instalação Chrome/Edge em modo desenvolvedor;
- orientação de uso e interpretação dos dois modelos;
- reforço da Guiding Constraint e do preenchimento do feedback;
- painel de resultados público em `/resultados-piloto`;
- URL antiga `/admin/resultados-piloto` redireciona para o painel público;
- pacote da extensão em `static/downloads/c1nc0-extension-piloto-v1.zip`.

## Arquivos alterados/novos

- `c1nc0-confiabilidade/app.py`
- `c1nc0-confiabilidade/templates/index.html`
- `c1nc0-confiabilidade/templates/resultados_piloto.html`
- `c1nc0-confiabilidade/static/downloads/c1nc0-extension-piloto-v1.zip`

A variável `C1NC0_ADMIN_PASSWORD` não é mais usada pelo código e pode ser removida da Vercel.
