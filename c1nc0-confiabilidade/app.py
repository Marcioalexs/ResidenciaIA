import os

from flask import Flask, jsonify, redirect, render_template, request, url_for

from ml.article_extractor import coletar_noticia
from ml.feature_extractor import calcular_features_url
from ml.model_service import analisar_com_modelos, modelos_disponiveis
from services.laboratorio_runtime import carregar_artefatos_etapa_b
from services.feedback_service import (
    registrar_analise,
    registrar_feedback,
    obter_resultados_piloto,
)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024  # 80 MB; dataset completo deve ser treinado localmente.


def indicios_observaveis(dados):
    """Monta os cartões de observação usados pela interface pública.

    Os campos extras (``id``, ``rotulo``, ``valor`` e ``tom``) servem apenas
    para apresentação. ``ok``, ``nome`` e ``texto`` permanecem compatíveis
    com a versão anterior do template.
    """
    return [
        {
            "id": "coerencia",
            "nome": "Coerência entre os títulos",
            "rotulo": "Título visível × metadados",
            "valor": (
                f"{dados.get('coerencia_h1_og')}%"
                if dados.get("coerencia_h1_og") is not None
                else "Sem comparação"
            ),
            "tom": "mint",
            "ok": dados.get("coerencia_h1_og") is not None,
            "texto": (
                "Semelhança entre o título visível e o título informado pela página: "
                f"{dados.get('coerencia_h1_og')}%."
                if dados.get("coerencia_h1_og") is not None
                else "Não havia dados suficientes para comparar H1 e og:title."
            ),
        },
        {
            "id": "links",
            "nome": "Links e referências externas",
            "rotulo": "Outros sites citados",
            "valor": f"{dados.get('qtd_dominios_externos', 0)} domínios",
            "tom": "blue",
            "ok": dados.get("qtd_links_externos", 0) > 0,
            "texto": (
                f"{dados.get('qtd_links_externos', 0)} links externos em "
                f"{dados.get('qtd_dominios_externos', 0)} domínios externos."
            ),
        },
        {
            "id": "origem",
            "nome": "Origem e autoria",
            "rotulo": "Site da publicação",
            "valor": dados.get("dominio") or "Não identificado",
            "tom": "violet",
            "ok": bool(dados.get("dominio")),
            "texto": (
                f"Domínio identificado: {dados.get('dominio') or 'não identificado'}. "
                f"Autor: {dados.get('autor') or 'não identificado'}."
            ),
        },
        {
            "id": "data",
            "nome": "Data",
            "rotulo": "Data informada pela página",
            "valor": "Encontrada" if dados.get("data") else "Não identificada",
            "tom": "amber",
            "ok": bool(dados.get("data")),
            "texto": f"Data recuperada: {dados.get('data') or 'não encontrada'}.",
        },
        {
            "id": "imagem",
            "nome": "Imagem principal",
            "rotulo": "Metadado visual",
            "valor": "Encontrada" if dados.get("imagem") else "Não identificada",
            "tom": "rose",
            "ok": bool(dados.get("imagem")),
            "texto": (
                "Imagem principal declarada nos metadados."
                if dados.get("imagem")
                else "Imagem principal não localizada nos metadados analisados."
            ),
        },
        {
            "id": "estrutura",
            "nome": "Dados de identificação da página",
            "rotulo": "Dados estruturados",
            "valor": "Disponíveis" if dados.get("json_ld") else "Não localizados",
            "tom": "slate",
            "ok": bool(dados.get("json_ld")),
            "texto": (
                "A página fornece dados estruturados de identificação."
                if dados.get("json_ld")
                else "Não foram localizados dados estruturados de identificação da página."
            ),
        },
    ]


def preparar_resumo_visual(dados):
    """Gera apenas informações de apresentação a partir do conteúdo coletado."""
    texto = " ".join((dados.get("texto") or "").split())
    resumo = dados.get("resumo") or texto[:620]
    if resumo and len(resumo) < len(texto) and not resumo.endswith((".", "!", "?", "…")):
        resumo = resumo.rstrip() + "…"

    return {
        "resumo": resumo or "Não foi possível extrair um resumo textual desta página.",
        "qtd_palavras": dados.get("qtd_palavras") or len(texto.split()),
        "qtd_paragrafos": dados.get("qtd_paragrafos"),
    }


def limites_da_analise():
    """
    Explicita aquilo que o C1NC0 não consegue concluir apenas
    a partir dos elementos coletados e dos modelos experimentais.
    """
    return [
        {
            "nome": "Veracidade factual",
            "texto": (
                "O painel não confirma se as afirmações da notícia "
                "são verdadeiras ou falsas."
            ),
        },
        {
            "nome": "Intenção do autor",
            "texto": (
                "Características do texto não permitem determinar "
                "automaticamente a intenção de quem publicou."
            ),
        },
        {
            "nome": "Contexto completo",
            "texto": (
                "Uma página isolada pode não conter todos os fatos, "
                "documentos e acontecimentos necessários para "
                "interpretar a informação."
            ),
        },
        {
            "nome": "Possíveis omissões",
            "texto": (
                "A ausência de determinada informação não permite "
                "concluir automaticamente por que ela foi omitida."
            ),
        },
        {
            "nome": "Independência das fontes",
            "texto": (
                "A existência de links externos não garante que "
                "as fontes sejam independentes nem que confirmem "
                "as afirmações apresentadas."
            ),
        },
    ]


def roteiro_checagem():
    """
    Roteiro de leitura lateral para estimular a investigação
    do usuário antes de formar seu julgamento.
    """
    return [
        {
            "titulo": "Quem publicou?",
            "texto": (
                "Identifique autor, veículo ou organização. "
                "Procure informações sobre quem é responsável "
                "pelo conteúdo e sobre a origem da publicação."
            ),
        },
        {
            "titulo": "Qual é a fonte original?",
            "texto": (
                "Se houver pesquisa, documento, entrevista, dado "
                "ou comunicado citado, tente chegar à fonte primária."
            ),
        },
        {
            "titulo": (
                "Outras fontes independentes relatam o mesmo fato?"
            ),
            "texto": (
                "Faça leitura lateral: abra novas abas e procure "
                "cobertura independente. Evite considerar simples "
                "republicações do mesmo conteúdo como confirmações "
                "independentes."
            ),
        },
        {
            "titulo": "A data e o contexto fazem sentido?",
            "texto": (
                "Verifique se conteúdo antigo, imagem anterior ou "
                "informação originalmente correta está sendo "
                "reutilizada fora de contexto."
            ),
        },
        {
            "titulo": "Os links sustentam as afirmações?",
            "texto": (
                "Abra as referências apresentadas e confira se elas "
                "realmente apoiam as principais alegações do texto."
            ),
        },
        {
            "titulo": "Título e conteúdo dizem a mesma coisa?",
            "texto": (
                "Compare o título com o conteúdo completo e observe "
                "possíveis exageros, simplificações ou conclusões "
                "não sustentadas pelo texto."
            ),
        },
    ]



VALID_MODES = {"gaussian", "multinomial", "comparar"}


def _modo_modelo():
    """Modo fixo da experiência pública: sempre comparar as duas abordagens."""
    return "comparar"


@app.route("/", methods=["GET", "POST"])
def pagina_inicial():
    resultado = None

    url = (
        request.form.get("url", "")
        if request.method == "POST"
        else request.args.get("url", "")
    ).strip()

    modo_modelo = _modo_modelo()
    analisar = request.method == "POST" or request.args.get("analisar") == "1"
    origem_extensao = request.args.get("origem") == "extensao"
    origem = "extensao" if origem_extensao else "site"

    if analisar and url:
        dados = coletar_noticia(url)

        resultado = {
            "dados": dados,
            "apresentacao": preparar_resumo_visual(dados),
            "indicios": indicios_observaveis(dados),
            "limites": limites_da_analise(),
            "checagens": roteiro_checagem(),
            "ml": None,
        }

        if dados.get("status") == "OK":
            texto_modelo = f"{dados.get('titulo', '')} {dados.get('texto', '')}".strip()

            if len(texto_modelo) < 100:
                resultado["ml"] = {
                    "disponivel": False,
                    "erro": (
                        "Não foi possível extrair texto suficiente "
                        "para executar a análise experimental."
                    ),
                }
            elif modelos_disponiveis():
                features = calcular_features_url(dados)
                resultado["ml"] = analisar_com_modelos(
                    dados,
                    features,
                    modo=modo_modelo,
                )
            else:
                resultado["ml"] = {
                    "disponivel": False,
                    "erro": (
                        "Modelos treinados ainda não foram publicados. "
                        "Execute ml/train_models.py e versione os artefatos da pasta models."
                    ),
                }

    return render_template(
        "index.html",
        resultado=resultado,
        url=url,
        origem_extensao=origem_extensao,
        origem=origem,
        modo_modelo=modo_modelo,
    )


def _json_payload():
    return request.get_json(silent=True) or {}


@app.post("/api/analises")
def api_analises():
    payload = _json_payload()
    obrigatorios = ("session_id", "origem", "modo_modelo")

    if any(not payload.get(campo) for campo in obrigatorios):
        return jsonify({"ok": False, "erro": "Campos obrigatórios ausentes."}), 400

    if payload["origem"] not in {"site", "extensao"}:
        return jsonify({"ok": False, "erro": "Origem inválida."}), 400

    if payload["modo_modelo"] not in VALID_MODES:
        return jsonify({"ok": False, "erro": "Modo de modelo inválido."}), 400

    try:
        registro = registrar_analise(payload)
        return jsonify({"ok": True, "id": registro.get("id")})
    except Exception as exc:
        app.logger.exception("Falha ao registrar análise no Supabase.")
        return jsonify({"ok": False, "erro": "Não foi possível registrar a análise."}), 503


@app.post("/api/feedback")
def api_feedback():
    payload = _json_payload()
    obrigatorios = (
        "session_id",
        "origem",
        "modo_modelo",
        "ajudou_analise",
        "buscaria_outras_fontes",
        "clareza_limites",
        "mudou_avaliacao",
        "parte_mais_util",
        "usaria_novamente",
    )

    if any(payload.get(campo) in (None, "") for campo in obrigatorios):
        return jsonify({"ok": False, "erro": "Preencha os campos obrigatórios."}), 400

    if payload["origem"] not in {"site", "extensao"}:
        return jsonify({"ok": False, "erro": "Origem inválida."}), 400

    if payload["modo_modelo"] not in VALID_MODES:
        return jsonify({"ok": False, "erro": "Modo de modelo inválido."}), 400

    try:
        registro = registrar_feedback(payload)
        return jsonify({"ok": True, "id": registro.get("id")})
    except Exception:
        app.logger.exception("Falha ao registrar feedback no Supabase.")
        return jsonify({"ok": False, "erro": "Não foi possível salvar o feedback."}), 503


@app.get("/resultados-piloto")
def resultados_piloto():
    """Painel público, somente leitura, com resultados agregados do piloto."""
    try:
        resumo = obter_resultados_piloto()
        erro = None
    except Exception:
        app.logger.exception("Falha ao carregar dashboard do piloto.")
        resumo = None
        erro = "Não foi possível carregar os dados do piloto."

    return render_template(
        "resultados_piloto.html",
        resumo=resumo,
        erro=erro,
    )


@app.get("/admin/resultados-piloto")
def resultados_piloto_legacy():
    """Mantém links antigos funcionando após tornar o painel público."""
    return redirect(url_for("resultados_piloto"), code=302)



@app.get("/laboratorio")
def laboratorio():
    """Etapa B: no web/Vercel, apenas apresenta artefatos gerados offline."""
    return render_template(
        "laboratorio.html",
        artefatos=carregar_artefatos_etapa_b(),
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_ready": modelos_disponiveis(),
        "version": os.getenv("C1NC0_APP_VERSION", "piloto-2026-10-03"),
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
