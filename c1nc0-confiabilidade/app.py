import os
from datetime import datetime, timezone

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


def _parse_data_publicacao(valor):
    """Tenta converter datas comuns/ISO para datetime sem exigir dependências extras."""
    if not valor:
        return None

    raw = str(valor).strip()
    if not raw:
        return None

    candidatos = [raw]
    if raw.endswith("Z"):
        candidatos.insert(0, raw[:-1] + "+00:00")

    for candidato in candidatos:
        try:
            return datetime.fromisoformat(candidato)
        except ValueError:
            pass

    for formato in (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%d-%m-%Y",
    ):
        try:
            return datetime.strptime(raw, formato)
        except ValueError:
            continue

    return None


def _descricao_data_publicacao(valor):
    """Formata a data e explica sua distância em relação ao dia da análise."""
    dt = _parse_data_publicacao(valor)
    if dt is None:
        return {
            "data_formatada": None,
            "texto": (
                "A página informou uma data, mas o formato não pôde ser "
                "interpretado com segurança pelo C1NC0."
                if valor
                else "Não foi localizada uma data de publicação identificável na página."
            ),
            "interpretacao": (
                "Vale conferir a data diretamente na notícia. A data ajuda a avaliar "
                "atualidade e contexto, mas não comprova se o conteúdo é verdadeiro."
            ),
        }

    data_publicacao = dt.date()
    hoje = datetime.now(timezone.utc).date()
    dias = (hoje - data_publicacao).days
    data_formatada = data_publicacao.strftime("%d/%m/%Y")

    if dias < 0:
        distancia = (
            "A data informada está no futuro em relação ao dia desta análise. "
            "Isso merece conferência na própria página."
        )
    elif dias == 0:
        distancia = "A publicação está datada de hoje, portanto é recente em relação à análise."
    elif dias == 1:
        distancia = "A publicação está datada de ontem, portanto é recente em relação à análise."
    elif dias <= 7:
        distancia = f"A publicação tem {dias} dias e é recente em relação à análise."
    elif dias <= 30:
        distancia = f"A publicação foi datada há {dias} dias."
    elif dias < 365:
        meses = max(1, round(dias / 30))
        if meses == 1:
            distancia = "A publicação tem cerca de 1 mês em relação à análise."
        else:
            distancia = f"A publicação tem cerca de {meses} meses em relação à análise."
    else:
        anos = max(1, int(dias / 365))
        if anos == 1:
            distancia = "A publicação tem mais de 1 ano em relação à análise."
        else:
            distancia = f"A publicação tem aproximadamente {anos} anos em relação à análise."

    return {
        "data_formatada": data_formatada,
        "texto": f"A página informa a data de publicação {data_formatada}. {distancia}",
        "interpretacao": (
            "A data ajuda a perceber se uma informação pode estar desatualizada ou "
            "sendo reutilizada fora de contexto. Ser recente ou antiga, por si só, "
            "não torna a notícia verdadeira ou falsa."
        ),
    }


def _descricao_coerencia_titulos(valor):
    if valor is None:
        return {
            "texto": (
                "Não havia informações suficientes para comparar o título visível "
                "com o título declarado nos metadados da página."
            ),
            "interpretacao": (
                "A ausência dessa comparação não indica problema na notícia; apenas "
                "significa que esse indício não pôde ser calculado."
            ),
        }

    try:
        percentual = float(valor)
    except (TypeError, ValueError):
        percentual = None

    if percentual is None:
        faixa = "A semelhança não pôde ser classificada."
    elif percentual >= 85:
        faixa = "Os dois títulos são muito semelhantes."
    elif percentual >= 60:
        faixa = "Os dois títulos têm semelhança moderada."
    else:
        faixa = (
            "Os títulos têm baixa semelhança; vale conferir se o título visível, "
            "o compartilhado e o conteúdo dizem a mesma coisa."
        )

    return {
        "texto": (
            f"Semelhança entre o título visível e o título informado pela página: "
            f"{valor}%. {faixa}"
        ),
        "interpretacao": (
            "Títulos consistentes ajudam a perceber se a página está sendo apresentada "
            "de modo coerente. Diferença entre títulos merece atenção, mas não prova "
            "que o conteúdo seja falso."
        ),
    }


def indicios_observaveis(dados):
    """
    Organiza os elementos observáveis em linguagem voltada a pessoas não técnicas.

    Cada item informa o que foi encontrado e como interpretar esse dado sem tratá-lo
    como prova de veracidade.
    """
    dominio = dados.get("dominio")
    autor = dados.get("autor")
    qtd_links = int(dados.get("qtd_links_externos", 0) or 0)
    qtd_dominios = int(dados.get("qtd_dominios_externos", 0) or 0)
    data_info = _descricao_data_publicacao(dados.get("data"))
    coerencia_info = _descricao_coerencia_titulos(dados.get("coerencia_h1_og"))

    if dominio and autor:
        origem_texto = (
            f"A notícia foi acessada no domínio {dominio} e a página informa "
            f"a autoria como {autor}."
        )
    elif dominio:
        origem_texto = (
            f"A notícia foi acessada no domínio {dominio}, mas o C1NC0 não encontrou "
            "uma autoria identificável nos campos analisados."
        )
    elif autor:
        origem_texto = (
            f"A página informa a autoria como {autor}, mas o domínio não pôde ser "
            "identificado com segurança."
        )
    else:
        origem_texto = (
            "O C1NC0 não conseguiu identificar com segurança o domínio e a autoria "
            "nos campos analisados."
        )

    if qtd_links > 0:
        links_texto = (
            f"Foram encontrados {qtd_links} links que apontam para fora do site, "
            f"distribuídos em {qtd_dominios} domínio(s) externo(s)."
        )
        links_interpretacao = (
            "Esses links oferecem caminhos para conferir documentos, fontes e referências. "
            "É importante abri-los: quantidade de links não garante qualidade, independência "
            "nem confirmação das afirmações."
        )
    else:
        links_texto = (
            "Não foram encontrados links externos nos elementos analisados da página."
        )
        links_interpretacao = (
            "Isso reduz os caminhos diretos de checagem oferecidos pela própria notícia, "
            "mas a ausência de links não significa, sozinha, que o conteúdo seja falso."
        )

    if dados.get("imagem"):
        imagem_texto = (
            "A página declara uma imagem principal nos metadados analisados."
        )
        imagem_interpretacao = (
            "Isso permite localizar a imagem associada à publicação, mas o C1NC0 não "
            "confirma automaticamente se ela é original, atual, editada ou usada no contexto correto."
        )
    else:
        imagem_texto = (
            "O C1NC0 não localizou uma imagem principal declarada nos metadados analisados."
        )
        imagem_interpretacao = (
            "Algumas páginas não informam esse campo. A ausência da imagem nos metadados "
            "não é, por si só, evidência de baixa confiabilidade."
        )

    if dados.get("json_ld"):
        estrutura_texto = (
            "A página fornece dados estruturados de identificação, como informações em JSON-LD."
        )
        estrutura_interpretacao = (
            "Esses dados ajudam buscadores e sistemas a identificar título, autor, data ou "
            "tipo de conteúdo. Como são fornecidos pelo próprio site, não funcionam como "
            "validação independente da notícia."
        )
    else:
        estrutura_texto = (
            "Não foram localizados dados estruturados de identificação nos campos analisados."
        )
        estrutura_interpretacao = (
            "A falta desses dados pode ser apenas uma escolha técnica do site e não permite "
            "concluir que a notícia seja verdadeira ou falsa."
        )

    return [
        {
            "nome": "Origem e autoria",
            "ok": bool(dominio or autor),
            "texto": origem_texto,
            "interpretacao": (
                "Saber onde e por quem algo foi publicado facilita rastrear a fonte, "
                "consultar o veículo e procurar outras publicações do autor. Identificar "
                "origem e autoria não confirma, sozinho, a veracidade do conteúdo."
            ),
        },
        {
            "nome": "Data da publicação",
            "ok": data_info["data_formatada"] is not None,
            "texto": data_info["texto"],
            "interpretacao": data_info["interpretacao"],
        },
        {
            "nome": "Links e referências externas",
            "ok": qtd_links > 0,
            "texto": links_texto,
            "interpretacao": links_interpretacao,
        },
        {
            "nome": "Imagem principal",
            "ok": bool(dados.get("imagem")),
            "texto": imagem_texto,
            "interpretacao": imagem_interpretacao,
        },
        {
            "nome": "Dados de identificação da página",
            "ok": bool(dados.get("json_ld")),
            "texto": estrutura_texto,
            "interpretacao": estrutura_interpretacao,
        },
        {
            "nome": "Coerência entre os títulos da página",
            "ok": dados.get("coerencia_h1_og") is not None,
            "texto": coerencia_info["texto"],
            "interpretacao": coerencia_info["interpretacao"],
        },
    ]


def limites_da_analise():
    """
    Explicita o que não pode ser concluído automaticamente e orienta como reduzir
    cada incerteza com checagem humana.
    """
    return [
        {
            "nome": "Veracidade factual",
            "texto": (
                "O C1NC0 não verifica, afirmação por afirmação, se os fatos narrados "
                "aconteceram exatamente como o texto descreve."
            ),
            "orientacao": (
                "Para reduzir essa dúvida, procure documentos, dados oficiais, a fonte "
                "primária citada e cobertura de fontes independentes."
            ),
        },
        {
            "nome": "Intenção de quem publicou",
            "texto": (
                "A análise do texto não permite saber se uma informação incorreta foi "
                "publicada por engano, interpretação, sátira, descuido ou tentativa deliberada de enganar."
            ),
            "orientacao": (
                "Evite atribuir intenção apenas pelo estilo da escrita. Considere o contexto, "
                "o histórico da publicação e evidências externas."
            ),
        },
        {
            "nome": "Contexto completo",
            "texto": (
                "Uma página mostra apenas parte do contexto. Fatos anteriores, acontecimentos "
                "posteriores, documentos ou trechos omitidos podem mudar a interpretação."
            ),
            "orientacao": (
                "Procure a fonte original e outras coberturas sobre o mesmo acontecimento, "
                "especialmente quando houver números, falas recortadas ou imagens."
            ),
        },
        {
            "nome": "Motivo de possíveis omissões",
            "texto": (
                "O C1NC0 pode perceber que determinada informação não aparece, mas não consegue "
                "concluir por que ela ficou de fora."
            ),
            "orientacao": (
                "A ausência de um dado não prova intenção de esconder algo. Compare a notícia "
                "com documentos e relatos independentes antes de tirar essa conclusão."
            ),
        },
        {
            "nome": "Independência das fontes",
            "texto": (
                "Ter vários links ou encontrar a mesma informação em vários sites não garante "
                "que existam várias confirmações independentes."
            ),
            "orientacao": (
                "Sites diferentes podem reproduzir a mesma agência, comunicado ou texto original. "
                "Ao checar, tente descobrir de onde cada informação realmente veio."
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
        data_info = _descricao_data_publicacao(dados.get("data"))
        dados["data_formatada"] = data_info["data_formatada"]

        resultado = {
            "dados": dados,
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
