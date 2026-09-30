from flask import Flask, render_template, request

from ml.article_extractor import coletar_noticia
from ml.feature_extractor import calcular_features_url
from ml.model_service import analisar_com_modelos, modelos_disponiveis


app = Flask(__name__)


def indicios_observaveis(dados):
    """
    Organiza os elementos que a aplicação conseguiu observar
    diretamente na página analisada.

    Estes elementos são indícios e não representam um veredito
    sobre a veracidade da notícia.
    """
    return [
        {
            "nome": "Origem e autoria",
            "ok": bool(dados.get("dominio")),
            "texto": (
                f"Domínio identificado: "
                f"{dados.get('dominio') or 'não identificado'}. "
                f"Autor: {dados.get('autor') or 'não identificado'}."
            ),
        },
        {
            "nome": "Data",
            "ok": bool(dados.get("data")),
            "texto": (
                f"Data recuperada: "
                f"{dados.get('data') or 'não encontrada'}."
            ),
        },
        {
            "nome": "Links e referências externas",
            "ok": dados.get("qtd_links_externos", 0) > 0,
            "texto": (
                f"{dados.get('qtd_links_externos', 0)} links externos "
                f"em {dados.get('qtd_dominios_externos', 0)} "
                f"domínios externos."
            ),
        },
        {
            "nome": "Imagem principal",
            "ok": bool(dados.get("imagem")),
            "texto": (
                "Imagem principal declarada nos metadados."
                if dados.get("imagem")
                else (
                    "Imagem principal não localizada "
                    "nos metadados analisados."
                )
            ),
        },
        {
            "nome": "Procedência estruturada",
            "ok": bool(dados.get("json_ld")),
            "texto": (
                "JSON-LD encontrado."
                if dados.get("json_ld")
                else "JSON-LD não localizado."
            ),
        },
        {
            "nome": "Coerência de títulos",
            "ok": dados.get("coerencia_h1_og") is not None,
            "texto": (
                f"Similaridade H1 × og:title: "
                f"{dados.get('coerencia_h1_og')}%."
                if dados.get("coerencia_h1_og") is not None
                else (
                    "Não havia dados suficientes para comparar "
                    "H1 e og:title."
                )
            ),
        },
    ]


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


@app.route("/", methods=["GET", "POST"])
def pagina_inicial():
    resultado = None
    url = ""

    if request.method == "POST":
        url = request.form.get("url", "").strip()

        if url:
            dados = coletar_noticia(url)

            resultado = {
                "dados": dados,
                "indicios": indicios_observaveis(dados),
                "limites": limites_da_analise(),
                "checagens": roteiro_checagem(),
                "ml": None,
            }

            if dados.get("status") == "OK":
                texto_modelo = (
                    f"{dados.get('titulo', '')} "
                    f"{dados.get('texto', '')}"
                ).strip()

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
                    )

                else:
                    resultado["ml"] = {
                        "disponivel": False,
                        "erro": (
                            "Modelos treinados ainda não foram "
                            "publicados. Execute ml/train_models.py "
                            "e versione os artefatos da pasta models."
                        ),
                    }

    return render_template(
        "index.html",
        resultado=resultado,
        url=url,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_ready": modelos_disponiveis(),
        "version": "C1NC0-Pensamento-Critico-V2",
    }


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )