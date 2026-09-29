from flask import Flask, render_template, request

from ml.article_extractor import coletar_noticia
from ml.feature_extractor import calcular_features_url
from ml.model_service import analisar_com_modelos, modelos_disponiveis

app = Flask(__name__)


def indicios_observaveis(dados):
    return [
        {"nome": "Origem", "ok": bool(dados.get("dominio")), "texto": f"Domínio identificado: {dados.get('dominio') or 'não identificado'}. Autor: {dados.get('autor') or 'não identificado'}."},
        {"nome": "Data", "ok": bool(dados.get("data")), "texto": f"Data recuperada: {dados.get('data') or 'não encontrada'}."},
        {"nome": "Links", "ok": dados.get("qtd_links_externos", 0) > 0, "texto": f"{dados.get('qtd_links_externos', 0)} links externos em {dados.get('qtd_dominios_externos', 0)} domínios externos."},
        {"nome": "Imagem", "ok": bool(dados.get("imagem")), "texto": "Imagem principal declarada nos metadados." if dados.get("imagem") else "Imagem principal não localizada nos metadados analisados."},
        {"nome": "Dados estruturados", "ok": bool(dados.get("json_ld")), "texto": "JSON-LD encontrado." if dados.get("json_ld") else "JSON-LD não localizado."},
        {"nome": "Coerência de títulos", "ok": dados.get("coerencia_h1_og") is not None, "texto": f"Similaridade H1 × og:title: {dados.get('coerencia_h1_og')}%." if dados.get("coerencia_h1_og") is not None else "Não havia dados suficientes para comparar H1 e og:title."},
    ]


@app.route("/", methods=["GET", "POST"])
def pagina_inicial():
    resultado = None
    url = ""
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if url:
            dados = coletar_noticia(url)
            resultado = {"dados": dados, "indicios": indicios_observaveis(dados), "ml": None}
            if dados.get("status") == "OK":
                texto_modelo = f"{dados.get('titulo', '')} {dados.get('texto', '')}".strip()
                if len(texto_modelo) < 100:
                    resultado["ml"] = {"disponivel": False, "erro": "Não foi possível extrair texto suficiente para executar a análise experimental."}
                elif modelos_disponiveis():
                    features = calcular_features_url(dados)
                    resultado["ml"] = analisar_com_modelos(dados, features)
                else:
                    resultado["ml"] = {"disponivel": False, "erro": "Modelos treinados ainda não foram publicados. Execute ml/train_models.py e versione os artefatos da pasta models."}
    return render_template("index.html", resultado=resultado, url=url)


@app.get("/health")
def health():
    return {"status": "ok", "models_ready": modelos_disponiveis(), "version": "C1NC0-Naive-Bayes-V1"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
