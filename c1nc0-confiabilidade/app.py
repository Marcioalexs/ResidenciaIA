# ============================================================
# C1NC0 - Protótipo de apoio à avaliação de confiabilidade
#
# Objetivo:
# Receber uma URL de notícia e coletar indícios observáveis
# relacionados a:
#
# - origem
# - data
# - links
# - imagem
# - coerência
#
# O programa NÃO determina se a notícia é verdadeira ou falsa.
# ============================================================


# ------------------------------------------------------------
# IMPORTAÇÕES
# ------------------------------------------------------------

# Flask cria a aplicação Web.
from flask import Flask, render_template, request

# Requests realiza as requisições HTTP aos sites.
import requests

# BeautifulSoup interpreta o HTML recebido.
from bs4 import BeautifulSoup

# urlparse permite separar domínio, caminho etc.
# urljoin transforma URLs relativas em absolutas.
from urllib.parse import urlparse, urljoin

# Biblioteca nativa para interpretar JSON-LD.
import json

# Usada para comparar textos.
from difflib import SequenceMatcher

# Usada para validar endereços IP.
import ipaddress

# Usada para resolver nomes de domínio.
import socket


# ------------------------------------------------------------
# CRIAÇÃO DA APLICAÇÃO FLASK
# ------------------------------------------------------------

# Cria o objeto principal da aplicação.
app = Flask(__name__)


# ------------------------------------------------------------
# CONFIGURAÇÕES DO CRAWLER / SCRAPER
# ------------------------------------------------------------

# Define como nosso programa se identifica ao acessar sites.
USER_AGENT = (
    "C1NC0-Residencia-IA/0.1 "
    "(projeto educacional)"
)

# Headers enviados nas requisições.
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8"
}

# Tempo máximo de espera por uma página.
TIMEOUT = 12


# ------------------------------------------------------------
# PESOS DO SCORE HEURÍSTICO
# ------------------------------------------------------------

# Esses pesos são apenas uma HIPÓTESE INICIAL.
#
# Eles ainda precisam ser discutidos, testados e validados.
#
# Não representam probabilidade de verdade.

PESOS = {
    "origem": 20,
    "data": 15,
    "links": 20,
    "imagem": 15,
    "coerencia": 20,
    "dados_estruturados": 10
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================


# ------------------------------------------------------------
# OBTER DOMÍNIO
# ------------------------------------------------------------

def obter_dominio(url):

    # Divide a URL.
    partes = urlparse(url)

    # Recupera apenas o domínio.
    dominio = partes.netloc.lower()

    # Remove eventual porta.
    dominio = dominio.split(":")[0]

    # Remove www. apenas para facilitar comparações.
    if dominio.startswith("www."):
        dominio = dominio[4:]

    return dominio


# ------------------------------------------------------------
# VALIDAR URL
# ------------------------------------------------------------

def validar_url_publica(url):

    """
    Impede o servidor público de acessar endereços locais
    ou redes privadas.

    Isso é importante porque o usuário digita a URL.
    """

    try:

        # Interpreta a URL.
        parsed = urlparse(url)

        # Aceitamos somente HTTP e HTTPS.
        if parsed.scheme not in ("http", "https"):
            return False

        # É obrigatório existir hostname.
        if not parsed.hostname:
            return False

        # Resolve o hostname para IP.
        ip_texto = socket.gethostbyname(parsed.hostname)

        # Converte para objeto IP.
        ip = ipaddress.ip_address(ip_texto)

        # Bloqueia IP privado.
        if ip.is_private:
            return False

        # Bloqueia localhost.
        if ip.is_loopback:
            return False

        # Bloqueia IP reservado.
        if ip.is_reserved:
            return False

        # Bloqueia link-local.
        if ip.is_link_local:
            return False

        return True

    except Exception:

        return False


# ------------------------------------------------------------
# OBTER META TAG
# ------------------------------------------------------------

def obter_meta(soup, chave):

    # Primeiro tenta:
    #
    # <meta property="og:title">

    elemento = soup.find(
        "meta",
        attrs={"property": chave}
    )

    # Caso não encontre, tenta:
    #
    # <meta name="author">

    if elemento is None:

        elemento = soup.find(
            "meta",
            attrs={"name": chave}
        )

    # Se encontrou...
    if elemento:

        # Recupera o atributo content.
        valor = elemento.get("content")

        if valor:

            return valor.strip()

    return None


# ------------------------------------------------------------
# EXTRAIR JSON-LD
# ------------------------------------------------------------

def extrair_json_ld(soup):

    # Lista dos objetos encontrados.
    objetos = []

    # Procura:
    #
    # <script type="application/ld+json">

    scripts = soup.find_all(
        "script",
        type="application/ld+json"
    )

    # Percorre cada bloco.
    for script in scripts:

        # Obtém conteúdo.
        texto = script.string

        # Ignora vazio.
        if not texto:
            continue

        try:

            # Converte JSON para objeto Python.
            dados = json.loads(texto)

            # Alguns sites têm lista.
            if isinstance(dados, list):

                objetos.extend(dados)

            # Outros têm objeto único.
            elif isinstance(dados, dict):

                objetos.append(dados)

        except (json.JSONDecodeError, TypeError):

            # JSON inválido não interrompe o crawler.
            continue

    return objetos


# ------------------------------------------------------------
# PERCORRER JSON RECURSIVAMENTE
# ------------------------------------------------------------

def percorrer_json(objeto):

    # Caso seja dicionário.
    if isinstance(objeto, dict):

        # Entrega o próprio dicionário.
        yield objeto

        # Percorre seus valores internos.
        for valor in objeto.values():

            yield from percorrer_json(valor)

    # Caso seja uma lista.
    elif isinstance(objeto, list):

        for item in objeto:

            yield from percorrer_json(item)


# ------------------------------------------------------------
# PROCURAR CAMPO NO JSON-LD
# ------------------------------------------------------------

def procurar_json_ld(objetos, campo):

    # Percorre os objetos principais.
    for objeto_principal in objetos:

        # Percorre estruturas internas.
        for objeto in percorrer_json(objeto_principal):

            # Verifica se o campo existe.
            if campo in objeto:

                return objeto[campo]

    return None


# ------------------------------------------------------------
# NORMALIZAR AUTOR
# ------------------------------------------------------------

def normalizar_autor(valor):

    # Nada informado.
    if valor is None:

        return None

    # Autor como string.
    if isinstance(valor, str):

        return valor.strip()

    # Autor como objeto JSON.
    if isinstance(valor, dict):

        nome = valor.get("name")

        if nome:

            return str(nome).strip()

    # Lista de autores.
    if isinstance(valor, list):

        nomes = []

        for item in valor:

            nome = normalizar_autor(item)

            if nome:

                nomes.append(nome)

        if nomes:

            return "; ".join(nomes)

    return None


# ------------------------------------------------------------
# NORMALIZAR IMAGEM
# ------------------------------------------------------------

def normalizar_imagem(valor):

    if valor is None:

        return None

    # Imagem como string.
    if isinstance(valor, str):

        return valor.strip()

    # Imagem como objeto.
    if isinstance(valor, dict):

        return (
            valor.get("url")
            or valor.get("contentUrl")
        )

    # Lista de imagens.
    if isinstance(valor, list) and valor:

        return normalizar_imagem(valor[0])

    return None


# ------------------------------------------------------------
# EXTRAIR LINKS EXTERNOS
# ------------------------------------------------------------

def extrair_links_externos(soup, url_base):

    # Descobre domínio da notícia.
    dominio_original = obter_dominio(url_base)

    # Set evita duplicatas.
    links = set()

    # Procura todos os <a href="">
    for tag in soup.find_all("a", href=True):

        href = tag.get("href")

        if not href:

            continue

        # Ignora tipos de links que não interessam.
        if href.startswith(
            (
                "javascript:",
                "mailto:",
                "tel:",
                "#"
            )
        ):

            continue

        # Resolve URL relativa.
        url_completa = urljoin(
            url_base,
            href
        )

        dominio_link = obter_dominio(
            url_completa
        )

        # Só queremos domínios diferentes.
        if (
            dominio_link
            and dominio_link != dominio_original
        ):

            links.add(url_completa)

    return sorted(links)


# ------------------------------------------------------------
# SIMILARIDADE ENTRE TÍTULOS
# ------------------------------------------------------------

def calcular_similaridade(texto1, texto2):

    # Se algum estiver ausente...
    if not texto1 or not texto2:

        return None

    # Normaliza caixa.
    texto1 = texto1.lower().strip()

    texto2 = texto2.lower().strip()

    # SequenceMatcher devolve valor 0..1.
    similaridade = SequenceMatcher(
        None,
        texto1,
        texto2
    ).ratio()

    # Transformamos em porcentagem.
    return round(similaridade * 100, 1)


# ============================================================
# COLETA DA NOTÍCIA
# ============================================================

def coletar_noticia(url):

    # Estrutura inicial.
    dados = {
        "url": url,
        "url_final": None,
        "dominio": None,
        "http_status": None,
        "titulo_html": None,
        "titulo_h1": None,
        "titulo_og": None,
        "autor": None,
        "data": None,
        "canonical": None,
        "imagem": None,
        "links_externos": [],
        "qtd_links_externos": 0,
        "json_ld": False,
        "coerencia": None,
        "status": None,
        "erro": None
    }

    # Valida a URL antes do acesso.
    if not validar_url_publica(url):

        dados["status"] = "URL_INVALIDA"

        dados["erro"] = (
            "URL inválida ou endereço não permitido."
        )

        return dados

    try:

        # Faz a requisição.
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=TIMEOUT,
            allow_redirects=True
        )

        # Guarda código HTTP.
        dados["http_status"] = (
            response.status_code
        )

        # Guarda URL final.
        dados["url_final"] = response.url

        # Guarda domínio.
        dados["dominio"] = obter_dominio(
            response.url
        )

        # Se servidor respondeu erro...
        if response.status_code >= 400:

            dados["status"] = (
                f"HTTP_{response.status_code}"
            )

            return dados

        # Interpreta HTML.
        soup = BeautifulSoup(
            response.text,
            "lxml"
        )


        # ====================================================
        # TÍTULO HTML
        # ====================================================

        if soup.title:

            dados["titulo_html"] = (
                soup.title.get_text(
                    " ",
                    strip=True
                )
            )


        # ====================================================
        # H1
        # ====================================================

        h1 = soup.find("h1")

        if h1:

            dados["titulo_h1"] = (
                h1.get_text(
                    " ",
                    strip=True
                )
            )


        # ====================================================
        # OPEN GRAPH TITLE
        # ====================================================

        dados["titulo_og"] = obter_meta(
            soup,
            "og:title"
        )


        # ====================================================
        # JSON-LD
        # ====================================================

        json_ld = extrair_json_ld(soup)

        dados["json_ld"] = bool(json_ld)


        # ====================================================
        # AUTOR
        # ====================================================

        # Primeiro tenta meta tag.
        autor = obter_meta(
            soup,
            "author"
        )

        # Depois tenta JSON-LD.
        if not autor:

            autor = normalizar_autor(
                procurar_json_ld(
                    json_ld,
                    "author"
                )
            )

        dados["autor"] = autor


        # ====================================================
        # DATA
        # ====================================================

        data = obter_meta(
            soup,
            "article:published_time"
        )

        if not data:

            data = procurar_json_ld(
                json_ld,
                "datePublished"
            )

        # Terceira tentativa: <time>
        if not data:

            tag_time = soup.find("time")

            if tag_time:

                data = (
                    tag_time.get("datetime")
                    or tag_time.get_text(
                        " ",
                        strip=True
                    )
                )

        dados["data"] = data


        # ====================================================
        # CANONICAL
        # ====================================================

        canonical = soup.find(
            "link",
            rel="canonical"
        )

        if canonical:

            dados["canonical"] = (
                canonical.get("href")
            )


        # ====================================================
        # IMAGEM
        # ====================================================

        imagem = obter_meta(
            soup,
            "og:image"
        )

        if not imagem:

            imagem = normalizar_imagem(
                procurar_json_ld(
                    json_ld,
                    "image"
                )
            )

        dados["imagem"] = imagem


        # ====================================================
        # LINKS EXTERNOS
        # ====================================================

        links = extrair_links_externos(
            soup,
            response.url
        )

        dados["links_externos"] = links

        dados["qtd_links_externos"] = len(
            links
        )


        # ====================================================
        # COERÊNCIA
        # ====================================================

        # Nesta versão, usamos a concordância entre:
        #
        # H1 da página
        # e
        # og:title
        #
        # como um primeiro indicador experimental.

        dados["coerencia"] = (
            calcular_similaridade(
                dados["titulo_h1"],
                dados["titulo_og"]
            )
        )


        # Se chegamos até aqui...
        dados["status"] = "OK"

        return dados


    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    except requests.exceptions.Timeout:

        dados["status"] = "TIMEOUT"

        dados["erro"] = (
            "O site demorou mais que o limite permitido."
        )


    # --------------------------------------------------------
    # ERRO DE REDE
    # --------------------------------------------------------

    except requests.exceptions.RequestException as erro:

        dados["status"] = "ERRO_REQUEST"

        dados["erro"] = str(erro)


    # --------------------------------------------------------
    # OUTRO ERRO
    # --------------------------------------------------------

    except Exception as erro:

        dados["status"] = "ERRO_PROCESSAMENTO"

        dados["erro"] = str(erro)


    return dados


# ============================================================
# FEATURE ENGINEERING + SCORE
# ============================================================

def calcular_score(dados):

    # Caso a coleta não tenha terminado...
    if dados["status"] != "OK":

        return {
            "score": None,
            "faixa": "Coleta inconclusiva",
            "indicios": [],
            "explicacao": (
                "Não foi possível calcular o score porque "
                "a coleta da página não foi concluída."
            )
        }


    # Lista que alimentará nosso painel.
    indicios = []

    # Score inicial.
    score = 0


    # ========================================================
    # 1. ORIGEM
    # ========================================================

    # Consideramos como sinais iniciais:
    #
    # domínio identificável
    # +
    # autor

    origem_ok = bool(
        dados["dominio"]
        and dados["autor"]
    )

    if origem_ok:

        score += PESOS["origem"]

        mensagem = (
            f"Domínio identificado ({dados['dominio']}) "
            f"e autor encontrado ({dados['autor']})."
        )

    elif dados["dominio"]:

        score += PESOS["origem"] * 0.5

        mensagem = (
            f"Domínio identificado ({dados['dominio']}), "
            "mas autor não encontrado."
        )

    else:

        mensagem = (
            "Não foi possível identificar adequadamente "
            "a origem."
        )

    indicios.append({
        "nome": "Origem",
        "ok": origem_ok,
        "texto": mensagem
    })


    # ========================================================
    # 2. DATA
    # ========================================================

    data_ok = bool(
        dados["data"]
    )

    if data_ok:

        score += PESOS["data"]

        mensagem = (
            f"Data encontrada: {dados['data']}."
        )

    else:

        mensagem = (
            "Data de publicação não encontrada."
        )

    indicios.append({
        "nome": "Data",
        "ok": data_ok,
        "texto": mensagem
    })


    # ========================================================
    # 3. LINKS
    # ========================================================

    quantidade_links = (
        dados["qtd_links_externos"]
    )

    if quantidade_links >= 3:

        score += PESOS["links"]

        links_ok = True

    elif quantidade_links > 0:

        score += PESOS["links"] * 0.5

        links_ok = True

    else:

        links_ok = False

    indicios.append({
        "nome": "Links",
        "ok": links_ok,
        "texto": (
            f"Foram encontrados "
            f"{quantidade_links} links externos."
        )
    })


    # ========================================================
    # 4. IMAGEM
    # ========================================================

    imagem_ok = bool(
        dados["imagem"]
    )

    if imagem_ok:

        score += PESOS["imagem"]

        mensagem = (
            "A página declara uma imagem principal "
            "em seus metadados."
        )

    else:

        mensagem = (
            "Não foi encontrada imagem principal "
            "nos metadados analisados."
        )

    indicios.append({
        "nome": "Imagem",
        "ok": imagem_ok,
        "texto": mensagem
    })


    # ========================================================
    # 5. COERÊNCIA
    # ========================================================

    coerencia = dados["coerencia"]

    if coerencia is None:

        coerencia_ok = False

        mensagem = (
            "Não havia informações suficientes "
            "para comparar os títulos."
        )

    elif coerencia >= 85:

        coerencia_ok = True

        score += PESOS["coerencia"]

        mensagem = (
            f"H1 e og:title apresentam "
            f"{coerencia}% de similaridade."
        )

    elif coerencia >= 60:

        coerencia_ok = True

        score += PESOS["coerencia"] * 0.5

        mensagem = (
            f"H1 e og:title apresentam "
            f"{coerencia}% de similaridade."
        )

    else:

        coerencia_ok = False

        mensagem = (
            f"H1 e og:title apresentam apenas "
            f"{coerencia}% de similaridade."
        )

    indicios.append({
        "nome": "Coerência",
        "ok": coerencia_ok,
        "texto": mensagem
    })


    # ========================================================
    # 6. DADOS ESTRUTURADOS
    # ========================================================

    json_ok = dados["json_ld"]

    if json_ok:

        score += PESOS[
            "dados_estruturados"
        ]

        mensagem = (
            "A página contém JSON-LD/dados estruturados."
        )

    else:

        mensagem = (
            "Não foi localizado JSON-LD."
        )

    indicios.append({
        "nome": "Dados estruturados",
        "ok": json_ok,
        "texto": mensagem
    })


    # ========================================================
    # ARREDONDAMENTO
    # ========================================================

    score = round(
        score,
        1
    )


    # ========================================================
    # FAIXA NÃO BINÁRIA
    # ========================================================

    if score < 40:

        faixa = (
            "Evidência insuficiente"
        )

    elif score < 70:

        faixa = (
            "Evidência parcial"
        )

    else:

        faixa = (
            "Evidência mais robusta"
        )


    # ========================================================
    # EXPLICAÇÃO
    # ========================================================

    explicacao = (
        "O resultado representa somente a presença de "
        "indícios observáveis de transparência e "
        "procedência encontrados automaticamente. "
        "Ele não determina se a informação é verdadeira "
        "ou falsa e não representa probabilidade de "
        "veracidade."
    )


    return {
        "score": score,
        "faixa": faixa,
        "indicios": indicios,
        "explicacao": explicacao
    }


# ============================================================
# PÁGINA WEB
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def pagina_inicial():

    # Resultado começa vazio.
    resultado = None

    # URL começa vazia.
    url = ""

    # Verifica se o usuário clicou em "Analisar".
    if request.method == "POST":

        # Recupera a URL digitada.
        url = request.form.get(
            "url",
            ""
        ).strip()

        # Só executa se houver URL.
        if url:

            # Faz a coleta da página.
            dados = coletar_noticia(
                url
            )

            # Calcula os indícios e o score.
            analise = calcular_score(
                dados
            )

            # Junta os resultados.
            resultado = {
                "dados": dados,
                "analise": analise
            }

    # Abre templates/index.html.
    return render_template(
        "index.html",
        resultado=resultado,
        url=url
    )

# ============================================================
# EXECUÇÃO LOCAL
# ============================================================

# Esta parte é usada quando executamos:
#
# python app.py
#
# Localmente.
#
# No Vercel, o próprio runtime carrega o objeto app.

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )