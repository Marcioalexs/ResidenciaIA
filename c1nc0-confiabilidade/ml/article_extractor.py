import ipaddress
import json
import re
import socket
from difflib import SequenceMatcher
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = "C1NC0-Residencia-IA/1.0 (projeto educacional)"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8"}
TIMEOUT = 15


def obter_dominio(url: str) -> str:
    dominio = urlparse(url).netloc.lower().split(":")[0]
    return dominio[4:] if dominio.startswith("www.") else dominio


def validar_url_publica(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        for info in socket.getaddrinfo(parsed.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if any((ip.is_private, ip.is_loopback, ip.is_reserved, ip.is_link_local, ip.is_multicast)):
                return False
        return True
    except Exception:
        return False


def obter_meta(soup, chave):
    elemento = soup.find("meta", attrs={"property": chave}) or soup.find("meta", attrs={"name": chave})
    if elemento and elemento.get("content"):
        return elemento["content"].strip()
    return None


def extrair_json_ld(soup):
    objetos = []
    for script in soup.find_all("script", type="application/ld+json"):
        texto = script.string
        if not texto:
            continue
        try:
            dados = json.loads(texto)
            objetos.extend(dados if isinstance(dados, list) else [dados])
        except (json.JSONDecodeError, TypeError):
            continue
    return objetos


def percorrer_json(objeto):
    if isinstance(objeto, dict):
        yield objeto
        for valor in objeto.values():
            yield from percorrer_json(valor)
    elif isinstance(objeto, list):
        for item in objeto:
            yield from percorrer_json(item)


def procurar_json_ld(objetos, campo):
    for principal in objetos:
        for objeto in percorrer_json(principal):
            if campo in objeto:
                return objeto[campo]
    return None


def normalizar_autor(valor):
    if valor is None:
        return None
    if isinstance(valor, str):
        return valor.strip()
    if isinstance(valor, dict):
        return str(valor.get("name", "")).strip() or None
    if isinstance(valor, list):
        nomes = [normalizar_autor(v) for v in valor]
        return "; ".join(n for n in nomes if n) or None
    return None


def normalizar_imagem(valor):
    if isinstance(valor, str):
        return valor.strip()
    if isinstance(valor, dict):
        return valor.get("url") or valor.get("contentUrl")
    if isinstance(valor, list) and valor:
        return normalizar_imagem(valor[0])
    return None


def extrair_links_externos(soup, url_base):
    origem = obter_dominio(url_base)
    links = set()
    dominios = set()
    for tag in soup.find_all("a", href=True):
        href = tag.get("href", "").strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        completa = urljoin(url_base, href)
        dominio = obter_dominio(completa)
        if dominio and dominio != origem and urlparse(completa).scheme in ("http", "https"):
            links.add(completa)
            dominios.add(dominio)
    return sorted(links), sorted(dominios)


def similaridade(a, b):
    if not a or not b:
        return None
    return round(SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio() * 100, 1)


def coletar_noticia(url: str):
    dados = {
        "url": url, "url_final": None, "dominio": None, "http_status": None,
        "titulo": "", "titulo_html": None, "titulo_h1": None, "titulo_og": None,
        "texto": "", "resumo": "", "qtd_paragrafos": 0, "qtd_palavras": 0,
        "autor": None, "data": None, "canonical": None, "imagem": None,
        "links_externos": [], "qtd_links_externos": 0, "dominios_externos": [],
        "qtd_dominios_externos": 0, "json_ld": False, "coerencia_h1_og": None,
        "status": None, "erro": None,
    }
    if not validar_url_publica(url):
        dados.update(status="URL_INVALIDA", erro="URL inválida ou endereço não permitido.")
        return dados
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        dados["http_status"] = response.status_code
        dados["url_final"] = response.url
        dados["dominio"] = obter_dominio(response.url)
        if not validar_url_publica(response.url):
            dados.update(status="REDIRECIONAMENTO_NAO_PERMITIDO", erro="O redirecionamento levou a um endereço não permitido.")
            return dados
        if response.status_code >= 400:
            dados["status"] = f"HTTP_{response.status_code}"
            return dados

        soup = BeautifulSoup(response.text, "lxml")
        dados["titulo_html"] = soup.title.get_text(" ", strip=True) if soup.title else None
        h1 = soup.find("h1")
        dados["titulo_h1"] = h1.get_text(" ", strip=True) if h1 else None
        dados["titulo_og"] = obter_meta(soup, "og:title") or obter_meta(soup, "twitter:title")
        dados["titulo"] = dados["titulo_og"] or dados["titulo_h1"] or dados["titulo_html"] or ""

        json_ld = extrair_json_ld(soup)
        dados["json_ld"] = bool(json_ld)
        dados["autor"] = obter_meta(soup, "author") or normalizar_autor(procurar_json_ld(json_ld, "author"))
        dados["data"] = obter_meta(soup, "article:published_time") or procurar_json_ld(json_ld, "datePublished")
        if not dados["data"]:
            tag_time = soup.find("time")
            if tag_time:
                dados["data"] = tag_time.get("datetime") or tag_time.get_text(" ", strip=True)
        canonical = soup.find("link", rel="canonical")
        canonical_href = canonical.get("href") if canonical else None
        dados["canonical"] = urljoin(response.url, canonical_href) if canonical_href else None

        imagem = obter_meta(soup, "og:image") or normalizar_imagem(procurar_json_ld(json_ld, "image"))
        dados["imagem"] = urljoin(response.url, imagem) if imagem else None

        descricao = (
            obter_meta(soup, "description")
            or obter_meta(soup, "og:description")
            or obter_meta(soup, "twitter:description")
        )

        # Mesma estratégia do notebook: parágrafos com ao menos 30 caracteres.
        textos = []
        for p in soup.find_all("p"):
            trecho = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
            if len(trecho) >= 30:
                textos.append(trecho)
        dados["texto"] = re.sub(r"\s+", " ", " ".join(textos)).strip()
        dados["qtd_paragrafos"] = len(textos)
        dados["qtd_palavras"] = len(dados["texto"].split())

        if descricao:
            dados["resumo"] = re.sub(r"\s+", " ", descricao).strip()[:700]
        elif textos:
            resumo = " ".join(textos[:2])
            dados["resumo"] = re.sub(r"\s+", " ", resumo).strip()[:700]

        links, dominios = extrair_links_externos(soup, response.url)
        dados["links_externos"] = links
        dados["qtd_links_externos"] = len(links)
        dados["dominios_externos"] = dominios
        dados["qtd_dominios_externos"] = len(dominios)
        dados["coerencia_h1_og"] = similaridade(dados["titulo_h1"], dados["titulo_og"])
        dados["status"] = "OK"
        return dados
    except requests.exceptions.Timeout:
        dados.update(status="TIMEOUT", erro="O site demorou mais que o limite permitido.")
    except requests.exceptions.RequestException as erro:
        dados.update(status="ERRO_REQUEST", erro=str(erro))
    except Exception as erro:
        dados.update(status="ERRO_PROCESSAMENTO", erro=str(erro))
    return dados
