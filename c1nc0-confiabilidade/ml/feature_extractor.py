import re
from functools import lru_cache

import numpy as np
import spacy

FEATURES_META = [
    "qtd_links_externos", "qtd_dominios_externos", "num_tokens", "num_palavras",
    "num_types", "num_verbos", "num_substantivos", "num_adjetivos", "num_adverbios",
    "num_pronomes", "num_verbos_subjuntivo_imperativo", "num_palavras_maiusculas",
    "num_caracteres", "comprimento_medio_sentenca", "comprimento_medio_palavra",
    "percentual_erros_ortograficos", "emotividade", "diversidade_lexical",
    "coerencia_titulos_pct",
]

NOMES_FEATURES = {
    "qtd_links_externos": "Links externos", "qtd_dominios_externos": "Domínios externos",
    "num_tokens": "Tokens", "num_palavras": "Palavras", "num_types": "Types",
    "num_verbos": "Verbos", "num_substantivos": "Substantivos", "num_adjetivos": "Adjetivos",
    "num_adverbios": "Advérbios", "num_pronomes": "Pronomes",
    "num_verbos_subjuntivo_imperativo": "Verbos subj./imper.",
    "num_palavras_maiusculas": "Palavras maiúsculas", "num_caracteres": "Caracteres",
    "comprimento_medio_sentenca": "Comprimento médio da sentença",
    "comprimento_medio_palavra": "Comprimento médio da palavra",
    "percentual_erros_ortograficos": "Erros ortográficos",
    "emotividade": "Emotividade", "diversidade_lexical": "Diversidade lexical",
    "coerencia_titulos_pct": "Coerência título/conteúdo",
}

@lru_cache(maxsize=1)
def _nlp():
    return spacy.load("pt_core_news_sm")


def calcular_features_url(dados):
    titulo = dados.get("titulo") or ""
    texto = dados.get("texto") or ""
    texto_completo = f"{titulo} {texto}".strip()
    palavras = re.findall(r"\b[\wÀ-ÿ]+\b", texto_completo, flags=re.UNICODE)
    num_palavras = len(palavras)
    num_tokens = num_palavras
    num_types = len(set(p.lower() for p in palavras))
    num_caracteres = len(texto_completo)
    sentencas = [s.strip() for s in re.split(r"[.!?]+", texto_completo) if s.strip()]
    num_sentencas = max(len(sentencas), 1)
    comprimento_medio_sentenca = num_palavras / num_sentencas if num_palavras else 0
    comprimento_medio_palavra = float(np.mean([len(p) for p in palavras])) if palavras else 0
    palavras_maiusculas = [p for p in palavras if len(p) > 1 and p.isupper()]

    doc = _nlp()(texto_completo)
    num_verbos = sum(1 for t in doc if t.pos_ == "VERB")
    num_substantivos = sum(1 for t in doc if t.pos_ in ("NOUN", "PROPN"))
    num_adjetivos = sum(1 for t in doc if t.pos_ == "ADJ")
    num_adverbios = sum(1 for t in doc if t.pos_ == "ADV")
    num_pronomes = sum(1 for t in doc if t.pos_ == "PRON")
    num_verbos_subjuntivo_imperativo = sum(
        1 for t in doc if t.pos_ == "VERB" and any(m in ("Sub", "Imp") for m in t.morph.get("Mood"))
    )
    denominador = num_substantivos + num_verbos
    emotividade = (num_adjetivos + num_adverbios) / denominador if denominador else 0
    tokens_conteudo = [
        t.lemma_.lower() for t in doc
        if t.pos_ in ("NOUN", "PROPN", "VERB", "ADJ", "ADV") and t.is_alpha
    ]
    diversidade_lexical = len(set(tokens_conteudo)) / len(tokens_conteudo) if tokens_conteudo else 0

    # O notebook usa LanguageTool. Em serverless, Java não é garantido; por isso esta
    # feature fica ausente e é imputada pela mediana aprendida no pipeline GaussianNB.
    percentual_erros_ortograficos = np.nan

    titulo_words = {p.lower() for p in re.findall(r"\b[\wÀ-ÿ]+\b", titulo, flags=re.UNICODE)}
    texto_words = {p.lower() for p in re.findall(r"\b[\wÀ-ÿ]+\b", texto, flags=re.UNICODE)}
    coerencia = len(titulo_words & texto_words) / len(titulo_words) * 100 if titulo_words else 0

    return {
        "qtd_links_externos": dados.get("qtd_links_externos", 0),
        "qtd_dominios_externos": dados.get("qtd_dominios_externos", 0),
        "num_tokens": num_tokens, "num_palavras": num_palavras, "num_types": num_types,
        "num_verbos": num_verbos, "num_substantivos": num_substantivos,
        "num_adjetivos": num_adjetivos, "num_adverbios": num_adverbios,
        "num_pronomes": num_pronomes,
        "num_verbos_subjuntivo_imperativo": num_verbos_subjuntivo_imperativo,
        "num_palavras_maiusculas": len(palavras_maiusculas), "num_caracteres": num_caracteres,
        "comprimento_medio_sentenca": comprimento_medio_sentenca,
        "comprimento_medio_palavra": comprimento_medio_palavra,
        "percentual_erros_ortograficos": percentual_erros_ortograficos,
        "emotividade": emotividade, "diversidade_lexical": diversidade_lexical,
        "coerencia_titulos_pct": coerencia,
    }
