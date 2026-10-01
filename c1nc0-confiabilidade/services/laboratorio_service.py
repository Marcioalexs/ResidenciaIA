import io
from collections import Counter

import pandas as pd

LABEL_CANDIDATES = ("classe", "label", "target", "rotulo", "rótulo")
TEXT_CANDIDATES = ("texto", "text", "conteudo", "conteúdo", "content")
TITLE_CANDIDATES = ("titulo", "título", "title")
URL_CANDIDATES = ("url", "link")


def _normalizar_coluna(nome):
    return str(nome).strip().lower()


def _achar_coluna(colunas, candidatos):
    mapa = {_normalizar_coluna(c): c for c in colunas}
    for candidato in candidatos:
        if candidato in mapa:
            return mapa[candidato]
    return None


def _ler_csv(upload):
    raw = upload.read()
    if not raw:
        raise ValueError("O arquivo enviado está vazio.")

    ultimo_erro = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        for sep in (None, ";", ",", "\t"):
            try:
                kwargs = {"encoding": encoding, "low_memory": False}
                if sep is None:
                    kwargs.update({"sep": None, "engine": "python"})
                else:
                    kwargs["sep"] = sep
                df = pd.read_csv(io.BytesIO(raw), **kwargs)
                if len(df.columns) >= 2:
                    return df
            except Exception as exc:  # tenta formatos comuns antes de falhar
                ultimo_erro = exc
    raise ValueError(f"Não foi possível interpretar o CSV: {ultimo_erro}")


def diagnosticar_dataset(upload):
    df = _ler_csv(upload)
    total = len(df)
    colunas = list(df.columns)
    label_col = _achar_coluna(colunas, LABEL_CANDIDATES)
    titulo_col = _achar_coluna(colunas, TITLE_CANDIDATES)
    texto_col = _achar_coluna(colunas, TEXT_CANDIDATES)
    url_col = _achar_coluna(colunas, URL_CANDIDATES)

    ausentes = []
    for coluna in colunas:
        qtd = int(df[coluna].isna().sum())
        if qtd:
            ausentes.append({
                "coluna": str(coluna),
                "qtd": qtd,
                "pct": round(qtd / total * 100, 2) if total else 0,
            })
    ausentes.sort(key=lambda x: x["qtd"], reverse=True)

    classes = []
    if label_col:
        contagem = Counter(df[label_col].fillna("<ausente>").astype(str).str.strip())
        classes = [
            {
                "classe": classe,
                "qtd": int(qtd),
                "pct": round(qtd / total * 100, 2) if total else 0,
            }
            for classe, qtd in contagem.most_common()
        ]

    duplicatas_linha = int(df.duplicated().sum())
    duplicatas_titulo = (
        int(df[titulo_col].fillna("").astype(str).str.strip().duplicated(keep=False).sum())
        if titulo_col else None
    )
    duplicatas_texto = (
        int(df[texto_col].fillna("").astype(str).str.strip().duplicated(keep=False).sum())
        if texto_col else None
    )

    alertas = []
    if not label_col:
        alertas.append("Nenhuma coluna de classe/rótulo foi reconhecida automaticamente.")
    if not titulo_col and not texto_col:
        alertas.append("Nenhuma coluna de título ou texto foi reconhecida automaticamente.")
    if duplicatas_linha:
        alertas.append("Há linhas exatamente duplicadas; revise-as antes de qualquer split.")
    if duplicatas_titulo:
        alertas.append("Há títulos repetidos; eles podem atravessar treino/teste se o split for aleatório simples.")
    if duplicatas_texto:
        alertas.append("Há textos repetidos; agrupe conteúdos relacionados antes do split.")

    return {
        "arquivo": upload.filename or "dataset.csv",
        "registros": total,
        "colunas": [str(c) for c in colunas],
        "qtd_colunas": len(colunas),
        "label_col": label_col,
        "titulo_col": titulo_col,
        "texto_col": texto_col,
        "url_col": url_col,
        "classes": classes,
        "ausentes": ausentes[:20],
        "total_ausentes": int(df.isna().sum().sum()),
        "duplicatas_linha": duplicatas_linha,
        "duplicatas_titulo": duplicatas_titulo,
        "duplicatas_texto": duplicatas_texto,
        "alertas": alertas,
    }
