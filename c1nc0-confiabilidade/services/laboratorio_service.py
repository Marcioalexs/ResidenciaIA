"""Serviços da Etapa B — Laboratório de Modelos C1NC0.

O módulo mantém a experimentação separada da inferência pública. Ele aceita CSV,
diagnostica qualidade, cria grupos para reduzir vazamento por conteúdo repetido,
faz split estratificado por grupos e executa três baselines supervisionados.

O resultado é experimental: métricas medem aderência às classes do dataset e não
provam veracidade factual.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from io import BytesIO
import re
import time
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.feature_extractor import FEATURES_META


LABEL_CANDIDATES = ("classe", "classe_normalizada", "label", "target", "rotulo")
TITLE_CANDIDATES = ("titulo", "título", "title", "titulo_og")
TEXT_CANDIDATES = ("texto", "text", "conteudo", "content")
URL_CANDIDATES = ("url_final", "url_dataset", "url_requisicao", "url", "link")


def _read_csv(data: bytes) -> pd.DataFrame:
    last_error = None
    for kwargs in (
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": None, "engine": "python", "encoding": "utf-8-sig"},
    ):
        try:
            df = pd.read_csv(BytesIO(data), low_memory=False, **kwargs)
            if len(df.columns) > 1:
                return df
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Não foi possível interpretar o CSV. {last_error or ''}".strip())


def _find_column(df: pd.DataFrame, candidates) -> str | None:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalized_labels(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _content_groups(df: pd.DataFrame, title_col: str | None, text_col: str | None) -> pd.Series:
    """Agrupa conteúdos exatos normalizados.

    Prioriza texto; quando texto está vazio, usa título. O hash impede carregar o
    conteúdo integral na coluna auxiliar. Similaridade semântica fica explicitamente
    fora deste incremento e deve ser avaliada em uma evolução posterior.
    """
    groups = []
    for idx, row in df.iterrows():
        title = _normalize_text(row.get(title_col, "")) if title_col else ""
        text = _normalize_text(row.get(text_col, "")) if text_col else ""
        basis = text if text else title
        if not basis:
            basis = f"__linha_unica__{idx}"
        groups.append(sha256(basis.encode("utf-8")).hexdigest())
    return pd.Series(groups, index=df.index, name="_grupo_conteudo")


def analisar_dataset(data: bytes, filename: str) -> dict:
    df = _read_csv(data)
    label_col = _find_column(df, LABEL_CANDIDATES)
    title_col = _find_column(df, TITLE_CANDIDATES)
    text_col = _find_column(df, TEXT_CANDIDATES)
    url_col = _find_column(df, URL_CANDIDATES)

    missing_total = int(df.isna().sum().sum())
    exact_duplicates = int(df.duplicated().sum())

    title_duplicates = 0
    if title_col:
        s = df[title_col].map(_normalize_text)
        title_duplicates = int(s[(s != "")].duplicated(keep=False).sum())

    text_duplicates = 0
    if text_col:
        s = df[text_col].map(_normalize_text)
        text_duplicates = int(s[(s != "")].duplicated(keep=False).sum())

    classes = {}
    if label_col:
        labels = _normalized_labels(df[label_col])
        counts = labels[labels != ""].value_counts()
        total = int(counts.sum()) or 1
        classes = {
            str(k): {"quantidade": int(v), "percentual": round(float(v) * 100 / total, 2)}
            for k, v in counts.items()
        }

    groups = _content_groups(df, title_col, text_col)
    group_counts = groups.value_counts()
    grouped_rows = int(group_counts[group_counts > 1].sum())
    unique_groups = int(groups.nunique())

    warnings = []
    if not label_col:
        warnings.append("Não foi possível identificar automaticamente a coluna de classe.")
    if not text_col:
        warnings.append("Não foi possível identificar automaticamente a coluna de texto.")
    if exact_duplicates:
        warnings.append(f"Há {exact_duplicates} linhas exatamente duplicadas; revise-as antes do treinamento.")
    if grouped_rows:
        warnings.append(
            f"{grouped_rows} registros pertencem a grupos de conteúdo exato repetido. "
            "O treinamento do laboratório mantém cada grupo inteiro em treino ou teste."
        )
    warnings.append(
        "O agrupamento deste incremento detecta conteúdo exato após normalização; "
        "similaridade semântica/republicações reescritas ainda exigem uma etapa posterior."
    )

    return {
        "arquivo": filename,
        "registros": int(len(df)),
        "colunas": int(len(df.columns)),
        "nomes_colunas": [str(c) for c in df.columns],
        "label_col": label_col,
        "title_col": title_col,
        "text_col": text_col,
        "url_col": url_col,
        "classes": classes,
        "ausentes_total": missing_total,
        "linhas_duplicadas": exact_duplicates,
        "titulos_repetidos_registros": title_duplicates,
        "textos_repetidos_registros": text_duplicates,
        "grupos_conteudo": unique_groups,
        "registros_em_grupos_repetidos": grouped_rows,
        "features_meta_disponiveis": sum(1 for f in FEATURES_META if f in df.columns),
        "features_meta_total": len(FEATURES_META),
        "avisos": warnings,
    }


def _prepare(df: pd.DataFrame):
    label_col = _find_column(df, LABEL_CANDIDATES)
    title_col = _find_column(df, TITLE_CANDIDATES)
    text_col = _find_column(df, TEXT_CANDIDATES)
    if not label_col:
        raise ValueError("Coluna de classe não identificada. Esperado: classe, classe_normalizada, label, target ou rotulo.")
    if not text_col:
        raise ValueError("Coluna de texto não identificada. Esperado: texto, text, conteudo ou content.")

    work = df.copy()
    work["_y"] = _normalized_labels(work[label_col])
    work = work[work["_y"] != ""].copy()
    if work["_y"].nunique() < 2:
        raise ValueError("O treinamento exige pelo menos duas classes.")

    title = work[title_col].fillna("").astype(str) if title_col else pd.Series("", index=work.index)
    text = work[text_col].fillna("").astype(str)
    work["_conteudo"] = (title + " " + text).str.strip()
    work = work[work["_conteudo"] != ""].copy()
    work["_grupo"] = _content_groups(work, title_col, text_col)
    return work, label_col, title_col, text_col


def _split_grouped(work: pd.DataFrame, test_size: float, random_state: int):
    if not 0.10 <= test_size <= 0.40:
        raise ValueError("test_size deve ficar entre 0.10 e 0.40.")
    # 20% -> 5 folds; outros valores são aproximados ao inverso.
    n_splits = max(2, min(10, round(1.0 / test_size)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    try:
        train_pos, test_pos = next(splitter.split(work, work["_y"], groups=work["_grupo"]))
    except ValueError as exc:
        raise ValueError(
            "Não foi possível criar split estratificado por grupos. "
            "Verifique quantidade de classes e grupos por classe."
        ) from exc
    return work.iloc[train_pos].copy(), work.iloc[test_pos].copy()


def _metrics(y_true, y_pred) -> dict:
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    pos = "fake" if "fake" in labels else labels[-1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_classe_foco": round(float(precision_score(y_true, y_pred, pos_label=pos, zero_division=0)), 6),
        "recall_classe_foco": round(float(recall_score(y_true, y_pred, pos_label=pos, zero_division=0)), 6),
        "f1_classe_foco": round(float(f1_score(y_true, y_pred, pos_label=pos, zero_division=0)), 6),
        "classe_foco": pos,
        "labels_matriz": labels,
        "matriz_confusao": cm.tolist(),
    }


def executar_experimento(
    data: bytes,
    filename: str,
    modelo: str = "multinomial",
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    df = _read_csv(data)
    work, _, _, _ = _prepare(df)
    train, test = _split_grouped(work, test_size, random_state)

    train_groups = set(train["_grupo"])
    test_groups = set(test["_grupo"])
    overlap = len(train_groups & test_groups)
    if overlap:
        raise RuntimeError("Falha de segurança metodológica: grupos atravessaram treino e teste.")

    modelo = modelo.strip().lower()
    start = time.perf_counter()

    if modelo == "gaussian":
        missing = [f for f in FEATURES_META if f not in work.columns]
        if missing:
            raise ValueError("GaussianNB requer as 19 features. Ausentes: " + ", ".join(missing))
        xtr = train[FEATURES_META].apply(pd.to_numeric, errors="coerce")
        xte = test[FEATURES_META].apply(pd.to_numeric, errors="coerce")
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("nb", GaussianNB()),
        ])
        estimator.fit(xtr, train["_y"])
        pred = estimator.predict(xte)
        representation = "19 features estruturais/linguísticas"

    elif modelo in {"multinomial", "logistic"}:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            max_features=100000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        xtr = vectorizer.fit_transform(train["_conteudo"])
        xte = vectorizer.transform(test["_conteudo"])
        if modelo == "multinomial":
            estimator = MultinomialNB()
            model_name = "MultinomialNB"
        else:
            estimator = LogisticRegression(max_iter=1000, random_state=random_state)
            model_name = "LogisticRegression"
        estimator.fit(xtr, train["_y"])
        pred = estimator.predict(xte)
        representation = f"título + texto / TF-IDF ({xtr.shape[1]} features)"
    else:
        raise ValueError("Modelo inválido. Use gaussian, multinomial ou logistic.")

    elapsed = time.perf_counter() - start
    model_name = {"gaussian": "GaussianNB", "multinomial": "MultinomialNB", "logistic": "LogisticRegression"}[modelo]
    result = {
        "arquivo": filename,
        "modelo": model_name,
        "representacao": representation,
        "random_state": random_state,
        "split_planejado": f"{round((1-test_size)*100)}% / {round(test_size*100)}%",
        "split_real": {
            "treino": int(len(train)),
            "teste": int(len(test)),
            "percentual_teste": round(len(test) * 100 / len(work), 2),
        },
        "grupos": {
            "treino": int(train["_grupo"].nunique()),
            "teste": int(test["_grupo"].nunique()),
            "sobreposicao": overlap,
        },
        "tempo_segundos": round(elapsed, 3),
        "metricas": _metrics(test["_y"], pred),
        "nota": (
            "Resultado experimental sobre as classes do dataset. Não representa prova "
            "nem probabilidade calibrada de veracidade factual."
        ),
    }
    return result
