"""Serviços da Etapa B — Laboratório "Entenda a IA" do C1NC0.

A Etapa B separa duas responsabilidades:

* offline/local: ler o dataset completo, validar, treinar/comparar modelos,
  gerar matriz de confusão, exemplos de erro, Model Card e K-Means + PCA;
* web/Vercel: apenas carregar e apresentar artefatos JSON já calculados.

O objetivo é permitir auditoria e aprendizado. Nenhuma métrica ou classe prevista é
tratada como prova de veracidade factual.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import platform
import re
import time
import unicodedata
from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sklearn

from ml.feature_extractor import FEATURES_META


LABEL_CANDIDATES = ("classe", "classe_normalizada", "label", "target", "rotulo")
TITLE_CANDIDATES = ("titulo", "título", "title", "titulo_og")
TEXT_CANDIDATES = ("texto", "text", "conteudo", "content")
URL_CANDIDATES = ("url_final", "url_dataset", "url_requisicao", "url", "link")
ORIGIN_CANDIDATES = (
    "origem",
    "origem_dataset",
    "dataset_origem",
    "fonte_dataset",
    "dataset_fonte",
    "source_dataset",
    "dataset_source",
    "base_origem",
    "origem_base",
    "arquivo_origem",
    "fonte_origem",
    "fonte",
    "source",
    "dataset",
    "base",
    "corpus",
)

MISSING_TOKENS = {"", "none", "null", "nan", "n/a", "na", "não informado", "nao informado"}

ARTIFACT_FILENAMES = {
    "manifest": "manifest.json",
    "dataset_summary": "dataset-summary.json",
    "model_comparison": "model-comparison.json",
    "confusion_matrix": "confusion-matrix.json",
    "error_examples": "error-examples.json",
    "model_card": "model-card.json",
    "clusters": "clusters.json",
}

METRIC_EXPLANATIONS = {
    "accuracy": (
        "De todas as notícias avaliadas, qual fração recebeu a mesma classe que o rótulo do dataset. "
        "Pode parecer alta mesmo quando uma classe é muito mais frequente que a outra."
    ),
    "precision": (
        "Entre as notícias que o modelo marcou como FAKE, quantas realmente estavam rotuladas como FAKE no dataset."
    ),
    "recall": (
        "Entre todas as notícias rotuladas como FAKE no dataset, quantas o modelo conseguiu encontrar."
    ),
    "f1": (
        "Combina Precision e Recall em uma única medida. É útil quando queremos equilibrar falsos alarmes e casos perdidos."
    ),
    "balanced_accuracy": (
        "Calcula o acerto médio entre as classes, dando o mesmo peso a cada uma. É especialmente útil quando o dataset é desbalanceado."
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_csv(data: bytes) -> pd.DataFrame:
    """Lê CSVs comuns do projeto sem assumir separador/encoding único."""
    last_error = None
    attempts = (
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "latin-1"},
        {"sep": ",", "encoding": "latin-1"},
        {"sep": None, "engine": "python", "encoding": "utf-8-sig"},
    )
    for kwargs in attempts:
        try:
            df = pd.read_csv(BytesIO(data), low_memory=False, **kwargs)
            if len(df.columns) > 1:
                return df
        except Exception as exc:  # pragma: no cover - apenas fallback de parsing
            last_error = exc
    raise ValueError(f"Não foi possível interpretar o CSV. {last_error or ''}".strip())


def _find_column(df: pd.DataFrame, candidates) -> str | None:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in normalized:
            return str(normalized[candidate])
    return None


def _normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text in MISSING_TOKENS else text


def _missing_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").fillna("").str.strip().str.lower()
    return series.isna() | text.isin(MISSING_TOKENS)


def _clean_string_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: "" if _normalize_text(value) == "" else str(value).strip())


def _find_origin_column(df: pd.DataFrame) -> str | None:
    exact = _find_column(df, ORIGIN_CANDIDATES)
    if exact:
        return exact
    for col in df.columns:
        normalized = str(col).strip().lower()
        if any(token in normalized for token in ("origem", "fonte_dataset", "dataset_fonte", "source_dataset")):
            return str(col)
    return None


def _normalized_labels(series: pd.Series) -> pd.Series:
    return series.map(_normalize_text)


def _content_groups(df: pd.DataFrame, title_col: str | None, text_col: str | None) -> pd.Series:
    """Cria grupos por conteúdo exato após normalização.

    É uma barreira contra vazamento por duplicatas exatas. Republicações reescritas ou
    semanticamente equivalentes ainda precisam de agrupamento semântico em evolução futura.
    """
    groups: list[str] = []
    for idx, row in df.iterrows():
        title = _normalize_text(row.get(title_col, "")) if title_col else ""
        text = _normalize_text(row.get(text_col, "")) if text_col else ""
        basis = text or title or f"__linha_unica__{idx}"
        groups.append(sha256(basis.encode("utf-8")).hexdigest())
    return pd.Series(groups, index=df.index, name="_grupo_conteudo")


def _column_presence(series: pd.Series | None) -> dict:
    if series is None:
        return {"preenchidos": 0, "percentual": 0.0}
    missing = _missing_mask(series)
    filled = int((~missing).sum())
    total = max(int(len(series)), 1)
    return {"preenchidos": filled, "percentual": round(filled * 100 / total, 2)}


def _top_origins(
    df: pd.DataFrame, origin_col: str | None, url_col: str | None, limit: int = 12
) -> tuple[list[dict], str | None]:
    if origin_col:
        series = _clean_string_series(df[origin_col])
        valid = series.map(_normalize_text) != ""
        counts = series[valid].value_counts().head(limit)
        total = int(valid.sum()) or 1
        return ([
            {
                "origem": str(name),
                "quantidade": int(count),
                "percentual": round(int(count) * 100 / total, 2),
            }
            for name, count in counts.items()
        ], "coluna_dataset")

    if url_col:
        domains = df[url_col].map(
            lambda value: urlparse(str(value).strip()).netloc.lower().removeprefix("www.")
            if _normalize_text(value)
            else ""
        )
        counts = domains[domains != ""].value_counts().head(limit)
        total = int((domains != "").sum()) or 1
        return ([
            {
                "origem": str(name),
                "quantidade": int(count),
                "percentual": round(int(count) * 100 / total, 2),
            }
            for name, count in counts.items()
        ], "dominio_url")
    return [], None


def analisar_dataset(data: bytes, filename: str) -> dict:
    """Incremento B1: descreve estrutura, qualidade, classes e origens do dataset."""
    df = _read_csv(data)
    label_col = _find_column(df, LABEL_CANDIDATES)
    title_col = _find_column(df, TITLE_CANDIDATES)
    text_col = _find_column(df, TEXT_CANDIDATES)
    url_col = _find_column(df, URL_CANDIDATES)
    origin_col = _find_origin_column(df)

    missing_by_column = []
    missing_total = 0
    for col in df.columns:
        missing = int(_missing_mask(df[col]).sum())
        missing_total += missing
        if missing:
            missing_by_column.append(
                {
                    "coluna": str(col),
                    "quantidade": missing,
                    "percentual": round(missing * 100 / max(len(df), 1), 2),
                }
            )
    missing_by_column.sort(key=lambda item: item["quantidade"], reverse=True)

    exact_duplicates = int(df.duplicated().sum())
    title_duplicates = 0
    if title_col:
        titles = df[title_col].map(_normalize_text)
        title_duplicates = int(titles[titles != ""].duplicated(keep=False).sum())

    text_duplicates = 0
    if text_col:
        texts = df[text_col].map(_normalize_text)
        text_duplicates = int(texts[texts != ""].duplicated(keep=False).sum())

    classes: list[dict] = []
    if label_col:
        labels = _normalized_labels(df[label_col])
        counts = labels[labels != ""].value_counts()
        total_labeled = int(counts.sum()) or 1
        classes = [
            {
                "classe": str(name),
                "quantidade": int(count),
                "percentual": round(int(count) * 100 / total_labeled, 2),
            }
            for name, count in counts.items()
        ]

    groups = _content_groups(df, title_col, text_col)
    group_counts = groups.value_counts()
    grouped_rows = int(group_counts[group_counts > 1].sum())
    conflicting_groups = 0
    if label_col:
        group_labels = pd.DataFrame({"grupo": groups, "classe": _normalized_labels(df[label_col])})
        group_labels = group_labels[group_labels["classe"] != ""]
        conflicting_groups = int((group_labels.groupby("grupo")["classe"].nunique() > 1).sum())

    origins, origins_type = _top_origins(df, origin_col, url_col)

    title_presence = _column_presence(df[title_col] if title_col else None)
    text_presence = _column_presence(df[text_col] if text_col else None)
    url_presence = _column_presence(df[url_col] if url_col else None)

    avg_title_chars = None
    if title_col:
        values = df[title_col].fillna("").astype(str)
        avg_title_chars = round(float(values.str.len().mean()), 1)
    avg_text_chars = None
    median_text_chars = None
    if text_col:
        lengths = df[text_col].fillna("").astype(str).str.len()
        avg_text_chars = round(float(lengths.mean()), 1)
        median_text_chars = round(float(lengths.median()), 1)

    warnings: list[str] = []
    if not label_col:
        warnings.append("Não foi possível identificar automaticamente a coluna de classe.")
    if not text_col:
        warnings.append("Não foi possível identificar automaticamente a coluna de texto.")
    if exact_duplicates:
        warnings.append(f"Há {exact_duplicates} linhas exatamente duplicadas.")
    if grouped_rows:
        warnings.append(
            f"{grouped_rows} registros pertencem a grupos de conteúdo exato repetido. "
            "O split agrupado mantém cada grupo inteiro em treino ou teste."
        )
    if conflicting_groups:
        warnings.append(
            f"Há {conflicting_groups} grupos de conteúdo com mais de um rótulo. "
            "Esses conflitos devem ser auditados, pois o mesmo conteúdo não deveria ensinar classes divergentes sem justificativa."
        )
    warnings.append(
        "O agrupamento atual controla duplicatas exatas após normalização, mas não garante "
        "separação de republicações semanticamente equivalentes."
    )

    return {
        "schema_version": 1,
        "arquivo": filename,
        "sha256": sha256(data).hexdigest(),
        "registros": int(len(df)),
        "colunas": int(len(df.columns)),
        "nomes_colunas": [str(c) for c in df.columns],
        "colunas_reconhecidas": {
            "classe": label_col,
            "titulo": title_col,
            "texto": text_col,
            "url": url_col,
            "origem": origin_col,
        },
        # aliases mantidos para compatibilidade com código anterior
        "label_col": label_col,
        "title_col": title_col,
        "text_col": text_col,
        "url_col": url_col,
        "origin_col": origin_col,
        "classes": classes,
        "origens": origins,
        "origens_tipo": origins_type,
        "ausentes_total": missing_total,
        "ausentes_por_coluna": missing_by_column[:20],
        "linhas_duplicadas": exact_duplicates,
        "titulos_repetidos_registros": title_duplicates,
        "textos_repetidos_registros": text_duplicates,
        "grupos_conteudo": int(groups.nunique()),
        "registros_em_grupos_repetidos": grouped_rows,
        "grupos_com_rotulos_conflitantes": conflicting_groups,
        "features_meta_disponiveis": [f for f in FEATURES_META if f in df.columns],
        "features_meta_total": len(FEATURES_META),
        "caracteristicas": {
            "titulos_preenchidos": title_presence,
            "textos_preenchidos": text_presence,
            "urls_preenchidas": url_presence,
            "media_caracteres_titulo": avg_title_chars,
            "media_caracteres_texto": avg_text_chars,
            "mediana_caracteres_texto": median_text_chars,
        },
        "avisos": warnings,
    }


def _prepare(df: pd.DataFrame):
    label_col = _find_column(df, LABEL_CANDIDATES)
    title_col = _find_column(df, TITLE_CANDIDATES)
    text_col = _find_column(df, TEXT_CANDIDATES)
    url_col = _find_column(df, URL_CANDIDATES)
    origin_col = _find_origin_column(df)

    if not label_col:
        raise ValueError(
            "Coluna de classe não identificada. Esperado: classe, classe_normalizada, label, target ou rotulo."
        )
    if not text_col:
        raise ValueError("Coluna de texto não identificada. Esperado: texto, text, conteudo ou content.")

    work = df.copy()
    work["_y"] = _normalized_labels(work[label_col])
    work = work[work["_y"] != ""].copy()
    if work["_y"].nunique() < 2:
        raise ValueError("O treinamento exige pelo menos duas classes.")

    title = _clean_string_series(work[title_col]) if title_col else pd.Series("", index=work.index)
    text = _clean_string_series(work[text_col])
    work["_conteudo"] = (title + " " + text).str.strip()
    work = work[work["_conteudo"] != ""].copy()
    work["_grupo"] = _content_groups(work, title_col, text_col)
    return work, label_col, title_col, text_col, url_col, origin_col


def _split_grouped(work: pd.DataFrame, test_size: float, random_state: int):
    if not 0.10 <= test_size <= 0.40:
        raise ValueError("test_size deve ficar entre 0.10 e 0.40.")
    n_splits = max(2, min(10, round(1.0 / test_size)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    try:
        train_pos, test_pos = next(splitter.split(work, work["_y"], groups=work["_grupo"]))
    except ValueError as exc:
        raise ValueError(
            "Não foi possível criar split estratificado por grupos. "
            "Verifique a quantidade de classes e de grupos por classe."
        ) from exc
    train = work.iloc[train_pos].copy()
    test = work.iloc[test_pos].copy()
    overlap = len(set(train["_grupo"]) & set(test["_grupo"]))
    if overlap:
        raise RuntimeError("Falha metodológica: grupos atravessaram treino e teste.")
    return train, test


def _focus_label(labels: list[str]) -> str:
    normalized = [str(x).lower() for x in labels]
    priorities = ("fake", "false", "falso", "falsa")
    for wanted in priorities:
        if wanted in normalized:
            return str(labels[normalized.index(wanted)])
    return str(sorted(labels)[0])


def _metrics(y_true, y_pred) -> dict:
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    focus = _focus_label(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, pos_label=focus, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, pos_label=focus, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, pos_label=focus, zero_division=0)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "classe_foco": focus,
        "labels_matriz": labels,
        "matriz_confusao": cm.tolist(),
    }


def _error_examples(
    test: pd.DataFrame,
    predictions,
    *,
    model_key: str,
    model_name: str,
    title_col: str | None,
    text_col: str | None,
    url_col: str | None,
    limit: int,
) -> dict:
    labels = sorted(set(map(str, test["_y"])) | set(map(str, predictions)))
    focus = _focus_label(labels)
    pred = pd.Series(predictions, index=test.index, dtype="object").astype(str)
    truth = test["_y"].astype(str)

    def serialize(indices) -> list[dict]:
        rows = []
        for idx in list(indices)[:limit]:
            row = test.loc[idx]
            title = str(row.get(title_col, "") or "") if title_col else ""
            text = str(row.get(text_col, "") or "") if text_col else str(row.get("_conteudo", "") or "")
            url = str(row.get(url_col, "") or "") if url_col else ""
            excerpt = re.sub(r"\s+", " ", text).strip()
            if len(excerpt) > 360:
                excerpt = excerpt[:357].rstrip() + "..."
            rows.append(
                {
                    "id": str(idx),
                    "titulo": title[:240],
                    "trecho": excerpt,
                    "url": url,
                    "classe_real": str(truth.loc[idx]),
                    "classe_prevista": str(pred.loc[idx]),
                }
            )
        return rows

    false_positive_idx = test.index[(truth != focus) & (pred == focus)]
    false_negative_idx = test.index[(truth == focus) & (pred != focus)]
    any_error_idx = test.index[truth != pred]

    return {
        "modelo_id": model_key,
        "modelo": model_name,
        "classe_foco": focus,
        "falsos_positivos_total": int(len(false_positive_idx)),
        "falsos_negativos_total": int(len(false_negative_idx)),
        "erros_total": int(len(any_error_idx)),
        "falsos_positivos": serialize(false_positive_idx),
        "falsos_negativos": serialize(false_negative_idx),
    }


def executar_experimento(
    data: bytes,
    filename: str,
    modelo: str = "multinomial",
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """Compatibilidade com o laboratório interativo anterior para datasets menores."""
    df = _read_csv(data)
    work, _, _, _, _, _ = _prepare(df)
    train, test = _split_grouped(work, test_size, random_state)
    modelo = modelo.strip().lower()
    start = time.perf_counter()

    if modelo == "gaussian":
        missing = [f for f in FEATURES_META if f not in work.columns]
        if missing:
            raise ValueError("GaussianNB requer as 19 features. Ausentes: " + ", ".join(missing))
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("nb", GaussianNB()),
            ]
        )
        estimator.fit(train[FEATURES_META].apply(pd.to_numeric, errors="coerce"), train["_y"])
        pred = estimator.predict(test[FEATURES_META].apply(pd.to_numeric, errors="coerce"))
        representation = "19 features estruturais/linguísticas"
        model_name = "GaussianNB"
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

    return {
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
            "sobreposicao": 0,
        },
        "tempo_segundos": round(time.perf_counter() - start, 3),
        "metricas": _metrics(test["_y"], pred),
        "nota": (
            "Resultado experimental sobre as classes do dataset. Não representa prova "
            "nem probabilidade calibrada de veracidade factual."
        ),
    }


def _fit_models(
    work: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    random_state: int,
    max_tfidf_features: int,
    error_limit: int,
    title_col: str | None,
    text_col: str | None,
    url_col: str | None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Treina abordagens comparáveis sobre o mesmo split agrupado."""
    comparison: list[dict] = []
    matrices: list[dict] = []
    errors: list[dict] = []

    def register(model_key: str, model_name: str, representation: str, pred, elapsed: float):
        metrics = _metrics(test["_y"], pred)
        comparison.append(
            {
                "id": model_key,
                "modelo": model_name,
                "representacao": representation,
                "metricas": {k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "balanced_accuracy")},
                "classe_foco": metrics["classe_foco"],
                "tempo_segundos": round(elapsed, 3),
                "status": "ok",
            }
        )
        matrices.append(
            {
                "modelo_id": model_key,
                "modelo": model_name,
                "labels": metrics["labels_matriz"],
                "valores": metrics["matriz_confusao"],
                "classe_foco": metrics["classe_foco"],
            }
        )
        errors.append(
            _error_examples(
                test,
                pred,
                model_key=model_key,
                model_name=model_name,
                title_col=title_col,
                text_col=text_col,
                url_col=url_col,
                limit=error_limit,
            )
        )

    missing_meta = [f for f in FEATURES_META if f not in work.columns]
    if not missing_meta:
        start = time.perf_counter()
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("nb", GaussianNB()),
            ]
        )
        xtr_meta = train[FEATURES_META].apply(pd.to_numeric, errors="coerce")
        xte_meta = test[FEATURES_META].apply(pd.to_numeric, errors="coerce")
        pipeline.fit(xtr_meta, train["_y"])
        pred_meta = pipeline.predict(xte_meta)
        register(
            "gaussian-meta",
            "GaussianNB",
            "19 features estruturais/linguísticas",
            pred_meta,
            time.perf_counter() - start,
        )
    else:
        comparison.append(
            {
                "id": "gaussian-meta",
                "modelo": "GaussianNB",
                "representacao": "19 features estruturais/linguísticas",
                "metricas": None,
                "status": "indisponivel",
                "motivo": "Features ausentes: " + ", ".join(missing_meta),
            }
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=max_tfidf_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        dtype=np.float32,
    )
    xtr_text = vectorizer.fit_transform(train["_conteudo"])
    xte_text = vectorizer.transform(test["_conteudo"])
    representation = f"título + texto / TF-IDF ({xtr_text.shape[1]} features)"

    start = time.perf_counter()
    multinomial = MultinomialNB()
    multinomial.fit(xtr_text, train["_y"])
    pred_multi = multinomial.predict(xte_text)
    register(
        "multinomial-tfidf",
        "MultinomialNB",
        representation,
        pred_multi,
        time.perf_counter() - start,
    )

    start = time.perf_counter()
    logistic = LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=None)
    logistic.fit(xtr_text, train["_y"])
    pred_logistic = logistic.predict(xte_text)
    register(
        "logistic-tfidf",
        "LogisticRegression",
        representation,
        pred_logistic,
        time.perf_counter() - start,
    )

    return comparison, matrices, errors


def _sample_indices_by_cluster(cluster_labels: np.ndarray, max_points: int, random_state: int) -> np.ndarray:
    total = len(cluster_labels)
    if total <= max_points:
        return np.arange(total)

    rng = np.random.default_rng(random_state)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    selected: list[int] = []
    remaining = max_points
    for position, (cluster, count) in enumerate(zip(unique, counts)):
        if position == len(unique) - 1:
            quota = remaining
        else:
            quota = max(1, round(max_points * int(count) / total))
            quota = min(quota, remaining - (len(unique) - position - 1))
        indices = np.flatnonzero(cluster_labels == cluster)
        chosen = rng.choice(indices, size=min(quota, len(indices)), replace=False)
        selected.extend(map(int, chosen))
        remaining = max_points - len(selected)
    return np.array(sorted(selected[:max_points]), dtype=int)


def _cluster_artifact(
    work: pd.DataFrame,
    *,
    title_col: str | None,
    random_state: int,
    n_clusters: int,
    max_points: int,
    max_text_features: int,
) -> dict:
    """Incremento B5: K-Means sem usar os rótulos + projeção PCA para visualização."""
    available_meta = [f for f in FEATURES_META if f in work.columns]

    if len(available_meta) == len(FEATURES_META):
        numeric = work[FEATURES_META].apply(pd.to_numeric, errors="coerce")
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        dense = scaler.fit_transform(imputer.fit_transform(numeric))
        cluster_matrix = dense
        pca_input = dense
        representation = "19 features estruturais/linguísticas padronizadas"
        preprocessing = "mediana + StandardScaler"
    else:
        # Fallback para datasets sem as 19 features: texto -> TF-IDF -> SVD -> PCA.
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            max_features=max_text_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            dtype=np.float32,
        )
        tfidf = vectorizer.fit_transform(work["_conteudo"])
        n_components = max(2, min(50, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
        if n_components < 2:
            raise ValueError("Dataset insuficiente para reduzir a representação textual para clustering.")
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        dense = svd.fit_transform(tfidf)
        cluster_matrix = dense
        pca_input = dense
        representation = f"título + texto / TF-IDF -> SVD ({n_components} dimensões)"
        preprocessing = f"TF-IDF até {max_text_features} features + TruncatedSVD"

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(cluster_matrix)

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(pca_input)
    selected = _sample_indices_by_cluster(cluster_labels, max_points, random_state)

    labels_series = work["_y"].astype(str).reset_index(drop=True)
    titles_series = (
        work[title_col].fillna("").astype(str).reset_index(drop=True)
        if title_col
        else pd.Series([""] * len(work))
    )

    points = []
    for i in selected:
        points.append(
            {
                "x": round(float(coordinates[i, 0]), 6),
                "y": round(float(coordinates[i, 1]), 6),
                "cluster": int(cluster_labels[i]),
                "classe": str(labels_series.iloc[i]),
                "titulo": str(titles_series.iloc[i])[:180],
            }
        )

    composition = []
    composition_df = pd.crosstab(pd.Series(cluster_labels, name="cluster"), labels_series.rename("classe"))
    for cluster_idx, row in composition_df.iterrows():
        total = int(row.sum()) or 1
        composition.append(
            {
                "cluster": int(cluster_idx),
                "total": total,
                "classes": [
                    {
                        "classe": str(label),
                        "quantidade": int(value),
                        "percentual": round(int(value) * 100 / total, 2),
                    }
                    for label, value in row.items()
                ],
            }
        )

    return {
        "schema_version": 1,
        "algoritmo": "K-Means",
        "n_clusters": n_clusters,
        "random_state": random_state,
        "rotulos_usados_no_treino": False,
        "representacao": representation,
        "preprocessamento": preprocessing,
        "pca": {
            "componentes": 2,
            "variancia_explicada": [round(float(v), 6) for v in pca.explained_variance_ratio_],
            "variancia_explicada_total": round(float(np.sum(pca.explained_variance_ratio_)), 6),
        },
        "registros_clusterizados": int(len(work)),
        "pontos_visualizados": int(len(points)),
        "amostra_visual_deterministica": len(points) < len(work),
        "pontos": points,
        "composicao_apos_revelar_rotulos": composition,
        "nota": (
            "O K-Means não recebeu TRUE/FAKE durante o agrupamento. Os rótulos aparecem apenas depois, "
            "para comparar os grupos encontrados com as classes já existentes no dataset."
        ),
    }


def gerar_artefatos_etapa_b(
    data: bytes,
    filename: str,
    output_dir: str | Path,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
    max_tfidf_features: int = 100000,
    error_limit: int = 6,
    n_clusters: int = 2,
    max_cluster_points: int = 2000,
    max_cluster_text_features: int = 20000,
) -> dict:
    """Gera os JSONs estáticos consumidos pelo /laboratorio no Vercel."""
    if not 2 <= n_clusters <= 8:
        raise ValueError("n_clusters deve ficar entre 2 e 8.")
    if max_tfidf_features < 1000:
        raise ValueError("max_tfidf_features deve ser pelo menos 1000.")

    generated_at = _utc_now()
    dataset_summary = analisar_dataset(data, filename)
    df = _read_csv(data)
    work, _, title_col, text_col, url_col, origin_col = _prepare(df)
    train, test = _split_grouped(work, test_size, random_state)

    comparison, matrices, errors = _fit_models(
        work,
        train,
        test,
        random_state=random_state,
        max_tfidf_features=max_tfidf_features,
        error_limit=error_limit,
        title_col=title_col,
        text_col=text_col,
        url_col=url_col,
    )

    valid_models = [m for m in comparison if m.get("metricas")]
    best_model = max(valid_models, key=lambda item: item["metricas"]["f1"]) if valid_models else None

    model_comparison = {
        "schema_version": 1,
        "gerado_em": generated_at,
        "classe_foco": valid_models[0]["classe_foco"] if valid_models else None,
        "split": {
            "metodo": "StratifiedGroupKFold — primeiro fold usado como holdout",
            "planejado": f"{round((1-test_size)*100)}% treino / {round(test_size*100)}% teste",
            "treino": int(len(train)),
            "teste": int(len(test)),
            "grupos_treino": int(train["_grupo"].nunique()),
            "grupos_teste": int(test["_grupo"].nunique()),
            "sobreposicao_grupos": 0,
            "random_state": random_state,
        },
        "explicacoes_metricas": METRIC_EXPLANATIONS,
        "modelos": comparison,
        "melhor_f1_observado": (
            {
                "modelo_id": best_model["id"],
                "modelo": best_model["modelo"],
                "f1": best_model["metricas"]["f1"],
            }
            if best_model
            else None
        ),
        "nota": (
            "A comparação mede aderência aos rótulos do dataset no mesmo split agrupado. "
            "Ela não mede veracidade factual fora desse contexto."
        ),
    }

    confusion_artifact = {
        "schema_version": 1,
        "gerado_em": generated_at,
        "modelos": matrices,
        "como_ler": {
            "falso_positivo": "O modelo marcou como FAKE um item cujo rótulo do dataset era outra classe.",
            "falso_negativo": "O modelo não marcou como FAKE um item cujo rótulo do dataset era FAKE.",
        },
    }

    errors_artifact = {
        "schema_version": 1,
        "gerado_em": generated_at,
        "limite_exemplos_por_tipo": error_limit,
        "modelos": errors,
        "nota": (
            "Os exemplos vêm exclusivamente do conjunto de teste do mesmo split usado nas métricas. "
            "Eles ajudam a inspecionar padrões de erro, não a julgar a notícia isoladamente."
        ),
    }

    cluster_artifact = _cluster_artifact(
        work.reset_index(drop=True),
        title_col=title_col,
        random_state=random_state,
        n_clusters=n_clusters,
        max_points=max_cluster_points,
        max_text_features=max_cluster_text_features,
    )
    cluster_artifact["gerado_em"] = generated_at

    model_card = {
        "schema_version": 1,
        "gerado_em": generated_at,
        "nome": "C1NC0 — Etapa B / Laboratório Entenda a IA",
        "objetivo": (
            "Comparar abordagens de classificação e tornar visíveis dados, métricas, erros, "
            "limitações e padrões exploratórios do dataset."
        ),
        "dataset": {
            "arquivo": filename,
            "sha256": dataset_summary["sha256"],
            "registros_lidos": dataset_summary["registros"],
            "registros_com_texto_e_classe": int(len(work)),
            "colunas": dataset_summary["colunas"],
            "classes": dataset_summary["classes"],
            "origens": dataset_summary["origens"],
            "duplicatas_linha": dataset_summary["linhas_duplicadas"],
            "registros_em_grupos_repetidos": dataset_summary["registros_em_grupos_repetidos"],
        },
        "representacoes": [
            "19 features estruturais/linguísticas, quando disponíveis",
            f"título + texto com TF-IDF de até {max_tfidf_features} features",
            cluster_artifact["representacao"] + " para K-Means/PCA",
        ],
        "validacao": model_comparison["split"],
        "modelos_avaliados": [
            {
                "id": model["id"],
                "modelo": model["modelo"],
                "representacao": model["representacao"],
                "metricas": model.get("metricas"),
                "status": model["status"],
            }
            for model in comparison
        ],
        "limitacoes": [
            "Os rótulos do dataset podem conter vieses, ruídos ou regras de coleta que o modelo aprende indiretamente.",
            "O agrupamento de treino/teste controla conteúdo exato normalizado, mas ainda não bloqueia toda similaridade semântica.",
            "Desempenho elevado no dataset não garante generalização para novos veículos, temas, períodos ou estilos de escrita.",
            "TF-IDF aprende associações de palavras e expressões; isso não equivale a compreender fatos ou causalidade.",
            "K-Means encontra proximidade na representação escolhida; os grupos não são classes verdadeiras nem prova de separação natural.",
        ],
        "o_que_este_modelo_nao_faz": [
            "Não verifica fatos em fontes externas nem confirma se uma afirmação é verdadeira ou falsa.",
            "Não substitui leitura lateral, checagem humana, investigação jornalística ou consulta à fonte primária.",
            "Não determina intenção, ironia, contexto completo ou motivação de quem publicou.",
            "Não transforma a saída TRUE/FAKE em probabilidade calibrada de veracidade.",
            "Não deve ser usado sozinho para decisões que afetem pessoas, reputação, direitos ou acesso a oportunidades.",
        ],
        "uso_recomendado": [
            "Ensino e auditoria de modelos supervisionados e não supervisionados.",
            "Comparação metodológica de representações e métricas.",
            "Geração de perguntas para investigação humana, sempre mantendo incerteza e contexto visíveis.",
        ],
        "reprodutibilidade": {
            "random_state": random_state,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "gerador": "services.laboratorio_service.gerar_artefatos_etapa_b",
        },
    }

    manifest = {
        "schema_version": 1,
        "gerado_em": generated_at,
        "dataset": {
            "arquivo": filename,
            "sha256": dataset_summary["sha256"],
            "registros": dataset_summary["registros"],
        },
        "artefatos": {key: value for key, value in ARTIFACT_FILENAMES.items() if key != "manifest"},
        "arquitetura": "offline-json-web-readonly",
    }

    artifacts = {
        "manifest": manifest,
        "dataset_summary": dataset_summary,
        "model_comparison": model_comparison,
        "confusion_matrix": confusion_artifact,
        "error_examples": errors_artifact,
        "model_card": model_card,
        "clusters": cluster_artifact,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for key, payload in artifacts.items():
        path = output / ARTIFACT_FILENAMES[key]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return artifacts


def carregar_artefatos_etapa_b(base_dir: str | Path | None = None) -> dict:
    """Carrega artefatos prontos sem executar treino no request web."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1] / "static" / "laboratorio"
    base = Path(base_dir)

    loaded: dict[str, object] = {}
    missing: list[str] = []
    errors: list[str] = []
    for key, filename in ARTIFACT_FILENAMES.items():
        path = base / filename
        if not path.exists():
            missing.append(filename)
            continue
        try:
            loaded[key] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{filename}: {exc}")

    return {
        "pronto": not missing and not errors,
        "parcial": bool(loaded),
        "diretorio": str(base),
        "faltando": missing,
        "erros": errors,
        **loaded,
    }
