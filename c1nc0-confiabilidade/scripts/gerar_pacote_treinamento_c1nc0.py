#!/usr/bin/env python3
r"""Treina o C1NC0 com um novo CSV e gera todos os artefatos para publicação.

Este script é o ponto único de atualização do modelo supervisionado + Laboratório B1–B5.
Ele:

1. lê o CSV com detecção de separador/encoding;
2. normaliza classes para ``true`` / ``fake``;
3. usa apenas ``extracao_elegivel=True`` por padrão, quando a coluna existe;
4. valida as 19 features estruturais/linguísticas;
5. treina e serializa GaussianNB + TF-IDF + MultinomialNB;
6. calcula também LogisticRegression para auditoria/comparação, sem publicá-la no runtime;
7. gera ``model_metadata.json`` e ``dataset_info.json``;
8. gera todos os JSONs estáticos B1–B5 em ``static/laboratorio``;
9. valida os artefatos produzidos;
10. remove repetições da mesma URL, mantendo URLs diferentes mesmo quando o conteúdo é igual;
11. agrupa conteúdos iguais para que nunca atravessem treino/teste;
12. cria um ZIP contendo somente os arquivos que precisam ser atualizados no C1NC0.

Uso (PowerShell, a partir de c1nc0-confiabilidade):

    python .\scripts\gerar_pacote_treinamento_c1nc0.py `
      --dataset "C:\caminho\dataset_crawler_ptbr_consolidado.csv"

Por padrão, se existir ``extracao_elegivel``, somente linhas elegíveis entram no
conjunto de treinamento. Use ``--incluir-nao-elegiveis`` apenas se quiser mudar
explicitamente essa regra metodológica.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unicodedata
from urllib.parse import urlparse
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.feature_extractor import FEATURES_META  # noqa: E402
from services.laboratorio_service import gerar_artefatos_etapa_b  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
LAB_DIR = PROJECT_ROOT / "static" / "laboratorio"
DIST_DIR = PROJECT_ROOT / "dist"

CLASS_MAP = {
    "true": "true",
    "verdadeira": "true",
    "verdadeiro": "true",
    "real": "true",
    "1": "true",
    "1.0": "true",
    "fake": "fake",
    "false": "fake",
    "falsa": "fake",
    "falso": "fake",
    "0": "fake",
    "0.0": "fake",
}

ELIGIBLE_TRUE = {"true", "1", "1.0", "sim", "yes", "y"}
ELIGIBLE_FALSE = {"false", "0", "0.0", "nao", "não", "no", "n"}

COLUMN_ALIASES = {
    "classe": ("classe", "class", "label", "rotulo", "rótulo", "target"),
    "titulo": ("titulo", "título", "title", "headline", "manchete"),
    "texto": ("texto", "text", "content", "conteudo", "conteúdo", "body", "noticia"),
    "extracao_elegivel": ("extracao_elegivel", "extração_elegível", "elegivel", "elegível", "eligible"),
    "url_dataset": ("url_dataset", "url", "link", "url_origem"),
}

RUNTIME_MODEL_FILES = (
    "gaussian_nb_pipeline.joblib",
    "tfidf_vectorizer.joblib",
    "multinomial_nb.joblib",
    "model_metadata.json",
    "dataset_info.json",
    "training_manifest.json",
)

LAB_FILES = (
    "manifest.json",
    "dataset-summary.json",
    "model-comparison.json",
    "confusion-matrix.json",
    "error-examples.json",
    "model-card.json",
    "clusters.json",
)

SOURCE_FILES_TO_PACKAGE = (
    ".gitignore",
    "vercel.json",
    "ml/feature_extractor.py",
    "ml/model_service.py",
    "ml/train_models.py",
    "templates/index.html",
    "scripts/gerar_pacote_treinamento_c1nc0.py",
    "scripts/gerar_treinamento_c1nc0.ps1",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_column_name(name: object) -> str:
    text = unicodedata.normalize("NFKD", str(name).strip().lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def read_csv_robust(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = path.read_bytes()
    attempts = (
        (";", "utf-8-sig"),
        (",", "utf-8-sig"),
        ("\t", "utf-8-sig"),
        ("|", "utf-8-sig"),
        (";", "cp1252"),
        (",", "cp1252"),
        (";", "latin-1"),
        (",", "latin-1"),
    )
    last_error: Exception | None = None
    for sep, encoding in attempts:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encoding, low_memory=False)
            if len(df.columns) > 1:
                return df, {"separador": sep, "encoding": encoding}
        except Exception as exc:  # fallback de parsing
            last_error = exc

    try:
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")
        if len(df.columns) > 1:
            return df, {"separador": "auto", "encoding": "utf-8-sig"}
    except Exception as exc:
        last_error = exc

    raise ValueError(f"Não foi possível interpretar o CSV: {last_error}")


def standardize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    original_to_normalized = {str(col): normalize_column_name(col) for col in df.columns}
    normalized = list(original_to_normalized.values())
    if len(normalized) != len(set(normalized)):
        raise ValueError("Há nomes de colunas que colidem depois da normalização.")

    work = df.rename(columns=original_to_normalized).copy()
    renames: dict[str, str] = {}

    for standard, aliases in COLUMN_ALIASES.items():
        if standard in work.columns:
            continue
        normalized_aliases = [normalize_column_name(a) for a in aliases]
        match = next((name for name in normalized_aliases if name in work.columns), None)
        if match:
            renames[match] = standard

    if renames:
        work = work.rename(columns=renames)

    return work, renames


def normalize_classes(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip().str.lower()
    normalized = raw.map(CLASS_MAP)
    unknown = sorted(raw[normalized.isna()].dropna().unique().tolist())
    if unknown:
        preview = ", ".join(map(str, unknown[:20]))
        raise ValueError(
            "Classes não reconhecidas: " + preview + ". "
            "Ajuste CLASS_MAP no script se o CSV usar outros rótulos."
        )
    return normalized


def eligible_mask(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip().str.lower()
    known = raw.isin(ELIGIBLE_TRUE | ELIGIBLE_FALSE)
    if not bool(known.all()):
        unknown = sorted(raw[~known].dropna().unique().tolist())
        raise ValueError(
            "Valores não reconhecidos em extracao_elegivel: " + ", ".join(map(str, unknown[:20]))
        )
    return raw.isin(ELIGIBLE_TRUE)


def _normalize_url_for_dedupe(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _dedupe_same_url(work: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove apenas repetições da mesma URL.

    URLs diferentes com o mesmo conteúdo são preservadas. O controle contra vazamento
    desses conteúdos acontece depois, no split agrupado pelo texto normalizado.
    """
    url_col = next(
        (c for c in ("url_dataset", "url_requisicao", "url_final", "url") if c in work.columns),
        None,
    )
    if not url_col:
        return work, {
            "coluna_url": None,
            "urls_repetidas_removidas": 0,
            "registros_antes": int(len(work)),
            "registros_depois": int(len(work)),
        }

    normalized = work[url_col].map(_normalize_url_for_dedupe)
    repeated = normalized.ne("") & normalized.duplicated(keep="first")
    removed = int(repeated.sum())
    result = work.loc[~repeated].copy()
    return result, {
        "coluna_url": url_col,
        "urls_repetidas_removidas": removed,
        "registros_antes": int(len(work)),
        "registros_depois": int(len(result)),
    }


def _content_duplicate_stats(work: pd.DataFrame) -> dict:
    normalized = work["texto"].map(normalize_content)
    nonempty = normalized[normalized != ""]
    counts = nonempty.value_counts()
    repeated_groups = counts[counts > 1]
    return {
        "grupos_conteudo_repetido": int(len(repeated_groups)),
        "registros_em_grupos_conteudo_repetido": int(repeated_groups.sum()) if len(repeated_groups) else 0,
        "repeticoes_conteudo_alem_primeiro": int((repeated_groups - 1).sum()) if len(repeated_groups) else 0,
        "conteudos_unicos": int(nonempty.nunique()),
    }


def prepare_training_dataframe(
    df: pd.DataFrame,
    *,
    only_eligible: bool,
) -> tuple[pd.DataFrame, dict]:
    work, alias_renames = standardize_columns(df)

    required = {"classe", "titulo", "texto", *FEATURES_META}
    missing = sorted(required - set(work.columns))
    if missing:
        raise ValueError(
            "O C1NC0 atual exige classe, título, texto e as 19 features. "
            "Colunas ausentes: " + ", ".join(missing)
        )

    rows_original = len(work)
    work["classe"] = normalize_classes(work["classe"])
    work["titulo"] = work["titulo"].fillna("").astype(str)
    work["texto"] = work["texto"].fillna("").astype(str)

    eligibility_column_present = "extracao_elegivel" in work.columns
    rows_before_eligibility = len(work)
    if only_eligible and eligibility_column_present:
        mask = eligible_mask(work["extracao_elegivel"])
        work = work[mask].copy()

    rows_after_eligibility = len(work)
    work, url_dedupe_stats = _dedupe_same_url(work)

    work["_conteudo_validacao"] = (work["titulo"] + " " + work["texto"]).str.strip()
    empty_content = int((work["_conteudo_validacao"] == "").sum())
    if empty_content:
        work = work[work["_conteudo_validacao"] != ""].copy()
    work = work.drop(columns=["_conteudo_validacao"])

    # Garante numérico para as 19 features; NaNs permanecem para o imputer do pipeline.
    for feature in FEATURES_META:
        work[feature] = pd.to_numeric(work[feature], errors="coerce")

    content_duplicate_stats = _content_duplicate_stats(work)
    class_counts = work["classe"].value_counts().to_dict()
    if set(class_counts) != {"true", "fake"}:
        raise ValueError(
            "Após preparação, o treinamento precisa conter exatamente as classes true e fake. "
            f"Encontrado: {class_counts}"
        )
    if min(class_counts.values()) < 5:
        raise ValueError("Uma das classes possui menos de 5 registros.")

    stats = {
        "linhas_arquivo_original": int(rows_original),
        "linhas_antes_filtro_elegibilidade": int(rows_before_eligibility),
        "linhas_apos_filtro_elegibilidade": int(rows_after_eligibility),
        "filtro_elegibilidade_solicitado": bool(only_eligible),
        "coluna_extracao_elegivel_presente": bool(eligibility_column_present),
        "deduplicacao_url": url_dedupe_stats,
        "conteudo_repetido": content_duplicate_stats,
        "linhas_treinamento": int(len(work)),
        "linhas_conteudo_vazio_removidas": int(empty_content),
        "classes": {k: int(v) for k, v in class_counts.items()},
        "aliases_renomeados": alias_renames,
    }
    return work, stats


def normalize_content(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    return re.sub(r"\s+", " ", text).strip()


def grouped_split(df: pd.DataFrame, random_state: int = 42):
    work = df.copy()
    work["_conteudo"] = (work["titulo"].fillna("").astype(str) + " " + work["texto"].fillna("").astype(str)).str.strip()
    groups = []
    for idx, row in work.iterrows():
        basis = normalize_content(row["texto"]) or normalize_content(row["titulo"]) or f"__linha_{idx}"
        groups.append(sha256(basis.encode("utf-8")).hexdigest())
    work["_grupo"] = groups

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    tr_pos, te_pos = next(splitter.split(work, work["classe"], groups=work["_grupo"]))
    train = work.iloc[tr_pos].copy()
    test = work.iloc[te_pos].copy()
    overlap = len(set(train["_grupo"]) & set(test["_grupo"]))
    if overlap:
        raise RuntimeError("Grupos de conteúdo atravessaram treino/teste.")
    return train, test, overlap


def classification_metrics(y_true, y_pred) -> dict:
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    focus = "fake"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_focus": float(precision_score(y_true, y_pred, pos_label=focus, zero_division=0)),
        "recall_focus": float(recall_score(y_true, y_pred, pos_label=focus, zero_division=0)),
        "f1_focus": float(f1_score(y_true, y_pred, pos_label=focus, zero_division=0)),
        "focus_class": focus,
        "confusion_labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def train_runtime_models(
    training_df: pd.DataFrame,
    *,
    random_state: int,
    max_tfidf_features: int,
) -> dict:
    """Treina apenas os modelos realmente carregados pelo site em produção."""
    train, test, overlap = grouped_split(training_df, random_state=random_state)

    print("  GaussianNB: 19 features...")
    gaussian = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("nb", GaussianNB()),
    ])
    xtr_meta = train[FEATURES_META].apply(pd.to_numeric, errors="coerce")
    xte_meta = test[FEATURES_META].apply(pd.to_numeric, errors="coerce")
    gaussian.fit(xtr_meta, train["classe"])
    pred_gaussian = gaussian.predict(xte_meta)
    joblib.dump(gaussian, MODELS_DIR / "gaussian_nb_pipeline.joblib", compress=3)

    print("  TF-IDF: título + texto...")
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
    xtr = vectorizer.fit_transform(train["_conteudo"])
    xte = vectorizer.transform(test["_conteudo"])
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib", compress=3)

    print("  MultinomialNB: TF-IDF...")
    multinomial = MultinomialNB().fit(xtr, train["classe"])
    pred_multinomial = multinomial.predict(xte)
    joblib.dump(multinomial, MODELS_DIR / "multinomial_nb.joblib", compress=3)

    metadata = {
        "version": "C1NC0_crawler_ptbr_grouped_v4_url_content",
        "dataset_rows_eligible": int(len(training_df)),
        "random_state": random_state,
        "split": "aprox. 80/20 estratificado por grupos de conteúdo exato normalizado",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "group_overlap": int(overlap),
        "gaussian_19_features": classification_metrics(test["classe"], pred_gaussian),
        "multinomial_tfidf_title_text": classification_metrics(test["classe"], pred_multinomial),
        "tfidf_features": int(xtr.shape[1]),
        "limitation": (
            "Repetições da mesma URL são removidas antes do treino. URLs diferentes com conteúdo "
            "exatamente igual são mantidas, porém agrupadas para nunca atravessarem treino/teste. "
            "Similaridade semântica entre republicações reescritas ainda não é controlada."
        ),
        "warning": "Saídas experimentais; não representam prova ou probabilidade calibrada de veracidade.",
    }
    (MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def origin_counts(df: pd.DataFrame, limit: int = 12) -> tuple[dict[str, int], int, str]:
    # Para o crawler, origem_dataset é intencionalmente constante. Preferimos domínios reais.
    candidates = ("dominio", "fonte", "source", "origem_dataset")
    for col in candidates:
        if col not in df.columns:
            continue
        series = df[col].fillna("").astype(str).str.strip()
        valid = series[series != ""]
        if valid.nunique() > 1 or col == "origem_dataset":
            counts = valid.value_counts()
            return (
                {str(k): int(v) for k, v in counts.head(limit).items()},
                int(valid.nunique()),
                col,
            )

    for col in ("url_final", "url_dataset", "url_requisicao", "url"):
        if col in df.columns:
            domains = df[col].fillna("").astype(str).map(
                lambda value: urlparse(value).netloc.lower().removeprefix("www.") if value else ""
            )
            domains = domains[domains != ""]
            counts = domains.value_counts()
            return (
                {str(k): int(v) for k, v in counts.head(limit).items()},
                int(domains.nunique()),
                f"dominio_derivado_de_{col}",
            )

    return {}, 0, "indisponivel"


def write_dataset_info(
    training_df: pd.DataFrame,
    *,
    dataset_path: Path,
    original_sha: str,
    training_sha: str,
    prep_stats: dict,
) -> dict:
    counts = training_df["classe"].value_counts().to_dict()
    total = max(len(training_df), 1)
    origins, origins_total, origin_source = origin_counts(training_df)

    info = {
        "schema_version": 1,
        "gerado_em": utc_now(),
        "nome": dataset_path.stem,
        "arquivo": dataset_path.name,
        "sha256_original": original_sha,
        "sha256_treinamento": training_sha,
        "registros_arquivo_original": int(prep_stats["linhas_arquivo_original"]),
        "registros": int(len(training_df)),
        "filtro_elegibilidade": bool(
            prep_stats["filtro_elegibilidade_solicitado"]
            and prep_stats["coluna_extracao_elegivel_presente"]
        ),
        "classes": {
            "true": int(counts.get("true", 0)),
            "fake": int(counts.get("fake", 0)),
        },
        "percentuais": {
            "true": round(int(counts.get("true", 0)) * 100 / total, 2),
            "fake": round(int(counts.get("fake", 0)) * 100 / total, 2),
        },
        "origens": origins,
        "origens_total": origins_total,
        "origens_campo": origin_source,
        "status_coleta_ok": int(len(training_df)),
        "duplicidades": {
            "coluna_url": prep_stats["deduplicacao_url"]["coluna_url"],
            "urls_repetidas_removidas": prep_stats["deduplicacao_url"]["urls_repetidas_removidas"],
            "registros_apos_elegibilidade": prep_stats["linhas_apos_filtro_elegibilidade"],
            "registros_apos_deduplicacao_url": prep_stats["linhas_treinamento"],
            **prep_stats["conteudo_repetido"],
        },
        "representacoes": {
            "gaussian": "19 features estruturais/linguísticas",
            "multinomial": (
                "título + texto com TF-IDF de unigramas e bigramas "
                "(até 100.000 características)"
            ),
        },
        "avaliacao": (
            "Filtro extracao_elegivel=True; remoção de repetições da mesma URL; "
            "StratifiedGroupKFold com primeiro fold como holdout ≈80/20; "
            "grupos por conteúdo exato normalizado; random_state=42"
        ),
        "limitacao": (
            "URLs diferentes com conteúdo exatamente igual são preservadas, mas ficam no mesmo grupo "
            "e nunca atravessam treino/teste. O método ainda não controla toda similaridade semântica "
            "entre textos reescritos. As classes refletem os rótulos do dataset e não constituem "
            "verificação factual."
        ),
    }

    path = MODELS_DIR / "dataset_info.json"
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def enrich_model_metadata(
    *,
    dataset_path: Path,
    original_sha: str,
    training_sha: str,
    prep_stats: dict,
    dataset_info: dict,
) -> dict:
    path = MODELS_DIR / "model_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "version": "C1NC0_crawler_ptbr_grouped_v4_url_content",
            "gerado_em": utc_now(),
            "dataset": {
                "arquivo": dataset_path.name,
                "sha256_original": original_sha,
                "sha256_treinamento": training_sha,
                "linhas_arquivo_original": prep_stats["linhas_arquivo_original"],
                "linhas_treinamento": prep_stats["linhas_treinamento"],
                "filtro_elegibilidade": dataset_info["filtro_elegibilidade"],
                "classes": dataset_info["classes"],
                "duplicidades": dataset_info["duplicidades"],
            },
        }
    )
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def validate_runtime_models() -> dict:
    gaussian_path = MODELS_DIR / "gaussian_nb_pipeline.joblib"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    multinomial_path = MODELS_DIR / "multinomial_nb.joblib"

    gaussian = joblib.load(gaussian_path)
    vectorizer = joblib.load(vectorizer_path)
    multinomial = joblib.load(multinomial_path)

    gaussian_classes = {str(x).lower() for x in gaussian.named_steps["nb"].classes_}
    multinomial_classes = {str(x).lower() for x in multinomial.classes_}
    expected = {"true", "fake"}
    if gaussian_classes != expected:
        raise RuntimeError(f"Classes inesperadas no GaussianNB: {gaussian_classes}")
    if multinomial_classes != expected:
        raise RuntimeError(f"Classes inesperadas no MultinomialNB: {multinomial_classes}")

    if not hasattr(vectorizer, "transform"):
        raise RuntimeError("TF-IDF inválido: transform() não disponível.")

    return {
        "gaussian_classes": sorted(gaussian_classes),
        "multinomial_classes": sorted(multinomial_classes),
        "tfidf_vocabulario": int(len(vectorizer.vocabulary_)),
    }


def write_training_manifest(
    *,
    dataset_path: Path,
    original_sha: str,
    training_sha: str,
    prep_stats: dict,
    dataset_info: dict,
    validation: dict,
) -> dict:
    artifact_paths = [MODELS_DIR / name for name in RUNTIME_MODEL_FILES if name != "training_manifest.json"]
    artifact_paths += [LAB_DIR / name for name in LAB_FILES]

    files = {}
    for path in artifact_paths:
        if path.exists():
            files[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }

    manifest = {
        "schema_version": 1,
        "gerado_em": utc_now(),
        "dataset_original": {
            "arquivo": dataset_path.name,
            "sha256": original_sha,
            "linhas": prep_stats["linhas_arquivo_original"],
        },
        "dataset_treinamento": {
            "sha256": training_sha,
            "linhas": prep_stats["linhas_treinamento"],
            "classes": dataset_info["classes"],
            "filtro_elegibilidade": dataset_info["filtro_elegibilidade"],
            "deduplicacao_url": prep_stats["deduplicacao_url"],
            "conteudo_repetido": prep_stats["conteudo_repetido"],
        },
        "validacao_runtime": validation,
        "arquivos": files,
    }

    path = MODELS_DIR / "training_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_update_zip(timestamp: str) -> Path:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DIST_DIR / f"c1nc0-evidencias-explicadas-dataset-revc1nc0-{timestamp}.zip"

    package_files: list[Path] = []
    package_files.extend(MODELS_DIR / name for name in RUNTIME_MODEL_FILES)
    package_files.extend(LAB_DIR / name for name in LAB_FILES)
    package_files.extend(PROJECT_ROOT / name for name in SOURCE_FILES_TO_PACKAGE)

    missing = [str(p.relative_to(PROJECT_ROOT)) for p in package_files if not p.exists()]
    if missing:
        raise RuntimeError("Arquivos esperados não encontrados: " + ", ".join(missing))

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in package_files:
            arcname = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            z.write(path, arcname=arcname)

    return zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina C1NC0, gera B1–B5 e cria ZIP pronto para atualizar o projeto."
    )
    parser.add_argument(
        "--dataset",
        help="Caminho para o CSV consolidado. Se omitido, o script pergunta no terminal.",
    )
    parser.add_argument(
        "--incluir-nao-elegiveis",
        action="store_true",
        help="Não filtra extracao_elegivel. Por padrão, somente elegíveis são treinados.",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-tfidf-features", type=int, default=100000)
    parser.add_argument("--error-examples", type=int, default=6)
    parser.add_argument("--clusters", type=int, default=2)
    parser.add_argument("--cluster-points", type=int, default=2000)
    parser.add_argument("--cluster-text-features", type=int, default=20000)
    parser.add_argument(
        "--sem-laboratorio",
        action="store_true",
        help="Treina runtime sem regenerar B1–B5 (não recomendado para publicação final).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_raw = args.dataset or input("Caminho completo do CSV consolidado: ").strip().strip('"')
    dataset_path = Path(dataset_raw).expanduser().resolve()
    if not dataset_path.exists():
        print(f"ERRO: dataset não encontrado: {dataset_path}", file=sys.stderr)
        return 2

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LAB_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("C1NC0 — TREINAMENTO + PACOTE DE ATUALIZAÇÃO")
    print("=" * 78)
    print("Dataset:", dataset_path)
    print("Somente elegíveis:", not args.incluir_nao_elegiveis)

    original_sha = file_sha256(dataset_path)
    print("SHA-256 original:", original_sha)

    print("\n[1/7] Lendo e preparando dataset...")
    df, parse_info = read_csv_robust(dataset_path)
    print(f"  Leitura: {len(df):,} linhas x {len(df.columns)} colunas".replace(",", "."))
    print(f"  Parser : sep={parse_info['separador']!r}, encoding={parse_info['encoding']}")

    training_df, prep_stats = prepare_training_dataframe(
        df,
        only_eligible=not args.incluir_nao_elegiveis,
    )
    del df

    print(
        f"  Treino : {len(training_df):,} linhas | "
        f"TRUE={prep_stats['classes'].get('true', 0):,} | "
        f"FAKE={prep_stats['classes'].get('fake', 0):,}".replace(",", ".")
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    with tempfile.TemporaryDirectory(prefix="c1nc0_training_") as tmp:
        temp_csv = Path(tmp) / "dataset_treinamento_normalizado.csv"
        training_df.to_csv(temp_csv, sep=";", index=False, encoding="utf-8-sig")
        training_sha = file_sha256(temp_csv)

        print("\n[2/7] Treinando modelos supervisionados do runtime...")
        train_runtime_models(
            training_df,
            random_state=args.random_state,
            max_tfidf_features=args.max_tfidf_features,
        )

        # Remove eventual LogisticRegression antiga: o runtime do site não a carrega.
        # A comparação com LogisticRegression continua presente nos JSONs B1–B5.
        logistic_path = MODELS_DIR / "logistic_regression.joblib"
        if logistic_path.exists():
            logistic_path.unlink()

        print("\n[3/7] Gerando informações dinâmicas do dataset...")
        dataset_info = write_dataset_info(
            training_df,
            dataset_path=dataset_path,
            original_sha=original_sha,
            training_sha=training_sha,
            prep_stats=prep_stats,
        )
        model_metadata = enrich_model_metadata(
            dataset_path=dataset_path,
            original_sha=original_sha,
            training_sha=training_sha,
            prep_stats=prep_stats,
            dataset_info=dataset_info,
        )

        if args.sem_laboratorio:
            print("\n[4/7] B1–B5 ignorados por --sem-laboratorio.")
        else:
            print("\n[4/7] Gerando artefatos estáticos B1–B5...")
            normalized_bytes = temp_csv.read_bytes()
            gerar_artefatos_etapa_b(
                normalized_bytes,
                dataset_path.name,
                LAB_DIR,
                test_size=args.test_size,
                random_state=args.random_state,
                max_tfidf_features=args.max_tfidf_features,
                error_limit=args.error_examples,
                n_clusters=args.clusters,
                max_cluster_points=args.cluster_points,
                max_cluster_text_features=args.cluster_text_features,
            )
            del normalized_bytes

    print("\n[5/7] Validando artefatos do runtime...")
    validation = validate_runtime_models()
    print("  Classes GaussianNB    :", validation["gaussian_classes"])
    print("  Classes MultinomialNB :", validation["multinomial_classes"])
    print(f"  Vocabulário TF-IDF    : {validation['tfidf_vocabulario']:,}".replace(",", "."))

    if not args.sem_laboratorio:
        missing_lab = [name for name in LAB_FILES if not (LAB_DIR / name).exists()]
        if missing_lab:
            raise RuntimeError("Artefatos B1–B5 ausentes: " + ", ".join(missing_lab))

    print("\n[6/7] Gravando manifesto de treinamento...")
    write_training_manifest(
        dataset_path=dataset_path,
        original_sha=original_sha,
        training_sha=model_metadata["dataset"]["sha256_treinamento"],
        prep_stats=prep_stats,
        dataset_info=dataset_info,
        validation=validation,
    )

    if args.sem_laboratorio:
        print("\n[7/7] ZIP final não criado porque faltam B1–B5.")
        print("Execute novamente sem --sem-laboratorio para o pacote completo.")
        return 0

    print("\n[7/7] Criando ZIP pronto para atualização do C1NC0...")
    zip_path = build_update_zip(timestamp)

    print("\n" + "=" * 78)
    print("CONCLUÍDO")
    print("=" * 78)
    print(f"Dataset original   : {prep_stats['linhas_arquivo_original']:,} registros".replace(",", "."))
    print(f"Após elegibilidade   : {prep_stats['linhas_apos_filtro_elegibilidade']:,} registros".replace(",", "."))
    print(f"URLs repetidas remov.: {prep_stats['deduplicacao_url']['urls_repetidas_removidas']:,}".replace(",", "."))
    print(f"Dataset treinamento  : {prep_stats['linhas_treinamento']:,} registros".replace(",", "."))
    print(
        "Classes             : "
        f"TRUE={dataset_info['classes']['true']:,} | FAKE={dataset_info['classes']['fake']:,}"
        .replace(",", ".")
    )
    print("Modelos             : models/")
    print("Laboratório B1–B5   : static/laboratorio/")
    print("Pacote              :", zip_path)
    print("\nArquivos do runtime:")
    for name in RUNTIME_MODEL_FILES:
        path = MODELS_DIR / name
        print(f"  - models/{name} ({path.stat().st_size:,} bytes)".replace(",", "."))
    print("\nJSONs do laboratório:")
    for name in LAB_FILES:
        print("  - static/laboratorio/" + name)
    print("\nImportante: o CSV completo NÃO é incluído no ZIP nem deve ir para o Vercel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
