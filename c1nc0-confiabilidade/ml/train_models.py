"""Treina e serializa os baselines supervisionados do C1NC0.

Uso:
  python -m ml.train_models "C:\\caminho\\dataset.csv"

Regras aplicadas nesta versão:
- normaliza classes TRUE/FALSE, true/fake etc. para ``true`` / ``fake``;
- se existir ``extracao_elegivel``, usa somente registros elegíveis;
- remove repetições da mesma URL;
- mantém URLs diferentes mesmo quando o conteúdo é exatamente igual;
- conteúdos exatamente iguais são agrupados para nunca atravessarem treino/teste.

Para publicação completa (modelos + dataset_info + B1–B5 + ZIP), prefira:
  python .\\scripts\\gerar_pacote_treinamento_c1nc0.py --dataset "..."
"""
from pathlib import Path
import io
import json
import re
import sys
import unicodedata
from hashlib import sha256

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_extractor import FEATURES_META

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)

CLASS_MAP = {
    "true": "true", "verdadeira": "true", "verdadeiro": "true", "real": "true", "1": "true", "1.0": "true",
    "fake": "fake", "false": "fake", "falsa": "fake", "falso": "fake", "0": "fake", "0.0": "fake",
}
ELIGIBLE_TRUE = {"true", "1", "1.0", "sim", "yes", "y"}


def _read_csv(path):
    raw = Path(path).read_bytes()
    attempts = ((";", "utf-8-sig"), (",", "utf-8-sig"), ("\t", "utf-8-sig"), ("|", "utf-8-sig"),
                (";", "cp1252"), (",", "cp1252"), (";", "latin-1"), (",", "latin-1"))
    for sep, encoding in attempts:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encoding, low_memory=False)
            if len(df.columns) > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")


def _norm(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).lower()
    return re.sub(r"\s+", " ", value).strip()


def _normalize_classes(series):
    raw = series.astype(str).str.strip().str.lower()
    normalized = raw.map(CLASS_MAP)
    if normalized.isna().any():
        unknown = sorted(raw[normalized.isna()].unique().tolist())
        raise ValueError("Classes não reconhecidas: " + ", ".join(map(str, unknown[:20])))
    return normalized


def _groups(df):
    result = []
    for idx, row in df.iterrows():
        text = _norm(row.get("texto", ""))
        title = _norm(row.get("titulo", ""))
        basis = text or title or f"__linha_{idx}"
        result.append(sha256(basis.encode("utf-8")).hexdigest())
    return pd.Series(result, index=df.index)


def _metricas(y, p):
    labels = sorted(set(map(str, y)) | set(map(str, p)))
    return {
        "accuracy": float(accuracy_score(y, p)),
        "precision_focus": float(precision_score(y, p, pos_label="fake", zero_division=0)),
        "recall_focus": float(recall_score(y, p, pos_label="fake", zero_division=0)),
        "f1_focus": float(f1_score(y, p, pos_label="fake", zero_division=0)),
        "focus_class": "fake",
        "confusion_labels": labels,
        "confusion_matrix": confusion_matrix(y, p, labels=labels).tolist(),
    }


def _prepare(df):
    obrigatorias = set(FEATURES_META + ["classe", "titulo", "texto"])
    faltantes = sorted(obrigatorias - set(df.columns))
    if faltantes:
        raise ValueError("Colunas ausentes no dataset: " + ", ".join(faltantes))

    work = df.copy()
    raw_rows = len(work)
    work["classe"] = _normalize_classes(work["classe"])
    work["titulo"] = work["titulo"].fillna("").astype(str)
    work["texto"] = work["texto"].fillna("").astype(str)

    if "extracao_elegivel" in work.columns:
        elig = work["extracao_elegivel"].astype(str).str.strip().str.lower()
        work = work[elig.isin(ELIGIBLE_TRUE)].copy()
    eligible_rows = len(work)

    url_col = next((c for c in ("url_dataset", "url_requisicao", "url_final", "url") if c in work.columns), None)
    removed_url = 0
    if url_col:
        urls = work[url_col].fillna("").astype(str).str.strip()
        dup = urls.ne("") & urls.duplicated(keep="first")
        removed_url = int(dup.sum())
        work = work[~dup].copy()

    work["conteudo"] = (work["titulo"] + " " + work["texto"]).str.strip()
    work = work[work["conteudo"] != ""].copy()
    work["_grupo"] = _groups(work)

    return work, {
        "raw_rows": int(raw_rows),
        "eligible_rows": int(eligible_rows),
        "same_url_removed": int(removed_url),
        "training_rows": int(len(work)),
        "url_column": url_col,
        "content_groups": int(work["_grupo"].nunique()),
    }


def main(csv_path):
    df = _read_csv(csv_path)
    df, prep = _prepare(df)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr_pos, te_pos = next(splitter.split(df, df["classe"], groups=df["_grupo"]))
    tr, te = df.iloc[tr_pos].copy(), df.iloc[te_pos].copy()

    overlap = len(set(tr["_grupo"]) & set(te["_grupo"]))
    if overlap:
        raise RuntimeError("Grupos de conteúdo atravessaram treino/teste.")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("nb", GaussianNB()),
    ])
    pipe.fit(tr[FEATURES_META].apply(pd.to_numeric, errors="coerce"), tr["classe"])
    pred_meta = pipe.predict(te[FEATURES_META].apply(pd.to_numeric, errors="coerce"))
    joblib.dump(pipe, OUT / "gaussian_nb_pipeline.joblib", compress=3)

    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", max_features=100000,
        ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
    )
    xtr = vectorizer.fit_transform(tr["conteudo"])
    xte = vectorizer.transform(te["conteudo"])
    joblib.dump(vectorizer, OUT / "tfidf_vectorizer.joblib", compress=3)

    mnb = MultinomialNB().fit(xtr, tr["classe"])
    pred_mnb = mnb.predict(xte)
    joblib.dump(mnb, OUT / "multinomial_nb.joblib", compress=3)

    logreg = LogisticRegression(max_iter=1000, random_state=42).fit(xtr, tr["classe"])
    pred_lr = logreg.predict(xte)
    joblib.dump(logreg, OUT / "logistic_regression.joblib", compress=3)

    metadata = {
        "version": "C1NC0_grouped_split_v4_url_content",
        "dataset_rows_original": prep["raw_rows"],
        "dataset_rows_eligible": prep["eligible_rows"],
        "same_url_duplicates_removed": prep["same_url_removed"],
        "dataset_rows_training": prep["training_rows"],
        "url_dedup_column": prep["url_column"],
        "content_groups": prep["content_groups"],
        "random_state": 42,
        "split": "aprox. 80/20 estratificado por grupos de conteúdo exato normalizado",
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "group_overlap": overlap,
        "gaussian_19_features": _metricas(te["classe"], pred_meta),
        "multinomial_tfidf_title_text": _metricas(te["classe"], pred_mnb),
        "logistic_regression_tfidf_title_text": _metricas(te["classe"], pred_lr),
        "tfidf_features": int(xtr.shape[1]),
        "limitation": (
            "Repetições da mesma URL são removidas. URLs diferentes com conteúdo exatamente igual "
            "são mantidas, mas permanecem no mesmo lado do split. Similaridade semântica entre "
            "textos reescritos ainda não é controlada."
        ),
        "warning": "Saídas experimentais; não representam prova ou probabilidade calibrada de veracidade.",
    }
    (OUT / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nArtefatos gravados em: {OUT}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit('Uso: python -m ml.train_models "caminho\\dataset.csv"')
    main(sys.argv[1])
