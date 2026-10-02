"""Treina e serializa baselines supervisionados do C1NC0 com split agrupado.

Uso:
  python -m ml.train_models "C:\\caminho\\dataset.csv"

O split mantém conteúdos exatamente iguais (após normalização) no mesmo lado.
A detecção de similaridade semântica entre republicações reescritas continua sendo
uma etapa metodológica futura.
"""
from pathlib import Path
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


def _read_csv(path):
    for sep in (";", ","):
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)
            if len(df.columns) > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")


def _norm(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).lower()
    return re.sub(r"\s+", " ", value).strip()


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
    pos = "fake" if "fake" in labels else labels[-1]
    return {
        "accuracy": float(accuracy_score(y, p)),
        "precision_focus": float(precision_score(y, p, pos_label=pos, zero_division=0)),
        "recall_focus": float(recall_score(y, p, pos_label=pos, zero_division=0)),
        "f1_focus": float(f1_score(y, p, pos_label=pos, zero_division=0)),
        "focus_class": pos,
        "confusion_labels": labels,
        "confusion_matrix": confusion_matrix(y, p, labels=labels).tolist(),
    }


def main(csv_path):
    df = _read_csv(csv_path)
    obrigatorias = set(FEATURES_META + ["classe", "titulo", "texto"])
    faltantes = sorted(obrigatorias - set(df.columns))
    if faltantes:
        raise ValueError("Colunas ausentes no dataset: " + ", ".join(faltantes))

    df = df.copy()
    df["classe"] = df["classe"].fillna("").astype(str).str.strip().str.lower()
    df["titulo"] = df["titulo"].fillna("").astype(str)
    df["texto"] = df["texto"].fillna("").astype(str)
    df["conteudo"] = (df["titulo"] + " " + df["texto"]).str.strip()
    df = df[(df["classe"] != "") & (df["conteudo"] != "")].copy()
    df["_grupo"] = _groups(df)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr_pos, te_pos = next(splitter.split(df, df["classe"], groups=df["_grupo"]))
    tr, te = df.iloc[tr_pos].copy(), df.iloc[te_pos].copy()

    overlap = len(set(tr["_grupo"]) & set(te["_grupo"]))
    if overlap:
        raise RuntimeError("Grupos de conteúdo atravessaram treino/teste.")

    # 1) GaussianNB — 19 features
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("nb", GaussianNB()),
    ])
    pipe.fit(tr[FEATURES_META].apply(pd.to_numeric, errors="coerce"), tr["classe"])
    pred_meta = pipe.predict(te[FEATURES_META].apply(pd.to_numeric, errors="coerce"))
    joblib.dump(pipe, OUT / "gaussian_nb_pipeline.joblib", compress=3)

    # 2 e 3) TF-IDF compartilhado: MultinomialNB + LogisticRegression
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
        "version": "C1NC0_grouped_split_v2",
        "dataset_rows_eligible": int(len(df)),
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
            "O agrupamento cobre duplicatas exatas normalizadas; similaridade semântica "
            "entre republicações reescritas ainda não é controlada."
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
