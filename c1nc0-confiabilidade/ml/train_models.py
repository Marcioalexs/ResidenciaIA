"""Treina e serializa os modelos usados pela aplicação Web.

Uso:
  python -m ml.train_models "C:\\caminho\\dataset.csv"

O CSV deve conter 'classe', 'titulo', 'texto' e as 19 features de FEATURES_META.
"""
from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .feature_extractor import FEATURES_META

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)


def main(csv_path):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    obrigatorias = set(FEATURES_META + ["classe", "titulo", "texto"])
    faltantes = sorted(obrigatorias - set(df.columns))
    if faltantes:
        raise ValueError("Colunas ausentes no dataset: " + ", ".join(faltantes))

    # Experimento 1 — mesmas condições do Naive_Bayes_V1.ipynb.
    x_meta = df[FEATURES_META].apply(pd.to_numeric, errors="coerce")
    y_meta = df["classe"]
    xtr, xte, ytr, yte = train_test_split(x_meta, y_meta, test_size=0.20, random_state=42, stratify=y_meta)
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("nb", GaussianNB())])
    pipe.fit(xtr, ytr)
    pred_meta = pipe.predict(xte)
    joblib.dump(pipe, OUT / "gaussian_nb_pipeline.joblib", compress=3)

    # Experimento 2 — título + texto, mesmas configurações do notebook.
    texto = df[["titulo", "texto", "classe"]].copy()
    texto["titulo"] = texto["titulo"].fillna("").astype(str)
    texto["texto"] = texto["texto"].fillna("").astype(str)
    texto["conteudo"] = (texto["titulo"] + " " + texto["texto"]).str.strip()
    texto = texto[texto["conteudo"] != ""].copy()
    idx_tr, idx_te = train_test_split(texto.index, test_size=0.20, random_state=42, stratify=texto["classe"])
    tr, te = texto.loc[idx_tr], texto.loc[idx_te]
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", max_features=100000,
                                 ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
    xtr_t = vectorizer.fit_transform(tr["conteudo"])
    xte_t = vectorizer.transform(te["conteudo"])
    model = MultinomialNB().fit(xtr_t, tr["classe"])
    pred_texto = model.predict(xte_t)
    joblib.dump(vectorizer, OUT / "tfidf_vectorizer.joblib", compress=3)
    joblib.dump(model, OUT / "multinomial_nb.joblib", compress=3)

    def metricas(y, p):
        return {
            "accuracy": accuracy_score(y, p),
            "precision_fake": precision_score(y, p, pos_label="fake"),
            "recall_fake": recall_score(y, p, pos_label="fake"),
            "f1_fake": f1_score(y, p, pos_label="fake"),
        }

    metadata = {
        "version": "Naive_Bayes_V1",
        "random_state": 42,
        "split": "80/20 estratificado",
        "gaussian_19_features": metricas(yte, pred_meta),
        "multinomial_tfidf_title_text": metricas(te["classe"], pred_texto),
        "tfidf_features": int(xtr_t.shape[1]),
        "warning": "Saídas experimentais; não representam prova ou probabilidade calibrada de veracidade.",
    }
    (OUT / "model_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nArtefatos gravados em: {OUT}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit('Uso: python -m ml.train_models "caminho\\dataset.csv"')
    main(sys.argv[1])
