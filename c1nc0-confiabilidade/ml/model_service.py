from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd

from .feature_extractor import FEATURES_META, NOMES_FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def modelos_disponiveis():
    return all((MODELS / n).exists() for n in ("gaussian_nb_pipeline.joblib", "tfidf_vectorizer.joblib", "multinomial_nb.joblib"))


def carregar_modelos():
    if not modelos_disponiveis():
        return None
    metadata = {}
    meta_path = MODELS / "model_metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "meta": joblib.load(MODELS / "gaussian_nb_pipeline.joblib"),
        "vectorizer": joblib.load(MODELS / "tfidf_vectorizer.joblib"),
        "texto": joblib.load(MODELS / "multinomial_nb.joblib"),
        "metadata": metadata,
    }


def _formatar_valor(feature, valor):
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "N/A"
    if feature in {"percentual_erros_ortograficos", "coerencia_titulos_pct"}:
        return f"{float(valor):.2f}%"
    if feature in {"comprimento_medio_sentenca", "comprimento_medio_palavra", "emotividade", "diversidade_lexical"}:
        return f"{float(valor):.2f}"
    return str(int(round(float(valor))))


def calcular_evidencias_features(pipeline, features_url):
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    nb = pipeline.named_steps["nb"]
    df_url = pd.DataFrame([features_url], columns=FEATURES_META)
    originais = df_url.iloc[0].copy()
    x_imp = imputer.transform(df_url)
    x_scaled = scaler.transform(x_imp)
    classes = [str(c).lower() for c in nb.classes_]
    idx_true, idx_fake = classes.index("true"), classes.index("fake")
    linhas = []
    for i, feature in enumerate(FEATURES_META):
        x = x_scaled[0, i]
        var_true = max(nb.var_[idx_true, i], 1e-12)
        var_fake = max(nb.var_[idx_fake, i], 1e-12)
        lt = -0.5 * (np.log(2 * np.pi * var_true) + ((x - nb.theta_[idx_true, i]) ** 2) / var_true)
        lf = -0.5 * (np.log(2 * np.pi * var_fake) + ((x - nb.theta_[idx_fake, i]) ** 2) / var_fake)
        evidencia = float(lt - lf)
        classe = "TRUE" if evidencia > 0 else "FAKE" if evidencia < 0 else "NEUTRA"
        valor = originais[feature]
        imputada = bool(pd.isna(valor))
        linhas.append({
            "feature": feature, "nome": NOMES_FEATURES.get(feature, feature),
            "valor": None if imputada else float(valor), "valor_formatado": _formatar_valor(feature, valor),
            "evidencia": evidencia, "classe_evidencia": classe, "foi_imputada": imputada,
        })
    return linhas


def analisar_com_modelos(dados, features_url):
    modelos = carregar_modelos()
    if modelos is None:
        return {"disponivel": False, "erro": "Artefatos treinados não encontrados em /models."}

    evidencias = calcular_evidencias_features(modelos["meta"], features_url)
    validas = [e for e in evidencias if not e["foi_imputada"]]
    true_like = sum(e["classe_evidencia"] == "TRUE" for e in validas)
    fake_like = sum(e["classe_evidencia"] == "FAKE" for e in validas)
    neutras = sum(e["classe_evidencia"] == "NEUTRA" for e in validas)
    direcionadas = true_like + fake_like
    pct_true = true_like / direcionadas * 100 if direcionadas else 0
    pct_fake = fake_like / direcionadas * 100 if direcionadas else 0

    conteudo = f"{dados.get('titulo', '')} {dados.get('texto', '')}".strip()
    x_texto = modelos["vectorizer"].transform([conteudo])
    classe_experimental = str(modelos["texto"].predict(x_texto)[0])

    return {
        "disponivel": True,
        "evidencias": evidencias,
        "true_like": true_like, "fake_like": fake_like, "neutras": neutras,
        "percentual_true": round(pct_true, 2), "percentual_fake": round(pct_fake, 2),
        # Mantida para rastreabilidade científica; a interface NÃO a exibe como veredito.
        "classe_textual_experimental": classe_experimental,
        "metadata": modelos["metadata"],
    }
