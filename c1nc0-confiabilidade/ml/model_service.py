from pathlib import Path
import json
import math

import joblib
import numpy as np

from .feature_extractor import FEATURES_META, NOMES_FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

DEFAULT_DATASET_INFO = {
    "nome": "DATASET_TERCEIROGENITO_V1_URL_UNIFORME_PT",
    "registros": 18217,
    "classes": {
        "true": 13513,
        "fake": 4704,
    },
    "percentuais": {
        "true": 74.18,
        "fake": 25.82,
    },
    "origens": {
        "DatasetFinal": 9974,
        "FakeRecogna": 4695,
        "Fake.Br Corpus": 3539,
        "Fake News Net": 9,
    },
    "status_coleta_ok": 18217,
    "representacoes": {
        "gaussian": "19 features estruturais/linguísticas",
        "multinomial": "título + texto com TF-IDF de unigramas e bigramas (até 100.000 características)",
    },
    "avaliacao": "divisão estratificada 80/20; random_state=42",
    "limitacao": (
        "Informações de compatibilidade com o dataset anterior. "
        "Execute o treinamento atualizado para gerar models/dataset_info.json."
    ),
}


def _carregar_dataset_info():
    """Carrega metadados do dataset usados no treinamento atual."""
    path = MODELS / "dataset_info.json"
    if not path.exists():
        return DEFAULT_DATASET_INFO

    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            return DEFAULT_DATASET_INFO
        return info
    except (OSError, json.JSONDecodeError):
        return DEFAULT_DATASET_INFO


DATASET_INFO = _carregar_dataset_info()


def modelos_disponiveis():
    return all(
        (MODELS / n).exists()
        for n in (
            "gaussian_nb_pipeline.joblib",
            "tfidf_vectorizer.joblib",
            "multinomial_nb.joblib",
        )
    )


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


def _valor_float(valor):
    """Converte feature numérica para float; ausentes viram NaN para o imputer."""
    if valor is None:
        return np.nan
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return np.nan
    return numero if math.isfinite(numero) else np.nan


def _vetor_features(features_url):
    """Mantém exatamente a ordem usada no treinamento, sem depender de pandas."""
    valores = [_valor_float(features_url.get(feature)) for feature in FEATURES_META]
    return np.asarray([valores], dtype=float)


def _formatar_valor(feature, valor):
    numero = _valor_float(valor)
    if math.isnan(numero):
        return "N/A"
    if feature in {"percentual_erros_ortograficos", "coerencia_titulos_pct"}:
        return f"{numero:.2f}%"
    if feature in {
        "comprimento_medio_sentenca",
        "comprimento_medio_palavra",
        "emotividade",
        "diversidade_lexical",
    }:
        return f"{numero:.2f}"
    return str(int(round(numero)))


def calcular_evidencias_features(pipeline, features_url):
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    nb = pipeline.named_steps["nb"]

    x_raw = _vetor_features(features_url)
    x_imp = imputer.transform(x_raw)
    x_scaled = scaler.transform(x_imp)

    classes = [str(c).lower() for c in nb.classes_]
    idx_true = classes.index("true")
    idx_fake = classes.index("fake")

    linhas = []

    for i, feature in enumerate(FEATURES_META):
        x = x_scaled[0, i]

        var_true = max(nb.var_[idx_true, i], 1e-12)
        var_fake = max(nb.var_[idx_fake, i], 1e-12)

        lt = -0.5 * (
            np.log(2 * np.pi * var_true)
            + ((x - nb.theta_[idx_true, i]) ** 2) / var_true
        )
        lf = -0.5 * (
            np.log(2 * np.pi * var_fake)
            + ((x - nb.theta_[idx_fake, i]) ** 2) / var_fake
        )

        evidencia = float(lt - lf)
        classe = "TRUE" if evidencia > 0 else "FAKE" if evidencia < 0 else "NEUTRA"

        valor_original = features_url.get(feature)
        valor_numerico = _valor_float(valor_original)
        imputada = bool(math.isnan(valor_numerico))

        linhas.append(
            {
                "feature": feature,
                "nome": NOMES_FEATURES.get(feature, feature),
                "valor": None if imputada else valor_numerico,
                "valor_formatado": _formatar_valor(feature, valor_original),
                "evidencia": evidencia,
                "classe_evidencia": classe,
                "foi_imputada": imputada,
            }
        )

    return linhas


def _resultado_gaussian(modelos, features_url):
    pipeline = modelos["meta"]
    evidencias = calcular_evidencias_features(pipeline, features_url)

    validas = [e for e in evidencias if not e["foi_imputada"]]
    true_like = sum(e["classe_evidencia"] == "TRUE" for e in validas)
    fake_like = sum(e["classe_evidencia"] == "FAKE" for e in validas)
    neutras = sum(e["classe_evidencia"] == "NEUTRA" for e in validas)

    direcionadas = true_like + fake_like
    pct_true = true_like / direcionadas * 100 if direcionadas else 0
    pct_fake = fake_like / direcionadas * 100 if direcionadas else 0

    # Executa explicitamente as etapas para não depender de DataFrame/pandas no runtime.
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    nb = pipeline.named_steps["nb"]
    x_raw = _vetor_features(features_url)
    classe = str(nb.predict(scaler.transform(imputer.transform(x_raw)))[0]).lower()

    return {
        "id": "gaussian_nb_v1",
        "nome": "Gaussian Naive Bayes",
        "representacao": "19 features estruturais/linguísticas",
        "classe_associada": classe,
        "evidencias": evidencias,
        "true_like": true_like,
        "fake_like": fake_like,
        "neutras": neutras,
        "percentual_true": round(pct_true, 2),
        "percentual_fake": round(pct_fake, 2),
    }


def _resultado_multinomial(modelos, dados):
    conteudo = f"{dados.get('titulo', '')} {dados.get('texto', '')}".strip()
    x_texto = modelos["vectorizer"].transform([conteudo])
    classe = str(modelos["texto"].predict(x_texto)[0]).lower()

    return {
        "id": "multinomial_nb_v1",
        "nome": "Multinomial Naive Bayes",
        "representacao": "Título + texto com TF-IDF",
        "classe_associada": classe,
    }


def analisar_com_modelos(dados, features_url, modo="comparar"):
    modelos = carregar_modelos()
    if modelos is None:
        return {
            "disponivel": False,
            "erro": "Artefatos treinados não encontrados em /models.",
        }

    if modo not in {"gaussian", "multinomial", "comparar"}:
        modo = "comparar"

    resultado = {
        "disponivel": True,
        "modo": modo,
        "gaussian": None,
        "multinomial": None,
        "modelos_concordaram": None,
        "metadata": modelos["metadata"],
        "dataset": DATASET_INFO,
    }

    if modo in {"gaussian", "comparar"}:
        resultado["gaussian"] = _resultado_gaussian(modelos, features_url)

    if modo in {"multinomial", "comparar"}:
        resultado["multinomial"] = _resultado_multinomial(modelos, dados)

    if modo == "comparar":
        resultado["modelos_concordaram"] = (
            resultado["gaussian"]["classe_associada"]
            == resultado["multinomial"]["classe_associada"]
        )

    return resultado
