from pathlib import Path
import json
import math

import joblib
import numpy as np

from .feature_extractor import DESCRICOES_FEATURES, FEATURES_META, NOMES_FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

FALLBACK_DATASET_INFO = {
    "nome": "Dataset de treinamento não identificado",
    "arquivo": None,
    "registros_arquivo_original": 0,
    "registros": 0,
    "classes": {"true": 0, "fake": 0},
    "percentuais": {"true": 0.0, "fake": 0.0},
    "origens": {},
    "origens_total": 0,
    "status_coleta_ok": 0,
    "representacoes": {
        "gaussian": "19 features estruturais/linguísticas",
        "multinomial": "título + texto com TF-IDF de unigramas e bigramas (até 100.000 características)",
    },
    "avaliacao": "split agrupado por conteúdo; random_state=42",
    "limitacao": (
        "As métricas são experimentais e dependem da composição do dataset. "
        "O modelo não verifica fatos nem produz probabilidade calibrada de veracidade."
    ),
    "duplicidades": {},
}


def carregar_dataset_info():
    """Carrega do treinamento as informações exibidas pela interface.

    O arquivo ``models/dataset_info.json`` é gerado pelo script unificado de
    treinamento. O fallback mantém a aplicação funcional caso o arquivo ainda
    não tenha sido publicado.
    """
    path = MODELS / "dataset_info.json"
    if not path.exists():
        return dict(FALLBACK_DATASET_INFO)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(FALLBACK_DATASET_INFO)

    info = dict(FALLBACK_DATASET_INFO)
    info.update(payload if isinstance(payload, dict) else {})
    info["classes"] = {
        **FALLBACK_DATASET_INFO["classes"],
        **(info.get("classes") or {}),
    }
    info["percentuais"] = {
        **FALLBACK_DATASET_INFO["percentuais"],
        **(info.get("percentuais") or {}),
    }
    info["representacoes"] = {
        **FALLBACK_DATASET_INFO["representacoes"],
        **(info.get("representacoes") or {}),
    }
    info["origens"] = info.get("origens") or {}
    info["duplicidades"] = info.get("duplicidades") or {}
    return info


DATASET_INFO = carregar_dataset_info()


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


def _formatar_numero(feature, numero):
    numero = _valor_float(numero)
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


def _formatar_valor(feature, valor):
    return _formatar_numero(feature, valor)


def _explicacao_evidencia(
    *,
    classe: str,
    feature: str,
    valor_usado: float,
    media_true: float,
    media_fake: float,
    foi_imputada: bool,
) -> str:
    nome = NOMES_FEATURES.get(feature, feature)
    valor_txt = _formatar_numero(feature, valor_usado)
    true_txt = _formatar_numero(feature, media_true)
    fake_txt = _formatar_numero(feature, media_fake)

    if classe == "NEUTRA":
        base = (
            f"Para {nome.lower()}, o valor usado ({valor_txt}) não inclinou a comparação "
            "de forma relevante para nenhum dos dois grupos."
        )
    else:
        base = (
            f"O modelo comparou o valor usado ({valor_txt}) com as distribuições aprendidas "
            f"nos grupos TRUE e FAKE. Neste caso, esse valor foi estatisticamente mais compatível "
            f"com o padrão {classe}, por isso esta característica favoreceu {classe}. "
            f"Como referência, as médias observadas no treinamento foram TRUE={true_txt} e FAKE={fake_txt}."
        )

    if foi_imputada:
        base += (
            " O valor desta característica não pôde ser calculado na página; o pipeline usou "
            "o valor de imputação aprendido no treinamento."
        )

    return base + " Isso é uma associação estatística do dataset, não uma prova de veracidade."


def calcular_evidencias_features(pipeline, features_url):
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    nb = pipeline.named_steps["nb"]

    x_raw = _vetor_features(features_url)
    x_imp = imputer.transform(x_raw)
    x_scaled = scaler.transform(x_imp)

    classes = [str(c).lower() for c in nb.classes_]
    if "true" not in classes or "fake" not in classes:
        raise RuntimeError(
            "O modelo publicado precisa usar as classes normalizadas 'true' e 'fake'. "
            f"Classes encontradas: {classes}"
        )
    idx_true = classes.index("true")
    idx_fake = classes.index("fake")

    # O GaussianNB foi treinado depois do StandardScaler. A inversão abaixo serve
    # apenas para apresentar as médias em unidades compreensíveis ao usuário.
    medias_originais = scaler.inverse_transform(nb.theta_)

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
        classe = "TRUE" if evidencia > 1e-12 else "FAKE" if evidencia < -1e-12 else "NEUTRA"

        valor_original = features_url.get(feature)
        valor_numerico = _valor_float(valor_original)
        imputada = bool(math.isnan(valor_numerico))
        valor_usado = float(x_imp[0, i])
        media_true = float(medias_originais[idx_true, i])
        media_fake = float(medias_originais[idx_fake, i])

        linhas.append(
            {
                "feature": feature,
                "nome": NOMES_FEATURES.get(feature, feature),
                "descricao": DESCRICOES_FEATURES.get(feature, "Característica numérica extraída do conteúdo."),
                "valor": None if imputada else valor_numerico,
                "valor_formatado": _formatar_valor(feature, valor_original),
                "valor_usado_formatado": _formatar_numero(feature, valor_usado),
                "media_true_formatada": _formatar_numero(feature, media_true),
                "media_fake_formatada": _formatar_numero(feature, media_fake),
                "evidencia": evidencia,
                "classe_evidencia": classe,
                "foi_imputada": imputada,
                "explicacao_leiga": _explicacao_evidencia(
                    classe=classe,
                    feature=feature,
                    valor_usado=valor_usado,
                    media_true=media_true,
                    media_fake=media_fake,
                    foi_imputada=imputada,
                ),
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
        "explicacao_leiga": (
            "O Gaussian Naive Bayes não decide se a notícia é verdadeira ou falsa. "
            "Ele compara cada característica da página com padrões estatísticos aprendidos "
            "nos grupos rotulados TRUE e FAKE do dataset. Cada característica pode inclinar "
            "a comparação para um desses grupos, e o conjunto delas produz a associação final."
        ),
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
        "explicacao_leiga": (
            "O modelo textual transforma palavras e combinações de palavras em números e compara "
            "esse padrão com o que aprendeu no treinamento. Uma associação com TRUE ou FAKE quer "
            "dizer semelhança com um dos grupos do dataset; o modelo não consulta a internet nem confirma fatos."
        ),
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
        "aviso_interpretacao": (
            "TRUE e FAKE são nomes dos grupos usados no treinamento. A saída mostra associação "
            "estatística com esses grupos e não funciona como checagem factual da notícia."
        ),
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
