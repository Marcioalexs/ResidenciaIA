"""Carregamento leve dos artefatos pré-calculados da Etapa B.

Este módulo é deliberadamente livre de pandas/scikit-learn. No Vercel, o
Laboratório apenas apresenta os JSONs gerados localmente; toda a análise do
CSV, treinamento, métricas, matriz de confusão e K-Means/PCA permanecem no
módulo offline ``services.laboratorio_service``.
"""
from __future__ import annotations

import json
from pathlib import Path


ARTIFACT_FILENAMES = {
    "manifest": "manifest.json",
    "dataset_summary": "dataset-summary.json",
    "model_comparison": "model-comparison.json",
    "confusion_matrix": "confusion-matrix.json",
    "error_examples": "error-examples.json",
    "model_card": "model-card.json",
    "clusters": "clusters.json",
}


def carregar_artefatos_etapa_b(base_dir: str | Path | None = None) -> dict:
    """Carrega os JSONs B1–B5 sem executar qualquer processamento pesado."""
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
