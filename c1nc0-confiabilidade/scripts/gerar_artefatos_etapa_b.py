r"""CLI para gerar os artefatos estáticos B1–B5 do C1NC0.

Exemplo (PowerShell, a partir de c1nc0-confiabilidade):
    python .\scripts\gerar_artefatos_etapa_b.py `
      --dataset "C:\caminho\03_Dataset_Final_C1NC0-Tabular.csv"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.laboratorio_service import gerar_artefatos_etapa_b  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera dataset-summary, comparação, erros, Model Card e K-Means/PCA para /laboratorio."
    )
    parser.add_argument("--dataset", required=True, help="Caminho para o CSV do dataset C1NC0.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "static" / "laboratorio"),
        help="Diretório de saída dos JSONs (padrão: static/laboratorio).",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-tfidf-features", type=int, default=100000)
    parser.add_argument("--error-examples", type=int, default=6)
    parser.add_argument("--clusters", type=int, default=2)
    parser.add_argument("--cluster-points", type=int, default=2000)
    parser.add_argument("--cluster-text-features", type=int, default=20000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        print(f"ERRO: dataset não encontrado: {dataset_path}", file=sys.stderr)
        return 2

    output = Path(args.output).expanduser().resolve()
    data = dataset_path.read_bytes()

    print("=" * 72)
    print("C1NC0 — GERADOR DE ARTEFATOS DA ETAPA B")
    print("=" * 72)
    print(f"Dataset : {dataset_path}")
    print(f"Saída   : {output}")
    print(f"Bytes   : {len(data):,}".replace(",", "."))
    print("\nGerando B1–B5 localmente...")

    artifacts = gerar_artefatos_etapa_b(
        data,
        dataset_path.name,
        output,
        test_size=args.test_size,
        random_state=args.random_state,
        max_tfidf_features=args.max_tfidf_features,
        error_limit=args.error_examples,
        n_clusters=args.clusters,
        max_cluster_points=args.cluster_points,
        max_cluster_text_features=args.cluster_text_features,
    )

    summary = artifacts["dataset_summary"]
    comparison = artifacts["model_comparison"]
    clusters = artifacts["clusters"]

    print("\nConcluído.")
    print(f"Registros: {summary['registros']:,}".replace(",", "."))
    print(f"Colunas  : {summary['colunas']}")
    print("Modelos  :")
    for model in comparison["modelos"]:
        if model.get("metricas"):
            metrics = model["metricas"]
            print(
                f"  - {model['modelo']}: accuracy={metrics['accuracy']:.2%}, "
                f"F1={metrics['f1']:.2%}, balanced_accuracy={metrics['balanced_accuracy']:.2%}"
            )
        else:
            print(f"  - {model['modelo']}: {model.get('status')} — {model.get('motivo', '')}")
    print(
        f"K-Means/PCA: {clusters['registros_clusterizados']:,} registros; "
        f"{clusters['pontos_visualizados']:,} pontos no gráfico".replace(",", ".")
    )
    print("\nArquivos gerados:")
    for path in sorted(output.glob("*.json")):
        print(f"  - {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
