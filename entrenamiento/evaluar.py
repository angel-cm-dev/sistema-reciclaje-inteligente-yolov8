"""Evalúa el modelo y genera las métricas para el capítulo de resultados de la tesis.

Uso:
    python entrenamiento/evaluar.py                      # usa modelos/best.pt y el split "test"
    python entrenamiento/evaluar.py --split val

Genera en resultados/evaluacion/:
    metricas.csv            precisión, recall, mAP50 y mAP50-95 por clase y global
    metricas.md             la misma tabla lista para pegar en el documento
    confusion_matrix*.png   matriz de confusión (absoluta y normalizada)
    *_curve.png             curvas precisión-recall, F1, etc.
"""

import argparse
import csv
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", default=str(RAIZ / "modelos" / "best.pt"))
    parser.add_argument("--datos", default=str(RAIZ / "entrenamiento" / "dataset.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--dispositivo", default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    salida = RAIZ / "resultados"
    metricas = YOLO(args.modelo).val(
        data=args.datos, split=args.split, device=args.dispositivo,
        project=str(salida), name="evaluacion", exist_ok=True, plots=True,
    )

    filas = [("Clase", "Precisión", "Recall", "mAP50", "mAP50-95")]
    for i, indice in enumerate(metricas.box.ap_class_index):
        p, r, map50, map5095 = metricas.box.class_result(i)
        filas.append((metricas.names[int(indice)], p, r, map50, map5095))
    filas.append(("Todas", metricas.box.mp, metricas.box.mr, metricas.box.map50, metricas.box.map))

    carpeta = salida / "evaluacion"
    with (carpeta / "metricas.csv").open("w", newline="", encoding="utf-8") as archivo:
        csv.writer(archivo).writerows(
            [filas[0]] + [(f[0], *(f"{v:.4f}" for v in f[1:])) for f in filas[1:]]
        )

    lineas = ["| " + " | ".join(filas[0]) + " |", "|" + "---|" * len(filas[0])]
    lineas += ["| " + " | ".join([f[0], *(f"{v:.3f}" for v in f[1:])]) + " |" for f in filas[1:]]
    (carpeta / "metricas.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))
    print(f"\nResultados guardados en {carpeta}")


if __name__ == "__main__":
    main()
