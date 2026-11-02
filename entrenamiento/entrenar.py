"""Entrena (o reentrena) el detector de residuos con YOLOv8.

Ejemplos:
    python entrenamiento/entrenar.py
    python entrenamiento/entrenar.py --modelo yolov8s.pt --epocas 100 --lote 16

El mejor modelo queda en runs/detect/<nombre>/weights/best.pt; cópielo a modelos/best.pt
para que la aplicación lo use.
"""

import argparse
from pathlib import Path

AQUI = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datos", default=str(AQUI / "dataset.yaml"), help="archivo YAML del dataset")
    parser.add_argument("--modelo", default="yolov8l.pt", help="pesos base (n, s, m, l, x)")
    parser.add_argument("--epocas", type=int, default=100)
    parser.add_argument("--lote", type=int, default=8, help="imágenes por lote; bájelo si falta memoria de GPU")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--paciencia", type=int, default=20, help="épocas sin mejora antes de detenerse")
    parser.add_argument("--dispositivo", default=None, help="'0' para GPU, 'cpu' para procesador")
    parser.add_argument("--nombre", default="reciclaje")
    parser.add_argument("--semilla", type=int, default=42, help="fija la aleatoriedad para poder repetir el experimento")
    args = parser.parse_args()

    from ultralytics import YOLO

    modelo = YOLO(args.modelo)
    modelo.train(
        data=args.datos,
        epochs=args.epocas,
        batch=args.lote,
        imgsz=args.imgsz,
        patience=args.paciencia,
        device=args.dispositivo,
        name=args.nombre,
        seed=args.semilla,
        deterministic=True,
        plots=True,  # guarda curvas de pérdida, PR y matriz de confusión
    )


if __name__ == "__main__":
    main()
