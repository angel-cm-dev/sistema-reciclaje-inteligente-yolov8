"""Prepara el "Trash Detection Dataset" (Kan, 2025) con las 5 categorías de la aplicación.

Fuente: https://data.mendeley.com/datasets/z732f9pwxt (DOI 10.17632/z732f9pwxt.1, CC BY 4.0)

Qué hace:
1. Reagrupa las 6 clases originales en las 5 de la interfaz:
   BIODEGRADABLE -> Organico, CARDBOARD y PAPER -> Papel_Carton,
   GLASS -> Vidrio, METAL -> Metal, PLASTIC -> Plastico.
2. Convierte las etiquetas de polígono a cajas (x_centro, y_centro, ancho, alto),
   porque el dataset mezcla ambos formatos y el detector usa cajas.
3. Copia todo a datos/ con la estructura de entrenamiento/dataset.yaml y
   muestra cuántos objetos hay por clase y por partición.

Uso:
    python entrenamiento/preparar_dataset.py --origen "datos_crudos/Trash Detection Dataset/CUSTOM_DATASET"
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

CLASES = ["Metal", "Vidrio", "Plastico", "Papel_Carton", "Organico"]
ORIGINALES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
REMAPEO = {
    "BIODEGRADABLE": "Organico",
    "CARDBOARD": "Papel_Carton",
    "PAPER": "Papel_Carton",
    "GLASS": "Vidrio",
    "METAL": "Metal",
    "PLASTIC": "Plastico",
}
PARTICIONES = {"train": "train", "valid": "val", "test": "test"}


def linea_a_caja(linea: str):
    """Devuelve (clase_nueva, xc, yc, w, h) o None si la línea está vacía o es inválida."""
    partes = linea.split()
    if len(partes) < 5:
        return None
    clase = CLASES.index(REMAPEO[ORIGINALES[int(partes[0])]])
    valores = [float(v) for v in partes[1:]]
    if len(valores) == 4:
        xc, yc, w, h = valores
    else:
        # Polígono x1 y1 x2 y2 ...: la caja es el rectángulo que lo contiene.
        xs, ys = valores[0::2], valores[1::2]
        x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
        y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
        xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return clase, xc, yc, w, h


def convertir_etiquetas(texto: str) -> list:
    return [c for c in (linea_a_caja(l) for l in texto.splitlines()) if c]


def preparar(origen: Path, destino: Path) -> dict:
    conteo = {p: Counter() for p in PARTICIONES.values()}
    imagenes = Counter()
    for nombre_origen, particion in PARTICIONES.items():
        dir_img, dir_lbl = destino / "images" / particion, destino / "labels" / particion
        dir_img.mkdir(parents=True, exist_ok=True)
        dir_lbl.mkdir(parents=True, exist_ok=True)
        for imagen in sorted((origen / nombre_origen / "images").iterdir()):
            etiqueta = origen / nombre_origen / "labels" / (imagen.stem + ".txt")
            cajas = convertir_etiquetas(etiqueta.read_text(encoding="utf-8")) if etiqueta.exists() else []
            shutil.copy2(imagen, dir_img / imagen.name)
            (dir_lbl / (imagen.stem + ".txt")).write_text(
                "".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for c, xc, yc, w, h in cajas),
                encoding="utf-8",
            )
            imagenes[particion] += 1
            conteo[particion].update(CLASES[c] for c, *_ in cajas)
    return {"imagenes": imagenes, "objetos": conteo}


def tabla(resumen: dict) -> str:
    filas = ["| Clase | train | val | test | Total |", "|---|---|---|---|---|"]
    for clase in CLASES:
        n = [resumen["objetos"][p][clase] for p in ("train", "val", "test")]
        filas.append(f"| {clase} | {n[0]} | {n[1]} | {n[2]} | {sum(n)} |")
    im = [resumen["imagenes"][p] for p in ("train", "val", "test")]
    filas.append(f"| **Imágenes** | {im[0]} | {im[1]} | {im[2]} | {sum(im)} |")
    return "\n".join(filas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--origen", default=str(RAIZ / "datos_crudos" / "Trash Detection Dataset" / "CUSTOM_DATASET"))
    parser.add_argument("--destino", default=str(RAIZ / "datos"))
    args = parser.parse_args()

    destino = Path(args.destino)
    if destino.exists():
        shutil.rmtree(destino)  # se regenera completo para no mezclar versiones
    print(tabla(preparar(Path(args.origen), destino)))


if __name__ == "__main__":
    main()
