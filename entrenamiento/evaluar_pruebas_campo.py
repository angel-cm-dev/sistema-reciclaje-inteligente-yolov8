"""Calcula los resultados de las pruebas con objetos reales (ver pruebas_campo/protocolo.md).

Uso: python entrenamiento/evaluar_pruebas_campo.py pruebas_campo/formato_pruebas.csv
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
NINGUNA = "ninguna"


def leer(ruta: Path) -> list:
    with ruta.open(encoding="utf-8") as archivo:
        filas = [f for f in csv.DictReader(archivo) if f.get("clase_real", "").strip()]
    for f in filas:
        f["clase_real"] = f["clase_real"].strip().lower()
        f["clase_predicha"] = (f.get("clase_predicha") or NINGUNA).strip().lower() or NINGUNA
        f["iluminacion"] = (f.get("iluminacion") or "normal").strip().lower() or "normal"
    return filas


def calcular(filas: list) -> dict:
    """Exactitud global, métricas por clase, matriz de confusión y exactitud por iluminación."""
    if not filas:
        raise ValueError("El archivo no tiene pruebas registradas.")
    clases = sorted({f["clase_real"] for f in filas})
    matriz = Counter((f["clase_real"], f["clase_predicha"]) for f in filas)

    por_clase = {}
    for c in clases:
        vp = matriz[(c, c)]
        predichas = sum(n for (_, p), n in matriz.items() if p == c)
        reales = sum(n for (r, _), n in matriz.items() if r == c)
        precision = vp / predichas if predichas else 0.0
        recall = vp / reales if reales else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        por_clase[c] = {"n": reales, "precision": precision, "recall": recall, "f1": f1}

    aciertos = defaultdict(lambda: [0, 0])
    for f in filas:
        aciertos[f["iluminacion"]][0] += f["clase_real"] == f["clase_predicha"]
        aciertos[f["iluminacion"]][1] += 1

    columnas = clases + ([NINGUNA] if any(p == NINGUNA for _, p in matriz) else [])
    return {
        "total": len(filas),
        "exactitud": sum(matriz[(c, c)] for c in clases) / len(filas),
        "por_clase": por_clase,
        "matriz": {r: {p: matriz[(r, p)] for p in columnas} for r in clases},
        "columnas": columnas,
        "por_iluminacion": {k: a / n for k, (a, n) in sorted(aciertos.items())},
    }


def a_markdown(r: dict) -> str:
    l = [f"**Pruebas:** {r['total']} · **Exactitud global:** {r['exactitud']:.1%}", "",
         "| Clase | Pruebas | Precisión | Recall | F1 |", "|---|---|---|---|---|"]
    l += [f"| {c} | {m['n']} | {m['precision']:.1%} | {m['recall']:.1%} | {m['f1']:.1%} |"
          for c, m in r["por_clase"].items()]
    l += ["", "**Matriz de confusión** (filas: clase real, columnas: clase predicha)", "",
          "| Real \\ Predicha | " + " | ".join(r["columnas"]) + " |", "|---|" + "---|" * len(r["columnas"])]
    l += [f"| {c} | " + " | ".join(str(v) for v in fila.values()) + " |" for c, fila in r["matriz"].items()]
    l += ["", "| Iluminación | Exactitud |", "|---|---|"]
    l += [f"| {k} | {v:.1%} |" for k, v in r["por_iluminacion"].items()]
    return "\n".join(l) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archivo", nargs="?", default=str(RAIZ / "pruebas_campo" / "formato_pruebas.csv"))
    args = parser.parse_args()

    filas = leer(Path(args.archivo))
    if not filas:
        print("Aún no hay pruebas registradas. Siga pruebas_campo/protocolo.md.")
        return
    texto = a_markdown(calcular(filas))
    salida = RAIZ / "resultados" / "pruebas_campo"
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "resultados.md").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Guardado en {salida / 'resultados.md'}")


if __name__ == "__main__":
    main()
