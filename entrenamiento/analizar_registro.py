"""Resume resultados/clasificaciones.csv (las clasificaciones hechas con la app en uso real).

Uso: python entrenamiento/analizar_registro.py
Muestra cuántos objetos se clasificaron por categoría y la confianza promedio.
"""

import csv
from collections import defaultdict
from pathlib import Path

RUTA = Path(__file__).resolve().parent.parent / "resultados" / "clasificaciones.csv"


def resumir(ruta: Path = RUTA) -> dict:
    conteo, suma = defaultdict(int), defaultdict(float)
    with ruta.open(encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            conteo[fila["categoria"]] += 1
            suma[fila["categoria"]] += float(fila["confianza"])
    return {c: (conteo[c], suma[c] / conteo[c]) for c in sorted(conteo)}


def main() -> None:
    if not RUTA.exists():
        print(f"Aún no hay registros en {RUTA}. Use la aplicación para generarlos.")
        return
    resumen = resumir()
    total = sum(n for n, _ in resumen.values())
    print(f"{'Categoría':<15}{'Cantidad':>10}{'%':>8}{'Confianza prom.':>18}")
    for categoria, (n, conf) in resumen.items():
        print(f"{categoria:<15}{n:>10}{n / total:>8.1%}{conf:>18.1%}")
    print(f"{'Total':<15}{total:>10}")


if __name__ == "__main__":
    main()
