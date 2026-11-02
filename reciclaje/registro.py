"""Registro CSV de las clasificaciones confirmadas: evidencia para el análisis de resultados."""

import csv
import time
from datetime import datetime
from pathlib import Path

from . import config

COLUMNAS = ["fecha_hora", "categoria", "clase_modelo", "confianza"]


class Registro:
    def __init__(self, ruta: Path = config.RUTA_REGISTRO,
                 segundos_entre_registros: float = config.SEGUNDOS_ENTRE_REGISTROS,
                 reloj=time.monotonic):
        self.ruta = Path(ruta)
        self.espera = segundos_entre_registros
        self._reloj = reloj
        self._ultimo = None  # (categoria, instante)

    def guardar(self, deteccion) -> bool:
        """Guarda la detección salvo que sea el mismo objeto registrado hace muy poco."""
        ahora = self._reloj()
        if self._ultimo and self._ultimo[0] == deteccion.categoria and ahora - self._ultimo[1] < self.espera:
            return False

        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        nuevo = not self.ruta.exists()
        with self.ruta.open("a", newline="", encoding="utf-8") as archivo:
            escritor = csv.writer(archivo)
            if nuevo:
                escritor.writerow(COLUMNAS)
            escritor.writerow([
                datetime.now().isoformat(timespec="seconds"),
                deteccion.categoria,
                deteccion.clase_modelo,
                f"{deteccion.confianza:.3f}",
            ])
        self._ultimo = (deteccion.categoria, ahora)
        return True
