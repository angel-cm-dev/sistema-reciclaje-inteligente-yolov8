"""Lógica de detección independiente de la cámara y de la interfaz (fácil de probar)."""

import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from . import config


@dataclass(frozen=True)
class Deteccion:
    clase_modelo: str          # nombre tal como lo entrega el modelo, p. ej. "Carton"
    categoria: Optional[str]   # clave de config.CATEGORIAS, o None si no se reconoce
    confianza: float
    caja: tuple                # (x1, y1, x2, y2) en píxeles del frame analizado

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.caja
        return max(0, x2 - x1) * max(0, y2 - y1)


def normalizar(texto: str) -> str:
    """'Papel Cartón' -> 'papel_carton': minúsculas, sin tildes y con guion bajo."""
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sin_tildes.strip().lower().replace(" ", "_").replace("-", "_")


def categoria_de(clase_modelo: str) -> Optional[str]:
    return config.ALIAS_CLASES.get(normalizar(clase_modelo))


def extraer_detecciones(resultado, nombres: dict) -> list:
    """Convierte un resultado de Ultralytics en una lista de Deteccion."""
    detecciones = []
    cajas = getattr(resultado, "boxes", None)
    if cajas is None:
        return detecciones
    for xyxy, conf, cls in zip(cajas.xyxy.tolist(), cajas.conf.tolist(), cajas.cls.tolist()):
        clase = nombres[int(cls)]
        detecciones.append(
            Deteccion(clase, categoria_de(clase), float(conf), tuple(int(v) for v in xyxy))
        )
    return detecciones


def elegir_principal(detecciones: Iterable, confianza_minima: float = config.CONFIANZA_MINIMA) -> Optional[Deteccion]:
    """El objeto a clasificar es el más confiable; si empatan, el más grande (más cercano)."""
    validas = [d for d in detecciones if d.confianza >= confianza_minima and d.categoria]
    if not validas:
        return None
    return max(validas, key=lambda d: (round(d.confianza, 2), d.area))


def evaluar_iluminacion(frame_gris: np.ndarray, minimo: int = config.BRILLO_MINIMO,
                        maximo: int = config.BRILLO_MAXIMO) -> Optional[str]:
    """Devuelve un aviso si hay poca o demasiada luz; None si la iluminación es adecuada."""
    brillo = float(np.mean(frame_gris))
    if brillo < minimo:
        return "Poca iluminación: encienda la luz o acerque el objeto a una fuente de luz."
    if brillo > maximo:
        return "Demasiada iluminación: evite la luz directa o el flash sobre la cámara."
    return None


class Estabilizador:
    """Evita que la clasificación parpadee entre clases frame a frame.

    Solo confirma una categoría cuando aparece en `frames_estables` frames seguidos,
    y la limpia tras `frames_sin_objeto` frames sin ninguna detección válida.
    """

    def __init__(self, frames_estables: int = config.FRAMES_ESTABLES,
                 frames_sin_objeto: int = config.FRAMES_SIN_OBJETO):
        self.frames_estables = frames_estables
        self.frames_sin_objeto = frames_sin_objeto
        self.actual: Optional[str] = None
        self._candidata: Optional[str] = None
        self._repeticiones = 0
        self._vacios = 0

    def actualizar(self, categoria: Optional[str]) -> Optional[str]:
        if categoria is None:
            self._vacios += 1
            self._candidata, self._repeticiones = None, 0
            if self._vacios >= self.frames_sin_objeto:
                self.actual = None
            return self.actual

        self._vacios = 0
        if categoria == self._candidata:
            self._repeticiones += 1
        else:
            self._candidata, self._repeticiones = categoria, 1
        if self._repeticiones >= self.frames_estables:
            self.actual = categoria
        return self.actual
