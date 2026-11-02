"""Pruebas con el modelo real. Se omiten si falta ultralytics o el archivo best.pt."""

import numpy as np
import pytest

from reciclaje import config
from reciclaje.deteccion import categoria_de, extraer_detecciones

ultralytics = pytest.importorskip("ultralytics")
pytestmark = pytest.mark.skipif(not config.RUTA_MODELO.is_file(), reason="falta modelos/best.pt")


@pytest.fixture(scope="module")
def modelo():
    return ultralytics.YOLO(str(config.RUTA_MODELO))


def test_todas_las_clases_del_modelo_tienen_categoria(modelo):
    sin_categoria = [n for n in modelo.names.values() if categoria_de(n) is None]
    assert sin_categoria == []


def test_el_modelo_predice_sobre_un_frame(modelo):
    frame = np.full((config.ALTO_VIDEO, config.ANCHO_VIDEO, 3), 127, dtype=np.uint8)
    resultado = modelo.predict(frame, conf=config.CONFIANZA_MINIMA, verbose=False)[0]
    detecciones = extraer_detecciones(resultado, modelo.names)
    assert isinstance(detecciones, list)
