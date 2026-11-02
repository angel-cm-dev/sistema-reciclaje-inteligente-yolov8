from types import SimpleNamespace

import numpy as np
import pytest

from reciclaje.deteccion import (
    Deteccion,
    Estabilizador,
    categoria_de,
    elegir_principal,
    evaluar_iluminacion,
    extraer_detecciones,
    normalizar,
)

NOMBRES_MODELO = {0: "Metal", 1: "Vidrio", 2: "Plastico", 3: "Papel_Carton", 4: "Organico"}


def det(categoria, confianza, caja=(0, 0, 10, 10), clase="X"):
    return Deteccion(clase, categoria, confianza, caja)


@pytest.mark.parametrize("texto, esperado", [
    ("Papel Cartón", "papel_carton"),
    ("  ORGÁNICO ", "organico"),
    ("Papel-Carton", "papel_carton"),
])
def test_normalizar(texto, esperado):
    assert normalizar(texto) == esperado


@pytest.mark.parametrize("clase, categoria", [
    ("Metal", "metal"), ("Glass", "vidrio"), ("Plastic", "plastico"),
    ("Carton", "papel_carton"), ("Papel_Carton", "papel_carton"), ("Orgánico", "organico"),
])
def test_categoria_de_traduce_clases_del_modelo(clase, categoria):
    assert categoria_de(clase) == categoria


def test_carton_ya_no_se_confunde_con_organico():
    # Error de la versión original: el índice 3 (Carton) se mostraba como "Organico".
    assert categoria_de("Carton") == "papel_carton"
    assert categoria_de(NOMBRES_MODELO[3]) == "papel_carton"
    assert categoria_de(NOMBRES_MODELO[4]) == "organico"


def test_clase_desconocida_devuelve_none():
    assert categoria_de("Battery") is None
    assert categoria_de("Medical") is None


def test_extraer_detecciones_desde_resultado_ultralytics():
    cajas = SimpleNamespace(
        xyxy=np.array([[10.4, 20.6, 110.0, 220.0], [0, 0, 5, 5]]),
        conf=np.array([0.91, 0.3]),
        cls=np.array([3.0, 0.0]),
    )
    detecciones = extraer_detecciones(SimpleNamespace(boxes=cajas), NOMBRES_MODELO)
    assert detecciones[0] == Deteccion("Papel_Carton", "papel_carton", pytest.approx(0.91), (10, 20, 110, 220))
    assert detecciones[1].categoria == "metal"


def test_extraer_detecciones_sin_cajas():
    assert extraer_detecciones(SimpleNamespace(boxes=None), NOMBRES_MODELO) == []


def test_elegir_principal_prefiere_mayor_confianza():
    assert elegir_principal([det("metal", 0.6), det("vidrio", 0.9)]).categoria == "vidrio"


def test_elegir_principal_desempata_por_tamano():
    pequena = det("metal", 0.8, caja=(0, 0, 10, 10))
    grande = det("plastico", 0.8, caja=(0, 0, 100, 100))
    assert elegir_principal([pequena, grande]) is grande


def test_elegir_principal_descarta_baja_confianza_y_desconocidas():
    assert elegir_principal([det("metal", 0.2), det(None, 0.99)], confianza_minima=0.5) is None


@pytest.mark.parametrize("brillo, hay_aviso, palabra", [
    (10, True, "Poca"), (128, False, None), (240, True, "Demasiada"),
])
def test_evaluar_iluminacion(brillo, hay_aviso, palabra):
    aviso = evaluar_iluminacion(np.full((10, 10), brillo, dtype=np.uint8))
    assert (aviso is not None) == hay_aviso
    if palabra:
        assert palabra in aviso


def test_estabilizador_confirma_tras_frames_consecutivos():
    e = Estabilizador(frames_estables=3, frames_sin_objeto=2)
    assert e.actualizar("metal") is None
    assert e.actualizar("metal") is None
    assert e.actualizar("metal") == "metal"


def test_estabilizador_ignora_parpadeos():
    e = Estabilizador(frames_estables=3, frames_sin_objeto=5)
    for _ in range(3):
        e.actualizar("metal")
    # un frame suelto de otra clase no cambia lo mostrado
    assert e.actualizar("vidrio") == "metal"
    assert e.actualizar("metal") == "metal"


def test_estabilizador_limpia_cuando_se_retira_el_objeto():
    e = Estabilizador(frames_estables=1, frames_sin_objeto=2)
    e.actualizar("plastico")
    assert e.actualizar(None) == "plastico"
    assert e.actualizar(None) is None
