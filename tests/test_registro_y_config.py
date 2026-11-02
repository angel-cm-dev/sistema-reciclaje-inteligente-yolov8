import csv

from entrenamiento.analizar_registro import resumir
from reciclaje import config
from reciclaje.deteccion import Deteccion
from reciclaje.registro import Registro


class Reloj:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_registro_crea_csv_con_encabezado(tmp_path):
    ruta = tmp_path / "salida" / "clasificaciones.csv"
    Registro(ruta).guardar(Deteccion("Glass", "vidrio", 0.876, (0, 0, 1, 1)))
    filas = list(csv.DictReader(ruta.open(encoding="utf-8")))
    assert filas[0]["categoria"] == "vidrio"
    assert filas[0]["clase_modelo"] == "Glass"
    assert filas[0]["confianza"] == "0.876"


def test_registro_no_duplica_el_mismo_objeto_en_poco_tiempo(tmp_path):
    reloj = Reloj()
    registro = Registro(tmp_path / "r.csv", segundos_entre_registros=3, reloj=reloj)
    d = Deteccion("Metal", "metal", 0.9, (0, 0, 1, 1))
    assert registro.guardar(d) is True
    reloj.t = 1
    assert registro.guardar(d) is False
    reloj.t = 5
    assert registro.guardar(d) is True


def test_resumen_del_registro(tmp_path):
    ruta = tmp_path / "r.csv"
    registro = Registro(ruta, segundos_entre_registros=0)
    registro.guardar(Deteccion("Metal", "metal", 0.8, (0, 0, 1, 1)))
    registro.guardar(Deteccion("Glass", "vidrio", 0.6, (0, 0, 1, 1)))
    registro.guardar(Deteccion("Metal", "metal", 1.0, (0, 0, 1, 1)))
    resumen = resumir(ruta)
    assert resumen["metal"][0] == 2
    assert abs(resumen["metal"][1] - 0.9) < 1e-9
    assert resumen["vidrio"][0] == 1


def test_todas_las_categorias_tienen_sus_imagenes():
    for categoria in config.CATEGORIAS.values():
        assert (config.CARPETA_RECURSOS / categoria.imagen).is_file(), categoria.imagen
        assert (config.CARPETA_RECURSOS / categoria.informacion).is_file(), categoria.informacion


def test_alias_apuntan_a_categorias_existentes():
    assert set(config.ALIAS_CLASES.values()) <= set(config.CATEGORIAS)


def test_fondos_de_la_interfaz_existen():
    for nombre in ("Canva.png", "bienvenida.png"):
        assert (config.CARPETA_RECURSOS / nombre).is_file()
