import pytest

from entrenamiento.preparar_dataset import CLASES, convertir_etiquetas, linea_a_caja, preparar
from reciclaje.deteccion import categoria_de


def test_remapea_clases_originales():
    # BIODEGRADABLE(0)->Organico, CARDBOARD(1) y PAPER(4)->Papel_Carton, GLASS(2)->Vidrio
    assert linea_a_caja("0 0.5 0.5 0.2 0.2")[0] == CLASES.index("Organico")
    assert linea_a_caja("1 0.5 0.5 0.2 0.2")[0] == CLASES.index("Papel_Carton")
    assert linea_a_caja("4 0.5 0.5 0.2 0.2")[0] == CLASES.index("Papel_Carton")
    assert linea_a_caja("2 0.5 0.5 0.2 0.2")[0] == CLASES.index("Vidrio")


def test_poligono_se_convierte_en_caja_contenedora():
    clase, xc, yc, w, h = linea_a_caja("3 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6")
    assert clase == CLASES.index("Metal")
    assert (xc, yc, w, h) == pytest.approx((0.3, 0.4, 0.4, 0.4))


def test_poligono_fuera_de_imagen_se_recorta():
    _, xc, yc, w, h = linea_a_caja("5 -0.1 0.0 0.4 0.0 0.4 0.5")
    assert xc - w / 2 == pytest.approx(0.0)


def test_ignora_lineas_vacias_e_invalidas():
    assert convertir_etiquetas("\n3 0.5 0.5 0 0.1\n\n") == []


def test_las_clases_nuevas_tienen_categoria_en_la_app():
    assert all(categoria_de(c) for c in CLASES)


def test_preparar_genera_estructura(tmp_path):
    origen = tmp_path / "orig"
    for particion in ("train", "valid", "test"):
        (origen / particion / "images").mkdir(parents=True)
        (origen / particion / "labels").mkdir(parents=True)
        (origen / particion / "images" / "a.jpg").write_bytes(b"x")
        (origen / particion / "labels" / "a.txt").write_text("3 0.5 0.5 0.2 0.2\n4 0.5 0.5 0.1 0.1\n")
    resumen = preparar(origen, tmp_path / "datos")
    assert (tmp_path / "datos" / "images" / "val" / "a.jpg").exists()
    assert resumen["objetos"]["val"]["Metal"] == 1
    assert resumen["objetos"]["test"]["Papel_Carton"] == 1
