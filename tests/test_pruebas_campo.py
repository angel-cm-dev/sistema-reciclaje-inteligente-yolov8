import pytest

from entrenamiento.evaluar_pruebas_campo import a_markdown, calcular, leer


def escribir(tmp_path, filas):
    ruta = tmp_path / "p.csv"
    ruta.write_text("id,objeto,clase_real,clase_predicha,confianza,iluminacion,observacion\n"
                    + "\n".join(filas) + "\n", encoding="utf-8")
    return ruta


def test_calcula_exactitud_precision_y_recall(tmp_path):
    ruta = escribir(tmp_path, [
        "1,lata,metal,metal,0.9,normal,",
        "2,lata,metal,metal,0.8,normal,",
        "3,botella,vidrio,metal,0.6,baja,",
        "4,botella,vidrio,vidrio,0.7,normal,",
        "5,frasco,vidrio,,,baja,no detectó",
    ])
    r = calcular(leer(ruta))
    assert r["total"] == 5
    assert r["exactitud"] == pytest.approx(3 / 5)
    assert r["por_clase"]["metal"]["precision"] == pytest.approx(2 / 3)
    assert r["por_clase"]["metal"]["recall"] == 1.0
    assert r["por_clase"]["vidrio"]["recall"] == pytest.approx(1 / 3)
    assert r["matriz"]["vidrio"]["ninguna"] == 1
    assert r["por_iluminacion"] == {"baja": 0.0, "normal": 1.0}


def test_markdown_incluye_tablas(tmp_path):
    r = calcular(leer(escribir(tmp_path, ["1,lata,metal,metal,0.9,normal,"])))
    texto = a_markdown(r)
    assert "Exactitud global:** 100.0%" in texto
    assert "| metal | 1 | 100.0% | 100.0% | 100.0% |" in texto


def test_formato_vacio_no_tiene_pruebas(tmp_path):
    assert leer(escribir(tmp_path, [])) == []
