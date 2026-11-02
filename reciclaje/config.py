"""Configuración central: rutas, umbrales, categorías y posiciones de la interfaz."""

from dataclasses import dataclass
from pathlib import Path

# Las rutas se resuelven desde la raíz del proyecto, no desde la carpeta
# donde se ejecuta el comando, para que la app funcione desde cualquier lugar.
RAIZ = Path(__file__).resolve().parent.parent
CARPETA_RECURSOS = RAIZ / "setUp"
RUTA_MODELO = RAIZ / "modelos" / "best.pt"
RUTA_REGISTRO = RAIZ / "resultados" / "clasificaciones.csv"

# Detección
CONFIANZA_MINIMA = 0.50       # descarta detecciones con menos del 50 % de confianza
FRAMES_ESTABLES = 5           # frames seguidos con la misma clase antes de mostrarla
FRAMES_SIN_OBJETO = 15        # frames sin detección antes de limpiar los paneles
SEGUNDOS_ENTRE_REGISTROS = 3  # evita registrar el mismo objeto muchas veces

# Iluminación (brillo promedio en escala de grises, 0-255)
BRILLO_MINIMO = 50
BRILLO_MAXIMO = 200

# Cámara e interfaz
INDICE_CAMARA = None          # None = usa la primera cámara que entregue imagen (0, 1, 2...)
INDICES_A_PROBAR = range(5)
ANCHO_VENTANA, ALTO_VENTANA = 1280, 720
ANCHO_VIDEO, ALTO_VIDEO = 571, 420
POSICION_VIDEO = (354, 140)          # dentro del panel central del fondo
POSICION_CLASIFICACION = (76, 280)   # panel "Clasificación" (izquierda)
POSICION_INFORMACION = (995, 335)    # panel "Información" (derecha)
POSICION_AVISO = (640, 600)          # centro del aviso de iluminación
SEGUNDOS_BIENVENIDA = 3
MS_ENTRE_FRAMES = 15


@dataclass(frozen=True)
class Categoria:
    """Categoría de residuo que la interfaz sabe mostrar."""

    clave: str
    nombre: str
    imagen: str           # tarjeta del panel "Clasificación"
    informacion: str      # tarjeta del panel "Información"
    contenedor: str       # color según NTP 900.058:2019
    color_bgr: tuple      # color del recuadro sobre el video


CATEGORIAS = {
    "metal": Categoria("metal", "Metales", "metal.png", "metaltxt.png", "amarillo", (0, 200, 255)),
    "vidrio": Categoria("vidrio", "Vidrio", "vidrio.png", "vidriotxt.png", "plomo", (200, 200, 200)),
    "plastico": Categoria("plastico", "Plástico", "plastico.png", "plasticotxt.png", "blanco", (255, 255, 255)),
    "papel_carton": Categoria("papel_carton", "Papel y cartón", "PapelyCarton.png", "papelycartontxt.png", "azul", (255, 120, 0)),
    "organico": Categoria("organico", "Orgánico", "Organico.png", "organicotxt.png", "marrón", (40, 80, 140)),
}

# Traduce los nombres de clase que entrega el modelo a nuestras categorías.
# Se leen del propio modelo (model.names) para no depender del orden de los
# índices: el error original fue suponer que el índice 3 era "Orgánico" cuando
# el modelo anterior tenía 3 Carton. El actual tiene 0 Metal, 1 Vidrio,
# 2 Plastico, 3 Papel_Carton, 4 Organico.
ALIAS_CLASES = {
    "metal": "metal",
    "glass": "vidrio",
    "vidrio": "vidrio",
    "plastic": "plastico",
    "plastico": "plastico",
    "carton": "papel_carton",
    "cardboard": "papel_carton",
    "paper": "papel_carton",
    "papel_carton": "papel_carton",
    "organic": "organico",
    "organico": "organico",
    "biodegradable": "organico",
}
