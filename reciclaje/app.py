"""Interfaz Tkinter: cámara en vivo, clasificación con YOLOv8 y paneles informativos."""

import platform
import threading
import tkinter as tk
from typing import Optional

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk

from . import config
from .deteccion import Estabilizador, elegir_principal, evaluar_iluminacion, extraer_detecciones
from .registro import Registro


def cargar_fuente(tamano: int) -> ImageFont.ImageFont:
    for nombre in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(nombre, tamano)
        except OSError:
            continue
    return ImageFont.load_default()


def abrir_camara():
    """Abre la cámara configurada o, si no hay una fija, la primera que entregue imagen.

    El índice 0 no siempre es la cámara física: con cámaras virtuales instaladas
    (NVIDIA Broadcast, OBS) puede estar en 1 o 2, y entonces el índice 0 falla.
    """
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    indices = [config.INDICE_CAMARA] if config.INDICE_CAMARA is not None else config.INDICES_A_PROBAR
    for indice in indices:
        camara = cv2.VideoCapture(indice, backend)
        if camara.isOpened() and camara.read()[0]:
            return camara
        camara.release()
    return None


def redondear(imagen: Image.Image, radio: int = 30) -> Image.Image:
    """Aplica esquinas redondeadas con transparencia para que se vea el fondo."""
    mascara = Image.new("L", imagen.size, 0)
    ImageDraw.Draw(mascara).rounded_rectangle([0, 0, *imagen.size], radio, fill=255)
    redondeada = imagen.convert("RGBA")
    redondeada.putalpha(mascara)
    return redondeada


class Clasificador(threading.Thread):
    """Ejecuta el modelo en segundo plano para que el video no se congele.

    YOLOv8-large tarda bastante por frame en CPU; si se ejecutara en el mismo hilo
    que la interfaz, la ventana dejaría de responder. Este hilo siempre analiza el
    frame más reciente y descarta los intermedios.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.modelo = None
        self.error: Optional[str] = None
        self.listo = threading.Event()
        self._activo = True
        self._condicion = threading.Condition()
        self._frame = None
        self.resultado = None      # lista de Deteccion del último frame analizado
        self.version = 0           # aumenta con cada resultado nuevo

    def enviar(self, frame_rgb) -> None:
        with self._condicion:
            self._frame = frame_rgb
            self._condicion.notify()

    def detener(self) -> None:
        self._activo = False
        with self._condicion:
            self._condicion.notify()

    def run(self) -> None:
        try:
            from ultralytics import YOLO  # import diferido: tarda varios segundos

            self.modelo = YOLO(str(config.RUTA_MODELO))
        except Exception as exc:  # modelo ausente, dañado o dependencias faltantes
            self.error = f"No se pudo cargar el modelo: {exc}"
            self.listo.set()
            return
        self.listo.set()

        while self._activo:
            with self._condicion:
                while self._frame is None and self._activo:
                    self._condicion.wait()
                frame, self._frame = self._frame, None
            if frame is None:
                continue
            # Ultralytics espera BGR cuando recibe un arreglo de numpy.
            prediccion = self.modelo.predict(
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), conf=config.CONFIANZA_MINIMA, verbose=False
            )[0]
            self.resultado = extraer_detecciones(prediccion, self.modelo.names)
            self.version += 1


class Aplicacion:
    def __init__(self):
        self.ventana = tk.Tk()
        self.ventana.title("Sistema de Reciclaje Inteligente")
        self.ventana.geometry(f"{config.ANCHO_VENTANA}x{config.ALTO_VENTANA}")
        # Tamaño fijo: los paneles del fondo están dibujados en posiciones exactas.
        self.ventana.resizable(False, False)
        self.ventana.protocol("WM_DELETE_WINDOW", self.cerrar)
        self.ventana.bind("<Escape>", lambda _evento: self.cerrar())

        self.lienzo = tk.Canvas(self.ventana, width=config.ANCHO_VENTANA, height=config.ALTO_VENTANA,
                                highlightthickness=0)
        self.lienzo.pack()

        self.fuente_caja = cargar_fuente(16)
        self.imagenes = {}           # referencias vivas: Tkinter borra las imágenes sin referencia
        self.camara = None
        self.clasificador = Clasificador()
        self.estabilizador = Estabilizador()
        self.registro = Registro()
        self.detecciones = []
        self.version_vista = 0
        self.categoria_mostrada = None

        self._mostrar_bienvenida()

    # ---------- recursos ----------
    def _imagen(self, nombre: str, tamano=None) -> ImageTk.PhotoImage:
        clave = (nombre, tamano)
        if clave not in self.imagenes:
            imagen = Image.open(config.CARPETA_RECURSOS / nombre)
            if tamano:
                imagen = imagen.resize(tamano)
            self.imagenes[clave] = ImageTk.PhotoImage(imagen)
        return self.imagenes[clave]

    # ---------- pantallas ----------
    def _mostrar_bienvenida(self) -> None:
        self.lienzo.create_image(0, 0, anchor="nw", image=self._imagen("bienvenida.png"), tags="fondo")
        self.clasificador.start()  # el modelo carga mientras se ve la bienvenida
        self.ventana.after(config.SEGUNDOS_BIENVENIDA * 1000, self._iniciar_principal)

    def _iniciar_principal(self) -> None:
        self.lienzo.delete("all")
        self.lienzo.create_image(0, 0, anchor="nw", image=self._imagen("Canva.png"))
        x, y = config.POSICION_VIDEO
        self.item_video = self.lienzo.create_image(x, y, anchor="nw")
        self.item_clasificacion = self.lienzo.create_image(*config.POSICION_CLASIFICACION, anchor="nw")
        self.item_informacion = self.lienzo.create_image(*config.POSICION_INFORMACION, anchor="nw")
        self.item_estado = self.lienzo.create_text(
            x + config.ANCHO_VIDEO // 2, y + config.ALTO_VIDEO // 2,
            text="Cargando modelo...", font=("Arial", 18, "bold"), fill="#1f2937",
        )
        self.item_aviso = self.lienzo.create_text(
            *config.POSICION_AVISO, text="", font=("Arial", 14, "bold"), fill="#b91c1c", width=560,
        )

        self.camara = abrir_camara()
        if self.camara is None:
            self._mostrar_estado("No se pudo acceder a la cámara.\nVerifique que esté conectada y no la use otra aplicación.")
            return
        self.camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._actualizar()

    def _mostrar_estado(self, texto: str) -> None:
        self.lienzo.itemconfigure(self.item_estado, text=texto)
        self.lienzo.tag_raise(self.item_estado)

    # ---------- ciclo principal ----------
    def _actualizar(self) -> None:
        ok, frame = self.camara.read()
        if not ok:
            self._mostrar_estado("Se perdió la señal de la cámara.")
            self.camara.release()
            return

        frame = cv2.flip(frame, 1)  # efecto espejo: más natural para el usuario
        frame = cv2.resize(frame, (config.ANCHO_VIDEO, config.ALTO_VIDEO))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        aviso = evaluar_iluminacion(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY))
        self.lienzo.itemconfigure(self.item_aviso, text=aviso or "")

        if self.clasificador.error:
            self._mostrar_estado(self.clasificador.error)
        elif self.clasificador.listo.is_set():
            self.lienzo.itemconfigure(self.item_estado, text="")
            self.clasificador.enviar(frame_rgb.copy())
            self._procesar_resultado()

        imagen = Image.fromarray(frame_rgb)
        self._dibujar_cajas(imagen)
        foto = ImageTk.PhotoImage(redondear(imagen))
        self.lienzo.itemconfigure(self.item_video, image=foto)
        self.imagenes["video"] = foto

        self.ventana.after(config.MS_ENTRE_FRAMES, self._actualizar)

    def _procesar_resultado(self) -> None:
        if self.clasificador.version == self.version_vista:
            return  # todavía no hay un resultado nuevo
        self.version_vista = self.clasificador.version
        self.detecciones = self.clasificador.resultado or []

        principal = elegir_principal(self.detecciones)
        categoria = self.estabilizador.actualizar(principal.categoria if principal else None)
        if categoria != self.categoria_mostrada:
            self.categoria_mostrada = categoria
            self._mostrar_categoria(categoria)
            if categoria and principal:
                self.registro.guardar(principal)

    def _mostrar_categoria(self, clave: Optional[str]) -> None:
        if clave is None:
            self.lienzo.itemconfigure(self.item_clasificacion, image="")
            self.lienzo.itemconfigure(self.item_informacion, image="")
            return
        categoria = config.CATEGORIAS[clave]
        self.lienzo.itemconfigure(self.item_clasificacion, image=self._imagen(categoria.imagen))
        self.lienzo.itemconfigure(self.item_informacion, image=self._imagen(categoria.informacion))

    def _dibujar_cajas(self, imagen: Image.Image) -> None:
        dibujo = ImageDraw.Draw(imagen)
        for d in self.detecciones:
            if not d.categoria:
                continue
            categoria = config.CATEGORIAS[d.categoria]
            b, g, r = categoria.color_bgr
            color = (r, g, b)
            dibujo.rectangle(d.caja, outline=color, width=3)
            etiqueta = f"{categoria.nombre} {d.confianza:.0%}"
            x1, y1 = d.caja[0], max(0, d.caja[1] - 24)
            ancho = dibujo.textlength(etiqueta, font=self.fuente_caja)
            dibujo.rectangle([x1, y1, x1 + ancho + 10, y1 + 22], fill=color)
            texto = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
            dibujo.text((x1 + 5, y1 + 2), etiqueta, font=self.fuente_caja, fill=texto)

    def cerrar(self) -> None:
        self.clasificador.detener()
        if self.camara is not None:
            self.camara.release()
        self.ventana.destroy()

    def ejecutar(self) -> None:
        self.ventana.mainloop()


def main() -> None:
    Aplicacion().ejecutar()
