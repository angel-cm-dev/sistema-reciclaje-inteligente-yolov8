# Ficha técnica del modelo

## Resumen

| Campo | Valor |
|---|---|
| Arquitectura | YOLOv8s (detección de objetos), 11.1 millones de parámetros, 28.4 GFLOPs |
| Pesos de partida | `yolov8s.pt` preentrenado en COCO (transfer learning) |
| Framework | Ultralytics 8.4.197 · PyTorch 2.11.0 + CUDA 12.8 · Python 3.12 |
| Hardware de entrenamiento | NVIDIA GeForce RTX 4060 (8 GB) |
| Épocas | 60 (0.97 horas) |
| Lote · tamaño de imagen | 16 · 640 × 640 |
| Optimizador | AdamW (lr 0.00111, elegido automáticamente por Ultralytics) |
| Semilla | 42 (entrenamiento determinista, reproducible) |
| Clases | 0 Metal · 1 Vidrio · 2 Plástico · 3 Papel y cartón · 4 Orgánico |
| Archivo | `modelos/best.pt` (22.5 MB) |
| Velocidad en la app | ~18 ms por imagen con GPU (~0.8 s en CPU) |

Parámetros completos: [`resultados/parametros_entrenamiento.yaml`](resultados/parametros_entrenamiento.yaml).

## Dataset

**Trash Detection Dataset** — Kan, P. (2025). *Mendeley Data*, V1. https://doi.org/10.17632/z732f9pwxt.1 — Licencia CC BY 4.0.

Cita sugerida (APA 7):
> Kan, P. (2025). *Trash Detection Dataset* (Versión 1) [Conjunto de datos]. Mendeley Data. https://doi.org/10.17632/z732f9pwxt.1

### Preparación (`entrenamiento/preparar_dataset.py`)
1. Las 6 clases originales se reagruparon en las 5 categorías de la interfaz:

| Clase original | Categoría del sistema |
|---|---|
| METAL | Metal |
| GLASS | Vidrio |
| PLASTIC | Plástico |
| CARDBOARD + PAPER | Papel y cartón |
| BIODEGRADABLE | Orgánico |

2. Las etiquetas en formato polígono se convirtieron a cajas delimitadoras (el dataset mezcla ambos formatos).
3. Se respetó la partición original del autor (train / valid / test).

### Distribución (objetos etiquetados)

| Clase | train | val | test | Total |
|---|---|---|---|---|
| Metal | 2595 | 789 | 436 | 3820 |
| Vidrio | 1538 | 432 | 437 | 2407 |
| Plástico | 2744 | 815 | 391 | 3950 |
| Papel y cartón | 3611 | 798 | 477 | 4886 |
| Orgánico | 13024 | 4450 | 1545 | 19019 |
| **Imágenes** | 3565 | 1008 | 505 | 5078 |

El dataset está **desbalanceado**: Orgánico tiene muchos más objetos porque sus imágenes suelen contener varios residuos pequeños.

## Resultados

### Conjunto de prueba (505 imágenes que el modelo nunca vio)

| Clase | Precisión | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Metal | 0.752 | 0.594 | 0.615 | 0.463 |
| Vidrio | 0.798 | 0.304 | 0.328 | 0.228 |
| Plástico | 0.494 | 0.494 | 0.490 | 0.334 |
| Papel y cartón | 0.643 | 0.507 | 0.561 | 0.415 |
| Orgánico | 0.808 | 0.478 | 0.600 | 0.319 |
| **Todas** | **0.699** | **0.475** | **0.519** | **0.352** |

### Conjunto de validación (1008 imágenes)

| Clase | Precisión | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Metal | 0.715 | 0.655 | 0.706 | 0.475 |
| Vidrio | 0.795 | 0.727 | 0.768 | 0.600 |
| Plástico | 0.663 | 0.504 | 0.558 | 0.382 |
| Papel y cartón | 0.711 | 0.557 | 0.599 | 0.483 |
| Orgánico | 0.748 | 0.443 | 0.541 | 0.270 |
| **Todas** | **0.726** | **0.577** | **0.634** | **0.442** |

### Figuras
- Matriz de confusión normalizada: [`resultados/confusion_matrix_normalized.png`](resultados/confusion_matrix_normalized.png)
- Curva precisión-recall: [`resultados/BoxPR_curve.png`](resultados/BoxPR_curve.png)
- Curva F1 vs confianza: [`resultados/BoxF1_curve.png`](resultados/BoxF1_curve.png)
- Pérdidas y métricas por época: [`resultados/curvas_entrenamiento.png`](resultados/curvas_entrenamiento.png) (datos en `entrenamiento_por_epoca.csv`)
- Ejemplo de predicciones: [`resultados/val_batch0_pred.jpg`](resultados/val_batch0_pred.jpg)

### Interpretación
- **Mejor clasificadas (matriz de confusión, test):** Papel y cartón (83 %), Plástico (80 %) y Orgánico (79 %) cuando el objeto es detectado.
- **Clase más débil: Vidrio.** En test, el 40 % de los objetos de vidrio no se detectan (se confunden con el fondo, por su transparencia) y el 20 % se clasifica como plástico. En validación su recall fue 0.727, lo que indica que las imágenes de vidrio del conjunto de prueba son más difíciles o distintas.
- **Falsos positivos:** parte del fondo se detecta como Orgánico o Papel y cartón; el umbral de confianza de 0.50 de la app y el estabilizador de 5 frames reducen este efecto en uso real.
- **Diferencia validación vs prueba** (mAP50 0.634 vs 0.519): es normal que baje en datos totalmente nuevos; en la tesis se deben reportar las métricas de **prueba**.

### Trabajo futuro sugerido
- Agregar más imágenes de vidrio, sobre todo transparente y con fondos variados.
- Balancear la clase Orgánico (limitar objetos por imagen o aumentar las demás clases).
- Probar YOLOv8m (más preciso, más lento) y comparar.
- Agregar imágenes propias tomadas con la cámara del sistema en su lugar de uso.

## Modelo anterior (descartado)

El proyecto original usaba un modelo YOLOv8-large entrenado el 15/10/2023 (Ultralytics 8.0.193, 30 épocas) con las clases Metal, Glass, Plastic, Carton y Medical, sin dataset ni métricas disponibles y de origen no documentado. Se reemplazó por el modelo actual, entrenado y evaluado de forma reproducible. Queda guardado en `respaldo/best_modelo_anterior_2023.pt` (fuera del control de versiones).
