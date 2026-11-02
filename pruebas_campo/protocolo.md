# Protocolo de pruebas con objetos reales

Objetivo: medir qué tan bien clasifica el sistema **en condiciones reales de uso**, como complemento de las métricas del dataset (`entrenamiento/evaluar.py`).

## Materiales
- La aplicación funcionando (`python main.py`).
- **20 objetos por categoría** (100 en total para 5 categorías), distintos entre sí. Ejemplos:

| Categoría | Ejemplos |
|---|---|
| Metal | latas de gaseosa, latas de atún, tapas metálicas, papel aluminio |
| Vidrio | botellas, frascos, vasos de vidrio |
| Plástico | botellas PET, envases de yogur, bolsas, cubiertos descartables |
| Papel y cartón | cajas, hojas, periódicos, rollos de papel higiénico |
| Médico | mascarillas, guantes, jeringas sin aguja, gasas (manipular con cuidado) |

## Condiciones controladas
Registre cada prueba en una de estas condiciones (columna `iluminacion`):
- `normal`: luz de ambiente de oficina o aula.
- `baja`: luz tenue (la app debe mostrar el aviso).
- `alta`: luz directa o contraluz.

Distancia recomendada: el objeto debe ocupar entre el 30 % y el 70 % del recuadro de video.

## Procedimiento por objeto
1. Coloque el objeto frente a la cámara y espere a que la clasificación se estabilice (~3 segundos).
2. Anote en `formato_pruebas.csv`:
   - `id`: número correlativo.
   - `objeto`: descripción corta (ej. "lata Coca-Cola 355 ml").
   - `clase_real`: metal, vidrio, plastico, papel_carton u organico.
   - `clase_predicha`: lo que mostró la app, o `ninguna` si no clasificó.
   - `confianza`: el % que aparece sobre el recuadro (ej. 0.87).
   - `iluminacion`: normal, baja o alta.
   - `observacion`: opcional.
3. Retire el objeto y continúe con el siguiente.

## Cálculo de resultados
```bash
python entrenamiento/evaluar_pruebas_campo.py pruebas_campo/formato_pruebas.csv
```
Genera exactitud global, precisión, recall y F1 por clase, la matriz de confusión y la exactitud por condición de iluminación, en `resultados/pruebas_campo/` (tabla en Markdown lista para la tesis).
