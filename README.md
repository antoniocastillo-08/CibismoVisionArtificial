# CibismoVisionArtificial

Sistema de visión artificial para la **detección y cuantificación de marcas de chicle** en suelos de institutos, desarrollado con YOLOv11 y OpenCV. Permite analizar vídeos grabados con dron y calcular la densidad de chicles por metro cuadrado, diferenciando entre zonas centrales y laterales del pasillo.

---

## Descripción del Proyecto

El proyecto nació de la necesidad de medir objetivamente el nivel de "cibismo" (suciedad por chicles pegados) en los patios e instalaciones de centros educativos. A través de un dron que graba el suelo a baja altura (~2 metros), el sistema detecta automáticamente cada marca de chicle, las geolocaliza en una cuadrícula de metros cuadrados reales y genera un informe con la densidad por zona.

El modelo está diseñado para **generalizarse a distintos tipos de suelo** — hormigón, asfalto, baldosas — sin necesidad de reentrenar desde cero para cada instituto.

<figure class="video_container">
  <iframe src="VIDEOS/resultados/resultado_video_grancapitan.mp4" allowfullscreen="true"> 
</iframe>
</figure>

---

## Características

- Detección de chicles mediante **YOLOv11** con fine-tuning personalizado
- Análisis sobre **vídeo de dron** (MP4, 4K)
- Análisis sobre **imagen estática**
- **Cuadrícula de 1 m²** a escala real sobre la imagen
- Clasificación por zonas: **central (60%)** y **laterales (20%+20%)**
- Contador acumulado de chicles únicos en tiempo real
---

## Estructura del Proyecto

```
CibismoVisionArtificial/
│
├── best.pt                    # Pesos del modelo entrenado (YOLOv11)
│
├── dataset_chicles/           # Dataset de entrenamiento
│   ├── train/
│   │   ├── images/            # Imágenes de entrenamiento (~35 frames)
│   │   └── labels/            # Anotaciones en formato YOLO (.txt)
│   ├── valid/
│   │   ├── images/            # Imágenes de validación (~9 frames)
│   │   └── labels/
│   └── data.yaml              # Configuración del dataset
│
├── VIDEOS/
│   ├── patio_fixed.mp4        # Vídeo de entrada (convertido con ffmpeg)
│   └── resultado.mp4          # Vídeo de salida con anotaciones
│
├── detector_chicles.py        # Script principal — análisis de vídeo
├── test-foto.py               # Script para análisis de imagen estática
└── README.md
```

---

### Instalación de dependencias

```bash
pip install ultralytics opencv-python numpy
```

Para usar GPU (recomendado):
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Para convertir el vídeo si da errores de lectura:
```bash
# Instalar ffmpeg: https://ffmpeg.org o con winget
winget install ffmpeg

# Convertir vídeo
ffmpeg -i patio.mp4 -c:v libx264 -crf 18 patio_fixed.mp4
```

---

## Uso

### Análisis de vídeo

```bash
python detector_chicles.py
```

Parámetros configurables al inicio del script:

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `ANCHO_REAL_METROS` | `3.0` | Ancho real del pasillo en metros |
| `conf` | `0.60` | Confianza mínima de detección |

### Análisis de imagen estática

```bash
python test-foto.py
```

Cambia la variable `imagen_path` por la ruta a tu imagen:
```python
imagen_path = r"C:\ruta\a\tu\imagen.jpg"
```

---

## Entrenamiento del Modelo

El modelo fue entrenado con **YOLOv11s** (nano) usando fine-tuning sobre un dataset propio anotado en [Roboflow](https://roboflow.com).

### Dataset

- **633 imágenes** totales (con augmentation)
- **549 train / 50 valid / 34 test**
- 1 clase: `chicle`
- Data Augmentation: flip horizontal, crop, rotación ±14°, variación de brillo

### Reentrenamiento

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="dataset_chicles/data.yaml",
    epochs=100,
    imgsz=640,
    batch= 32,
    patience=20,
    augment=True,
    dropout=0.1,
    weight_decay=0.0005,
    optimizer="AdamW",
    device="0",          # GPU; quitar para CPU
    name="chicles_v1"
)
```

Los pesos quedan en `runs/detect/chicles_v1/weights/best.pt`.

---

> Obtenidas con el entrenamiento inicial en Roboflow (YOLOv11n, dataset base sin augmentation completa).

---

## Zonas de Análisis

El pasillo se divide en tres zonas con distinto grado de importancia:

```
|   LATERAL   |        CENTRAL        |   LATERAL   |
     20%               60%                  20%
   (naranja)          (verde)             (naranja)
```

Los chicles en zona central tienen mayor relevancia al ser la zona de mayor tránsito peatonal. El sistema permite ajustar los porcentajes modificando estas variables:

```python
izq_arriba = int(ancho * 0.30)   # límite superior izquierdo
der_arriba = int(ancho * 0.70)   # límite superior derecho
izq_abajo  = int(ancho * 0.15)   # límite inferior izquierdo (perspectiva)
der_abajo  = int(ancho * 0.85)   # límite inferior derecho (perspectiva)
```

---

##  Salida del Sistema

### En vídeo/imagen
- **Recuadros verdes**: chicles en zona central
- **Recuadros naranjas**: chicles en zona lateral
- **Cuadrícula gris**: división en celdas de 1 m²
- **Color de celda**: densidad (amarillo < 3 | naranja < 7 | rojo ≥ 7 chicles/m²)
- **Panel superior**: conteo acumulado en tiempo real

### En consola al finalizar
```
══════════════════════════════════════════
  RESULTADO FINAL
══════════════════════════════════════════
  Chicles CENTRAL:      312
  Chicles LATERAL:      189
  Total únicos:         501
  Densidad central:     3.66 chicles/m²
  Densidad lateral:     2.09 chicles/m²
══════════════════════════════════════════
```
