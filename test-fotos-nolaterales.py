from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

# --- CONFIGURACIÓN DE PANTALLA ---
MAX_ANCHO_PANTALLA = 1366
MAX_ALTO_PANTALLA  = 768

puntos_calibracion = []

def capturar_clics(event, x, y, flags, param):
    """Función callback para registrar los dos clics de calibración."""
    global puntos_calibracion
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos_calibracion.append((x, y))
        # Dibujar un punto donde el usuario ha hecho clic para darle feedback visual
        cv2.circle(img_calibracion, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("PASO 1: Haz clic en el Punto 1 y luego en el Punto 2", img_calibracion)

# 1. Cargar el modelo YOLO
model = YOLO("best.pt")

# 2. Cargar Imagen Original
imagen_path = "VIDEOS/frames_ruta_entrada/frame_250.jpg"
frame = cv2.imread(imagen_path)
if frame is None:
    print(f"No se pudo cargar la imagen en: {imagen_path}")
    exit()

alto_orig, ancho_orig = frame.shape[:2]

escala_pantalla = min(MAX_ANCHO_PANTALLA / ancho_orig, MAX_ALTO_PANTALLA / alto_orig, 1.0)
ancho_vista = int(ancho_orig * escala_pantalla)
alto_vista = int(alto_orig * escala_pantalla)

img_calibracion = cv2.resize(frame, (ancho_vista, alto_vista))

# 4. Ventana interactiva para calibración
cv2.namedWindow("PASO 1: Haz clic en el Punto 1 y luego en el Punto 2")
cv2.setMouseCallback("PASO 1: Haz clic en el Punto 1 y luego en el Punto 2", capturar_clics)

print("--> Selecciona dos puntos en la imagen que se ha abierto.")
while len(puntos_calibracion) < 2:
    cv2.imshow("PASO 1: Haz clic en el Punto 1 y luego en el Punto 2", img_calibracion)
    # Si presionas 'q' o ESC, cancela
    if cv2.waitKey(1) & 0xFF == 27:
        print("Calibración cancelada.")
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# 5. Calcular la distancia en píxeles reales (escalando de vuelta al tamaño original)
p1 = (int(puntos_calibracion[0][0] / escala_pantalla), int(puntos_calibracion[0][1] / escala_pantalla))
p2 = (int(puntos_calibracion[1][0] / escala_pantalla), int(puntos_calibracion[1][1] / escala_pantalla))

distancia_px = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

print(f"\nPuntos registrados en la imagen original: P1={p1}, P2={p2} ({distancia_px:.1f} px)")
distancia_real_cm = float(input("Introduce la distancia real entre esos dos puntos (en centímetros): "))

# 6. Calcular nueva escala (Píxeles por Metro)
# Si 'distancia_real_cm' equivale a 'distancia_px', entonces 100 cm (1 metro) serán:
PX_POR_METRO = (distancia_px / distancia_real_cm) * 100.0

CELDA_PX        = int(PX_POR_METRO)
DISTANCIA_DEDUP = int(PX_POR_METRO * 0.05)  # 5cm para evitar duplicados


celdas_x = int(np.ceil(ancho_orig / CELDA_PX))
celdas_y = int(np.ceil(alto_orig  / CELDA_PX))

# Montando la cuadrícula en el frame original
for gx in range(celdas_x):
    for gy in range(celdas_y):
        x0 = gx * CELDA_PX
        y0 = gy * CELDA_PX
        cv2.rectangle(frame, (x0, y0),
                      (min(x0 + CELDA_PX, ancho_orig), min(y0 + CELDA_PX, alto_orig)),
                      (200, 200, 200), 1)

# Detección del modelo
resultados = model.predict(frame, conf=0.5, iou=0.4, verbose=False)

chicles_unicos = {}
conteo_celdas  = defaultdict(int)

def es_nuevo(cx, cy):
    for (px, py) in chicles_unicos:
        if abs(cx - px) < DISTANCIA_DEDUP and abs(cy - py) < DISTANCIA_DEDUP:
            return False
    return True

if resultados[0].boxes is not None:
    for box in resultados[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        celda = (cx // CELDA_PX, cy // CELDA_PX)
        conteo_celdas[celda] += 1

        if es_nuevo(cx, cy):
            chicles_unicos[(cx, cy)] = "detected"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# Pintar el conteo por cada celda
for gx in range(celdas_x):
    for gy in range(celdas_y):
        x0 = gx * CELDA_PX
        y0 = gy * CELDA_PX
        x1 = min(x0 + CELDA_PX, ancho_orig)
        y1 = min(y0 + CELDA_PX, alto_orig)

        # Consultar cuántos chicles hay en esta celda específica (0 por defecto)
        n = conteo_celdas[(gx, gy)]

        # Lógica de colores según densidad
        if n == 0:
            color_celda = (0, 255, 0)  # Verde para celdas completamente limpias
        elif n < 3:
            color_celda = (0, 255, 255)  # Amarillo (1 o 2 chicles)
        elif n < 7:
            color_celda = (0, 165, 255)  # Naranja (de 3 a 6 chicles)
        else:
            color_celda = (0, 0, 255)  # Rojo (7 o más chicles)

        # Dibujar el rectángulo exterior de la celda con su color correspondiente
        cv2.rectangle(frame, (x0, y0), (x1, y1), color_celda, 2)

        # Poner el texto con el conteo en el centro de la celda (ej: "0/m2", "4/m2")
        cv2.putText(frame, f"{n}/m2",
                    (x0 + CELDA_PX // 2 - 25, y0 + CELDA_PX // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_celda, 2)
# Conteo Resumen
total_global = len(chicles_unicos)

# Panel resumen en la imagen original
cv2.rectangle(frame, (15, alto_orig - 100), (450, alto_orig - 10), (0, 0, 0), -1)
cv2.putText(frame, f"Total chicles: {total_global}",
            (20, alto_orig - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.putText(frame, f"Escala manual: 1 celda = 1 m² ({CELDA_PX}px)",
            (20, alto_orig - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

# Guardar la imagen a resolución completa
cv2.imwrite("VIDEOS//resultados/resultado_imagen.jpg", frame)

# Mostrar el resultado final adaptado a la pantalla para poder verlo entero
img_final_vista = cv2.resize(frame, (ancho_vista, alto_vista))
cv2.imshow("Detección chicles - Resultado Final", img_final_vista)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("-" * 50)
print("  RESULTADO")
print("-" * 50)
print(f"  Total chicles:            {total_global}")
print(f"  Celdas detectadas con chicles: {len(conteo_celdas)}")
print(f"  Píxeles calculados por metro:  {CELDA_PX} px")
print("-" * 50)