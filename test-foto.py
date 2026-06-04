from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

model = YOLO("best.pt")

# Cargar Imagen
imagen_path = ("VIDEOS/MedinaAzahara/medina_azahara_1.jpg")
frame = cv2.imread(imagen_path)
alto, ancho = frame.shape[:2]

# Escala
ANCHO_REAL_METROS = 3.0
PX_POR_METRO      = ancho / ANCHO_REAL_METROS
CELDA_PX          = int(PX_POR_METRO)
DISTANCIA_DEDUP   = int(PX_POR_METRO * 0.05)  # 5cm

# Zonas marcadas
izq_arriba = int(ancho * 0.30)
der_arriba = int(ancho * 0.70)
izq_abajo  = int(ancho * 0.15)
der_abajo  = int(ancho * 0.85)

celdas_x = int(np.ceil(ancho / CELDA_PX))
celdas_y = int(np.ceil(alto  / CELDA_PX))

def zona_chicle(cx, cy):
    limite_izq = izq_arriba + (cy / alto) * (izq_abajo - izq_arriba)
    limite_der = der_arriba + (cy / alto) * (der_abajo - der_arriba)
    return "lateral" if (cx < limite_izq or cx > limite_der) else "central"

# Sombras laterales
mask = np.zeros_like(frame)
pts_izq = np.array([[0, 0], [izq_arriba, 0], [izq_abajo, alto], [0, alto]], np.int32)
pts_der = np.array([[der_arriba, 0], [ancho, 0], [ancho, alto], [der_abajo, alto]], np.int32)
cv2.fillPoly(mask, [pts_izq], (255, 100, 0))
cv2.fillPoly(mask, [pts_der], (255, 100, 0))
cv2.addWeighted(mask, 0.15, frame, 1.0, 0, frame)

cv2.line(frame, (izq_arriba, 0), (izq_abajo, alto), (255, 150, 0), 2)
cv2.line(frame, (der_arriba, 0), (der_abajo, alto), (255, 150, 0), 2)
cv2.putText(frame, "LATERAL", (10, alto // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 150, 0), 2)
cv2.putText(frame, "LATERAL", (ancho - 130, alto // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 150, 0), 2)
cv2.putText(frame, "CENTRAL", (ancho // 2 - 60, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

# Montando la cuadrícula
for gx in range(celdas_x):
    for gy in range(celdas_y):
        x0 = gx * CELDA_PX
        y0 = gy * CELDA_PX
        cv2.rectangle(frame, (x0, y0),
                      (min(x0 + CELDA_PX, ancho), min(y0 + CELDA_PX, alto)),
                      (200, 200, 200), 1)

# Detección del modelo
resultados = model.predict(frame, conf=0.60, iou=0.4, verbose=False)

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

        zona  = zona_chicle(cx, cy)
        color = (0, 255, 0) if zona == "central" else (255, 150, 0)

        celda = (cx // CELDA_PX, cy // CELDA_PX)
        conteo_celdas[celda] += 1

        if es_nuevo(cx, cy):
            chicles_unicos[(cx, cy)] = zona

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{'LAT ' if zona == 'lateral' else ''}{conf:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# Pintar el conteo por cada celda
for (gx, gy), n in conteo_celdas.items():
    x0 = gx * CELDA_PX
    y0 = gy * CELDA_PX
    x1 = min(x0 + CELDA_PX, ancho)
    y1 = min(y0 + CELDA_PX, alto)

    if n < 3:
        color_celda = (0, 255, 255)    # amarillo
    elif n < 7:
        color_celda = (0, 165, 255)    # naranja
    else:
        color_celda = (0, 0, 255)      # rojo

    cv2.rectangle(frame, (x0, y0), (x1, y1), color_celda, 2)
    cv2.putText(frame, f"{n}/m2",
                (x0 + CELDA_PX // 2 - 25, y0 + CELDA_PX // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_celda, 2)

# Conteo Resumen
total_central = sum(1 for z in chicles_unicos.values() if z == "central")
total_lateral = sum(1 for z in chicles_unicos.values() if z == "lateral")
total_global  = total_central + total_lateral

# Panel resumen en la imagen
cv2.rectangle(frame, (15, alto - 100), (420, alto - 10), (0, 0, 0), -1)
cv2.putText(frame, f"Central: {total_central}  Lateral: {total_lateral}  Total: {total_global}",
            (20, alto - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.putText(frame, f"Escala: 1 celda = 1 m²  ({CELDA_PX}px)",
            (20, alto - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

# Mostrar y descargar la imagen
cv2.imwrite("VIDEOS/resultados/resultado_medinaazahara.jpg", frame)

cv2.imshow("Detección chicles", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("-" * 50)
print("  RESULTADO")
print("-" * 50)
print(f"  Chicles CENTRAL:  {total_central}")
print(f"  Chicles LATERAL:  {total_lateral}")
print(f"  Total:            {total_global}")
print(f"  Celdas detectadas con chicles: {len(conteo_celdas)}")
print("-" * 50)
