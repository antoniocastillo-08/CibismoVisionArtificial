from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("best.pt")

cap = cv2.VideoCapture("VIDEOS/patio_fixed.mp4")
ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("VIDEOS/resultados/resultado.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))

# CONFIGURACIÓN DE LA PERSPECTIVA
izq_arriba = int(ancho * 0.30)
der_arriba = int(ancho * 0.70)
izq_abajo = int(ancho * 0.15)
der_abajo = int(ancho * 0.85)


chicles_contados_centro = set()
chicles_contados_lateral = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Sombreado semitransparente de los laterales
    overlay = frame.copy()
    mask = np.zeros_like(frame)
    pts_lateral_izq = np.array([[0, 0], [izq_arriba, 0], [izq_abajo, alto], [0, alto]], np.int32)
    pts_lateral_der = np.array([[der_arriba, 0], [ancho, 0], [ancho, alto], [der_abajo, alto]], np.int32)
    cv2.fillPoly(mask, [pts_lateral_izq], (255, 100, 0))
    cv2.fillPoly(mask, [pts_lateral_der], (255, 100, 0))
    cv2.addWeighted(mask, 0.15, frame, 1.0, 0, frame)

    # Líneas divisorias
    cv2.line(frame, (izq_arriba, 0), (izq_abajo, alto), (255, 150, 0), 2)
    cv2.line(frame, (der_arriba, 0), (der_abajo, alto), (255, 150, 0), 2)

    # Etiquetas de zona
    cv2.putText(frame, "LATERAL", (10, alto // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
    cv2.putText(frame, "LATERAL", (ancho - 110, alto // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
    cv2.putText(frame, "CENTRAL", (ancho // 2 - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # USAMOS .track() EN LUGAR DE .predict() para activar el rastreador (ByteTrack por defecto)
    resultados = model.track(frame, conf=0.60, iou=0.4, persist=True, verbose=False)

    # Contadores locales (solo para lo que se ve en el FRAME ACTUAL)
    actual_central = 0
    actual_laterales = 0

    # Verificamos si YOLO ha encontrado cajas y si tienen IDs asignados
    if resultados[0].boxes is not None and resultados[0].boxes.id is not None:
        boxes = resultados[0].boxes.xyxy.cpu().numpy()
        confs = resultados[0].boxes.conf.cpu().numpy()
        track_ids = resultados[0].boxes.id.int().cpu().numpy()

        for box, conf, track_id in zip(boxes, confs, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Cálculo de límites dinámicos por perspectiva
            limite_izq_dinamico = izq_arriba + ((cy - 0) / (alto - 0)) * (izq_abajo - izq_arriba)
            limite_der_dinamico = der_arriba + ((cy - 0) / (alto - 0)) * (der_abajo - der_arriba)

            # Clasificación por zona
            if cx < limite_izq_dinamico or cx > limite_der_dinamico:
                color = (255, 150, 0)  # naranja → lateral
                zona = "LAT"
                actual_laterales += 1
                chicles_contados_lateral.add(track_id)  # Se añade al set global lateral
            else:
                color = (0, 255, 0)  # verde → central
                zona = ""
                actual_central += 1
                chicles_contados_centro.add(track_id)  # Se añade al set global central

            # Dibujar caja e información del track_id
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            etiqueta = f"ID:{track_id} {zona} {conf:.2f}".strip()
            cv2.putText(frame, etiqueta, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # --- PANEL DE LOGS EN EL VIDEO ---
    # Mostramos tanto lo del frame actual como el conteo histórico acumulado sin duplicados
    cv2.putText(frame, f"En Frame - Centro: {actual_central} | Lats: {actual_laterales}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"TOTAL ACUMULADO:",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"-> Centrales:  {len(chicles_contados_centro)}",
                (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"-> Laterales: {len(chicles_contados_lateral)}",
                (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)

    out.write(frame)
    cv2.imshow("Chicles", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n--- PROCESAMIENTO FINALIZADO ---")
print(f"Total chicles únicos detectados en zona CENTRAL:  {len(chicles_contados_centro)}")
print(f"Total chicles únicos detectados en zona LATERAL:  {len(chicles_contados_lateral)}")
print(f"Total chicles globales en todo el patio: {len(chicles_contados_centro | chicles_contados_lateral)}")