import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
# 1. Configuración de la página
st.set_page_config(page_title="Detector de Chicles - Selección Interactiva", layout="wide")
st.title("🔍 Detección con Escala Dinámica e Interacción Directa")


@st.cache_resource
def load_model():
    # Asegúrate de que best.pt está en la misma carpeta que este script
    return YOLO("best.pt")


model = load_model()

# Barra lateral para parámetros fijos
st.sidebar.header("⚙️ Ajustes del Modelo YOLO")
conf_threshold = st.sidebar.slider("Confianza", 0.10, 1.00, 0.60, 0.05)
iou_threshold = st.sidebar.slider("IoU", 0.10, 1.00, 0.40, 0.05)

# 2. SELECCIÓN DE IMAGEN
uploaded_file = st.file_uploader("Sube la imagen para el análisis...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convertir a imagen de OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Mantener una copia original limpia y otra para procesar
    imagen_procesada = frame.copy()
    alto, ancho = imagen_procesada.shape[:2]

    # --- PASO 1: DEFINIR LA ESCALA INTERACTIVAMENTE ---
    st.header("📐 Paso 1: Definir la Escala por Puntos de Referencia")
    st.write(
        "Usa el canvas a continuación para **hacer clic en dos puntos** que definan una distancia conocida en la imagen (ej: una baldosa de 1m).")

    # Configurar el canvas interactivo
    col_canvas, col_medida = st.columns([2, 1])

    with col_canvas:
        # Convertir BGR a RGB para el canvas
        img_rgb_canvas = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Este componente crea un lienzo interactivo sobre la imagen
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Color de relleno de las formas
            stroke_width=3,
            stroke_color="#FF0000",  # Rojo para la línea de medición
            background_image=Image.fromarray(img_rgb_canvas),
            update_streamlit=True,
            height=alto,  # Ajustar la altura al tamaño real
            width=ancho,  # Ajustar el ancho al tamaño real
            drawing_mode="line",  # Solo permitir dibujar líneas
            point_display_radius=5,  # Radio de los puntos finales
            key="canvas_escala",
        )

    # Inicializar PX_POR_METRO
    px_por_metro = 0

    with col_medida:
        st.subheader("Configuración de la Medición")

        # Procesar los resultados del canvas
        if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
            # Obtener el último objeto dibujado (debería ser nuestra línea)
            ultima_linea = canvas_result.json_data["objects"][-1]

            # Solo si es una línea o tiene puntos definidos
            if "x1" in ultima_linea:
                x1, y1 = ultima_linea["x1"], ultima_linea["y1"]
                x2, y2 = ultima_linea["x2"], ultima_linea["y2"]

                # Calcular distancia en píxeles (Pitágoras)
                distancia_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # Mostrar coordenadas y distancia medida en píxeles
                st.write(f"📏 **Línea medida en pantalla:**")
                st.write(f"Punto 1: [{int(x1)}, {int(y1)}], Punto 2: [{int(x2)}, {int(y2)}]")
                st.write(f"Distancia: **{int(distancia_px)} píxeles**")

                # Ingreso de la distancia real
                distancia_real_m = st.number_input("Distancia REAL de esa línea (en metros):",
                                                   min_value=0.01, value=1.00, step=0.1, format="%.2f")

                # Botón para confirmar la escala
                if st.button("✅ Confirmar Escala e Iniciar Análisis"):
                    if distancia_px > 0:
                        px_por_metro = distancia_px / distancia_real_m
                        # Guardar la escala en session_state para que persista
                        st.session_state['escala_px_metro'] = px_por_metro
                        st.session_state['analisis_completo'] = False  # Resetear análisis anterior
                        st.success(f"Escala definida: **{int(px_por_metro)} px/m**. Iniciando análisis...")
            else:
                st.warning("Por favor, dibuja una línea clara sobre el objeto de referencia.")
        else:
            st.info("Dibuja una línea en el canvas izquierdo conectando dos puntos de referencia conocidos.")

    # --- PASO 2: ANÁLISIS DE DENSIDAD (Dependiente de la escala) ---
    if 'escala_px_metro' in st.session_state and st.session_state['escala_px_metro'] > 0:
        PX_POR_METRO = st.session_state['escala_px_metro']

        st.write("---")
        st.header("📊 Paso 2: Resultado del Análisis de Densidad")

        # Reutilizar el resto de tu lógica original, adaptada a Streamlit
        CELDA_PX = int(PX_POR_METRO)
        DISTANCIA_DEDUP = int(PX_POR_METRO * 0.05)  # 5cm se adapta

        # Evitar errores si la celda es muy chica
        if CELDA_PX <= 10:
            st.error("El tamaño de celda calculado es demasiado pequeño. Redibuja la línea de referencia.")
            st.stop()



        # Detección YOLO
        resultados = model.predict(imagen_procesada, conf=conf_threshold, iou=iou_threshold, verbose=False)

        chicles_unicos = {}
        conteo_celdas = defaultdict(int)


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

                cv2.rectangle(imagen_procesada, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(imagen_procesada, f"{conf:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Pintar el conteo por celda
        for (gx, gy), n in conteo_celdas.items():
            x0 = gx * CELDA_PX
            y0 = gy * CELDA_PX
            x1 = min(x0 + CELDA_PX, ancho)
            y1 = min(y0 + CELDA_PX, alto)

            if n < 3:
                color_celda = (0, 255, 255)  # amarillo
            elif n < 7:
                color_celda = (0, 165, 255)  # naranja
            else:
                color_celda = (0, 0, 255)  # rojo

            cv2.rectangle(imagen_procesada, (x0, y0), (x1, y1), color_celda, 2)
            cv2.putText(imagen_procesada, f"{n}/m2",
                        (x0 + 10, y0 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_celda, 2)

        total_global = len(chicles_unicos)

        # Panel resumen actualizado dinámicamente en la imagen
        cv2.rectangle(imagen_procesada, (15, alto - 100), (450, alto - 10), (0, 0, 0), -1)
        cv2.putText(imagen_procesada, f"Total chicles: {total_global}",
                    (20, alto - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(imagen_procesada, f"Escala: 1m = {CELDA_PX}px (Interactivo)",
                    (20, alto - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Mostrar Resultados Finales en Streamlit
        col1, col2 = st.columns(2)
        col1.metric("Total Chicles", total_global)
        col2.metric("Resolución de Escala", f"{CELDA_PX} px/m")

        st.image(cv2.cvtColor(imagen_procesada, cv2.COLOR_BGR2RGB), use_container_width=True)