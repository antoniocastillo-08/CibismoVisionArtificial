import cv2
import os

# Ruta del video
video_path = 'VIDEOS/video_cortado.mp4'
# Carpeta donde guardar los frames
output_folder = 'frames_extraidos_2'

# Crear la carpeta si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Capturar el video
cam = cv2.VideoCapture(video_path)

current_frame = 0
while True:
    # Leer el frame
    success, frame = cam.read()

    if not success:
        break

    # Definir el nombre del archivo
    name = f'{output_folder}/frame_{current_frame}.jpg'
    print(f'Creando... {name}')

    # Guardar el frame
    cv2.imwrite(name, frame)

    current_frame += 1

# Liberar la captura
cam.release()
cv2.destroyAllWindows()