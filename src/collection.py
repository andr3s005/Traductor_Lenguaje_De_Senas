import cv2
import mediapipe as mp
import os
import csv
import copy
import itertools

# --- CONFIGURACIÓN INICIAL ---
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Crea la carpeta data si no existe
if not os.path.exists('data'):
    os.makedirs('data')

# Función para normalizar las coordenadas (hacerlas relativas a la muñeca)
# Esto ayuda a que no importe si la mano está cerca o lejos de la cámara
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convertir a coordenadas relativas
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Aplanar la lista (convertir de pares [x,y] a una lista larga [x,y,x,y...])
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalización (valor máximo absoluto)
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

# --- INICIO DE CAPTURA ---
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    print("--- INSTRUCCIONES ---")
    print("1. Pon tu mano frente a la cámara.")
    print("2. Haz la seña de una letra (ej. 'A').")
    print("3. Presiona la tecla de esa letra en tu teclado (ej. presiona 'a').")
    print("   El sistema guardará esa posición.")
    print("4. Repite varias veces moviendo un poco la mano para tener variedad.")
    print("5. Presiona 'ESC' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        debug_image = copy.deepcopy(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # 1. Dibujar mano
                mp_drawing.draw_landmarks(
                    debug_image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS)
                
                # 2. Procesar datos
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # 3. Escuchar teclado para guardar
                key = cv2.waitKey(10)
                if 97 <= key <= 122: # Código ASCII para a-z
                    letter = chr(key).upper()
                    
                    # Guardar en CSV: [LETRA, p1_x, p1_y, ... p21_x, p21_y]
                    with open('data/keypoints.csv', 'a', newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([letter, *pre_processed_landmark_list])
                    
                    print(f"Dato guardado para la letra: {letter}")
                    
                    # Feedback visual (parpadeo verde)
                    cv2.putText(debug_image, f"Guardado: {letter}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Recoleccion de Datos LSM', debug_image)
        
        if cv2.waitKey(1) == 27: # ESC para salir
            break

cap.release()
cv2.destroyAllWindows()