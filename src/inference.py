import pickle
import cv2
import mediapipe as mp
import numpy as np
import copy
import itertools

# --- 1. CONFIGURACIÓN E IMPORTACIÓN DEL MODELO ---
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Copiamos las mismas funciones de pre-procesamiento para que el modelo entienda los datos
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
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

# --- 2. INICIO DE VIDEO EN TIEMPO REAL ---
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, # Por ahora solo detectamos una mano
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) # Espejo
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # A. Dibujar la mano
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # B. Pre-procesar (Convertir imagen a números)
                landmark_list = calc_landmark_list(frame, hand_landmarks)
                processed_data = pre_process_landmark(landmark_list)

                # C. PREDICCIÓN (Aquí ocurre la magia)
                # El modelo recibe los datos y nos devuelve la letra
                prediction = model.predict([processed_data])
                predicted_character = prediction[0]

                # D. Dibujar un rectángulo y la letra
                # Obtenemos coordenadas para el cuadrado alrededor de la mano
                x_vals = [lm[0] for lm in landmark_list]
                y_vals = [lm[1] for lm in landmark_list]
                x1, y1 = min(x_vals) - 20, min(y_vals) - 20
                x2, y2 = max(x_vals) + 20, max(y_vals) + 20

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4) # Borde negro
                cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('Traductor LSM - Final', frame)
        if cv2.waitKey(1) == 27: # ESC para salir
            break

cap.release()
cv2.destroyAllWindows()