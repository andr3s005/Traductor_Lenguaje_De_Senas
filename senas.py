import cv2
import mediapipe as mp

def distancia_euclidiana(p1, p2):
    d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return d

def draw_bounding_box(image, hand_landmarks):
    image_height, image_width, _ = image.shape
    x_min, y_min = image_width, image_height
    x_max, y_max = 0, 0
    
    # Iterate through the landmarks to find the bounding box coordinates
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y
    
    # Draw the bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Sensibilidad a la distancia en píxeles. Ajusta estos valores si la detección es inestable.
DELTA_Y = 50 
DELTA_DIST_CLOSE = 40 # Distancia para considerar dedos juntos/doblados
DELTA_DIST_FAR = 100 # Distancia para considerar dedos separados


cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_height, image_width, _ = image.shape
        predicted_letter = "" 

        if results.multi_hand_landmarks:
            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Draw bounding box
                draw_bounding_box(image, hand_landmarks)

                # PUNTAS (TIP)
                thumb_tip = (int(hand_landmarks.landmark[4].x * image_width), int(hand_landmarks.landmark[4].y * image_height))
                index_tip = (int(hand_landmarks.landmark[8].x * image_width), int(hand_landmarks.landmark[8].y * image_height))
                middle_tip = (int(hand_landmarks.landmark[12].x * image_width), int(hand_landmarks.landmark[12].y * image_height))
                ring_tip = (int(hand_landmarks.landmark[16].x * image_width), int(hand_landmarks.landmark[16].y * image_height))
                pinky_tip = (int(hand_landmarks.landmark[20].x * image_width), int(hand_landmarks.landmark[20].y * image_height))
                
                # FALANGES MEDIAS (PIP)
                thumb_pip = (int(hand_landmarks.landmark[2].x * image_width), int(hand_landmarks.landmark[2].y * image_height))
                index_pip = (int(hand_landmarks.landmark[6].x * image_width), int(hand_landmarks.landmark[6].y * image_height))
                middle_pip = (int(hand_landmarks.landmark[10].x * image_width), int(hand_landmarks.landmark[10].y * image_height))
                ring_pip = (int(hand_landmarks.landmark[14].x * image_width), int(hand_landmarks.landmark[14].y * image_height))
                pinky_pip = (int(hand_landmarks.landmark[18].x * image_width), int(hand_landmarks.landmark[18].y * image_height))

                # OTROS PUNTOS
                wrist = (int(hand_landmarks.landmark[0].x * image_width), int(hand_landmarks.landmark[0].y * image_height))
                
                is_index_extended = index_tip[1] < index_pip[1] - DELTA_Y
                is_middle_extended = middle_tip[1] < middle_pip[1] - DELTA_Y
                is_ring_extended = ring_tip[1] < ring_pip[1] - DELTA_Y
                is_pinky_extended = pinky_tip[1] < pinky_pip[1] - DELTA_Y
                
                is_index_folded = index_tip[1] > index_pip[1] + DELTA_Y
                is_middle_folded = middle_tip[1] > middle_pip[1] + DELTA_Y
                is_ring_folded = ring_tip[1] > ring_pip[1] + DELTA_Y
                is_pinky_folded = pinky_tip[1] > pinky_pip[1] + DELTA_Y
                
                # Lógica del pulgar (más difícil por la rotación)
                is_thumb_folded = thumb_tip[1] > thumb_pip[1] + 30 or distancia_euclidiana(thumb_tip, thumb_pip) < 50
                is_thumb_separated = distancia_euclidiana(thumb_tip, index_tip) > DELTA_DIST_FAR

                # A: Puño cerrado, pulgar a un lado.
                if is_index_folded and is_middle_folded and is_ring_folded and is_pinky_folded and \
                   thumb_tip[1] < index_pip[1] and thumb_tip[0] > index_pip[0]: 
                    predicted_letter = 'A'

                # B: Mano abierta y extendida, pulgar doblado (o cerca de la palma).
                elif is_index_extended and is_middle_extended and is_ring_extended and is_pinky_extended and \
                     not is_thumb_separated: 
                    predicted_letter = 'B'

                # C: Mano curvada, como si sostuviera una taza.
                elif not is_index_extended and not is_pinky_extended and \
                     distancia_euclidiana(index_tip, wrist) > DELTA_DIST_FAR and \
                     distancia_euclidiana(pinky_tip, wrist) > DELTA_DIST_FAR and \
                     distancia_euclidiana(index_tip, middle_tip) < DELTA_DIST_CLOSE * 1.5: 
                    predicted_letter = 'C'
                        
        cv2.imshow('MediaPipe Hands - Vocales (ASL)', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()