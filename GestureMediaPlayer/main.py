'''
Odtwarzacz sterowany gestami

Instrukcja uruchomienia:
Pobrać biblioteki
-openCV
-MediaPipe
_PyAutoGUI
można użyć następującego polecenia polecenia:
~pip install opencv-python mediapipe pyautogui

'''

import cv2
import mediapipe as mp
#Computer Vision i Mediapipe do rozpoznawania gestów
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #Przekonwertowanie obrazu do RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    #Rozpoznawanie puntów
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#

