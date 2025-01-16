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
import time

import cv2
import mediapipe as mp
import pyautogui

# Computer Vision i Mediapipe do rozpoznawania gestów
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)
# Czas i opóźnienia
gesture_start = 0.0  # Czas rozpoczęcia wykonywania gestu
last_gesture_time = 0.0  # Czas wykonania ostatniego gestu
gesture_threshold = 5.0  # Czas przytrzymania gestu
gesture_cooldown = 2.0  # Czas, który musi minąć od ostatniego wykonania gestu

gesture = None


# Obsługa gestów pyautogui
def recognize_gesture(hand_landmarks, frame):
    global gesture_start, last_gesture_time, gesture_threshold, gesture_cooldown, gesture

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

    # Pauza/Play: Gest w porządku kciuk i palec wskazujący złączone
    if abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05:
        if gesture != "play_pause":
            gesture_start = 0.0
        gesture = "play_pause"

        if gesture_start == 0.0:
            gesture_start = time.time()
        elif time.time() - gesture_start > gesture_threshold and time.time() - last_gesture_time > gesture_cooldown:
            pyautogui.press('space')
            cv2.putText(frame, 'Pause/Play', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            gesture_start = 0.0
            last_gesture_time = time.time()
    # Wyprostowany palec wskazujący
    elif index_tip.x < index_pip.y:
        if gesture != "next":
            gesture_start = 0.0
        gesture = "next"

        if gesture_start == 0.0:
            gesture_start = time.time()
        elif time.time() - gesture_start > gesture_threshold and time.time() - last_gesture_time > gesture_cooldown:
            pyautogui.hotkey('ctrl', 'right')  # pyautogui.press('right')
            cv2.putText(frame, 'Next', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            gesture_start = 0.0
            last_gesture_time = time.time()
    # Poprzedni utwór: Kciuk w górę "LIKE"
    elif thumb_tip.y < thumb_ip.y:
        if gesture != "previous":
            gesture_start = 0.0
        gesture = "previous"
        
        if gesture_start == 0.0:
            gesture_start = time.time()
        elif time.time() - gesture_start > gesture_threshold and time.time() - last_gesture_time > gesture_cooldown:
            pyautogui.hotkey('ctrl', 'left')  # pyautogui.press('left')
            cv2.putText(frame, 'Previous', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            gesture_start = 0.0
            last_gesture_time = time.time()
    # Wyciszenie poprzez złączenie wszystkich palców
    # !!! POPRAWIĆ MUTE !!! NIE ROZPOZNAJE
    elif abs(thumb_tip.x - pinky_tip.x) < 0.05 and abs(thumb_tip.y - pinky_tip.y) < 0.05:
        if gesture != "mute":
            gesture_start = 0.0
        gesture = "mute"
        
        if gesture_start == 0.0:
            gesture_start = time.time()
        elif time.time() - gesture_start > gesture_threshold and time.time() - last_gesture_time > gesture_cooldown:
            pyautogui.press('m')
            cv2.putText(frame, 'Mute', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            gesture_start = 0.0
            last_gesture_time = time.time()
    else:
        gesture_start = 0.0

    print(f"gesture: {gesture}")
    print(f"gesture_hold: {time.time() - gesture_start}")
    print(f"last_gesture: {time.time() - last_gesture_time}")


# inicjalizacja kamery
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Przekonwertowanie obrazu do RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Rozpoznawanie puntów
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            recognize_gesture(hand_landmarks, frame)  # Wywołanie funkcji
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
