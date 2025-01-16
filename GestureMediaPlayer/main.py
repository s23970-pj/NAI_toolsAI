'''
Odtwarzacz sterowany gestami

Instrukcja uruchomienia:
Pobrać biblioteki
-openCV
-MediaPipe
_PyAutoGUI
można użyć następującego polecenia:
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

# Gesture timing variables
gesture_start = 0.0  # Czas rozpoczęcia wykonywania gestu
last_gesture_time = 0.0  # Czas wykonania ostatniego gestu
gesture_threshold = 1.0  # Czas przytrzymania gestu
gesture_cooldown = 1.5  # Czas, który musi minąć od ostatniego wykonania gestu

gesture = None


# Recognize gestures
def recognize_gesture(hand_landmarks, frame):
    global gesture_start, last_gesture_time, gesture_threshold, gesture_cooldown, gesture

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

    # Display debugging info for landmarks
    cv2.putText(frame, f"Pinky Tip Y: {pinky_tip.y:.3f}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Pinky PIP Y: {pinky_pip.y:.3f}", (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
    # Następny utwór: Wyprostowany palec wskazujący
    elif index_tip.y < index_pip.y:
        if gesture != "next":
            gesture_start = 0.0
        gesture = "next"

        if gesture_start == 0.0:
            gesture_start = time.time()
        elif time.time() - gesture_start > gesture_threshold and time.time() - last_gesture_time > gesture_cooldown:
            pyautogui.hotkey('ctrl', 'right')  # pyautogui.press('right')
            cv2.putText(frame, 'Next', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            gesture_start = 0.0
            last_gesture_time = time.time()
    # Poprzedni utwór: Kciuk w górę "LIKE"
    elif thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y:
        if gesture != "previous":
            gesture_start = 0.0
        gesture = "previous"

        if gesture_start == 0.0:
            gesture_start = time.time()
        elif time.time() - gesture_start > gesture_threshold and time.time() - last_gesture_time > gesture_cooldown:
            pyautogui.hotkey('ctrl', 'left')  # pyautogui.press('left')
            cv2.putText(frame, 'Previous', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            gesture_start = 0.0
            last_gesture_time = time.time()
    # Wyciszenie: Mały palec uniesiony i wyraźnie wyżej niż inne palce
    if pinky_tip.y < pinky_pip.y - 0.02 and pinky_tip.y < index_tip.y and pinky_tip.y < thumb_tip.y:
        if gesture != "mute":
            gesture_start = 0.0
        gesture = "mute"

        if gesture_start == 0.0:
            gesture_start = time.time()
        elif time.time() - gesture_start > gesture_threshold and time.time() - last_gesture_time > gesture_cooldown:
            pyautogui.press('m')
            cv2.putText(frame, 'Mute', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            gesture_start = 0.0
            last_gesture_time = time.time()

    # Display gesture info on the frame
    cv2.putText(frame, f'Gesture: {gesture}', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Hold: {time.time() - gesture_start:.2f}s', (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Cooldown: {time.time() - last_gesture_time:.2f}s', (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# Initialize camera
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Detect and process landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            recognize_gesture(hand_landmarks, frame)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
