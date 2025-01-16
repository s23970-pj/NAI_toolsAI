"""
Odtwarzacz sterowany gestami

Autorzy: Adrian Goik, Łukasz Soldatke

Opis problemu:
Sterowanie podstawowymi funkcjami Spotify za pomocą gestów dłoni rozpoznawanych przy użyciu MediaPipe i OpenCV.

Instrukcja uruchomienia:
1. Pobrać wymagane biblioteki:
   - opencv-python
   - mediapipe
   - pyautogui (opcjonalnie, jeśli używane są klawisze)
   - spotipy
   Można użyć następującego polecenia:
   ~pip install opencv-python mediapipe pyautogui spotipy

2. KONFIGURACJA SPOTIFY API:
   - Wejdź na stronę https://developer.spotify.com/documentation/web-api.
   - W zakładce Dashboard uzyskaj swój własny Client ID i Client Secret (tylko dla użytkowników premium).
   - Wstaw odpowiednio do zmiennych SPOTIPY_CLIENT_ID i SPOTIPY_CLIENT_SECRET.

Działanie programu:
- Program używa kamery internetowej do śledzenia dłoni i rozpoznawania gestów.
- Rozpoznane gesty sterują odtwarzaczem Spotify (pauza/play, następny utwór, poprzedni utwór, wyciszenie).
-    Pauza/Play: Gest  kciuk i palec wskazujący złączone (jakby szczypanie)
-    Następny utwór: Wyprostowany palec wskazujący
-    Poprzedni utwór: Znak LIKE, kciuk w górę
-    Wyciszenie: Mały palec uniesiony i wyraźnie wyżej niż inne palce

"""

import time
import cv2
import mediapipe as mp
import pyautogui
import spotipy
from spotipy.oauth2 import SpotifyOAuth


# Konfiguracja Spotify API
SPOTIPY_CLIENT_ID = ''
SPOTIPY_CLIENT_SECRET = ''
SPOTIPY_REDIRECT_URI = 'http://localhost:3000/callback'

# Zakres uprawnień wymagany przez spotify API
scope = "user-modify-playback-state user-read-playback-state"
spotify = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope=scope,
    ),
    requests_timeout=10  # Ustawienie limitu czasu na 10 sekund
)

# Computer Vision i Mediapipe do rozpoznawania gestów
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# Dodanie do gestów opóźnień czasowych
gesture_start = 0.0  # Czas rozpoczęcia wykonywania gestu
last_gesture_time = 0.0  # Czas wykonania ostatniego gestu
gesture_threshold = 1.0  # Czas przytrzymania gestu
gesture_cooldown = 1.5  # Czas, który musi minąć od ostatniego wykonania gestu

gesture = None

is_muted = False

devices = spotify.devices()
print("Available devices:")
for device in devices['devices']:
    print(f"Name: {device['name']}, Type: {device['type']}, ID: {device['id']}")


# Funkcja do rozpoznawania gestów
def recognize_gesture(hand_landmarks, frame):
    """
      Rozpoznaje gesty na podstawie landmarków dłoni i wykonuje odpowiednie akcje.

      Args:
          hand_landmarks: Wynik przetwarzania landmarków dłoni przez MediaPipe.
          frame: Aktualna klatka obrazu z kamery.

      Gesty obsługiwane:
          - Pauza/Play: Kciuk i palec wskazujący dotykają się.
          - Następny utwór: Palec wskazujący wyprostowany.
          - Poprzedni utwór: Kciuk skierowany w górę.
          - Wyciszenie: Mały palec uniesiony wyżej niż pozostałe palce.
      """
    global gesture_start, last_gesture_time, gesture_threshold, gesture_cooldown, gesture, is_muted

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
            playback_state = spotify.current_playback()
            if playback_state and playback_state['is_playing']:
                spotify.pause_playback()
            else:
                spotify.start_playback()
            cv2.putText(frame, 'Pause/Play', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            gesture_start = 0.0
            last_gesture_time = time.time()

    # Następny utwór: Dłoń w pionie
    elif index_tip.y < index_pip.y:
        if gesture != "next":
            gesture_start = 0.0
        gesture = "next"

        if gesture_start == 0.0:
            gesture_start = time.time()
        elif time.time() - gesture_start > gesture_threshold and time.time() - last_gesture_time > gesture_cooldown:
            # pyautogui.hotkey('ctrl', 'right')  # pyautogui.press('right')
            spotify.next_track()
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
            # pyautogui.hotkey('ctrl', 'left')  # pyautogui.press('left')
            spotify.previous_track()
            cv2.putText(frame, 'Previous', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            gesture_start = 0.0
            last_gesture_time = time.time()
    # Wyciszenie: Dłoń w poziomie
    elif pinky_tip.y < pinky_pip.y - 0.02 and pinky_tip.y < index_tip.y and pinky_tip.y < thumb_tip.y:
        if gesture != "mute":
            gesture_start = 0.0
        gesture = "mute"

        if gesture_start == 0.0:
            gesture_start = time.time()
        elif time.time() - gesture_start > gesture_threshold and time.time() - last_gesture_time > gesture_cooldown:
            devices = spotify.devices()
            active_device = next((d for d in devices['devices'] if d['is_active']), None)

            if not active_device:
                # Jeśli brak aktywnego urządzenia, spróbuj przenieść odtwarzanie na pierwsze dostępne urządzenie
                if devices['devices']:
                    device_id = devices['devices'][0]['id']
                    spotify.transfer_playback(device_id=device_id, force_play=True)
                    active_device = devices['devices'][0]
                    print(f"Transferred playback to device: {active_device['name']}")
                else:
                    print("No active devices found. Ensure Spotify is running on a device.")
                    cv2.putText(frame, 'No Active Device', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    return

            try:
                # Przełączanie wyciszenia
                if not is_muted:
                    spotify.volume(0, device_id=active_device['id'])  # Ustaw głośność na 0
                else:
                    spotify.volume(50, device_id=active_device['id'])  # Przywróć głośność
                is_muted = not is_muted
                cv2.putText(frame, 'Mute Toggled', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error setting volume: {e}")
                cv2.putText(frame, 'Error Muting', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            gesture_start = 0.0
            last_gesture_time = time.time()

    cv2.putText(frame, f'Gesture: {gesture}', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Hold Time: {time.time() - gesture_start:.2f}s', (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    cv2.putText(frame, f'Cooldown: {time.time() - last_gesture_time:.2f}s', (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)


# Initialize camera
while cap.isOpened():
    """
    Odczytuje obraz z kamery, przetwarza go za pomocą MediaPipe
    oraz wywołuje funkcję rozpoznawania gestów.
    """
    ret, frame = cap.read()
    if not ret:
        break

    # Konwertuje obraz do RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Rozpoznawanie tzw. landmarków (gestów)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            recognize_gesture(hand_landmarks, frame)
#Wyświetlanie obrazu
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Zwalnianie zasobów
cap.release()
cv2.destroyAllWindows()
