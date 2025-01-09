#OpenCV
#PyAutoGUI??

#opencv-python, mediapipe, pyautogui.
#Krok 3: Inicjalizacja kamery i MediaPipe

  #  Otwórz kamerę
  #  Wykorzystaj model detekcji rąk MediaPipe

#Krok 4: Wykrywanie dłoni i punktów charakterystycznych
#
#Każda dłoń ma 21 punktów charakterystycznych, które MediaPipe potrafi wykryć. Kluczowe punkty to:

#    kciuk: THUMB_TIP
  #  palec wskazujący: INDEX_FINGER_TIP
 #   środkowy: MIDDLE_FINGER_TIP

 #Krok 5: Definiowanie gestów

#Zaproponowane gesty do obsługi odtwarzacza multimedialnego:

 #   Pauza/Odtwarzanie: zbliżenie kciuka i palca wskazującego.
  #  Następny utwór: wyprostowany palec wskazujący.
   # Poprzedni utwór: wyprostowany palec środkowy.
    #Wyciszenie: złączenie kciuka i małego palca