'''
Autorzy: Adrian Goik, Łukasz Soldatke
Zasady: Gra turowa o sumie zerowej polegająca na wyświetleniu planszy w postaci macierzy 5x5,
której pola posiadają losowe wartości od -10 do 10. Wartości z wybranych pól są dodawane do punktów gracza
i odejmowane od punktów przeciwnika. Gra kończy się po zajęciu wszystkich pól, a wygrywa gracz z większą liczbą punktów.


-komentarze, dokumentacja

'''

from easyAI import AI
import random


def create_matrix():
    return [[random.randint(-10,10) for _ in range(5)] for _ in range(5)]
def main():
