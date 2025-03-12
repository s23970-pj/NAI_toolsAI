from random import randint
from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax

"""
Autor: Adrian Goik

Zasady: Gra turowa o sumie zerowej polegająca na wyświetleniu planszy w postaci macierzy NxN,
której pola posiadają losowe wartości od -10 do 10. Wartości z wybranych pól są dodawane do punktów gracza
i odejmowane od punktów przeciwnika. Gra kończy się po zajęciu wszystkich pól, a wygrywa gracz z większą liczbą
punktów. 

Aby uruchomić grę, musisz mieć zainstalowane Python 3.12 oraz bibliotekę easyAI. Postępuj zgodnie z poniższymi krokami:

Krok 1: Zainstaluj Pythona
Jeśli nie masz zainstalowanego Pythona, pobierz i zainstaluj go z oficjalnej strony: https://www.python.org/downloads/

Pobierz PyCharm->otwórz plik -> naciśnij Run-> w terminalu otworzy się gra
"""


class MatrixGame(TwoPlayerGame):
    def __init__(self, players=None):
        """
           Inicjalizuje obiekt klasy PointGame, tworzy macierz o wymiarach N x N i przypisuje punkty graczy.
            Aby zmienić rozmiar macierzy wystarczy zmienić atrybut dimension na N.
        Parameters:
        players (list): Lista graczy, którzy będą grać (człowiek i AI).

        Returns:
        None
        """
        self.players = players
        self.dimension = 3
        self.moves = [[randint(-10, 10) for _ in range(self.dimension)] for _ in range(self.dimension)]
        self.playerOnePoints = 0
        self.playerTwoPoints = 0
        self.current_player = 1

    def possible_moves(self):
        """
        Zwraca listę dostępnych współrzędnych ruchów, które nie zostały jeszcze zajęte.

        Returns:
        list: Lista dostępnych ruchów w formacie 'wiersz-kolumna' (np. '1-2').
        """
        return [f"{i}-{j}" for i in range(self.dimension) for j in range(self.dimension) if self.moves[i][j] != "X"]

    def make_move(self, move):
        """
      Przetwarza ruch gracza. Dodaje wartość z wybranego pola do punktów aktualnego gracza i odejmuje tę wartość od punktów przeciwnika.

        Parameters:
        move (str): Współrzędne w formacie 'wiersz-kolumna', które odpowiadają pozycji w macierzy.

        Returns:
        None
        """
        # Zamiana ruchu (np. '1-2') na współrzędne (wiersz, kolumna)
        rzad, kolumna = map(int, move.split('-'))
        value = self.moves[rzad][kolumna]  # Wybrana wartość z planszy

        # Dodajemy punkty aktualnemu graczowi, a odejmujemy przeciwnikowi
        if self.current_player == 1:
            self.playerOnePoints += value
            self.playerTwoPoints -= value
        else:
            self.playerTwoPoints += value
            self.playerOnePoints -= value

        # Usuwamy wybrany ruch z macierzy (oznaczamy jako "X")
        self.moves[rzad][kolumna] = "X"

    def is_over(self):
        """
         Sprawdza, czy gra się zakończyła. Gra kończy się, gdy wszystkie pola są zajęte (oznaczone jako "X").

        Returns:
        bool: Zwraca True, jeśli wszystkie pola są zajęte, w przeciwnym razie False.
        """
        return all(self.moves[i][j] == "X" for i in range(self.dimension) for j in range(self.dimension))

    def show(self):
        """
        Wyświetla stan gry, w tym planszę (dostępne wartości na planszy) oraz aktualne punkty graczy.

        Returns:
        None
        """
        print("\nPlansza:")
        for rzad in self.moves:
            print(" | ".join(f"{x:3}" for x in rzad))  # Wyświetla każdy wiersz macierzy
        print(f"\nPunkty Gracz 1: {self.playerOnePoints}")
        print(f"Punkty AI: {self.playerTwoPoints}")

    def scoring(self):
        """
    Funkcja zwraca wynik, który używany jest przez algorytm AI do oceny stanu gry.

        Returns:
        int: Różnica punktów gracza 2 (AI) i gracza 1.
        """
        return self.playerTwoPoints - self.playerOnePoints
    def winner(self):
        '''
         Określa zwycięzcę na podstawie punktów obu graczy. Wyświetla, kto wygrał po zakończeniu gry.

        Returns:
        None
        '''
        if self.playerOnePoints > self.playerTwoPoints:
            print("\nGracz 1 WYGRYWA!!!")
        elif self.playerTwoPoints > self.playerOnePoints:
            print("\nAI WYGRYWA!!!")
        else:
            print("\nRemis")

# Definicja AI z głębokością 13 dla algorytmu Negamax
ai = Negamax(6)

# Tworzenie gry z graczem i AI
game = MatrixGame([Human_Player(), AI_Player(ai)])

# Rozpoczęcie gry
history = game.play()
#Określenie zwycięzcy
game.winner()
