'''
Autorzy: Adrian Goik, Łukasz Soldatke
Zasady: Gra turowa o sumie zerowej polegająca na wyświetleniu planszy w postaci macierzy 5x5,
której pola posiadają losowe wartości od -10 do 10. Wartości z wybranych pól są dodawane do punktów gracza
i odejmowane od punktów przeciwnika. Gra kończy się po zajęciu wszystkich pól, a wygrywa gracz z większą liczbą punktów.


-komentarze, dokumentacja

'''

from random import randint

from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax


class PointGame(TwoPlayerGame):
    """In turn, players choose some negative value. The first player to reach zero points wins."""

    def __init__(self, players=None):
        self.players = players
        self.dimension = 5
        self.moves = [randint(-10, 10) for _ in range(self.dimension ** 2)]
        self.playerOnePoints = 0
        self.playerTwoPoints = 0
        self.current_player = 1

    def possible_moves(self):
        return self.moves

    def make_move(self, move):
        if self.current_player == 1:
            self.playerOnePoints += move
            self.playerTwoPoints -= move
        if self.current_player == 2:
            self.playerTwoPoints += move
            self.playerOnePoints -= move

        self.moves.remove(move)
        self.moves = [randint(-10, 10) for _ in range(len(self.moves) - 1)]

    def win(self):
        return len(self.moves) == 0 and self.playerTwoPoints > self.playerOnePoints

    def is_over(self):
        return self.win() or self.playerOnePoints <= 0

    def show(self):
        print(self.moves)
        print("Player 1 score: %d\nPlayer 2 score: %d" % (self.playerOnePoints, self.playerTwoPoints))

    def scoring(self):
        return 1 if game.win() else 0


ai = Negamax(13)
game = PointGame([Human_Player(), AI_Player(ai)])
history = game.play()
