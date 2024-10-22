from random import randint

from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax


class PointGame(TwoPlayerGame):
    """In turn, players choose some negative value. The first player to reach zero points wins."""

    def __init__(self, players=None):
        self.players = players
        self.dimension = 3
        self.moves = [randint(0, 100) for _ in range(self.dimension ** 2)]
        self.playerOnePoints = 100
        self.playerTwoPoints = 100
        self.current_player = 1

    def possible_moves(self):
        return self.moves

    def make_move(self, move):
        if self.current_player == 1:
            self.playerOnePoints -= move
        if self.current_player == 2:
            self.playerTwoPoints -= move

        self.moves = [randint(0, 100) for _ in range(self.dimension ** 2)]

    def win(self):
        return self.playerTwoPoints <= 0

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
