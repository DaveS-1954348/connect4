from random import randrange

from connectfour import Game
from player import Player
from model import Model
from connectfour import starting_player
from copy import copy, deepcopy


def print_results(amount, p1_wins, p2_wins, draws):
    print(f"results from {amount} game(s):\nplayer1 wins: {int(p1_wins/amount*100)}%\nplayer2 wins: {int(p2_wins/amount*100)}%\ndraws: {int(draws/amount*100)}%")


class Controller:

    def __init__(self, p1: Player, p2: Player, model: Model):
        self.p1 = p1
        self.p2 = p2
        self.model = model

    def play_games(self, amount, startingPlayer=None):
        p1_wins = 0
        p2_wins = 0
        draws = 0
        for i in range(amount):
            starting = startingPlayer if startingPlayer is not None else starting_player()  # selects starting player or selects a random player
            self.game = Game()
            result = self.play(starting)
            if result == -1:
                p1_wins += 1
            elif result == 1:
                p2_wins += 1
            else:
                draws += 1
        print_results(amount, p1_wins, p2_wins, draws)

    def play(self, player):
        while self.game.status is None:
            board_copy = deepcopy(self.game.board)
            max_value = 0
            best_move = 0
            for col in range(self.game.width):
                lowest_row = 0
                for row in range(self.game.height):
                    if board_copy[row][col] == 0:
                        lowest_row = row
                        break
                board_copy = deepcopy(self.game.board)
                if player == -1:
                    if self.p1.strategy == "neural":
                        board_copy[lowest_row][col] = player
                        value = self.get_neural_move(board_copy, 2)
                        #print(f"player {player} predicted value for col {col} is {value}")
                        if value > max_value:
                            best_move = col
                            max_value = value
                    else:
                        best_move = self.get_random_move()
                else:
                    if self.p2.strategy == "neural":
                        board_copy[lowest_row][col] = player
                        value = self.model.predict(board_copy, 1)
                        pr_zero = self.model.predict(board_copy, 0)
                        pred_two = self.model.predict(board_copy, 2)
                        #print(f"player {player} predicted value for col {col} is {value}")
                        #print(f"prediction 0: {pr_zero}")
                        #print(f"prediction 2: {pred_two}")

                        if value > max_value:
                            best_move = col
                            max_value = value
                    else:
                        best_move = self.get_random_move()
            #print(f"player {player} best move: {best_move}")
            self.game.play_move(player, best_move)
            #print(f"updated_board:\n{self.game.board}")
            player *= -1
        #print(f"results:\n {self.game.board}")
        return self.game.status

    def get_random_move(self):
        return randrange(self.game.width)

    def get_neural_move(self, board_copy, value):
        return self.model.predict(board_copy, value)

    def set_players(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
