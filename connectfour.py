from random import random, randint
from copy import copy, deepcopy
import numpy as np
from sys import stderr
from random import randrange

from model import Model


def check_consecutive(board, consecutive=4):
    for row in board:
        count = 1
        for col in range(1, len(row)):
            if row[col - 1] == row[col]:
                count += 1
            else:
                count = 1
            if count == consecutive and row[col] != 0:
                return row[col]
    return 0


def diagonals(L):
    h, w = len(L), len(L[0])
    return [[L[h - p + q - 1][q] for q in range(max(p - h + 1, 0), min(p + 1, w))] for p in range(h + w - 1)]


def antidiagonals(L):
    h, w = len(L), len(L[0])
    return [[L[p - q][q] for q in range(max(p - h + 1, 0), min(p + 1, w))] for p in range(h + w - 1)]


def argmax(x, key=lambda x: x):
    (k, i, v) = max(((key(v), i, v) for i, v in enumerate(x)))
    return (i, v)


def starting_player():
    return 1 if randint(0, 1) == 0 else -1


class Game:
    width = 7
    height = 6
    consecutive = 4

    def __init__(self, board=None):
        if board is None:
            self.board = np.zeros((self.height, self.width), dtype='int')
        else:
            self.board = np.array(board)
        self.status = self.check_status()

    def random_action(self, legal_only=True):
        column = randint(0, self.width - 1)
        if legal_only:
            while not self.is_legal_move(column):
                column = randint(0, self.width - 1)
        return column

    def is_full(self):
        return np.all(self.board != 0)

    def is_legal_move(self, column):
        for row in range(self.height):
            if self.board[row, column] == 0:
                return True
        return False

    def play_move(self, player, column):
        legal_move = False
        for row in range(self.height):
            if self.board[row, column] == 0:
                self.board[row, column] = player
                legal_move = True
                break

        if not legal_move:
            status = player * -1
        else:
            status = self.check_status()
        self.status = status

    def check_status(self):
        h = check_consecutive(self.board, self.consecutive)
        if h != 0:
            return h

        v = check_consecutive(self.board.T, self.consecutive)
        if v != 0:
            return v

        d = check_consecutive(diagonals(self.board), self.consecutive)
        if d != 0:
            return d

        ad = check_consecutive(antidiagonals(self.board), self.consecutive)
        if ad != 0:
            return ad

        if self.is_full():
            return 0

        return None

    def __repr__(self):
        return repr(self.board)

    def random_play(self, starting=None, legal_only=True, f=None):
        player = starting if starting is not None else starting_player()

        f is not None and f(self, None, None)
        while self.status is None:
            move = self.random_action(legal_only=legal_only)
            self.play_move(player, move)
            f is not None and f(self, player, move)
            player = player * -1
        return self.status

    def winning(self, player, legal_only=True, n=1000):
        other = player * -1
        p = np.empty(self.width)
        for col in range(self.width):
            wins = losses = 0
            for i in range(n):
                game = deepcopy(self)
                game.play_move(player, col)
                status = game.random_play(other, legal_only=legal_only)
                if status == player:
                    wins += 1
                elif status != 0:
                    losses += 1

            ratio = (wins / losses) if losses != 0 else wins
            p[col] = ratio
        return p

    def smart_action(self, player, legal_only=True, n=100):
        p = self.winning(player, legal_only=legal_only, n=n)
        return argmax(p)

    def smart_play(self, starting=None, legal_only=True, n=100, f=None):
        player = starting if starting is not None else starting_player()

        f is not None and f(self, None, None)
        while self.status is None:
            move, p = self.smart_action(player, legal_only=legal_only, n=n)
            if not self.is_legal_move(move):
                print("illegal move", player, p, move, file=stderr)
            self.play_move(player, move)
            f is not None and f(self, player, move)
            player = player * -1
        return self.status

    def neural_play(self, model, startingPlayer=None):
        """
        idea:   1# copy the board
                2# make all possible moves on the board
                3# predict the best move based on the values
        """

        player = startingPlayer if startingPlayer is not None else starting_player()  # selects starting player or selects a random player

        while self.status is None:
            board_copy = deepcopy(self.board)

            max_value = 0
            best_move = 0

            for col in range(self.width):
                lowest_row = 0
                for row in range(self.height):
                    if board_copy[row][col] == 0:
                        lowest_row = row
                        break
                board_copy = deepcopy(self.board)
                if player == -1:
                    board_copy[lowest_row][col] = player
                    value = model.predict(board_copy, 2)
                    print(f"predicted value for player {player} is {value}")
                else:
                    board_copy[lowest_row][col] = player
                    value = model.predict(board_copy, 1)
                    print(f"predicted value for player {player} is {value}")
                    """best_move = randrange(7)
                    print(f"player {player} inserts a coin in randomly chosen column {best_move+1}")"""
                    break
                if value > max_value:
                    best_move = col
                    max_value = value

            print(f"beste move for player {player} is column {best_move+1}")
            #print(f"before board:\n {self.board}")
            self.play_move(player, best_move)
            print(f"after board:\n {self.board}")
            player *= -1
            #print(f"switching into player {player}")
        print(f"game ended, results: {self.status}")
        return self.status



# game = Game([[ 0, -1,  1,  1, -1,  0,  0],
#       [ 0,  0,  0,  1, -1,  0,  0],
#       [ 0,  0,  0,  1,  0,  0,  0],
#       [ 0,  0,  0,  0,  0,  0,  0],
#       [ 0,  0,  0,  0,  0,  0,  0],
#       [ 0,  0,  0,  0,  0,  0,  0]])
# print(game.status, game)
# print(game.winning(-1, n=1000))
def create_model():
    model = Model(42, 3, 50, 100)
    data = np.load("c4.npy")
    input = []

    for i in range(len(data)):
        winner = int(data[i][42])
        matrix = []
        for j in range(6):
            row = []
            for k in range(7):
                row.append(int(data[i][j * 6 + k]))
            matrix.append(row)
        input.append((winner, matrix))

    #model.train_model(input)
    #model.save("savedModel/model")
    model.load("savedModel/model")

    return model


if __name__ == "__main__":
    from io import StringIO
    from mpi4py.futures import MPIPoolExecutor, MPICommExecutor


    def play_game(gameid, starting=None, legal_only=False):
        states = []

        def add_state(game, player, move):
            if player is not None and move is not None:
                states.append((player, move))

        game = Game()
        model = create_model()
        # model = None
        # status = game.random_play(starting, legal_only=legal_only, f=add_state)
        # status = game.smart_play(starting, legal_only=legal_only, n=10, f=add_state)
        status = game.neural_play(model)

        io = StringIO()
        for idx, move in enumerate(states):
            print(gameid, idx, move[0], move[1], status, sep=",", file=io)
        return [gameid,status]


    def repeat(x):
        while True:
            yield x


    player1 = 0
    player2 = 0
    draw = 0
    total = 0
    with MPICommExecutor() as pool:
        for result in pool.map(play_game, range(10), unordered=True):
            total += 1
            if result[1] == -1:
                player1 += 1
            elif result[1] == 1:
                player2 += 1
            elif result[1] == 0:
                draw += 1
    print(f"results from the gameq:\ntotal amount of game: {total}\n player 1 wins: {player1}\n player 2 wins: {player2} \n draws: {draw} ")
