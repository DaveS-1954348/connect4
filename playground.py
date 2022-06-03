import model
from connectfour import Game
from io import StringIO
from mpi4py.futures import MPIPoolExecutor, MPICommExecutor
import convert as c
import numpy as np
import pandas as pd


def play_game(gameid, starting=None, legal_only=False):
    states = []

    def add_state(game, player, move):
        if player is not None and move is not None:
            states.append((player, move))

    game = Game()
    status = game.random_play(starting, legal_only=legal_only, f=add_state)
    # status = game.smart_play(starting, legal_only=legal_only, n=10, f=add_state)

    io = StringIO()
    for idx, move in enumerate(states):
        print(gameid, idx, move[0], move[1], status, sep=",", file=io)
    return io.getvalue()


def repeat(x):
    while True:
        yield x


player1_counter = 0
player2_counter = 0
draws = 0
with MPICommExecutor() as pool:
    for result in pool.map(play_game, range(1), unordered=True):
        winner = result.split("\n")[0].split(',')[4]
        if winner == "0":
            draws += 1
        elif winner == "1":
            player2_counter += 1
        elif winner == "-1":
            player1_counter += 1

print(f"\nfinal results\n----------\nplayer-1 wins: {player1_counter}\nplayer1 wins: {player2_counter}\ndraws: {draws}")