from controller import Controller
from player import Player
from model import Model
from connectfour import Game

if __name__ == "__main__":
    connect4_model = Model(42, 3, 50, 100)
    trained_model = connect4_model.get_trained_model()

    # 1. random player vs random player
    print("Random player (-1) vs Random player (1)")
    p1 = Player("random", -1)
    p2 = Player("random", 1)
    controller = Controller(p1, p2, connect4_model)
    controller.play_games(100)

    # 2. neural player vs random player
    print("Neural player (-1) vs Random player (1)")
    p1.set_strategy("neural")
    controller.set_players(p1,p2)
    controller.play_games(100)