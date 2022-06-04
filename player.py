
class Player:

    def __init__(self, strategy:str, value:int):
        """
        :param strategy: can only be 'neural' or 'random'
        :param value: the player id (e.g., p1 = -1 and p2 = 1)
        """
        self.strategy = strategy
        self.value = value

    def set_strategy(self, strategy:str):
        self.strategy = strategy
