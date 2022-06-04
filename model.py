import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from convert_data import convert

"""
RED = -1    predict 2
YELLOW = 1  predict 0
DRAW = 2

output neurons :     (0) draw
                     (1) player 1 wins
                     (2) player -1 wins
"""


class Model:

    def __init__(self, numberOfInputs, numberOfOutputs, batchSize, epochs):
        self.numberOfInputs = numberOfInputs
        self.numberOfOutputs = numberOfOutputs
        self.batchSize = batchSize
        self.epochs = epochs
        self.model = Sequential()
        self.model.add(Dense(42, activation='relu', input_shape=(numberOfInputs,)))
        self.model.add(Dense(42, activation='relu'))
        self.model.add(Dense(numberOfOutputs, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])

    def get_trained_model(self, dataset="c4-10k.csv", load=True, filepath="savedModel/model", save=True):
        if load:
            return self.load(filepath)
        else:
            data = np.load(dataset)
            train_input = []
            for i in range(len(data)):
                winner = int(data[i][42])
                matrix = []
                for j in range(6):
                    row = []
                    for k in range(7):
                        row.append(int(data[i][j * 6 + k]))
                    matrix.append(row)
                train_input.append((winner, matrix))
            self.train_model(train_input)
            if save:
                self.save(filepath)
        return self.model

    def train_model(self, dataset):
        converted_data = convert(dataset)
        input = []
        output = []
        for data in converted_data:
            input.append(data[1])
            output.append(data[0])

        X = np.array(input).reshape((-1, self.numberOfInputs))
        y = to_categorical(output, num_classes=3)
        limit = int(0.8 * len(X))
        X_train = X[:limit]
        X_test = X[limit:]
        y_train = y[:limit]
        y_test = y[limit:]
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=self.epochs,
                       batch_size=self.batchSize)

    def predict(self, data, index):
        return self.model.predict(np.array(data).reshape(-1, self.numberOfInputs))[0][index]

    def save(self, filepath):
        self.model.save_weights(filepath)

    def load(self, filepath):
        return self.model.load_weights(filepath)
