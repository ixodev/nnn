import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, lr, epochs, roundx):
        self.weights = np.zeros(input_size + 1)
        self.input_size = input_size
        self.lr = lr
        self.epochs = epochs
        self.roundx = roundx

    def relu(self, x):
        return np.maximum(0, x)
    
    def predict(self, input):
        prediction = self.relu(np.dot(np.insert(input, 0, 1), self.weights))
        return int(round(prediction)) if prediction < self.roundx else int(prediction)
    
    def train(self, train_data, labels):
        for i in range(self.epochs):
            for data, label in zip(train_data, labels):
                result = self.predict(data)
                error = label - result

                self.weights[0] += self.lr * error
                self.weights[1:] += self.lr * error * data

    def save(self, file_name):
        file = open(file_name, "w")
        
        for wi in range(len(self.weights)):
            file.write(f"{wi}")
            file.write(',' if wi != len(self.weights) - 1 else '')
        
        file.close()

    def load(self, file_name):
        file = open(file_name, "r")

        self.weights = np.array([int(weight) for weight in int(file.read().split(','))])

        file.close()


if __name__ == "__main__":
    # Logic OR Data
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 1, 1, 1])

    model = NeuralNetwork(2, 0.2, 10000, 10e-17)
    model.train(data, labels)
    model.save("mymodel.model")

    for d in data:
        print(f"{d}  ->  {model.predict(d)}")