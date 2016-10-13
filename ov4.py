import random
import operator


class Perceptron(object):
    MAX_ITERATIONS = 1000

    def __init__(self, num_weights):
        self.threshold = 2
        self.learning_rate = 0.1
        self.num_weights = num_weights
        self.weights = self.get_weights_array(num_weights)

    @staticmethod
    def get_weights_array(number_of_weights):
        i = 0
        array = []
        while i < number_of_weights:
            i += 1
            array.append(random.uniform(-0.5, 0.5))
        return array

    def learning_algorithm(self, data_set):
        iteration = 0
        while True:
            iteration += 1
            print('Iteration no.', iteration)
            print('Current weights:', self.weights)
            errors = 0
            for row in data_set:
                result = self.activation(row[0])
                error = row[1] - result
                print('Input:', row[0], 'Desired:', row[1], 'Actual:', result)

                if error != 0:
                    print('Error detected.')
                    self.train_weights(error, row[0])
                    print('Changed weights to:', self.weights)
                    errors += 1

            if not errors or iteration > self.MAX_ITERATIONS:
                break
        print('Done after', iteration, 'iterations')

    def train_weights(self, error, list_of_numbers):
        i = 0
        for number in list_of_numbers:
            self.weights[i] += self.learning_rate * error * number
            i += 1

    def activation(self, variables):
        # Maps first var in variables to first var in weights, then second to second and so on.
        # The sets are then multiplied together.
        # The products are then summed together.
        # (dot-product)
        result = sum(map(operator.mul, variables, self.weights))
        result -= self.threshold
        if result < 0:
            return 0
        elif result >= 0:
            return 1
        else:
            print('What just happened?')
            return


# Datasets for AND and OR
AND = [
    [[0, 0], 0],
    [[1, 0], 0],
    [[0, 1], 0],
    [[1, 1], 1]
]
OR = [
    [[0, 0], 0],
    [[1, 0], 1],
    [[0, 1], 1],
    [[1, 1], 1]
]


a = Perceptron(2)
a.learning_algorithm(AND)