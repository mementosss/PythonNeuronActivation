import numpy as np


class HebbNeuron:
    def __init__(self, input_dim):
        """
        Инициализация нейрона:
        - Весовые коэффициенты (weights) случайны, близки к нулю.
        - Смещение (bias) равно нулю.
        """
        self.weights = np.random.uniform(-0.1, 0.1, input_dim)  # Малые случайные веса
        self.bias = 0  # Порог инициализируется нулем

    def activation(self, x):
        """
        Функция активации:
        Возвращает -1 или 1 на основе знака входного значения.
        """
        return 1 if x > 0 else -1

    def predict(self, inputs):
        """
        Предсказание выхода сети:
        - Скалярное произведение весов и входа + смещение.
        """
        net_input = np.dot(inputs, self.weights) - self.bias
        return self.activation(net_input)

    def train(self, X, Y, epochs=1):
        """
        Обучение по правилу Хебба:
        - Обновляет веса и смещение, если предсказание не совпадает с эталоном.
        """
        for epoch in range(epochs):
            print(f"Эпоха {epoch + 1}:")
            for i in range(len(X)):
                x, y_true = X[i], Y[i]
                y_pred = self.predict(x)
                print(f"  Вход: {x}, Ожидаемый: {y_true}, Предсказанный: {y_pred}")

                if y_pred != y_true:
                    # Обновление весов и смещения по правилу Хебба
                    self.weights += x * y_true
                    self.bias -= y_true
                    print(f"    Обновление весов: {self.weights}, Смещение: {self.bias}")
            print()

    def evaluate(self, X):
        """
        Оценивает входы после обучения.
        """
        results = []
        for x in X:
            results.append(self.predict(x))
        return results


# Пример использования алгоритма Хебба
if __name__ == "__main__":
    # Обучающая выборка: входы и эталонные выходы
    X = np.array([
        [-1, -1],  # Вектор 1
        [-1, 1],  # Вектор 2
        [1, -1],  # Вектор 3
        [1, 1],  # Вектор 4
    ])
    Y = np.array([-1, -1, -1, 1])  # Эталонные выходы

    # Инициализация нейрона
    neuron = HebbNeuron(input_dim=2)

    # Вывод начальных весов
    print(f"Начальные веса: {neuron.weights}, Смещение: {neuron.bias}\n")

    # Обучение
    neuron.train(X, Y, epochs=2)

    # Проверка сети
    outputs = neuron.evaluate(X)
    print("Результаты после обучения:")
    for i, x in enumerate(X):
        print(f"  Вход: {x}, Выход: {outputs[i]}")
