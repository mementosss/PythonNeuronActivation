import numpy as np


# Функция активации (бинарная функция)
def activation(x):
    return 1 if x >= 0 else -1


def hebbian_learning(inputs, outputs, epochs=10, initial_bias=0.1, learning_rate=0.01):
    # Инициализация весов и порога случайными значениями, близкими к нулю
    weights = np.random.uniform(-0.1, 0.1, inputs.shape[1])  # 3 веса для 3 переменных (x1, x2, смещение)
    bias = initial_bias  # Используем значение смещения, переданное в функцию

    print(f"Начальные веса: {weights}, начальный порог: {bias}")

    # Обучение
    for epoch in range(epochs):
        total_error = 0  # Счетчик ошибок на эпохе
        print(f"\nЭпоха {epoch + 1}:")

        for x, target in zip(inputs, outputs):
            # Сумма взвешенных входов
            net_input = np.dot(x, weights) + bias  # Включаем смещение в вычисления
            # Применяем активационную функцию
            y_pred = activation(net_input)

            # Ошибка
            error = target - y_pred
            total_error += abs(error)

            if error != 0:  # Если ошибка есть, обновляем веса и порог
                weights += learning_rate * error * x  # Обновление весов с коэффициентом обучения
                bias += -learning_rate * error  # Обновление порога (по правилу Хебба)

            # Выводим информацию о текущем шаге
            print(f"  Вход: {x}, Прогноз: {y_pred}, Ошибка: {error}, Обновленные веса: {weights}, Обновленный порог: {bias}")

        # Выводим ошибки на каждой эпохе
        print(f"  Ошибки на эпохе {epoch + 1}: {total_error}")

        # Если ошибка равна 0, завершить обучение
        if total_error == 0:
            print(f"\nОбучение завершено на эпохе {epoch + 1}.")
            break

    return weights, bias

# Основная программа
if __name__ == "__main__":
    # Параметры для настройки
    learning_rate = 0.01  # Коэффициент обучения
    initial_bias = 0.9    # Начальный порог

    # Входные данные для OR (с учетом смещения)
    inputs = np.array([
        [-1, -1, 1],  # x1 = -1, x2 = -1, смещение = 1
        [-1,  1, 1],  # x1 = -1, x2 =  1, смещение = 1
        [ 1, -1, 1],  # x1 =  1, x2 = -1, смещение = 1
        [ 1,  1, 1]   # x1 =  1, x2 =  1, смещение = 1
    ])

    # Ожидаемые выходные значения для OR
    outputs = np.array([-1, 1, 1, 1])  # OR: -1, 1, 1, 1

    # Обучение
    weights, bias = hebbian_learning(inputs, outputs, epochs=100, initial_bias=initial_bias, learning_rate=learning_rate)

    print(f"\nОбученные веса: {weights}")
    print(f"Обученный порог: {bias}")

    # Проверим результат
    print("\nПроверка обучения:")
    for x, target in zip(inputs, outputs):
        net_input = np.dot(x, weights) + bias
        y_pred = activation(net_input)
        print(f"Вход: {x}, Прогноз: {y_pred}, Ожидаемый результат: {target}")
