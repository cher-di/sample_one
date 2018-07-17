# Используем нейронную сеть
import Experiments
import numpy
import time

# Количество входных, скрытых и выходных узлов
input_nodes = 784
hidden_nodes = 90
output_nodes = 10

# Коэффициент обучения равен 0.1
learning_rate = 0.1

# Создать экземпляр нейронной сети
n = Experiments.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Загрузить в список тестовый набор данных CSV - файла набора MNIST
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
    
# Тренировка нейронной сети

# время обучения
learning_time = 0

# Переменная epochs указывает, сколько раз тренировочный
# набор данных используется для тренировки сети
epochs = 5

for e in range(epochs):
    # время начала очередной эпохи тренировки
    before = time.time()
    
    # Перебрать все записи в тренировочном наборе данных
    for record in training_data_list:
        # Получить список значений, используя символы запятой (',')
        # в качестве разделителей
        all_values = record.split(',')
        # Масштабировать и сместить входных значения
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # Создать целевые выходные значения (все равны 0.01 за исключением
        # желаемого маркерного значения, равного 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] - целевое маркерное значение для данной записи
        targets[int(all_values[0])] = 0.99

        # тренируем сеть
        n.train(inputs, targets)
        
        pass
    # время конца очередной эпохи тренировки
    after = time.time()

    print('Эпоха', e + 1)
    print('Время обучения:', after - before, '\n')
    learning_time += after - before

    # Тестирование нейронной сети

    # Устанваливаем время перед опросом сети
    time_before_query = time.time()

    # Загрузить в список тестовый набор данных CSV - файла набора MNIST
    test_data_file = open("mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # Журнал оценок работы сети, первоначально пустой
    scorecard = []

    # Перебрать все записи в тестовом наборе данных
    for record in test_data_list:
        # Получить список значений из записи, используя символы
        # запятой (',') в качестве разделителей
        all_values = record.split(',')
        # Правильный ответ - первое значение
        correct_label = int(all_values[0])
        # print(correct_label, "истинный маркер")
        # Масштабировать и сместить входные значения
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # Опрос сети
        outputs = n.query(inputs)
        # Индекс наибольшего значения является маркерным значением
        label = numpy.argmax(outputs)
        # print(label, "ответ сети")
        # Присоединить оценку ответа сети к концу списка
        if (label == correct_label):
            # В случае правильного ответа сети присоединить
            # к списку значение 1
            scorecard.append(1)
        else:
            # В случае неправильного ответа сети присоединить
            # к списку значение 0
            scorecard.append(0)
            pass
        pass

    query_time = time.time() - time_before_query
    print("Сеть опрошена")
    print("Время опросов:", query_time)

    # Рассчитать показатель эфеективности в виде
    # доли правильных ответов
    scorecard_array = numpy.asarray(scorecard)
    print("Эффективность =", scorecard_array.sum() / scorecard_array.size, '\n\n')
    
    pass

print("Обучение завершено")
print("Время обучения:", learning_time)
input()
