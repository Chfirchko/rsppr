import numpy as np
from numpy.random import randn
import random

train_data = {
  'Украины быть не должно. Если мы сохраним Украину такой, как она есть сейчас, допустим, все затихнет, то потом украинские диверсанты на всей территории России будут устраивать теракты': True,
  'Мы должны помочь ополченцам двух республик — Донецкой и Луганской — через пять минут после начала обстрела с их стороны, со стороны украинских вооруженных сил, нанести такой удар по всем вооруженным силам Украины, чтобы там больше ничего не осталось': True,
  'Турцию создал Ленин, Украину создали Ленин, Сталин, Хрущев. Это же мы все сами сделали, это не они. Мы ничего как страна не проиграли, это наша пятая колонна все испортила': True,
  'Подождите, через двадцать лет Украина, действительно, будет анти-Россия. Они будут проклинать русских. Ни слова русского не раздастся. Все будут говорить на украинском. Насильно их заставят стать украинцами. Поэтому время играет роль. Через 25 лет, если я скажу о вхождении украинских областей в состав России и белорусских, там, конечно, будет сопротивление': True,
  'Мы, ЛДПР и я лично, всегда не признавали Украину после большевистского переворота и насильственную украинизацию с 1920 по 1939 год прошлого века, и в 1991 году мы не признали незаконные «Беловежские соглашения» о расчленении страны.': True,
  'сегда с подозрением относились к президентам Кравчуку, Кучме, Ющенко, Януковичу, Порошенко, Зеленскому. Они все обманывали избирателей, к власти приходили за счет поддержки русскоязычных жителей Юго-Востока Украины': True,
  'Я вас уверяю, и над Харьковом, и над Николаевом, и над Житомиром, придет время, взовьются русские флаги. Это будет': True,
  'Россия никогда не будет вводить войска на Украину, но если кто-то попытается втянуть Украину в НАТО, а население будет против, а над ними будут издеваться; если кто-то будет бомбить Луганск, Донецк, мы защитим эти города': True,
  'Не волнуйтесь, вы захлебнетесь от наших туристов': False,
  'Мы за 102 года столько законов перебрали, что-то отменили, что-то приняли — и ничего хорошего в итоге нет. Нужно так принимать законы, чтобы они веками действовали': False,
  'Диктатура — это запор, демократия — понос. Выбирайте, что вам больше нравится': False,
  'От чего коровы с ума сходят? От британской демократии': False,
  'О какой демократии вы говорите, если 65% территории России — вечная мерзлота?!': False,
  'И на следующих, и до самой смерти буду участвовать в выборах. Даже с кладбища буду участвовать, еще и оттуда буду давать вам сигналы, что я там лежу': False,
  'У меня чистые руки, но они будут в крови, если я стану президентом': False,
  'Только под русские песни плачет планета': False,
  'Не надо заставлять детей учить английский. Пускай лучше изучают автомат Калашникова. И тогда скоро весь мир заговорит по-русски': False,
  'Я такой, какой я есть. В этом моя прелесть!': False,
  'Я русский Клинтон. То же образование, те же манеры, тот же возраст': False,
  'Не надо целоваться. Рот – самое поганое место': False,

}

test_data = {
  'Украины быть не должно. Если мы сохраним Украину такой, как она есть сейчас, допустим, все затихнет, то потом украинские диверсанты на всей территории России будут устраивать теракты': True,
  'Мы должны помочь ополченцам двух республик — Донецкой и Луганской — через пять минут после начала обстрела с их стороны, со стороны украинских вооруженных сил, нанести такой удар по всем вооруженным силам Украины, чтобы там больше ничего не осталось': True,
  'Я русский Клинтон. То же образование, те же манеры, тот же возраст': False,
  'Не надо целоваться. Рот – самое поганое место': False,
}


class RNN:
    # Классическая рекуррентная нейронная сеть

    def __init__(self, input_size, output_size, hidden_size=64):
        # Вес
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000

        # Смещения
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        '''
        Выполнение фазы прямого распространения нейронной сети с
        использованием введенных данных.
        Возврат итоговой выдачи и скрытого состояния.
        - Входные данные в массиве однозначного вектора с формой (input_size, 1).
        '''
        h = np.zeros((self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = {0: h}

        # Выполнение каждого шага нейронной сети RNN
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h

        # Подсчет вывода
        y = self.Why @ h + self.by

        return y, h

    def backprop(self, d_y, learn_rate=2e-2):
        '''
        Выполнение фазы обратного распространения RNN.
        - d_y (dL/dy) имеет форму (output_size, 1).
        - learn_rate является вещественным числом float.
        '''
        n = len(self.last_inputs)

        # Вычисление dL/dWhy и dL/dby.
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        # Инициализация dL/dWhh, dL/dWxh, и dL/dbh к нулю.
        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        # Вычисление dL/dh для последнего h.
        d_h = self.Why.T @ d_y

        # Обратное распространение во времени.
        for t in reversed(range(n)):
            # Среднее значение: dL/dh * (1 - h^2)
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)

            # dL/db = dL/dh * (1 - h^2)
            d_bh += temp

            # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
            d_Whh += temp @ self.last_hs[t].T

            # dL/dWxh = dL/dh * (1 - h^2) * x
            d_Wxh += temp @ self.last_inputs[t].T

            # Далее dL/dh = dL/dh * (1 - h^2) * Whh
            d_h = self.Whh @ temp

        # Отсекаем, чтобы предотвратить разрыв градиентов.
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        # Обновляем вес и смещение с использованием градиентного спуска.
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by


vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)

print('%d unique words found' % vocab_size)  # найдено 18 уникальных слов

# Назначить индекс каждому слову
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}



def createInputs(text):
    '''
    Возвращает массив унитарных векторов
    которые представляют слова в введенной строке текста
    - текст является строкой string
    - унитарный вектор имеет форму (vocab_size, 1)
    '''

    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)

    return inputs


def softmax(xs):
    # Применение функции Softmax для входного массива
    return np.exp(xs) / sum(np.exp(xs))


# Инициализация нашей рекуррентной нейронной сети RNN
rnn = RNN(vocab_size, 2)



for x, y in train_data.items():
    inputs = createInputs(x)
    target = int(y)

    # Прямое распространение
    out, _ = rnn.forward(inputs)
    probs = softmax(out)

    # Создание dL/dy
    d_L_d_y = probs
    d_L_d_y[target] -= 1

    # Обратное распространение
    rnn.backprop(d_L_d_y)


def processData(data, backprop=True):
    '''
    Возврат потери рекуррентной нейронной сети и точности для данных
    - данные представлены как словарь, что отображает текст как True или False.
    - backprop определяет, нужно ли использовать обратное распределение
    '''
    items = list(data.items())
    random.shuffle(items)

    loss = 0
    num_correct = 0

    for x, y in items:
        inputs = createInputs(x)
        target = int(y)

        # Прямое распределение
        out, _ = rnn.forward(inputs)
        probs = softmax(out)

        # Вычисление потери / точности
        loss -= np.log(probs[target])
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            # Создание dL/dy
            d_L_d_y = probs
            d_L_d_y[target] -= 1

            # Обратное распределение
            rnn.backprop(d_L_d_y)

    return loss / len(data), num_correct / len(data)



for epoch in range(100):
    train_loss, train_acc = processData(train_data)

    if epoch % 10 == 9:
        print('--- Epoch %d' % (epoch + 1))
        print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

        test_loss, test_acc = processData(test_data, backprop=False)
        print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))


inputs = createInputs('Насильно')
out, h = rnn.forward(inputs)
probs = softmax(out)
print(probs)  # [[0.50000095], [0.49999905]]