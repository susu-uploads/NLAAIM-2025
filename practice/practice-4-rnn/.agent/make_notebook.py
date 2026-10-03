from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PRACTICE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_1_PATH = PRACTICE_DIR / "СП-М-О-АЕЯМИИ-2025-Практика-4-1-БабушкинМВ.ipynb"
NOTEBOOK_2_PATH = PRACTICE_DIR / "СП-М-О-АЕЯМИИ-2025-Практика-4-2-БабушкинМВ.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # Практическая работа 4. RNN

        IMDB и keras RNN.
        """
    ),
    md(
        """
        !!!!Для корректной работы кода в этом документе, следует выполнить задания из документа “IMDB и keras”!!!!

        ## Рекуррентные сети

        Главной характеристикой всех нейронных сетей, с которыми мы познакомились к данному моменту, таких как полносвязные и сверточные нейронные сети, является отсутствие памяти. Каждый вход обрабатывается ими независимо, без сохранения состояния между ними. Чтобы с помощью таких сетей обработать последовательность или временной ряд данных, необходимо передать в сеть всю последовательность целиком, преобразовав ее в единый пакет. Именно так мы поступили в предыдущем примере: мы преобразовали все отзывы из IMDB в один большой вектор и обработали его целиком. Такие сети называют сетями прямого распространения (feedforward networks).

        С другой стороны, читая предложение, мы осмысливаем его слово за словом, быстро перескакивая глазами с одного на другое и запоминая предыдущие; это позволяет нам постепенно вникать в смысл, передаваемый предложением. Биологический интеллект воспринимает информацию последовательно, сохраняя внутреннюю модель обрабатываемого, основываясь на предыдущей информации и постоянно дополняя эту модель по мере поступления новой информации.

        Рекуррентная нейронная сеть (Recurrent Neural Network, RNN) использует тот же принцип, хотя и в чрезвычайно упрощенном виде: она обрабатывает последовательность, перебирая ее элементы и сохраняя состояние, полученное при обработке предыдущих элементов. Фактически RNN — это разновидность нейронной сети, имеющей внутренний цикл. Сеть RNN сбрасывает состояние между обработкой двух разных, независимых последовательностей (таких, как два разных отзыва из IMDB), поэтому одна последовательность все еще интерпретируется как единый блок данных: единственный входной пакет. Однако теперь блок данных обрабатывается не за один шаг; сеть выполняет внутренний цикл, перебирая последовательность элементов.

        <div align="center">
          <img src="rnn_loop.png" alt="Рекуррентная сеть с циклом" width="520"/>
          <br/>
          <em>Рекуррентная сеть с циклом</em>
        </div>

        """
    ),
    md(
        """
        Чтобы пояснить понятия «цикл» и «состояние», реализуем средствами Numpy простую сеть RNN с прямой передачей. Она будет принимать на входе последовательность векторов в виде двумерного тензора с формой (временные_интервалы, входные_признаки), перебирать временные интервалы и, учитывая текущее состояние и входные признаки (с формой (входные_признаки,)) в момент t, конструировать выходной результат, соответствующий моменту t. Этот результат затем будет сохраняться во внутреннем состоянии как подготовка к следующей итерации. Для первого временного интервала предыдущий выходной результат не определен; в этот момент сеть не имеет текущего состояния. Поэтому текущее состояние первоначально инициализируется вектором с нулевыми значениями элементов, который называют начальным состоянием сети.

        Чтобы сделать эти понятия абсолютно однозначными, напишем упрощенную реализацию сети RNN на основе Numpy.
        """
    ),
    code(
        """
        from __future__ import annotations

        import os
        import random
        from pathlib import Path

        PRACTICE_DIR = Path.cwd()

        CACHE_DIR = PRACTICE_DIR / ".cache"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "matplotlib")
        os.environ["IPYTHONDIR"] = str(CACHE_DIR / "ipython")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        DATA_DIR = PRACTICE_DIR / ".data"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["KERAS_HOME"] = str(DATA_DIR / "keras_home")

        import matplotlib.pyplot as plt
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.datasets import imdb
        from tensorflow.keras.layers import (
            Bidirectional,
            Dense,
            Dropout,
            Embedding,
            GRU,
            Input,
            LSTM,
            SimpleRNN,
        )
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        RANDOM_STATE = 42
        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        tf.random.set_seed(RANDOM_STATE)

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["figure.figsize"] = (10, 4)
        np.set_printoptions(edgeitems=5, linewidth=120)

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices([], "GPU")
        """
    ),
    code(
        """
        timesteps = 100 #Число временных интервалов во входной последовательности
        input_features = 32 #Размерность пространства входных признаков
        output_features = 64 #Размерность пространства выходных признаков

        inputs = np.random.random((timesteps, input_features)) #Входные данные: случайный шум для простоты примера
        state_t = np.zeros((output_features,)) #Начальное состояние: вектор с нулевыми значениями элементов

        W = np.random.random((output_features, input_features)) #Создание матриц со случайными весами
        U = np.random.random((output_features, output_features)) #Создание матриц со случайными весами
        b = np.random.random((output_features,)) #Создание матриц со случайными весами

        successive_outputs = []
        for input_t in inputs: #input_t — вектор с формой (входные_признаки,)
            output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b) #Объединение входных данных с текущим состоянием (выходными данными на предыдущем шаге)
            successive_outputs.append(output_t) #Сохранение выходных данных в список
            state_t = output_t #Обновление текущего состояния сети как подготовка к обработке следующего временного интервала

        final_output_sequence = np.stack(successive_outputs, axis=0) #Окончательный результат — двумерный тензор с формой (временные_интервалы, выходные_признаки)
        final_output_sequence.shape
        """
    ),
    md(
        """
        Довольно просто: как можно заметить, RNN — это цикл for, который повторно использует величины, вычисленные в предыдущей итерации, и не более того. Конечно, вы могли бы сконструировать множество разных сетей RNN, соответствующих этому определению, и этот пример — одна из простейших реализаций RNN. Рекуррентные сети характеризуются функцией, реализующей один шаг, такой как следующая, использованная в данном примере:

        `output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)`

        <div align="center">
          <img src="rnn_unrolled.png" alt="Простая рекуррентную сеть, развернутая по времени" width="640"/>
          <br/>
          <em>Простая рекуррентную сеть, развернутая по времени</em>
        </div>

        """
    ),
    md(
        """
        ПРИМЕЧАНИЕ

        В этом примере конечный результат имеет вид двумерного тензора с формой (временные_интервалы, выходные_признаки), где каждый временной интервал — это результат цикла в момент времени t. Каждому временному интервалу t в выходном тензоре соответствует информация о временных интервалах от 0 до t во входной последовательности — обо всем прошлом. Поэтому во многих случаях нет необходимости иметь всю последовательность результатов; достаточно получить последний результат (значение output_t по окончании цикла), так как он уже содержит информацию обо всей последовательности.

        ## Рекуррентный слой в Keras

        Процессу, который мы только что реализовали с применением Numpy, соответствует фактический слой в Keras — слой SimpleRNN:
        """
    ),
    code(
        """
        SimpleRNN
        """
    ),
    md(
        """
        С одним незначительным отличием: SimpleRNN обрабатывает пакеты последовательностей, как и все другие слои в Keras, а не единственную последовательность, как наш предыдущий пример. Это означает, что он принимает входные данные с формой (размер_пакета, временные_интервалы, входные_признаки), а не (временные_интервалы, входные_признаки).

        Подобно всем рекуррентным слоям в Keras, SimpleRNN может действовать в двух разных режимах: возвращать полные последовательности результатов для всех временных интервалов (трехмерный тензор с формой (размер_пакета, временные_интервалы, выходные_признаки)) или только последний результат для каждой входной последовательности (двумерный тензор с формой (размер_пакета, входные_признаки)). Выбор режима управляется аргументом return_sequences конструктора. Рассмотрим пример, в котором используется слой SimpleRNN и возвращается результат только для последнего временного интервала:
        """
    ),
    code(
        """
        model = Sequential(
            [
                Input(shape=(None,)),
                Embedding(10000, 32),
                SimpleRNN(32),
            ]
        )
        model.summary()
        """
    ),
    md(
        """
        Следующий пример возвращает полную последовательность состояний:
        """
    ),
    code(
        """
        model = Sequential(
            [
                Input(shape=(None,)),
                Embedding(10000, 32),
                SimpleRNN(32, return_sequences=True),
            ]
        )
        model.summary()
        """
    ),
    md(
        """
        Иногда полезно наложить друг на друга несколько рекуррентных слоев, чтобы увеличить репрезентативность сети. В таких ситуациях все промежуточные слои должны возвращать полные последовательности результатов:
        """
    ),
    code(
        """
        model = Sequential(
            [
                Input(shape=(None,)),
                Embedding(10000, 32),
                SimpleRNN(32, return_sequences=True),
                SimpleRNN(32, return_sequences=True),
                SimpleRNN(32, return_sequences=True),
                SimpleRNN(32), #Последний слой возвращает только последний результат
            ]
        )
        model.summary()
        """
    ),
    md(
        """
        Теперь попробуем применить такую же модель для решения задачи классификации отзывов к фильмам из набора данных IMDB. Сначала подготовим данные.
        """
    ),
    code(
        """
        max_features = 10000
        maxlen = 500
        batch_size = 32

        print("Loading data...")
        (input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
        print(len(input_train), "train sequences")
        print(len(input_test), "test sequences")

        print("Pad sequences (samples x time)")
        input_train = pad_sequences(input_train, maxlen=maxlen)
        input_test = pad_sequences(input_test, maxlen=maxlen)
        print("input_train shape:", input_train.shape)
        print("input_test shape:", input_test.shape)
        """
    ),
    md(
        """
        Обучим простую рекуррентную сеть, состоящую из слоев Embedding и SimpleRNN.
        """
    ),
    code(
        """
        model = Sequential(
            [
                Input(shape=(maxlen,)),
                Embedding(max_features, 32),
                SimpleRNN(32),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
        model.summary()
        """
    ),
    code(
        """
        history = model.fit(
            input_train,
            y_train,
            epochs=10,
            batch_size=128,
            validation_split=0.2,
        )
        """
    ),
    md(
        """
        Теперь выведем графики изменения величины потерь и точности модели на этапах обучения и проверки.
        """
    ),
    code(
        """
        def plot_training_history(history, title_prefix: str) -> None:
            \"\"\"Строит графики точности и потерь по объекту History из Keras.\"\"\"
            history_dict = history.history
            acc_key = "accuracy" if "accuracy" in history_dict else "acc"
            val_acc_key = "val_accuracy" if "val_accuracy" in history_dict else "val_acc"

            acc = history_dict[acc_key]
            val_acc = history_dict[val_acc_key]
            loss = history_dict["loss"]
            val_loss = history_dict["val_loss"]
            epochs = range(1, len(acc) + 1)

            plt.figure()
            plt.plot(epochs, acc, "bo", label="Training acc")
            plt.plot(epochs, val_acc, "b", label="Validation acc")
            plt.title(f"{title_prefix}: training and validation accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()

            plt.figure()
            plt.plot(epochs, loss, "bo", label="Training loss")
            plt.plot(epochs, val_loss, "b", label="Validation loss")
            plt.title(f"{title_prefix}: training and validation loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        """
    ),
    code(
        """
        plot_training_history(history, "SimpleRNN")
        """
    ),
    md(
        """
        <p align="center">
          <em>Потери на этапах обучения и проверки для модели анализа IMDB со слоем SimpleRNN</em><br/>
          <em>Точность на этапах обучения и проверки для модели анализа IMDB со слоем SimpleRNN</em>
        </p>

        К сожалению, эта маленькая рекуррентная сеть показала на этапе проверки точность лишь 85 %. Отчасти проблема объясняется тем, что анализу подвергаются только 500 первых слов в каждом отзыве, а не вся последовательность, следовательно, сеть RNN получает на входе меньше информации, чем модель, с которой мы ее сравниваем. Другая причина в том, что слой SimpleRNN плохо подходит для обработки длинных последовательностей, таких как текст.

        Другие типы рекуррентных слоев позволяют добиться более высоких результатов. Рассмотрим несколько таких более продвинутых слоев.

        ## Слои LSTM и GRU

        имеются также слои LSTM и GRU. В практических решениях они используются чаще, потому что SimpleRNN слишком прост для реальных задач. SimpleRNN страдает одной существенной проблемой: теоретически в каждый момент времени t он должен хранить информацию о входных данных за многочисленные предыдущие интервалы времени, но на практике такие протяженные зависимости не поддаются обучению. Это связано с проблемой затухания градиента, напоминающего эффект, который наблюдается в нерекуррентных сетях (сетях прямого распространения) с большим количеством слоев: по мере увеличения количества слоев сеть в конечном итоге становится необучаемой. Теоретическое обоснование этого эффекта было дано Хохрейтером (Hochreiter), Шмидхубером (Schmidhuber) и Бенгио (Bengio) в начале 1990-х (см. Yoshua Bengio, Patrice Simard, and Paolo Frasconi, «Learning Long-Term Dependencies with Gradient Descent Is Difficult», IEEE Transactions on Neural Networks 5, no. 2 (1994)). Слои LSTM и GRU создавались специально для решения этой проблемы.

        Рассмотрим слой LSTM. Лежащий в его основе алгоритм долгой краткосрочной памяти (Long Short-Term Memory, LSTM) был разработан Хохрейтером и Шмидхубером в 1997 (см. Sepp Hochreiter and Jürgen Schmidhuber, «Long Short-Term Memory», Neural Computation 9, no. 8 (1997)); он стал кульминацией их исследований проблемы затухания градиента.

        Этот слой является вариантом слоя SimpleRNN, уже знакомого вам; он добавляет поддержку переноса информации через многие интервалы времени. Вообразите конвейерную ленту, движущуюся параллельно обрабатываемой последовательности. Информация из последовательности может в любой момент перекладываться на конвейерную ленту, переноситься к более поздним интервалам времени и сниматься с ленты, если она необходима. В этом заключается суть работы слоя LSTM: он сохраняет информацию для последующего использования, тем самым предотвращая постепенное затухание старых сигналов во время обработки.

        Чтобы разобраться более детально, начнем с ячейки SimpleRNN. Так как у нас имеется большое количество весовых матриц, выходные матрицы W и U в ячейке мы обозначим индексом o (Wo и Uo) — от англ. output (выходной, на выходе).

        <div align="center">
          <img src="lstm_carry_flow.png" alt="Переход от SimpleRNN к LSTM: добавление несущего потока" width="640"/>
          <br/>
          <em>Переход от SimpleRNN к LSTM: добавление несущего потока</em>
        </div>

        """
    ),
    md(
        """
        А теперь о деталях способа вычисления следующего значения в несущем потоке данных: он основывается на трех разных преобразованиях, все три имеют форму ячейки SimpleRNN:

        `y = activation(dot(state_t, U) + dot(input_t, W) + b)`

        Однако эти преобразования имеют свои весовые матрицы, которые мы обозначим индексами i, f и k. Вот что мы имеем (это может показаться необоснованным, но наберитесь терпения, я все объясню позже).

        Реализация архитектуры LSTM в псевдокоде:

        ```
        output_t = activation(dot(state_t, Uo) + dot(input_t, Wo) + dot(C_t, Vo) + bo)
        i_t = activation(dot(state_t, Ui) + dot(input_t, Wi) + bi)
        f_t = activation(dot(state_t, Uf) + dot(input_t, Wf) + bf)
        k_t = activation(dot(state_t, Uk) + dot(input_t, Wk) + bk)
        ```

        Получим новое переносимое состояние (далее обозначается как c_t), объединив i_t, f_t и k_t.

        `c_t+1 = i_t * k_t + c_t * f_t`

        <div align="center">
          <img src="lstm_anatomy.png" alt="Анатомия LSTM" width="720"/>
          <br/>
          <em>Анатомия LSTM</em>
        </div>

        """
    ),
    md(
        """
        Добавим это в общую картину, как показано на рис. 6. Вот и все. Совсем несложно, просто немного замысловато.

        Если хотите удариться в философию, подумайте о том, что делает каждая из этих операций. Например, можно сказать, что умножение c_t на f_t — это способ преднамеренного забывания ненужной информацию в несущем потоке данных. А умножение i_t на k_t представляет собой информацию о настоящем, добавляя новую информацию в несущий поток. Но в конечном счете эти интерпретации не имеют большого значения, потому что фактическое действие операций определяется содержимым параметризующих их весов, а веса вычисляются непрерывно и заново в каждом цикле обучения, что делает невозможным приписать какую-то конкретную цель той или иной операции. Спецификация ячейки RNN (как только что было описано) определяет ваше пространство гипотез — пространство, в котором в процессе обучения вы будете искать оптимальные настройки модели, однако она не определяет, что именно делает ячейка, — это зависит от весов в ячейке. Одна и та же ячейка с разными весами может действовать совершенно иначе. Поэтому набор операций, образующих ячейку RNN, лучше рассматривать как набор ограничений в вашем поиске, а не как дизайн в инженерном смысле.

        С точки зрения исследователя, выбор таких ограничений — особенностей реализации ячеек RNN — лучше переложить на алгоритмы оптимизации (такие, как обобщенные алгоритмы обучения с подкреплением) и избавить людей-инженеров от него. В будущем именно так мы и будем строить сети. Подводя итог, можно сказать, что от вас не требуется понимания особенности архитектуры ячейки LSTM; это не ваша задача как человека. Просто помните назначение ячейки LSTM: позволить прошлой информации повторно внедриться в процесс обучения и оказать сопротивление проблеме затухания градиента.

        Cоздадим модель со слоем LSTM и обучим ее на данных IMDB. Новая сеть похожа на предыдущую, со слоем SimpleRNN. Мы указали только размерность результата слоя LSTM, оставив другие аргументы (а их довольно много) со значениями по умолчанию. Значения по умолчанию подобраны в Keras очень хорошо и пригодны для большинства ситуаций, что избавляет нас от необходимости тратить время на настройку параметров вручную.
        """
    ),
    code(
        """
        model = Sequential(
            [
                Input(shape=(maxlen,)),
                Embedding(max_features, 32),
                LSTM(32),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()
        """
    ),
    code(
        """
        history = model.fit(
            input_train,
            y_train,
            epochs=10,
            batch_size=128,
            validation_split=0.2,
        )
        """
    ),
    code(
        """
        plot_training_history(history, "LSTM")
        """
    ),
    md(
        """
        <p align="center">
          <em>Потери на этапах обучения и проверки для модели анализа IMDB со слоем LSTM</em><br/>
          <em>Точность на этапах обучения и проверки для модели анализа IMDB со слоем LSTM</em>
        </p>

        На этот раз мы достигли точности 89 % на этапе проверки. Неплохой результат: намного лучше, чем с сетью на основе слоя SimpleRNN, что в значительной степени объясняется меньшей подверженностью LSTM проблеме затухания градиента.

        Однако этот результат не является впечатляющим для подхода с таким большим объемом вычислений. Почему решение на основе LSTM не смогло добиться лучшего результата? Одна из причин — мы даже не пытались настроить гиперпараметры, такие как размерность векторных представлений или размерность результата, возвращаемого слоем LSTM. Другой причиной может быть отсутствие регуляризации. Однако, если быть честными, главная причина в том, что анализ глобальной протяженной структуры отзывов (с чем прекрасно справляется LSTM) плохо помогает в решении задачи определения эмоциональной окраски. Такие простые задачи хорошо решаются путем определения частот слов, которые встречаются в отзывах. Именно на этом было основано первое полносвязное решение. Однако существуют другие, намного более сложные задачи обработки естественного языка, где мощь LSTM проявляется более очевидно: например, в диалоговых системах типа «вопрос/ответ» и в машинном переводе.

        ## Задание

        Получить точность классификации более 92%. Например, с помощью увеличения числа слов, участвующих в обучении, настройки параметров сети и тд.
        """
    ),
    code(
        """
        from tensorflow.keras.layers import GlobalMaxPooling1D, SpatialDropout1D

        tf.random.set_seed(42)

        assignment_max_features = 20000
        assignment_maxlen = 500
        assignment_batch_size = 64

        assignment_model = Sequential(
            [
                Input(shape=(assignment_maxlen,)),
                Embedding(assignment_max_features, 128),
                SpatialDropout1D(0.2),
                Bidirectional(LSTM(64, return_sequences=True)),
                GlobalMaxPooling1D(),
                Dense(64, activation="relu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )

        assignment_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["acc"],
        )

        assignment_callbacks = [
            EarlyStopping(
                monitor="val_acc",
                patience=2,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=1,
                min_lr=1e-5,
            ),
        ]

        assignment_model.summary()
        """
    ),
    code(
        """
        assignment_history = assignment_model.fit(
            input_train,
            y_train,
            epochs=10,
            batch_size=assignment_batch_size,
            validation_split=0.2,
            callbacks=assignment_callbacks,
            verbose=1,
        )
        """
    ),
    code(
        """
        plot_training_history(assignment_history, "Bidirectional LSTM")
        """
    ),
    code(
        """
        acc_key = "acc" if "acc" in assignment_history.history else "accuracy"
        val_acc_key = "val_acc" if "val_acc" in assignment_history.history else "val_accuracy"

        best_train_accuracy = max(assignment_history.history[acc_key])
        best_validation_accuracy = max(assignment_history.history[val_acc_key])

        print(f"Максимальная точность на обучении: {best_train_accuracy:.4f}")
        print(f"Максимальная точность на проверке: {best_validation_accuracy:.4f}")
        print("Целевой порог 92% на обучении достигнут:", best_train_accuracy > 0.92)
        """
    ),
    md(
        """
        ## Вывод

        В работе были рассмотрены рекуррентные нейронные сети, их связь с внутренним состоянием, слой `SimpleRNN`, режим `return_sequences`, а также слой `LSTM`, предназначенный для обработки более длинных зависимостей. На задаче классификации отзывов `IMDB` простая рекуррентная модель используется как базовый вариант, после чего качество улучшается за счет более сильной рекуррентной архитектуры и увеличения числа учитываемых слов.
        """
    ),
]


NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
    },
}

assignment_start = next(
    i
    for i, cell in enumerate(cells)
    if cell.cell_type == "markdown" and "## Задание" in cell.source
)

notebook_1 = nbf.v4.new_notebook()
notebook_1["cells"] = cells[:assignment_start]
notebook_1["metadata"] = NOTEBOOK_METADATA

assignment_cells = [
    md(
        """
        # Практическая работа 4.2. RNN

        Выполнение задания.
        """
    ),
    md(
        """
        ## Задание

        Получить точность классификации более 92%. Например, с помощью увеличения числа слов, участвующих в обучении, настройки параметров сети и тд.
        """
    ),
    code(
        """
        from __future__ import annotations

        import os
        import random
        from pathlib import Path

        PRACTICE_DIR = Path.cwd()

        CACHE_DIR = PRACTICE_DIR / ".cache"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "matplotlib")
        os.environ["IPYTHONDIR"] = str(CACHE_DIR / "ipython")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        DATA_DIR = PRACTICE_DIR / ".data"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["KERAS_HOME"] = str(DATA_DIR / "keras_home")

        import matplotlib.pyplot as plt
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.datasets import imdb
        from tensorflow.keras.layers import (
            Bidirectional,
            Dense,
            Dropout,
            Embedding,
            GlobalMaxPooling1D,
            Input,
            LSTM,
            SpatialDropout1D,
        )
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.regularizers import l2

        RANDOM_STATE = 42
        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        tf.random.set_seed(RANDOM_STATE)

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["figure.figsize"] = (10, 4)

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices([], "GPU")
        """
    ),
    code(
        """
        def plot_training_history(history, title_prefix: str) -> None:
            \"\"\"Строит графики точности и потерь по объекту History из Keras.\"\"\"
            history_dict = history.history
            acc_key = "acc" if "acc" in history_dict else "accuracy"
            val_acc_key = "val_acc" if "val_acc" in history_dict else "val_accuracy"

            acc = history_dict[acc_key]
            val_acc = history_dict[val_acc_key]
            loss = history_dict["loss"]
            val_loss = history_dict["val_loss"]
            epochs = range(1, len(acc) + 1)

            plt.figure()
            plt.plot(epochs, acc, "bo", label="Training acc")
            plt.plot(epochs, val_acc, "b", label="Validation acc")
            plt.title(f"{title_prefix}: training and validation accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()

            plt.figure()
            plt.plot(epochs, loss, "bo", label="Training loss")
            plt.plot(epochs, val_loss, "b", label="Validation loss")
            plt.title(f"{title_prefix}: training and validation loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        """
    ),
    code(
        """
        max_features = 20000
        maxlen = 500
        batch_size = 64

        (input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
        input_train = pad_sequences(input_train, maxlen=maxlen)
        input_test = pad_sequences(input_test, maxlen=maxlen)

        print("input_train shape:", input_train.shape)
        print("input_test shape:", input_test.shape)
        """
    ),
    md(
        """
        За основу взята более сильная схема с двунаправленным `LSTM`, `SpatialDropout1D` и `GlobalMaxPooling1D`. Для добора качества увеличены размерность скрытого слоя и добавлена L2-регуляризация полносвязного слоя. Контролируется именно максимальная точность на проверочной выборке.
        """
    ),
    code(
        """
        model = Sequential(
            [
                Input(shape=(maxlen,)),
                Embedding(max_features, 128),
                SpatialDropout1D(0.2),
                Bidirectional(LSTM(96, return_sequences=True, dropout=0.2)),
                GlobalMaxPooling1D(),
                Dense(128, activation="relu", kernel_regularizer=l2(1e-4)),
                Dropout(0.45),
                Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
            loss="binary_crossentropy",
            metrics=["acc"],
        )

        callbacks = [
            EarlyStopping(
                monitor="val_acc",
                patience=3,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=1,
                min_lr=1e-5,
            ),
        ]

        model.summary()
        """
    ),
    code(
        """
        history = model.fit(
            input_train,
            y_train,
            epochs=12,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1,
        )
        """
    ),
    code(
        """
        plot_training_history(history, "Bidirectional LSTM")
        """
    ),
    code(
        """
        best_train_accuracy = max(history.history["acc"])
        best_validation_accuracy = max(history.history["val_acc"])

        print(f"Максимальная точность на обучении: {best_train_accuracy:.4f}")
        print(f"Максимальная точность на проверке: {best_validation_accuracy:.4f}")
        print("Целевой порог 92% на проверке достигнут:", best_validation_accuracy > 0.92)
        """
    ),
    md(
        """
        ## Вывод

        В задании используется увеличенный словарь, последовательности длиной `500` токенов и двунаправленный слой `LSTM`. Качество оценивается по максимальной точности на проверочной выборке; целевым считается значение выше `92%`.
        """
    ),
]

notebook_2 = nbf.v4.new_notebook()
notebook_2["cells"] = assignment_cells
notebook_2["metadata"] = NOTEBOOK_METADATA

nbf.write(notebook_1, NOTEBOOK_1_PATH)
nbf.write(notebook_2, NOTEBOOK_2_PATH)
print(f"Notebook written to {NOTEBOOK_1_PATH}")
print(f"Notebook written to {NOTEBOOK_2_PATH}")
