from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PRACTICE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PRACTICE_DIR / "СП-М-О-АЕЯМИИ-2025-Практика-3-БабушкинМВ.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # Практическая работа 3. Keras

        IMDB и Keras.
        """
    ),
    md(
        """
        ## Прямое кодирование

        Прямое кодирование (`one-hot encoding`) — наиболее используемый и простой способ преобразования токенов в векторы. Вы уже видели его в действии в первых примерах с наборами данных IMDB и Reuters. Он заключается в присваивании каждому слову уникального целочисленного индекса `i` с последующим его преобразованием в бинарный вектор размера `N` (размер словаря); все элементы этого вектора содержат нули, кроме `i`-го элемента, которому присваивается `1`.

        Конечно, прямое кодирование можно выполнить и на уровне символов.
        """
    ),
    code(
        """
        from __future__ import annotations

        import os
        import random
        import shutil
        import ssl
        import string
        import tarfile
        import urllib.request
        import zipfile
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
        from tensorflow.keras.datasets import imdb
        from tensorflow.keras.layers import Dense, Embedding, Flatten, Input
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import Tokenizer

        IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"

        RAW_IMDB_ARCHIVE = DATA_DIR / "aclImdb_v1.tar.gz"
        RAW_IMDB_DIR = DATA_DIR / "aclImdb"
        GLOVE_ARCHIVE = DATA_DIR / "glove.6B.zip"
        GLOVE_DIR = DATA_DIR / "glove.6B"
        MODEL_DIR = DATA_DIR / "models"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        RANDOM_STATE = 42
        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        tf.random.set_seed(RANDOM_STATE)

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["figure.figsize"] = (10, 4)
        np.set_printoptions(edgeitems=5, linewidth=120)
        """
    ),
    code(
        """
        samples = ["The cat sat on the mat.", "The dog ate my homework."]

        token_index = {}
        for sample in samples:
            for word in sample.split():
                if word not in token_index:
                    token_index[word] = len(token_index) + 1

        max_length = 10
        results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))

        for i, sample in enumerate(samples):
            for j, word in list(enumerate(sample.split()))[:max_length]:
                index = token_index.get(word)
                results[i, j, index] = 1.0

        print("Индекс токенов:", token_index)
        print("Форма результата:", results.shape)
        """
    ),
    md(
        """
        ## Прямое кодирование на уровне символов
        """
    ),
    code(
        """
        samples = ["The cat sat on the mat.", "The dog ate my homework."]

        characters = string.printable
        token_index = dict(zip(characters, range(1, len(characters) + 1)))

        max_length = 50
        results = np.zeros((len(samples), max_length, len(token_index) + 1))

        for i, sample in enumerate(samples):
            for j, character in enumerate(sample[:max_length]):
                index = token_index.get(character)
                results[i, j, index] = 1.0

        print("Количество символов:", len(token_index))
        print("Форма результата:", results.shape)
        """
    ),
    md(
        """
        Следует отметить, что во фреймворке Keras имеются встроенные утилиты для прямого кодирования простых текстовых данных на уровне слов и символов. Воспользуйтесь этими утилитами, поскольку они обладают рядом важных свойств, таких как удаление из строк специальных символов и возможность принимать в расчет только `N` слов, наиболее часто встречающихся в наборе данных (типичное ограничение, помогающее избежать создания слишком обширного пространства входных векторов).
        """
    ),
    md(
        """
        ## Использование Keras для прямого кодирования слов
        """
    ),
    code(
        """
        samples = ["The cat sat on the mat.", "The dog ate my homework."]

        tokenizer = Tokenizer(num_words=1000)
        tokenizer.fit_on_texts(samples)

        sequences = tokenizer.texts_to_sequences(samples)
        one_hot_results = tokenizer.texts_to_matrix(samples, mode="binary")
        word_index = tokenizer.word_index

        print("Последовательности:", sequences)
        print("Форма матрицы:", one_hot_results.shape)
        print("Found %s unique tokens." % len(word_index))
        """
    ),
    md(
        """
        Прием прямого кодирования имеет разновидность — так называемое прямое хеширование признаков (`one-hot hashing trick`), которое можно использовать, когда словарь содержит слишком большое количество токенов, чтобы его можно было использовать явно. Вместо явного присваивания индекса каждому слову и сохранения ссылок на эти индексы в словаре можно хешировать слова в векторы фиксированного размера. Обычно для этого используются очень легковесные функции хеширования. Главное достоинство этого метода — отсутствие необходимости хранить индексы слов, что позволяет сэкономить память и кодировать данные по мере необходимости (векторы токенов можно генерировать сразу же, по мере их обхода, до просмотра всех имеющихся данных). Единственный недостаток — этот метод восприимчив к хеш-коллизиям: два разных слова могут получить одинаковые хеш-значения, и впоследствии любая модель машинного обучения не сможет различить эти слова. Вероятность хеш-коллизий снижается, когда размер пространства хеширования намного больше общего количества уникальных токенов, подвергаемых хешированию.
        """
    ),
    md(
        """
        ## Прямое кодирование на уровне слов с использованием хеширования
        """
    ),
    code(
        """
        samples = ["The cat sat on the mat.", "The dog ate my homework."]

        dimensionality = 1000
        max_length = 10
        results = np.zeros((len(samples), max_length, dimensionality))

        for i, sample in enumerate(samples):
            for j, word in list(enumerate(sample.split()))[:max_length]:
                index = abs(hash(word)) % dimensionality
                results[i, j, index] = 1.0

        print("Форма результата:", results.shape)
        """
    ),
    md(
        """
        ## Векторное представление слов

        Другим популярным и мощным способом связывания вектора со словом является использование плотных векторов слов, или векторного представления слов (word embeddings). В отличие от векторов, полученных прямым кодированием, — бинарных, разреженных (почти полностью состоящих из нулей) и с большой размерностью (их размерность совпадает с количеством слов в словаре) — векторные представления слов являются малоразмерными векторами вещественных чисел (то есть плотными векторами, в противоположность разреженным. В отличие от векторов, полученных прямым кодированием, векторные представления слов конструируются из данных. При работе с огромными словарями размерность векторов слов нередко может достигать 256, 512 или 1024. С другой стороны, прямое кодирование слов обычно влечет за собой создание векторов с числом измерений 20 000 или больше (при использовании словаря с 20 000 токенов, как в данном случае). Иначе говоря, векторное представление слов позволяет уместить больший объем информации в меньшее число измерений.

        Получить векторные представления слов можно двумя способами:

        • Конструировать векторные представления в процессе решения основной задачи (такой, как классификация документа или определение эмоциональной окраски). В этом случае изначально создаются случайные векторы слов, которые затем постепенно конструируются (обучаются), как это происходит с весами нейронной сети.
        • Загрузить в модель векторные представления, полученные с использованием другой задачи машинного обучения, отличной от решаемой. Такие представления называют предварительно обученными векторными представлениями слов.

        Рассмотрим оба способа.
        """
    ),
    md(
        """
        ## Конструирование векторных представлений слов с помощью слоя Embedding

        Простейший способ связать плотный вектор со словом — выбрать случайный вектор. Однако пространство векторов, которое получится в этом случае, не имеет структуры: например, слова *accurate* и *exact* могут в конечном счете получить совершенно разные векторные представления, даже при том, что в большинстве случаев они взаимозаменяемы1. Глубокой нейронной сети трудно будет понять такое искаженное, неструктурированное пространство векторов.

        Говоря более абстрактно, геометрические отношения между векторами слов должны отражать семантические связи между соответствующими им словами. Как предполагается, векторные представления слов должны отображать человеческий язык в геометрическое пространство. Например, от правильно сконструированного пространства векторных представлений разумно ожидать, что синонимы будут представлены похожими векторами и в целом геометрическое расстояние (*L2*-расстояние) между любыми двумя векторами будет зависеть от семантического расстояния между соответствующими словами (слова с далеким друг от друга смыслом будут представлены далекими друг от друга точками, а слова со схожим смыслом — близкими). Кроме расстояния, может оказаться желательным наделить определенным смыслом конкретные *направления* в пространстве векторов. Поясним это на конкретном примере.

        ![Двумерная плоскость с векторными представлениями слов](2d_plot.png)

        На рис. изображена двумерная плоскость с четырьмя векторными представлениями слов: *кошка*, *собака*, *волк* и *тигр*. С выбранными здесь векторными представлениями некоторые семантические отношения между словами можно выразить в виде геометрических преобразований. Например, один и тот же вектор позволяет перейти от *кошки* к *тигру* и от *собаки* к *волку*: этот вектор можно было бы интерпретировать как вектор «от домашнего животного к дикому». Аналогично, другой вектор позволяет перейти от *собаки* к *кошке* и от *волка* к *тигру*, и его можно интерпретировать как вектор «от псовых к кошачьим».

        В настоящих векторных пространствах слов типичными примерами осмысленных геометрических преобразований могут служить векторы «половая принадлежность» и «много». Например, сложив векторы «женщина» и «король», мы получили бы вектор «королева». Сложив векторы «много» и «король», мы получили бы вектор «короли». В векторных пространствах слов обычно существуют тысячи таких интерпретируемых и потенциально полезных векторов.

        Существует ли идеальное векторное пространство слов, точно отражающее человеческий язык, которое можно было бы использовать для решения любых задач обработки естественного языка? Возможно, однако нам еще предстоит вычислить нечто подобное. Кроме того, нет такого понятия, как *человеческий язык*, — есть много разных языков, и они не изоморфны, потому что каждый язык является отражением конкретной культуры и контекста. Пригодность векторного пространства слов для практического применения в значительной степени зависит от конкретной задачи: идеальное векторное пространство слов для англоязычной модели анализа эмоциональной окраски отзывов к фильмам может отличаться от идеального векторного пространства для англоязычной модели классификации юридических документов, потому что важность определенных семантических отношений различна для разных задач.

        Как следствие, представляется разумным обучать новое векторное пространство слов для каждой новой задачи. К счастью, прием обратного распространения ошибки помогает легко добиться этого, а Keras еще больше упрощает реализацию. Речь идет об обучении весов слоя: в данном случае слоя `Embedding`.
        """
    ),
    md(
        """
        ## Создание слоя Embedding
        """
    ),
    code(
        """
        embedding_layer = Embedding(1000, 64)
        embedding_layer
        """
    ),
    md(
        """
        Слой Embedding лучше всего воспринимать как словарь, отображающий целочисленные индексы (обозначающие конкретные слова) в плотные векторы. Он принимает целые числа на входе, отыскивает их во внутреннем словаре и возвращает соответствующие векторы. Это эффективная операция поиска в словаре.

        Слой Embedding получает на входе двумерный тензор с целыми числами и с формой (образцы, длина_последовательности), каждый элемент которого является последовательностью целых чисел. Он может работать с последовательностями разной длины: например, слою Embedding из предыдущего примера можно передавать пакеты с формой (32, 10) (пакет с 32 последовательностями, каждая длиной 10) или (64, 15) (пакет с 64 последовательностями, каждая длиной 15). Все последовательности в пакете должны иметь одинаковую длину, потому что упаковываются в один тензор, поэтому короткие последовательности, если они есть, нужно дополнить нулями, а длинные — усечь.

        Этот слой возвращает трехмерный тензор с вещественными числами и с формой (образцы, длина_последовательности, размерность_векторного_представления). Такой трехмерный тензор можно затем обработать слоем RNN или одномерным сверточным слоем (оба будут представлены в следующих разделах).

        При создании слоя Embedding, его веса (внутренний словарь векторов токенов) инициализируются случайными значениями, как в случае с любым другим слоем. В процессе обучения векторы слов постепенно корректируются посредством обратного распространения ошибки, и пространство превращается в структурированную модель, пригодную к использованию. После полного обучения пространство векторов приобретет законченную структуру, специализированную под решение конкретной задачи.
        """
    ),
    md(
        """
        Используем эту идею для решения задачи определения эмоциональной окраски отзывов к фильмам в IMDB, которую мы уже пытались решить ранее. Сначала подготовим исходные данные. Ограничимся набором из 10 000 слов, наиболее часто встречающихся в отзывах к фильмам (так же, как мы делали это в первый раз), и выберем из каждого отзыва только первые 20 слов. Сеть будет обучать 8-мерные векторные представления, по 10 000 слов в каждом, преобразовывать входные последовательности целых чисел (двумерный тензор с целыми числами) в векторные последовательности (трехмерный тензор с вещественными числами), преобразовывать трехмерный тензор в двумерный и обучать единственный слой Dense на вершине, предназначенный для классификации.
        """
    ),
    code(
        """
        max_features = 10000
        maxlen = 20

        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

        x_train = pad_sequences(x_train, maxlen=maxlen)
        x_test = pad_sequences(x_test, maxlen=maxlen)

        print("Форма обучающих данных:", x_train.shape)
        print("Форма тестовых данных:", x_test.shape)
        """
    ),
    code(
        """
        model = Sequential(
            [
                Input(shape=(maxlen,)),
                Embedding(max_features, 8),
                Flatten(),
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
            x_train,
            y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
        )
        """
    ),
    md(
        """
        На проверочных данных мы получили точность ~76%. Это очень хорошо, если учесть, что мы исследовали только первые 20 слов из каждого отзыва. Однако обратите внимание на то, что в результате простого сокращения размерности векторных последовательностей и обучения единственного слоя Dense получается модель, которая отдельно интерпретирует каждое слово во входной последовательности, не учитывая связей между словами и структуры предложений (например, эта модель наверняка расценит оба отзыва — «this movie is a bomb» и «this movie is the bomb» — как отрицательные1). Лучший результат можно получить, если добавить рекуррентные или одномерные сверточные слои поверх векторных последовательностей для извлечения признаков, которые учитывают целые последовательности слов.
        """
    ),
    md(
        """
        ## Использование предварительно обученных векторных представлений слов

        Иногда обучающих данных оказывается слишком мало, чтобы можно было обучить векторное представление слов для конкретной задачи. Что можно сделать в этом случае?

        Вместо обучения векторного представления совместно с решением задачи можно загрузить предварительно сформированные векторные представления, хорошо организованные и обладающие полезными свойствами, которые охватывают основные аспекты языковой структуры. Использование предварительно обученных векторных представлений слов в обработке естественного языка обосновывается почти так же, как использование предварительно обученных сверточных нейронных сетей в классификации изображений: отсутствием достаточного объема данных для выделения по-настоящему мощных признаков. Также предполагается, что для решения задачи достаточно обобщенных признаков, то есть обобщенных визуальных и семантических признаков. В данном случае есть смысл повторно использовать признаки, выделенные в ходе решения другой задачи.

        Такие векторные представления обычно вычисляются с использованием статистик встречаемости слов (наблюдений совместной встречаемости слов в предложениях или документах) и применением разнообразных методик, иногда с привлечением нейронных сетей, иногда нет. Идея плотных, малоразмерных пространств векторных представлений слов, обучаемых без учителя, первоначально была исследована Йошуа Бенгио (Yoshua Bengio) с коллегами в начале 2000-х годов, но более основательное ее исследование и практическое применение началось только после выхода одной из самых известных и успешных схем реализации векторного представления слов — алгоритма Word2vec (*https://code. google.com/archive/p/word2vec*), разработанного в 2013 году Томасом Миколовым (Tomas Mikolov) из компании Google. Измерения Word2vec охватывают такие семантические признаки, как пол.

        Существует множество разнообразных предварительно обученных векторных представлений слов, которые можно загрузить и использовать в слое Embedding. Word2vec — одно из них. Другое популярное представление называется «глобальные векторы представления слов» (Global Vectors for Word Representation, GloVe, *https://nlp.stanford.edu/projects/glove*) и разработано исследователями из Стэнфорда в 2014-м. Это представление основано на факторизации матрицы статистик совместной встречаемости слов. Его создатели включили в представление миллионы токенов из английского языка, полученных из Википедии и данных компании Common Crawl.

        Давайте посмотрим, как можно использовать представления GloVe в моделях Keras. Ту же методику можно применить к Word2vec и другим предварительно обученным векторным представлениям слов. Этот пример также можно использовать для усовершенствования приемов токенизации текста, представленных несколькими абзацами выше: вы начинаете с простого текста и постепенно движетесь вверх.
        """
    ),
    md(
        """
        ## Объединение всего вместе: от исходного текста к векторному представлению слов

        Мы будем использовать модель, похожую на ту, по которой только что прошлись: преобразуем предложения в последовательности векторов, снизим их размерность и обучим слой `Dense` сверху. Но за основу мы возьмем предварительно обученные векторные представления слов и вместо использования предварительно токенизированных данных IMDB, входящих в состав Keras, пройдем весь процесс с самого начала — с загрузки исходных текстовых данных.
        """
    ),
    md(
        """
        ## Загрузка данных из IMDB в виде простого текста

        Сначала загрузите архив с исходным набором данных IMDB, доступный по адресу `http://mng.bz/0tIo`. Распакуйте его.

        Теперь соберем отдельные обучающие отзывы в список строк, по одной строке на отзыв. Также соберем метки отзывов (положительный/отрицательный) в список `labels`.
        """
    ),
    code(
        """
        def ensure_raw_imdb_dataset(
            dataset_url: str = IMDB_URL,
            archive_path: Path = RAW_IMDB_ARCHIVE,
            dataset_dir: Path = RAW_IMDB_DIR,
        ) -> Path:
            \"\"\"Возвращает локальный каталог `aclImdb`, при необходимости копируя или скачивая датасет.

            Parameters
            ----------
            dataset_url : str, default=IMDB_URL
                Ссылка на архив исходного датасета IMDB.
            archive_path : Path, default=RAW_IMDB_ARCHIVE
                Локальный путь к архиву.
            dataset_dir : Path, default=RAW_IMDB_DIR
                Локальный путь к распакованному датасету.

            Returns
            -------
            Path
                Каталог с распакованным датасетом `aclImdb`.
            \"\"\"
            if dataset_dir.exists():
                return dataset_dir

            practice1_dataset_dir = PRACTICE_DIR.parent / "Практика 1 - Тональность" / ".data" / "aclImdb"
            if practice1_dataset_dir.exists():
                shutil.copytree(practice1_dataset_dir, dataset_dir, dirs_exist_ok=True)
                return dataset_dir

            practice1_archive_path = PRACTICE_DIR.parent / "Практика 1 - Тональность" / ".data" / "aclImdb_v1.tar.gz"
            if practice1_archive_path.exists() and not archive_path.exists():
                shutil.copy2(practice1_archive_path, archive_path)

            if not archive_path.exists():
                try:
                    urllib.request.urlretrieve(dataset_url, archive_path)
                except Exception:
                    ssl_context = ssl._create_unverified_context()
                    with urllib.request.urlopen(dataset_url, context=ssl_context) as response:
                        archive_path.write_bytes(response.read())

            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=archive_path.parent)

            return dataset_dir
        """
    ),
    code(
        """
        def load_imdb_split(split_dir: Path) -> tuple[list[str], list[int]]:
            \"\"\"Загружает тексты и метки из части датасета IMDB.

            Parameters
            ----------
            split_dir : Path
                Каталог `train` или `test`.

            Returns
            -------
            tuple[list[str], list[int]]
                Список отзывов и список меток классов.
            \"\"\"
            labels = []
            texts = []

            for label_type in ["neg", "pos"]:
                dir_name = split_dir / label_type
                for file_path in sorted(dir_name.glob("*.txt")):
                    texts.append(file_path.read_text(encoding="utf-8", errors="replace").replace("<br />", " "))
                    labels.append(0 if label_type == "neg" else 1)

            return texts, labels
        """
    ),
    code(
        """
        imdb_dir = ensure_raw_imdb_dataset()
        train_texts, train_labels = load_imdb_split(imdb_dir / "train")

        print("Количество обучающих отзывов:", len(train_texts))
        print("Количество меток:", len(train_labels))
        print(train_texts[0][:1000])
        """
    ),
    md(
        """
        ## Токенизация данных

        Токенизация текста из исходного набора данных IMDB
        """
    ),
    code(
        """
        def prepare_imdb_sequences(
            texts: list[str],
            labels: list[int],
            maxlen: int = 100,
            training_samples: int = 200,
            validation_samples: int = 10000,
            max_words: int = 10000,
            seed: int = RANDOM_STATE,
        ) -> dict[str, object]:
            \"\"\"Токенизирует тексты IMDB и формирует обучающую и проверочную выборки.

            Parameters
            ----------
            texts : list[str]
                Исходные тексты отзывов.
            labels : list[int]
                Метки классов.
            maxlen : int, default=100
                Максимальная длина последовательности.
            training_samples : int, default=200
                Количество обучающих образцов.
            validation_samples : int, default=10000
                Количество проверочных образцов.
            max_words : int, default=10000
                Максимальное число слов в словаре.
            seed : int, default=RANDOM_STATE
                Случайное состояние для перемешивания.

            Returns
            -------
            dict[str, object]
                Токенизатор, индекс слов и подготовленные массивы.
            \"\"\"
            tokenizer = Tokenizer(num_words=max_words)
            tokenizer.fit_on_texts(texts)

            sequences = tokenizer.texts_to_sequences(texts)
            word_index = tokenizer.word_index

            data = pad_sequences(sequences, maxlen=maxlen)
            labels_array = np.asarray(labels)

            indices = np.arange(data.shape[0])
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

            data = data[indices]
            labels_array = labels_array[indices]

            x_train = data[:training_samples]
            y_train = labels_array[:training_samples]
            x_val = data[training_samples : training_samples + validation_samples]
            y_val = labels_array[training_samples : training_samples + validation_samples]

            return {
                "tokenizer": tokenizer,
                "word_index": word_index,
                "data": data,
                "labels": labels_array,
                "x_train": x_train,
                "y_train": y_train,
                "x_val": x_val,
                "y_val": y_val,
            }
        """
    ),
    code(
        """
        maxlen = 100 #Отсечение остатка отзывов после 100-го слова
        training_samples = 200 #Обучение на выборке из 200 образцов
        validation_samples = 10000 #Проверка на выборке из 10 000 образцов
        max_words = 10000 #Рассмотрение только 10 000 наиболее часто используемых слов

        prepared = prepare_imdb_sequences(
            texts=train_texts,
            labels=train_labels,
            maxlen=maxlen,
            training_samples=training_samples,
            validation_samples=validation_samples,
            max_words=max_words,
        )

        print("Found %s unique tokens." % len(prepared["word_index"]))
        print("Shape of data tensor:", prepared["data"].shape)
        print("Shape of label tensor:", prepared["labels"].shape)
        print("Shape of x_train:", prepared["x_train"].shape)
        print("Shape of x_val:", prepared["x_val"].shape)
        """
    ),
    md(
        """
        ## Загрузка векторного представления GloVe

        Откройте в браузере страницу `https://nlp.stanford.edu/projects/glove` и загрузите векторные представления, предварительно обученные на данных из англоязычной Википедии за 2014 год. Этот ZIP-архив размером `822` Мбайт с именем `glove.6B.zip` содержит `100`-мерные векторы с `400000` слов (токенов). Распакуйте его.
        """
    ),
    code(
        """
        def ensure_glove_file(
            glove_url: str = GLOVE_URL,
            archive_path: Path = GLOVE_ARCHIVE,
            target_dir: Path = GLOVE_DIR,
            filename: str = "glove.6B.100d.txt",
        ) -> Path:
            \"\"\"Возвращает путь к файлу GloVe с векторами размерности 100.

            Parameters
            ----------
            glove_url : str, default=GLOVE_URL
                Ссылка на архив GloVe.
            archive_path : Path, default=GLOVE_ARCHIVE
                Локальный путь к архиву.
            target_dir : Path, default=GLOVE_DIR
                Каталог распаковки.
            filename : str, default="glove.6B.100d.txt"
                Имя файла с векторами.

            Returns
            -------
            Path
                Путь к текстовому файлу GloVe.
            \"\"\"
            target_dir.mkdir(parents=True, exist_ok=True)
            glove_path = target_dir / filename

            if glove_path.exists():
                return glove_path

            if not archive_path.exists():
                try:
                    urllib.request.urlretrieve(glove_url, archive_path)
                except Exception:
                    ssl_context = ssl._create_unverified_context()
                    with urllib.request.urlopen(glove_url, context=ssl_context) as response:
                        archive_path.write_bytes(response.read())

            with zipfile.ZipFile(archive_path) as archive:
                archive.extract(filename, path=target_dir)

            return glove_path
        """
    ),
    md(
        """
        ## Предварительная обработка векторных представлений

        Обработаем распакованный файл `.txt` и создадим индекс, отображающий слова (в виде строк) в их векторные представления (в виде векторов с числами).
        """
    ),
    code(
        """
        def load_glove_embeddings(glove_path: Path) -> dict[str, np.ndarray]:
            \"\"\"Загружает GloVe-векторы в словарь.

            Parameters
            ----------
            glove_path : Path
                Путь к файлу `glove.6B.100d.txt`.

            Returns
            -------
            dict[str, np.ndarray]
                Словарь `слово -> вектор`.
            \"\"\"
            embeddings_index = {}

            with glove_path.open(encoding="utf-8") as glove_file:
                for line in glove_file:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype="float32")
                    embeddings_index[word] = coefs

            return embeddings_index
        """
    ),
    code(
        """
        glove_path = ensure_glove_file()
        embeddings_index = load_glove_embeddings(glove_path)

        print("Путь к GloVe:", glove_path)
        print("Found %s word vectors." % len(embeddings_index))
        """
    ),
    md(
        """
        Теперь создадим матрицу векторных представлений, которую можно будет передать на вход слоя `Embedding`. Это должна быть матрица с формой `(максимальное_число_слов, размерность_представления)`, каждый элемент `i` которой содержит вектор с размером, равным размерности представления, соответствующий слову с индексом `i` в индексе (созданном в ходе токенизации).

        Обратите внимание: индекс `0` не должен соответствовать никакому слову или токену — это пустой элемент.
        """
    ),
    code(
        """
        def build_embedding_matrix(
            word_index: dict[str, int],
            embeddings_index: dict[str, np.ndarray],
            max_words: int = 10000,
            embedding_dim: int = 100,
        ) -> tuple[np.ndarray, int]:
            \"\"\"Строит матрицу векторных представлений GloVe для словаря токенизатора.

            Parameters
            ----------
            word_index : dict[str, int]
                Индекс слов, построенный `Tokenizer`.
            embeddings_index : dict[str, np.ndarray]
                Словарь предварительно обученных векторов.
            max_words : int, default=10000
                Максимальное число слов.
            embedding_dim : int, default=100
                Размерность векторного представления.

            Returns
            -------
            tuple[np.ndarray, int]
                Матрица эмбеддингов и число найденных слов.
            \"\"\"
            embedding_matrix = np.zeros((max_words, embedding_dim))
            found_words = 0

            for word, i in word_index.items():
                if i < max_words:
                    embedding_vector = embeddings_index.get(word)
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
                        found_words += 1

            return embedding_matrix, found_words
        """
    ),
    code(
        """
        embedding_dim = 100

        embedding_matrix, found_words = build_embedding_matrix(
            word_index=prepared["word_index"],
            embeddings_index=embeddings_index,
            max_words=max_words,
            embedding_dim=embedding_dim,
        )

        print("Форма embedding_matrix:", embedding_matrix.shape)
        print("Слов с найденными векторами:", found_words)
        """
    ),
    md(
        """
        ## Определение модели

        Используем модель с той же архитектурой, как было показано выше.
        """
    ),
    code(
        """
        def build_imdb_model(max_words: int, embedding_dim: int, maxlen: int) -> Sequential:
            \"\"\"Создает модель классификации IMDB на основе слоя Embedding.

            Parameters
            ----------
            max_words : int
                Размер словаря.
            embedding_dim : int
                Размерность эмбеддинга.
            maxlen : int
                Длина входной последовательности.

            Returns
            -------
            Sequential
                Модель Keras.
            \"\"\"
            model = Sequential(
                [
                    Input(shape=(maxlen,)),
                    Embedding(max_words, embedding_dim),
                    Flatten(),
                    Dense(32, activation="relu"),
                    Dense(1, activation="sigmoid"),
                ]
            )
            return model
        """
    ),
    code(
        """
        model = build_imdb_model(
            max_words=max_words,
            embedding_dim=embedding_dim,
            maxlen=maxlen,
        )

        model.summary()
        """
    ),
    md(
        """
        ## Загрузка представлений GloVe в модель

        Уровень `Embedding` имеет единственную весовую матрицу: двумерную матрицу с вещественными числами, каждый `i`-й элемент которой — это вектор, связанный с `i`-м словом в индексе. Все довольно просто. Загрузим подготовленную матрицу GloVe в слой `Embedding` — первый слой модели.

        Мы также заморозили слой `Embedding` (присвоив атрибуту `trainable` значение `False`). Причины те же, что были описаны, когда мы знакомились с особенностями применения предварительно обученных сверточных нейронных сетей: когда в модели имеются уже обученные части (как наш слой `Embedding`) и части, инициализированные случайными значениями (как наш классификатор), обученные части не должны изменяться в ходе обучения, чтобы не потерять свои знания. Большие изменения градиента, вызванные случайными начальными значениями в необученных слоях, могут оказать разрушительное влияние на обученные слои.
        """
    ),
    code(
        """
        model.layers[0].set_weights([embedding_matrix])
        model.layers[0].trainable = False

        model.compile(
            optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        """
    ),
    md(
        """
        ## Обучение и оценка модели

        Скомпилируем и обучим модель.
        """
    ),
    code(
        """
        history = model.fit(
            prepared["x_train"],
            prepared["y_train"],
            epochs=10,
            batch_size=32,
            validation_data=(prepared["x_val"], prepared["y_val"]),
        )

        pretrained_weights_path = MODEL_DIR / "pre_trained_glove_model.weights.h5"
        model.save_weights(pretrained_weights_path)
        pretrained_weights_path
        """
    ),
    md(
        """
        Теперь выведем графики изменения качества модели

        Потери на этапах обучения и проверки при использовании уже обученных векторных представлений слов

        Точность на этапах обучения и проверки при использовании уже обученных векторных представлений слов

        Вывод результатов
        """
    ),
    code(
        """
        def plot_training_history(history: tf.keras.callbacks.History, title: str) -> None:
            \"\"\"Строит графики точности и функции потерь.

            Parameters
            ----------
            history : tf.keras.callbacks.History
                История обучения модели.
            title : str
                Заголовок для графиков.

            Returns
            -------
            None
                Выводит графики обучения.
            \"\"\"
            accuracy_key = "accuracy" if "accuracy" in history.history else "acc"
            val_accuracy_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"

            acc = history.history[accuracy_key]
            val_acc = history.history[val_accuracy_key]
            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            epochs = range(1, len(acc) + 1)

            plt.plot(epochs, acc, "bo", label="Training acc")
            plt.plot(epochs, val_acc, "b", label="Validation acc")
            plt.title(f"{title}: accuracy")
            plt.legend()
            plt.figure()

            plt.plot(epochs, loss, "bo", label="Training loss")
            plt.plot(epochs, val_loss, "b", label="Validation loss")
            plt.title(f"{title}: loss")
            plt.legend()
            plt.show()
        """
    ),
    code(
        """
        plot_training_history(history, "GloVe")
        """
    ),
    md(
        """
        Модель быстро достигает состояния переобучения, что неудивительно при таком малом объеме обучающих данных. По этой причине оценка точности демонстрирует высокую изменчивость, но все же достигает уровня `50%`.

        Имейте в виду, что ваши результаты могут несколько отличаться от представленных, поскольку при таком небольшом объеме обучающих данных результаты сильно зависят от того, какие именно `200` образцов попадут в обучающую выборку при случайном выборе. Если вы получили худший результат, чем мы, попробуйте ради эксперимента отобрать другой набор из `200` случайных образцов (в реальной жизни у вас не будет такой возможности).
        """
    ),
    md(
        """
        Эту же модель можно обучить без загрузки предварительно обученных векторных представлений слов и без замораживания слоя `Embedding`. В этом случае будет обучено представление, узкоспециализированное для входного набора токенов. В такой ситуации обычно получается более мощное представление, чем предварительно обученное, если имеется большой объем обучающих данных. Но в нашем случае имеется всего `200` обучающих образцов. Тем не менее давайте попробуем проделать это.
        """
    ),
    md(
        """
        ## Обучение той же модели без использования уже обученных векторных представлений
        """
    ),
    code(
        """
        model_without_glove = build_imdb_model(
            max_words=max_words,
            embedding_dim=embedding_dim,
            maxlen=maxlen,
        )

        model_without_glove.compile(
            optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        history_without_glove = model_without_glove.fit(
            prepared["x_train"],
            prepared["y_train"],
            epochs=10,
            batch_size=32,
            validation_data=(prepared["x_val"], prepared["y_val"]),
        )
        """
    ),
    code(
        """
        plot_training_history(history_without_glove, "Без GloVe")
        """
    ),
    md(
        """
        Точность на этапе проверки замерла на уровне, близком к `50%`. То есть в данном случае предварительно обученные векторные представления слов выигрывают у вновь обученных. Если увеличить число обучающих образцов, ситуация быстро изменится на противоположную, — попробуйте ради эксперимента.
        """
    ),
    md(
        """
        Наконец, оценим модель на контрольной выборке. Сначала токенизируем контрольные данные.
        """
    ),
    md(
        """
        ## Токенизация данных из контрольной выборки
        """
    ),
    code(
        """
        test_texts, test_labels = load_imdb_split(imdb_dir / "test")

        sequences = prepared["tokenizer"].texts_to_sequences(test_texts)
        x_test = pad_sequences(sequences, maxlen=maxlen)
        y_test = np.asarray(test_labels)

        print("Shape of x_test:", x_test.shape)
        print("Shape of y_test:", y_test.shape)
        """
    ),
    md(
        """
        А затем загрузим и оценим первую модель.
        """
    ),
    md(
        """
        ## Оценка модели на контрольном наборе данных
        """
    ),
    code(
        """
        model.load_weights(pretrained_weights_path)
        test_loss, test_accuracy = model.evaluate(x_test, y_test)

        print(f"Test accuracy: {test_accuracy:.4f}")
        """
    ),
    md(
        """
        Мы получили удручающе низкую точность `56%`. Сложно добиться хорошего результата, имея лишь горстку обучающих образцов!
        """
    ),
    md(
        """
        ## Задание

        Попробуйте увеличить обучающую выборку в самом начале и получить результаты классификации лучше чем `50%`.
        """
    ),
    code(
        """
        assignment_training_samples = 2000

        prepared_assignment = prepare_imdb_sequences(
            texts=train_texts,
            labels=train_labels,
            maxlen=maxlen,
            training_samples=assignment_training_samples,
            validation_samples=validation_samples,
            max_words=max_words,
        )

        assignment_embedding_matrix, assignment_found_words = build_embedding_matrix(
            word_index=prepared_assignment["word_index"],
            embeddings_index=embeddings_index,
            max_words=max_words,
            embedding_dim=embedding_dim,
        )

        print("Размер обучающей выборки:", prepared_assignment["x_train"].shape)
        print("Размер проверочной выборки:", prepared_assignment["x_val"].shape)
        print("Слов с найденными GloVe-векторами:", assignment_found_words)
        """
    ),
    code(
        """
        assignment_model = build_imdb_model(
            max_words=max_words,
            embedding_dim=embedding_dim,
            maxlen=maxlen,
        )

        assignment_model.layers[0].set_weights([assignment_embedding_matrix])
        assignment_model.layers[0].trainable = False

        assignment_model.compile(
            optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        assignment_history = assignment_model.fit(
            prepared_assignment["x_train"],
            prepared_assignment["y_train"],
            epochs=10,
            batch_size=32,
            validation_data=(prepared_assignment["x_val"], prepared_assignment["y_val"]),
        )
        """
    ),
    code(
        """
        plot_training_history(assignment_history, "Увеличенная обучающая выборка")
        """
    ),
    code(
        """
        assignment_test_sequences = prepared_assignment["tokenizer"].texts_to_sequences(test_texts)
        x_test_assignment = pad_sequences(assignment_test_sequences, maxlen=maxlen)

        assignment_test_loss, assignment_test_accuracy = assignment_model.evaluate(
            x_test_assignment,
            y_test,
        )

        print("Размер обучающей выборки:", assignment_training_samples)
        print(f"Test accuracy: {assignment_test_accuracy:.4f}")
        print(f"Лучше 50%: {assignment_test_accuracy > 0.5}")
        """
    ),
    md(
        """
        ## Вывод

        В работе были последовательно рассмотрены прямое кодирование текста, хеширование признаков, обучаемые векторные представления слов и предварительно обученные эмбеддинги `GloVe`. На малой обучающей выборке модель быстро переобучается, поэтому итоговое качество нестабильно. Увеличение обучающей выборки дает модели больше размеченных примеров и позволяет получить результат классификации выше `50%`.
        """
    ),
]


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3.12 (.venv)",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {
    "name": "python",
    "version": "3.12",
}

NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
print(NOTEBOOK_PATH)
