from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PRACTICE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PRACTICE_DIR / "СП-М-О-АЕЯМИИ-2025-Практика-5-2-БабушкинМВ.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # Практическая работа 5.2. Генерация текста на русском языке

        Генерация текста на основе корпуса произведений А. С. Пушкина.
        """
    ),
    md(
        """
        ## Задание

        Подобрать длинное художественное произведение на русском и на английском языке, сравнимое с исходным документом – трудами Ницше. Произвести все шаги из этой практики и получит сгенерированные последовательности на основе ваших документов.

        !!Обратите внимание на представление текстовых данных в документе с трудами Ницше. Ваш документ должен быть представлен в таком же фориате.
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

        import numpy as np
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        RANDOM_STATE = 42
        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        tf.random.set_seed(RANDOM_STATE)

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices([], "GPU")
        """
    ),
    md(
        """
        ## Подготовка данных

        В качестве русского корпуса используется текстовый файл с произведениями А. С. Пушкина. Текст загружается в формате `txt`, приводится к нижнему регистру и далее обрабатывается посимвольно.
        """
    ),
    code(
        """
        pushkin_url = "https://raw.githubusercontent.com/MerkulovDaniil/TensorFlow_and_Keras_crash_course/master/pushkin.txt"
        path = keras.utils.get_file(
            "pushkin.txt",
            origin=pushkin_url,
        )

        text = Path(path).read_text(encoding="utf-8").lower()
        print("Corpus length:", len(text))
        print(text[:500])
        """
    ),
    code(
        """
        maxlen = 60
        step = 3
        sentences = []
        next_chars = []

        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])

        print("Number of sequences:", len(sentences))

        chars = sorted(list(set(text)))
        print("Unique characters:", len(chars))

        char_indices = dict((char, chars.index(char)) for char in chars)
        indices_char = dict((index, char) for char, index in char_indices.items())

        print("Vectorization...")
        x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
        y = np.zeros((len(sentences), len(chars)), dtype=bool)

        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1
        """
    ),
    md(
        """
        ## Конструирование сети

        Используется та же структура, что и в базовом примере: один слой `LSTM` и полносвязный слой `Dense` с функцией активации `softmax` для выбора следующего символа.
        """
    ),
    code(
        """
        model = keras.models.Sequential()
        model.add(keras.Input(shape=(maxlen, len(chars))))
        model.add(layers.LSTM(128))
        model.add(layers.Dense(len(chars), activation="softmax"))
        model.summary()
        """
    ),
    code(
        """
        optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer)
        """
    ),
    code(
        """
        def sample(preds, temperature=1.0):
            \"\"\"Возвращает индекс следующего символа с учетом температуры softmax.\"\"\"
            preds = np.asarray(preds).astype("float64")
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)
        """
    ),
    md(
        """
        ## Обучение модели и генерация текста

        После каждой эпохи генерируются последовательности при разных значениях температуры. Полный текст генерации сохраняется во внешний файл, а в notebook выводится только ход выполнения.
        """
    ),
    code(
        """
        epochs_count = 20
        output_path = PRACTICE_DIR / "СП-М-О-АЕЯМИИ-2025-Практика-5-2-БабушкинМВ.txt"

        with output_path.open("w", encoding="utf-8") as output_file:
            for epoch in range(1, epochs_count + 1):
                print(f"Эпоха {epoch}/{epochs_count}")
                history = model.fit(x, y, batch_size=128, epochs=1, verbose=0)
                print(f"loss: {history.history['loss'][-1]:.4f}")

                output_file.write(f"\\n\\nЭпоха {epoch}\\n")

                start_index = random.randint(0, len(text) - maxlen - 1)
                generated_text = text[start_index: start_index + maxlen]
                output_file.write('--- Generating with seed: "' + generated_text + '"\\n')

                for temperature in [0.2, 0.5, 1.0, 1.2]:
                    print("  temperature:", temperature)
                    output_file.write(f"\\n------ temperature: {temperature}\\n")
                    output_file.write(generated_text)

                    for i in range(400):
                        sampled = np.zeros((1, maxlen, len(chars)))
                        for t, char in enumerate(generated_text):
                            sampled[0, t, char_indices[char]] = 1.0

                        preds = model.predict(sampled, verbose=0)[0]
                        next_index = sample(preds, temperature)
                        next_char = chars[next_index]

                        generated_text += next_char
                        generated_text = generated_text[1:]

                        output_file.write(next_char)
                    output_file.write("\\n")
                output_file.flush()
                print()

        print(f"Результаты генерации сохранены в файл: {output_path.name}")
        """
    ),
    md(
        """
        ## Вывод

        На русском художественном корпусе была обучена посимвольная модель `LSTM`, предсказывающая следующий символ по предыдущим символам последовательности. Температура управляет случайностью выбора: при малых значениях текст становится более предсказуемым, при больших — более разнообразным и менее устойчивым.
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


notebook = nbf.v4.new_notebook()
notebook["cells"] = cells
notebook["metadata"] = NOTEBOOK_METADATA

nbf.write(notebook, NOTEBOOK_PATH)
print(f"Notebook written to {NOTEBOOK_PATH}")
