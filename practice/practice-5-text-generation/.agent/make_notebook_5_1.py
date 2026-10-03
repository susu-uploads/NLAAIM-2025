from __future__ import annotations

import zipfile
from pathlib import Path
from textwrap import dedent
import xml.etree.ElementTree as ET

import nbformat as nbf


PRACTICE_DIR = Path(__file__).resolve().parent.parent
DOCX_PATH = PRACTICE_DIR / "Генерирование текста с помощью LSTM keras.docx"
NOTEBOOK_PATH = PRACTICE_DIR / "СП-М-О-АЕЯМИИ-2025-Практика-5-1-БабушкинМВ.ipynb"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


def read_docx_paragraphs(path: Path) -> dict[int, str]:
    """Возвращает непустые абзацы Word-документа по индексам элементов body."""
    with zipfile.ZipFile(path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))

    body = document.find("w:body", NS)
    paragraphs: dict[int, str] = {}

    for index, element in enumerate(list(body)):
        if element.tag.split("}")[-1] != "p":
            continue

        parts: list[str] = []
        for node in element.iter():
            tag = node.tag.split("}")[-1]
            if tag == "t":
                parts.append(node.text or "")
            elif tag == "tab":
                parts.append("\t")
            elif tag == "br":
                parts.append("\n")

        text = "".join(parts).strip()
        if text:
            paragraphs[index] = text

    return paragraphs


PARAGRAPHS = read_docx_paragraphs(DOCX_PATH)


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def md_doc(*indices: int):
    return md("\n\n".join(PARAGRAPHS[index] for index in indices))


def figure_caption(index: int) -> str:
    return PARAGRAPHS[index].removeprefix("Рис. ").strip()


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        f"""
        # Практическая работа 5.1. Генерация текста

        {PARAGRAPHS[0]}
        """
    ),
    md_doc(2, 3, 4, 5, 6, 7),
    md(
        f"""
        ## {PARAGRAPHS[9]}

        {PARAGRAPHS[10]}

        {PARAGRAPHS[11]}

        <div align="center">
          <img src="text_generation_process.png" alt="{figure_caption(13)}" width="720"/>
          <br/>
          <em>{figure_caption(13)}</em>
        </div>
        """
    ),
    md(
        f"""
        ## {PARAGRAPHS[15]}

        {PARAGRAPHS[16]}

        {PARAGRAPHS[17]}

        {PARAGRAPHS[18]}

        {PARAGRAPHS[19]}

        {PARAGRAPHS[20]}
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
    code(
        """
        import numpy as np

        def reweight_distribution(original_distribution, temperature=0.5): #original_distribution — это одномерный массив Numpy значений вероятностей, сумма которых должна быть равна 1
            \"\"\"Возвращает взвешенную версию распределения вероятностей.\"\"\"
            distribution = np.log(original_distribution) / temperature
            distribution = np.exp(distribution)
            return distribution / np.sum(distribution) #Возвращает новую, взвешенную версию оригинального распределения. Сумма вероятностей в новом распределении может получиться больше 1, поэтому разделим элементы вектора на сумму, чтобы получить новое распределение
        """
    ),
    md(
        f"""
        {PARAGRAPHS[27]}

        <div align="center">
          <img src="temperature_sampling.png" alt="{figure_caption(29)}" width="720"/>
          <br/>
          <em>{figure_caption(29)}</em>
        </div>
        """
    ),
    md(
        f"""
        ## {PARAGRAPHS[31]}

        {PARAGRAPHS[32]}

        ## {PARAGRAPHS[34]}

        {PARAGRAPHS[35]}

        {PARAGRAPHS[36]}
        """
    ),
    code(
        """
        path = keras.utils.get_file(
            "nietzsche.txt",
            origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt",
        )
        text = Path(path).read_text(encoding="utf-8").lower()
        print("Corpus length:", len(text))
        """
    ),
    md_doc(44, 45),
    code(
        """
        maxlen = 60 #Извлечение последовательностей по 60 символов
        step = 3 #Новые последовательности выбираются через каждые 3 символа
        sentences = [] #Хранение извлеченных последовательностей
        next_chars = [] #Хранение целей (символов, следующих за последовательностями)
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
        print("Number of sequences:", len(sentences))

        chars = sorted(list(set(text))) #Список уникальных символов в корпусе
        print("Unique characters:", len(chars))
        char_indices = dict((char, chars.index(char)) for char in chars) # Словарь, отображающий уникальные символы в их индексы в списке «chars»

        print("Vectorization...")
        x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
        y = np.zeros((len(sentences), len(chars)), dtype=bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1 #Прямое кодирование символов в бинарные массивы
        """
    ),
    md_doc(65, 66, 67),
    code(
        """
        model = keras.models.Sequential()
        model.add(keras.Input(shape=(maxlen, len(chars))))
        model.add(layers.LSTM(128))
        model.add(layers.Dense(len(chars), activation="softmax"))
        model.summary()
        """
    ),
    md_doc(73, 74),
    code(
        """
        optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer)
        """
    ),
    md_doc(78, 79, 80, 81, 82, 83, 84, 85),
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
    md_doc(94, 95),
    code(
        """
        import random

        output_path = PRACTICE_DIR / "СП-М-О-АЕЯМИИ-2025-Практика-5-1-БабушкинМВ.txt"

        with output_path.open("w", encoding="utf-8") as output_file:
            for epoch in range(1, 60):    # Обучение модели в течение 60 эпох
                print(f"Эпоха {epoch}/59")
                history = model.fit(x, y, batch_size=128, epochs=1, verbose=0)
                print(f"loss: {history.history['loss'][-1]:.4f}")

                output_file.write(f"\\n\\nЭпоха {epoch}\\n")

                # выбор случайного начального текста
                start_index = random.randint(0, len(text) - maxlen - 1)
                generated_text = text[start_index: start_index + maxlen]
                output_file.write('--- Generating with seed: "' + generated_text + '"\\n')

                for temperature in [0.2, 0.5, 1.0, 1.2]:
                    print("  temperature:", temperature)
                    output_file.write(f"\\n------ temperature: {temperature}\\n")
                    output_file.write(generated_text)

                    # Генерация 400 символов, начиная с начального текста
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
    md_doc(181, 182),
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
