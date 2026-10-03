from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PRACTICE_DIR = Path(__file__).resolve().parent.parent
METHODIC_NOTEBOOK = PRACTICE_DIR / "СП-М-О-АЕЯМИИ-2025-Практика-2-1-БабушкинМВ.ipynb"
ASSIGNMENT_NOTEBOOK = PRACTICE_DIR / "СП-М-О-АЕЯМИИ-2025-Практика-2-2-БабушкинМВ.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


COMMON_IMPORTS = """
from __future__ import annotations

import os
import re
import ssl
import tarfile
import urllib.request
from pathlib import Path

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

DATA_DIR = Path(".data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
np.random.seed(0)
"""


COMMON_HELPERS = """
def get_feature_names(vectorizer) -> np.ndarray:
    \"\"\"Возвращает имена признаков обученного текстового векторизатора.

    Parameters
    ----------
    vectorizer : Any
        Обученный объект `CountVectorizer`.

    Returns
    -------
    np.ndarray
        Массив текстовых признаков.
    \"\"\"
    return np.array(vectorizer.get_feature_names_out())


def topic_words_df(
    components: np.ndarray,
    feature_names: np.ndarray,
    topics: np.ndarray | list[int],
    n_words: int = 10,
) -> pd.DataFrame:
    \"\"\"Формирует таблицу наиболее важных слов для выбранных тем.

    Parameters
    ----------
    components : np.ndarray
        Матрица весов слов по темам.
    feature_names : np.ndarray
        Имена признаков.
    topics : np.ndarray | list[int]
        Индексы тем для вывода.
    n_words : int, default=10
        Количество слов на тему.

    Returns
    -------
    pd.DataFrame
        Таблица с описанием тем через их ключевые слова.
    \"\"\"
    sorting = np.argsort(components, axis=1)[:, ::-1]
    rows = []
    for topic_idx in topics:
        rows.append(
            {
                "Тема": int(topic_idx),
                "Ключевые слова": ", ".join(feature_names[sorting[topic_idx, :n_words]]),
            }
        )
    return pd.DataFrame(rows)


def print_top_documents(
    texts: list[str],
    document_topics: np.ndarray,
    topic_idx: int,
    n_docs: int = 10,
    n_sentences: int = 2,
) -> None:
    \"\"\"Печатает документы, наиболее связанные с указанной темой.

    Parameters
    ----------
    texts : list[str]
        Список исходных документов.
    document_topics : np.ndarray
        Матрица тем документов.
    topic_idx : int
        Индекс темы.
    n_docs : int, default=10
        Число выводимых документов.
    n_sentences : int, default=2
        Количество первых предложений для показа.

    Returns
    -------
    None
        Печатает фрагменты документов.
    \"\"\"
    ranking = np.argsort(document_topics[:, topic_idx])[::-1]
    for doc_idx in ranking[:n_docs]:
        sentences = re.split(r"(?<=[.!?])\\s+", texts[doc_idx].strip())
        excerpt = " ".join(sentences[:n_sentences])
        print(excerpt)
        print()


def plot_topic_weights(
    document_topics: np.ndarray,
    feature_names: np.ndarray,
    sorting: np.ndarray,
    title: str,
) -> None:
    \"\"\"Строит диаграммы суммарных весов тем в корпусе.

    Parameters
    ----------
    document_topics : np.ndarray
        Матрица тем документов.
    feature_names : np.ndarray
        Имена признаков.
    sorting : np.ndarray
        Индексы слов, отсортированных по важности в темах.
    title : str
        Заголовок рисунка.

    Returns
    -------
    None
        Отрисовывает две горизонтальные столбчатые диаграммы.
    \"\"\"
    fig, ax = plt.subplots(1, 2, figsize=(12, 12))
    topic_names = [
        f"{i:>2} " + " ".join(words)
        for i, words in enumerate(feature_names[sorting[:, :2]])
    ]
    topic_weights = np.sum(document_topics, axis=0)

    half = len(topic_names) // 2
    slices = [(0, half), (half, len(topic_names))]
    for col, (start, end) in enumerate(slices):
        ax[col].barh(np.arange(end - start), topic_weights[start:end])
        ax[col].set_yticks(np.arange(end - start))
        ax[col].set_yticklabels(topic_names[start:end], ha="left", va="top")
        ax[col].invert_yaxis()
        yax = ax[col].get_yaxis()
        yax.set_tick_params(pad=140)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
"""


METHODIC_CELLS = [
    md(
        """
        # Практическая работа 2.1. Моделирование тем и кластеризация документов

        Повторение методички на датасете IMDB.
        """
    ),
    md(
        """
        Моделирование тем — это процедура присвоения каждому документу одной или нескольких тем, которая, как правило, выполняется без учителя.

        В данной работе используется метод латентного размещения Дирихле (`Latent Dirichlet Allocation`, `LDA`), который пытается находить группы слов, часто появляющихся вместе, и интерпретировать каждый документ как смесь нескольких тем.
        """
    ),
    code(COMMON_IMPORTS),
    code(
        """
        from sklearn.datasets import load_files

        DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        ARCHIVE_PATH = DATA_DIR / "aclImdb_v1.tar.gz"
        DATASET_DIR = DATA_DIR / "aclImdb"
        """
    ),
    code(COMMON_HELPERS),
    md("## Загрузка данных IMDB"),
    code(
        """
        def download_imdb_dataset(
            dataset_url: str = DATASET_URL,
            archive_path: Path = ARCHIVE_PATH,
            dataset_dir: Path = DATASET_DIR,
        ) -> Path:
            \"\"\"Скачивает и распаковывает набор IMDB при его отсутствии.

            Parameters
            ----------
            dataset_url : str, default=DATASET_URL
                Ссылка на архив с датасетом.
            archive_path : Path, default=ARCHIVE_PATH
                Путь к локальному архиву.
            dataset_dir : Path, default=DATASET_DIR
                Путь к распакованному датасету.

            Returns
            -------
            Path
                Каталог с данными `aclImdb`.
            \"\"\"
            if dataset_dir.exists():
                return dataset_dir

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
        def load_imdb_train(split_dir: Path) -> list[str]:
            \"\"\"Загружает обучающие отзывы IMDB.

            Parameters
            ----------
            split_dir : Path
                Путь к каталогу `train`.

            Returns
            -------
            list[str]
                Список текстов отзывов.
            \"\"\"
            dataset = load_files(
                str(split_dir),
                categories=["neg", "pos"],
                encoding="utf-8",
                decode_error="replace",
            )
            return [doc.replace("<br />", " ") for doc in dataset.data]
        """
    ),
    code(
        """
        dataset_dir = download_imdb_dataset()
        text_train = load_imdb_train(dataset_dir / "train")

        print("Количество документов: {}".format(len(text_train)))
        print("Пример документа:\\n{}".format(text_train[0][:1200]))
        """
    ),
    md(
        """
        Для моделей неконтролируемого обучения, применяющихся к текстовым документам, часто бывает полезно удалить наиболее часто употребляемые слова. Удалим слова, которые появляются по крайней мере в `15%` документов, и ограничим словарь `10000` наиболее частыми словами.
        """
    ),
    code(
        """
        vect = CountVectorizer(max_features=10000, max_df=0.15)
        X = vect.fit_transform(text_train)

        print("форма X: {}".format(X.shape))
        """
    ),
    md(
        """
        Построим модель, выделив `10` тем. Используем метод обучения `batch` и зададим `max_iter=25`.
        """
    ),
    code(
        """
        lda = LatentDirichletAllocation(
            n_components=10,
            learning_method="batch",
            max_iter=25,
            random_state=0,
        )
        document_topics = lda.fit_transform(X)
        """
    ),
    code(
        """
        lda.components_.shape
        """
    ),
    md("## Наиболее важные слова для 10 тем"),
    code(
        """
        sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
        feature_names = get_feature_names(vect)

        display(topic_words_df(lda.components_, feature_names, topics=np.arange(10), n_words=10))
        """
    ),
    md(
        """
        Полученные `10` тем описывают корпус достаточно крупными блоками. На этом шаге обычно видны общие сюжетные линии и часто встречающиеся группы слов, однако границы между темами остаются довольно широкими.
        """
    ),
    md(
        """
        Теперь построим еще одну модель, на этот раз выделив `100` тем.
        """
    ),
    code(
        """
        lda100 = LatentDirichletAllocation(
            n_components=100,
            learning_method="batch",
            max_iter=25,
            random_state=0,
        )
        document_topics100 = lda100.fit_transform(X)
        """
    ),
    md(
        """
        Вывод всех `100` тем был бы слишком громоздким, поэтому выберем лишь некоторые интересные и характерные темы.
        """
    ),
    code(
        """
        topics = np.array([7, 16, 24, 25, 28, 36, 37, 45, 51, 53, 54, 63, 89, 97])
        sorting = np.argsort(lda100.components_, axis=1)[:, ::-1]
        feature_names = get_feature_names(vect)

        display(topic_words_df(lda100.components_, feature_names, topics=topics, n_words=20))
        """
    ),
    md(
        """
        При `100` темах модель выделяет более узкие и специфичные наборы слов. Это помогает точнее рассмотреть структуру корпуса, но интерпретация таких тем становится менее очевидной и требует дополнительной проверки по самим документам.
        """
    ),
    md(
        """
        Посмотрим, какие документы были отнесены к теме `45`.
        """
    ),
    code(
        """
        print_top_documents(text_train, document_topics100, topic_idx=45, n_docs=10, n_sentences=2)
        """
    ),
    md(
        """
        Просмотр документов нужен для проверки содержательной интерпретации темы. Если ключевые слова действительно отражают смысл темы, то и документы с наибольшим весом этой темы должны быть тематически близкими.
        """
    ),
    md(
        """
        Еще один способ исследовать темы — посмотреть, какой вес получает каждая тема в целом, просуммировав `document_topics` по всем документам.
        """
    ),
    code(
        """
        plot_topic_weights(
            document_topics100,
            feature_names,
            sorting,
            title="Суммарные веса тем для IMDB",
        )
        """
    ),
    md(
        """
        Суммарные веса показывают, какие темы наиболее заметны во всем корпусе IMDB. Чем выше столбец, тем больший вклад соответствующая тема вносит в описание всего набора отзывов.
        """
    ),
    md(
        """
        ## Вывод

        На датасете IMDB были построены модели LDA с `10` и `100` темами. При малом числе тем они получаются более широкими и обобщенными, а при большом числе тем — более конкретными, но и более сложными для интерпретации.
        """
    ),
]


ASSIGNMENT_CELLS = [
    md(
        """
        # Практическая работа 2.2. Задание 1

        Моделирование тем на другом общедоступном датасете с текстовыми данными.
        """
    ),
    md(
        """
        В качестве другого общедоступного датасета выбран `20 Newsgroups`. Ниже повторяются те же шаги, что и в методичке: векторизация, LDA на `10` и `100` тем, просмотр ключевых слов, анализ документов выбранной темы и суммарных весов тем.
        """
    ),
    code(COMMON_IMPORTS),
    code(
        """
        from sklearn.datasets import fetch_20newsgroups

        SKLEARN_DATA_HOME = DATA_DIR / "sklearn_data"
        SKLEARN_DATA_HOME.mkdir(parents=True, exist_ok=True)
        ssl._create_default_https_context = ssl._create_unverified_context
        """
    ),
    code(COMMON_HELPERS),
    md("## Загрузка датасета 20 Newsgroups"),
    code(
        """
        def load_newsgroups() -> list[str]:
            \"\"\"Загружает тексты из датасета 20 Newsgroups.

            Returns
            -------
            list[str]
                Список документов из набора новостей.
            \"\"\"
            dataset = fetch_20newsgroups(
                subset="all",
                remove=("headers", "footers", "quotes"),
                data_home=str(SKLEARN_DATA_HOME),
            )
            return [doc for doc in dataset.data if doc.strip()]
        """
    ),
    code(
        """
        text_train = load_newsgroups()

        print("Количество документов: {}".format(len(text_train)))
        print("Пример документа:\\n{}".format(text_train[0][:1200]))
        """
    ),
    md(
        """
        Повторим ту же схему векторизации: удалим слишком частые слова (`max_df=0.15`) и ограничим словарь `10000` наиболее частыми признаками.
        """
    ),
    code(
        """
        vect = CountVectorizer(max_features=10000, max_df=0.15, stop_words="english")
        X = vect.fit_transform(text_train)

        print("форма X: {}".format(X.shape))
        """
    ),
    md("## Модель LDA с 10 темами"),
    code(
        """
        lda = LatentDirichletAllocation(
            n_components=10,
            learning_method="batch",
            max_iter=25,
            random_state=0,
        )
        document_topics = lda.fit_transform(X)
        """
    ),
    code(
        """
        sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
        feature_names = get_feature_names(vect)

        display(topic_words_df(lda.components_, feature_names, topics=np.arange(10), n_words=10))
        """
    ),
    md(
        """
        При выделении `10` тем корпус `20 Newsgroups` тоже разбивается на достаточно крупные смысловые блоки. На этом уровне удобно увидеть общую структуру коллекции, но отдельные темы еще могут смешивать близкие по содержанию сюжеты.
        """
    ),
    md("## Модель LDA с 100 темами"),
    code(
        """
        lda100 = LatentDirichletAllocation(
            n_components=100,
            learning_method="batch",
            max_iter=25,
            random_state=0,
        )
        document_topics100 = lda100.fit_transform(X)
        """
    ),
    code(
        """
        topic_strength = np.sum(document_topics100, axis=0)
        topics = np.argsort(topic_strength)[-14:]
        topics = np.sort(topics)
        sorting = np.argsort(lda100.components_, axis=1)[:, ::-1]
        feature_names = get_feature_names(vect)

        display(topic_words_df(lda100.components_, feature_names, topics=topics, n_words=20))
        """
    ),
    md(
        """
        При `100` темах модель выделяет уже более узкие подтемы. Это делает представление корпуса детальнее, но одновременно повышает требования к интерпретации: часть тем может отличаться лишь небольшими наборами характерных слов.
        """
    ),
    md(
        """
        Посмотрим документы для наиболее весомой темы.
        """
    ),
    code(
        """
        dominant_topic = int(np.argmax(np.sum(document_topics100, axis=0)))
        print("Выбранная тема:", dominant_topic)
        print_top_documents(text_train, document_topics100, topic_idx=dominant_topic, n_docs=10, n_sentences=2)
        """
    ),
    md(
        """
        Анализ документов с наибольшим весом темы помогает убедиться, что найденная тема действительно имеет содержательный смысл, а не описывается только формальным набором часто встречающихся слов.
        """
    ),
    md(
        """
        Просуммируем веса тем по всем документам корпуса и построим диаграммы.
        """
    ),
    code(
        """
        plot_topic_weights(
            document_topics100,
            feature_names,
            sorting,
            title="Суммарные веса тем для 20 Newsgroups",
        )
        """
    ),
    md(
        """
        Диаграмма суммарных весов показывает, какие темы оказываются наиболее выраженными во всем корпусе новостных сообщений. Это позволяет увидеть, какие направления обсуждений доминируют в выбранном датасете.
        """
    ),
    md(
        """
        ## Вывод

        На другом публичном текстовом датасете был повторен тот же сценарий тематического моделирования. Модель LDA снова показала, что при большем числе тем структура корпуса раскрывается подробнее, а ключевые слова и документы с высоким весом темы помогают интерпретировать найденные топики.
        """
    ),
]


def build_notebook(path: Path, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {
        "name": "python",
        "version": "3.14",
    }
    path.write_text(nbf.writes(nb), encoding="utf-8")


build_notebook(METHODIC_NOTEBOOK, METHODIC_CELLS)
build_notebook(ASSIGNMENT_NOTEBOOK, ASSIGNMENT_CELLS)

print(METHODIC_NOTEBOOK)
print(ASSIGNMENT_NOTEBOOK)
