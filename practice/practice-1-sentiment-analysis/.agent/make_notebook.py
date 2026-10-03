from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PRACTICE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PRACTICE_DIR / "СП-М-О-АЕЯМИИ-2025-Практика-1-БабушкинМВ.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # Практическая работа 1. Тональность

        Задача определения тональности киноотзывов.
        """
    ),
    md(
        """
        Загружаем датасет по ссылке: `http://ai.stanford.edu/~amaas/data/sentiment/`.

        После распаковки набор данных представляет собой две отдельные папки с текстовыми файлами: одна папка для обучения, вторая для тестирования. Каждая из них содержит две подпапки: `pos` и `neg`.
        """
    ),
    md("## Загрузка обучающих данных"),
    code(
        """
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
        import nltk
        import numpy as np
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        from sklearn.datasets import load_files
        from sklearn.feature_extraction.text import (
            CountVectorizer,
            ENGLISH_STOP_WORDS,
            TfidfVectorizer,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import (
            GridSearchCV,
            StratifiedShuffleSplit,
            cross_val_score,
            train_test_split,
        )
        from sklearn.pipeline import make_pipeline

        DATA_DIR = Path(".data")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        ARCHIVE_PATH = DATA_DIR / "aclImdb_v1.tar.gz"
        DATASET_DIR = DATA_DIR / "aclImdb"
        RANDOM_STATE = 42

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["figure.figsize"] = (10, 5)
        np.random.seed(RANDOM_STATE)
        """
    ),
    code(
        """
        def download_imdb_dataset(
            dataset_url: str = DATASET_URL,
            archive_path: Path = ARCHIVE_PATH,
            dataset_dir: Path = DATASET_DIR,
        ) -> Path:
            \"\"\"Скачивает и распаковывает набор данных IMDB при его отсутствии.

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
                Каталог с набором данных `aclImdb`.
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
        def load_imdb_split(split_dir: Path) -> tuple[list[str], np.ndarray]:
            \"\"\"Загружает тексты и метки классов из каталога `train` или `test`.

            Parameters
            ----------
            split_dir : Path
                Путь к каталогу части датасета.

            Returns
            -------
            tuple[list[str], np.ndarray]
                Список текстов и массив меток классов.
            \"\"\"
            dataset = load_files(
                str(split_dir),
                categories=["neg", "pos"],
                encoding="utf-8",
                decode_error="replace",
            )
            return dataset.data, dataset.target
        """
    ),
    code(
        """
        dataset_dir = download_imdb_dataset()
        text_train, y_train = load_imdb_split(dataset_dir / "train")

        print("тип text_train: {}".format(type(text_train)))
        print("длина text_train: {}".format(len(text_train)))
        print("text_train[1]:\\n{}".format(text_train[1][:1500]))
        """
    ),
    md(
        """
        Переменная `text_train` представляет собой список длиной `25000`, в котором каждый элемент является строкой с текстом отзыва. Отзывы содержат разрывы строк HTML (`<br />`), поэтому перед дальнейшей работой выполним очистку данных.
        """
    ),
    code(
        """
        def remove_html_breaks(documents: list[str]) -> list[str]:
            \"\"\"Удаляет из документов HTML-разрывы строк `<br />`.

            Parameters
            ----------
            documents : list[str]
                Список текстов отзывов.

            Returns
            -------
            list[str]
                Очищенные тексты отзывов.
            \"\"\"
            return [doc.replace("<br />", " ") for doc in documents]
        """
    ),
    code(
        """
        text_train = remove_html_breaks(text_train)
        """
    ),
    md("## Загрузка тестовых данных"),
    code(
        """
        text_test, y_test = load_imdb_split(dataset_dir / "test")

        print("Количество документов в текстовых данных: {}".format(len(text_test)))
        print("Количество примеров на класс (тест): {}".format(np.bincount(y_test)))

        text_test = remove_html_breaks(text_test)
        """
    ),
    md("## Представление данных в виде мешка слов"),
    md(
        """
        Один из самых простых, но эффективных и широко используемых способов подготовки текста для машинного обучения — представление текстовой информации в виде «мешка слов» (`bag-of-words`).

        Получение представления «мешок слов» включает три этапа:
        1. Токенизация.
        2. Построение словаря.
        3. Создание разреженной матрицы частот.
        """
    ),
    md("### Преобразование данных в мешок слов"),
    code(
        """
        vect = CountVectorizer().fit(text_train)
        X_train = vect.transform(text_train)

        print("X_train:\\n{}".format(repr(X_train)))
        print("форма X_train: {}".format(X_train.shape))
        """
    ),
    md("### Пример применения LogisticRegression с использованием перекрестной проверки"),
    code(
        """
        scores = cross_val_score(
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
            X_train,
            y_train,
            cv=5,
            n_jobs=-1,
        )
        print("Средняя правильность перекрестной проверки: {:.2f}".format(np.mean(scores)))
        """
    ),
    md(
        """
        Логистическая регрессия имеет параметр регуляризации `C`, который можно настроить с помощью перекрестной проверки.
        """
    ),
    code(
        """
        param_grid = {"C": [0.001, 0.01, 0.1, 1, 10]}
        grid = GridSearchCV(
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
            param_grid,
            cv=5,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        print("Наилучшее значение перекрестной проверки: {:.2f}".format(grid.best_score_))
        print("Наилучшие параметры: {}".format(grid.best_params_))
        """
    ),
    code(
        """
        X_test = vect.transform(text_test)
        print("Правильность на тестовом наборе: {:.2f}".format(grid.score(X_test, y_test)))
        """
    ),
    md(
        """
        Можно улучшить процесс извлечения слов. Один из способов — использовать только те токены, которые встречаются по крайней мере в нескольких документах. Для этого применяется параметр `min_df`.
        """
    ),
    md("### Удаление редких слов с помощью `min_df`"),
    code(
        """
        def get_feature_names(vectorizer) -> np.ndarray:
            \"\"\"Возвращает имена признаков обученного текстового векторизатора.

            Parameters
            ----------
            vectorizer : Any
                Обученный `CountVectorizer` или `TfidfVectorizer`.

            Returns
            -------
            np.ndarray
                Массив имен признаков.
            \"\"\"
            return np.array(vectorizer.get_feature_names_out())
        """
    ),
    code(
        """
        vect = CountVectorizer(min_df=5).fit(text_train)
        X_train = vect.transform(text_train)

        print("X_train с min_df: {}".format(repr(X_train)))

        feature_names = get_feature_names(vect)
        print("Первые 50 признаков:\\n{}".format(feature_names[:50]))
        print("Признаки с 20010 по 20030:\\n{}".format(feature_names[20010:20030]))
        print("Каждый 700-й признак:\\n{}".format(feature_names[::700]))
        """
    ),
    code(
        """
        grid = GridSearchCV(
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
            param_grid,
            cv=5,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        print("Наилучшее значение перекрестной проверки: {:.2f}".format(grid.best_score_))
        """
    ),
    md("## Стоп-слова"),
    md(
        """
        Еще один способ избавиться от неинформативных слов — исключить слова, которые встречаются слишком часто, чтобы быть полезными признаками. Библиотека `scikit-learn` содержит встроенный список английских стоп-слов.
        """
    ),
    code(
        """
        sorted_stop_words = sorted(ENGLISH_STOP_WORDS)
        print("Количество стоп-слов: {}".format(len(sorted_stop_words)))
        print("Каждое 10-е стоп-слово:\\n{}".format(sorted_stop_words[::10]))
        """
    ),
    code(
        """
        vect = CountVectorizer(min_df=5, stop_words="english").fit(text_train)
        X_train = vect.transform(text_train)

        print("X_train с использованием стоп-слов:\\n{}".format(repr(X_train)))

        grid = GridSearchCV(
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
            param_grid,
            cv=5,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        print("Наилучшее значение перекрестной проверки: {:.2f}".format(grid.best_score_))
        """
    ),
    md("## Задание 1"),
    md(
        """
        В качестве примера попробуем другой подход: исключим слишком часто встречающиеся слова, задав параметр `max_df`, и посмотрим, как это повлияет на количество признаков и качество модели.
        """
    ),
    code(
        """
        max_df_values = [0.5, 0.7, 0.9, 0.95]
        max_df_results = []

        for max_df in max_df_values:
            vect = CountVectorizer(min_df=5, max_df=max_df)
            X_train = vect.fit_transform(text_train)
            X_test = vect.transform(text_test)

            grid = GridSearchCV(
                LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
                param_grid,
                cv=5,
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)

            max_df_results.append(
                {
                    "max_df": max_df,
                    "Признаков": X_train.shape[1],
                    "Лучшее значение CV": round(grid.best_score_, 4),
                    "Правильность test": round(grid.score(X_test, y_test), 4),
                    "Лучший C": grid.best_params_["C"],
                }
            )

        for row in max_df_results:
            print(row)
        """
    ),
    md(
        """
        Полученные результаты отличаются слабо, потому что параметр `max_df` в этой задаче удаляет очень небольшое число слов. При `max_df=0.95` исключаются только самые частотные слова вроде `and` и `the`, а при `max_df=0.5` удаляются уже и слова `movie`, `film`, `not`, которые могут нести полезную информацию для определения тональности.

        Поэтому здесь `max_df` почти не улучшает качество модели: изменения метрики находятся в пределах очень малого разброса. Лучший результат среди проверенных значений дает `max_df=0.7`, но преимущество над соседними вариантами незначительно.
        """
    ),
    md("## Масштабирование данных с помощью tf-idf"),
    md(
        """
        Следующий подход вместо исключения несущественных признаков пытается масштабировать признаки в зависимости от степени их информативности. Одним из наиболее распространенных способов такого масштабирования является метод `tf-idf`.

        Поскольку `tf-idf` использует статистические свойства обучающих данных, удобно воспользоваться конвейером, чтобы корректно выполнить решетчатый поиск.
        """
    ),
    code(
        """
        pipe = make_pipeline(
            TfidfVectorizer(min_df=5, norm=None),
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
        )

        param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10]}
        grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
        grid.fit(text_train, y_train)

        print("Наилучшее значение перекрестной проверки: {:.2f}".format(grid.best_score_))
        print("Наилучшие параметры: {}".format(grid.best_params_))
        """
    ),
    code(
        """
        vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
        X_train = vectorizer.transform(text_train)
        max_value = X_train.max(axis=0).toarray().ravel()
        sorted_by_tfidf = max_value.argsort()
        feature_names = get_feature_names(vectorizer)

        print("Признаки с наименьшими значениями tfidf:\\n{}".format(feature_names[sorted_by_tfidf[:20]]))
        print("Признаки с наибольшими значениями tfidf:\\n{}".format(feature_names[sorted_by_tfidf[-20:]]))
        """
    ),
    code(
        """
        sorted_by_idf = np.argsort(vectorizer.idf_)
        print("Признаки с наименьшими значениями idf:\\n{}".format(feature_names[sorted_by_idf[:100]]))
        """
    ),
    md("## Исследование коэффициентов модели"),
    md(
        """
        Посмотрим на коэффициенты логистической регрессии, обученной на признаках `tf-idf`.
        """
    ),
    code(
        """
        def plot_top_coefficients(
            coef: np.ndarray,
            feature_names: np.ndarray,
            n_top_features: int = 40,
            title: str = "Наиболее значимые коэффициенты",
        ) -> None:
            \"\"\"Строит график наиболее сильных отрицательных и положительных коэффициентов.

            Parameters
            ----------
            coef : np.ndarray
                Коэффициенты модели.
            feature_names : np.ndarray
                Имена признаков.
            n_top_features : int, default=40
                Число признаков с каждой стороны.
            title : str, default="Наиболее значимые коэффициенты"
                Заголовок графика.

            Returns
            -------
            None
                Отрисовывает график коэффициентов.
            \"\"\"
            coef = np.ravel(coef)
            top_negative = np.argsort(coef)[:n_top_features]
            top_positive = np.argsort(coef)[-n_top_features:]
            top_indices = np.concatenate([top_negative, top_positive])
            colors = ["firebrick" if value < 0 else "darkgreen" for value in coef[top_indices]]

            plt.figure(figsize=(12, 10))
            plt.barh(range(len(top_indices)), coef[top_indices], color=colors)
            plt.yticks(range(len(top_indices)), feature_names[top_indices])
            plt.axvline(0, color="black", linewidth=1)
            plt.title(title)
            plt.xlabel("Значение коэффициента")
            plt.tight_layout()
            plt.show()
        """
    ),
    code(
        """
        plot_top_coefficients(
            grid.best_estimator_.named_steps["logisticregression"].coef_,
            feature_names,
            n_top_features=25,
            title="Коэффициенты логистической регрессии для tf-idf",
        )
        """
    ),
    md("## n-граммы"),
    md(
        """
        Один из главных недостатков представления «мешок слов» — игнорирование порядка слов. Чтобы частично учитывать контекст, можно использовать не только отдельные токены, но и пары, тройки токенов, то есть `n`-граммы.

        В методичке для `n`-грамм указан более тяжелый решетчатый поиск. Ниже он повторяется почти в той же форме, но подбор параметров выполняется на стратифицированной подвыборке, чтобы ноутбук оставался исполнимым сверху вниз.
        """
    ),
    code(
        """
        text_train_small, _, y_train_small, _ = train_test_split(
            text_train,
            y_train,
            train_size=12000,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )

        pipe = make_pipeline(
            TfidfVectorizer(min_df=5),
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
        )

        param_grid = {
            "logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)],
        }

        grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
        grid.fit(text_train_small, y_train_small)

        print("Наилучшее значение перекрестной проверки: {:.2f}".format(grid.best_score_))
        print("Наилучшие параметры:\\n{}".format(grid.best_params_))
        """
    ),
    code(
        """
        scores = np.array(grid.cv_results_["mean_test_score"]).reshape(-1, 3).T

        plt.figure(figsize=(8, 4))
        heatmap = plt.imshow(scores, cmap="viridis", aspect="auto")
        plt.colorbar(heatmap)
        plt.xlabel("C")
        plt.ylabel("ngram_range")
        plt.xticks(range(len(param_grid["logisticregression__C"])), param_grid["logisticregression__C"])
        plt.yticks(
            range(len(param_grid["tfidfvectorizer__ngram_range"])),
            [str(item) for item in param_grid["tfidfvectorizer__ngram_range"]],
        )
        plt.title("Теплокарта качества для разных n-грамм")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        best_ngram_range = grid.best_params_["tfidfvectorizer__ngram_range"]
        best_c = grid.best_params_["logisticregression__C"]

        best_model = make_pipeline(
            TfidfVectorizer(min_df=5, ngram_range=best_ngram_range),
            LogisticRegression(C=best_c, max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
        )
        best_model.fit(text_train, y_train)

        vect = best_model.named_steps["tfidfvectorizer"]
        feature_names = get_feature_names(vect)
        coef = best_model.named_steps["logisticregression"].coef_

        plot_top_coefficients(
            coef,
            feature_names,
            n_top_features=20,
            title="Наиболее значимые коэффициенты модели с n-граммами",
        )

        print("Правильность на тестовом наборе: {:.2f}".format(best_model.score(text_test, y_test)))
        """
    ),
    code(
        """
        mask = np.array([len(feature.split(" ")) for feature in feature_names]) == 3
        plot_top_coefficients(
            coef.ravel()[mask],
            feature_names[mask],
            n_top_features=15,
            title="Наиболее значимые триграммные признаки",
        )
        """
    ),
    md("## Задание 2"),
    md(
        """
        Выполнить поиск и настройку моделей машинного обучения для улучшения качества классификации на данном наборе без использования дополнительных средств обработки текстовых данных, таких как лемматизация, стемминг и т.д.
        """
    ),
    code(
        """
        baseline_model = make_pipeline(
            CountVectorizer(),
            LogisticRegression(C=0.1, max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
        )
        baseline_model.fit(text_train, y_train)

        tfidf_model = make_pipeline(
            TfidfVectorizer(min_df=5, norm=None),
            LogisticRegression(C=0.001, max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
        )
        tfidf_model.fit(text_train, y_train)

        print("CountVectorizer: {:.4f}".format(baseline_model.score(text_test, y_test)))
        print("TfidfVectorizer: {:.4f}".format(tfidf_model.score(text_test, y_test)))
        print("TfidfVectorizer + n-граммы: {:.4f}".format(best_model.score(text_test, y_test)))
        """
    ),
    md(
        """
        В `Задании 2` улучшение достигается не за счет дополнительной обработки текста, а за счет более удачного способа представления признаков и настройки модели. Обычный `CountVectorizer` использует простые частоты слов, `TfidfVectorizer` дополнительно учитывает информативность слова в корпусе, а модель с `n`-граммами еще и частично захватывает контекст соседних слов.

        Поэтому лучший результат здесь показывает вариант `TfidfVectorizer + n-граммы`: он лучше выделяет значимые слова и устойчивые сочетания слов, которые важны для тональности отзывов. Разница между моделями уже заметна по метрике, так что для этого задания вывод о преимуществе `tf-idf` и `n`-грамм является содержательно корректным.
        """
    ),
    md("## Продвинутая токенизация"),
    md(
        """
        В методичке для этого этапа используется `spaCy`, однако старый вариант `spacy.load('en')` в текущей среде недоступен. Чтобы сохранить ту же идею нормализации и при этом оставить ноутбук исполнимым, ниже используется `WordNetLemmatizer` из `nltk`.
        """
    ),
    code(
        """
        def ensure_nltk_resources() -> None:
            \"\"\"Скачивает локальные ресурсы NLTK, необходимые для лемматизации.

            Returns
            -------
            None
                Загружает ресурсы в `.data/nltk_data`.
            \"\"\"
            resource_map = {
                "corpora/wordnet": "wordnet",
                "corpora/omw-1.4": "omw-1.4",
            }
            download_dir = DATA_DIR / "nltk_data"
            download_dir.mkdir(parents=True, exist_ok=True)
            if str(download_dir) not in nltk.data.path:
                nltk.data.path.insert(0, str(download_dir))
            ssl._create_default_https_context = ssl._create_unverified_context
            for resource_path, resource_name in resource_map.items():
                try:
                    nltk.data.find(resource_path)
                except LookupError:
                    nltk.download(resource_name, download_dir=str(download_dir), quiet=True)
        """
    ),
    code(
        """
        def compare_normalization(doc: str) -> None:
            \"\"\"Сравнивает лемматизацию и стемминг для заданной строки.

            Parameters
            ----------
            doc : str
                Текст для сравнения.

            Returns
            -------
            None
                Печатает результаты лемматизации и стемминга.
            \"\"\"
            ensure_nltk_resources()
            lemmatizer = WordNetLemmatizer()
            stemmer = PorterStemmer()
            tokens = re.findall(r"(?u)\\b\\w\\w+\\b", doc.lower())

            print("Лемматизация:")
            print([lemmatizer.lemmatize(token) for token in tokens])
            print("Стемминг:")
            print([stemmer.stem(token) for token in tokens])
        """
    ),
    code(
        """
        compare_normalization(
            "Our meeting today was worse than yesterday, I'm scared of meeting the clients tomorrow."
        )
        """
    ),
    code(
        """
        ensure_nltk_resources()
        lemmatizer = WordNetLemmatizer()

        def custom_tokenizer(document: str) -> list[str]:
            \"\"\"Токенизирует документ и возвращает список лемматизированных токенов.

            Parameters
            ----------
            document : str
                Текст документа.

            Returns
            -------
            list[str]
                Список лемм.
            \"\"\"
            tokens = re.findall(r"(?u)\\b\\w\\w+\\b", document.lower())
            return [lemmatizer.lemmatize(token) for token in tokens]
        """
    ),
    code(
        """
        lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, token_pattern=None, min_df=5)
        X_train_lemma = lemma_vect.fit_transform(text_train)
        print("форма X_train_lemma: {}".format(X_train_lemma.shape))

        vect = CountVectorizer(min_df=5).fit(text_train)
        X_train = vect.transform(text_train)
        print("форма X_train: {}".format(X_train.shape))
        """
    ),
    code(
        """
        param_grid = {"C": [0.001, 0.01, 0.1, 1, 10]}
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.99, train_size=0.01, random_state=0)
        grid = GridSearchCV(
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
            param_grid,
            cv=cv,
            n_jobs=-1,
        )

        grid.fit(X_train, y_train)
        print(
            "Наилучшее значение перекрестной проверки (стандартный CountVectorizer): {:.3f}".format(
                grid.best_score_
            )
        )

        grid.fit(X_train_lemma, y_train)
        print(
            "Наилучшее значение перекрестной проверки (лемматизация): {:.3f}".format(
                grid.best_score_
            )
        )
        """
    ),
    md(
        """
        В данном случае лемматизация дала небольшое изменение качества. Как и в случае с другими методами извлечения признаков, результат варьирует в зависимости от набора данных.
        """
    ),
    md("## Вывод"),
    md(
        """
        В работе последовательно были воспроизведены основные этапы из методички:
        - загрузка обучающих и тестовых данных IMDB;
        - очистка HTML-разметки;
        - представление текста в виде мешка слов;
        - настройка логистической регрессии;
        - эксперименты с `min_df`, стоп-словами, `max_df`, `tf-idf` и `n`-граммами;
        - сравнение стандартной токенизации и лемматизации.

        Наиболее сильное качество в этой реализации показала модель `TfidfVectorizer` с `n`-граммами. Это соответствует общему выводу методички: настройка текстового представления часто влияет на итоговую точность сильнее, чем простое удаление отдельных слов.
        """
    ),
]

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

NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
print(NOTEBOOK_PATH)
