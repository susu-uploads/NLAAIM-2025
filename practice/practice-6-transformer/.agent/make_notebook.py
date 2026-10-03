from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PRACTICE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PRACTICE_DIR / "СП-М-О-АЕЯМИИ-2025-Практика-6-БабушкинМВ.ipynb"


def md(text: str) -> nbf.NotebookNode:
    """Создает Markdown-ячейку."""
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    """Создает code-ячейку."""
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # Практическая работа 6. Трансформер

        Natural Language Processing with Disaster Tweets.
        """
    ),
    md(
        """
        ## Постановка задачи

        В соревновании необходимо определить, относится ли твит к реальной катастрофе. Обучающая выборка содержит текст твита и целевую переменную `target`: `1` для сообщения о реальной катастрофе и `0` для остальных сообщений. Для отправки результата на Kaggle нужно сформировать файл `submission.csv` с колонками `id` и `target`.
        """
    ),
    md(
        """
        ## План решения

        Для классификации используется `vinai/bertweet-base` и библиотека HuggingFace Transformers. Тексты приводятся к формату твитов: ссылки заменяются на `HTTPURL`, упоминания пользователей — на `@USER`. Модель дообучается через `Trainer`, качество контролируется по `F1-score`.
        """
    ),
    code(
        """
        from __future__ import annotations

        import os
        import random
        import re
        from html import unescape
        from pathlib import Path

        import numpy as np
        import pandas as pd

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
        os.environ.setdefault("HF_HOME", str(Path.cwd() / ".cache" / "huggingface"))
        os.environ.setdefault("HF_HUB_CACHE", str(Path.cwd() / ".cache" / "huggingface" / "hub"))
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)

        import ftfy
        import torch
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, classification_report, f1_score
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
        """
    ),
    code(
        """
        RANDOM_STATE = 42
        MODEL_NAME = "vinai/bertweet-base"
        MAX_LENGTH = 128
        BATCH_SIZE = 16
        EPOCHS = 4
        LEARNING_RATE = 1e-5
        VALIDATION_SIZE = 0.15
        WARMUP_RATIO = 0.10

        DATA_DIR = Path.cwd() / ".data"
        OUTPUT_DIR = Path.cwd() / ".cache" / "practice_6_bertweet"
        SUBMISSION_PATH = Path.cwd() / "submission.csv"

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        """
    ),
    code(
        """
        def set_seed(seed: int = RANDOM_STATE) -> None:
            \"\"\"Фиксирует генераторы случайных чисел.

            Args:
                seed: Значение seed для Python, NumPy и PyTorch.
            \"\"\"
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)


        set_seed()
        """
    ),
    md(
        """
        ## Данные
        """
    ),
    code(
        """
        def get_data_dir(data_dir: Path = DATA_DIR) -> Path:
            \"\"\"Проверяет директорию с данными соревнования.

            Args:
                data_dir: Ожидаемый путь к директории с CSV-файлами.

            Returns:
                Путь к директории с файлами `train.csv`, `test.csv`, `sample_submission.csv`.

            Raises:
                FileNotFoundError: Если один из обязательных файлов отсутствует.
            \"\"\"
            required_files = ["train.csv", "test.csv", "sample_submission.csv"]
            missing_files = [file_name for file_name in required_files if not (data_dir / file_name).exists()]

            if missing_files:
                raise FileNotFoundError(f"Не найдены файлы: {missing_files}")

            return data_dir
        """
    ),
    code(
        """
        data_dir = get_data_dir()

        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

        print("Train:", train_df.shape)
        print("Test:", test_df.shape)
        train_df.head()
        """
    ),
    code(
        """
        train_df["target"].value_counts(normalize=True).rename("share")
        """
    ),
    code(
        """
        def remove_conflicting_tweets(data: pd.DataFrame) -> pd.DataFrame:
            \"\"\"Удаляет одинаковые тексты с противоречивой разметкой.

            Args:
                data: Обучающая таблица с колонками `text` и `target`.

            Returns:
                Таблица без конфликтных текстов.
            \"\"\"
            target_counts = data.groupby("text")["target"].nunique()
            conflict_texts = target_counts[target_counts > 1].index
            print("Конфликтных твитов:", len(conflict_texts))
            return data[~data["text"].isin(conflict_texts)].reset_index(drop=True)
        """
    ),
    code(
        """
        train_df = remove_conflicting_tweets(train_df)
        print("Train после очистки:", train_df.shape)
        """
    ),
    md(
        """
        ## Подготовка текста
        """
    ),
    code(
        """
        def clean_text(value: str) -> str:
            \"\"\"Готовит текст твита в формате, близком к BERTweet.

            Args:
                value: Исходный текст.

            Returns:
                Текст с нормализованными ссылками, упоминаниями и пробелами.
            \"\"\"
            text = ftfy.fix_text(str(value))
            text = unescape(text)
            text = re.sub(r"https?://\\S+|www\\.\\S+", " HTTPURL ", text)
            text = re.sub(r"@\\w+", " @USER ", text)
            text = re.sub(r"\\s+", " ", text)
            return text.strip()
        """
    ),
    code(
        """
        def build_input_text(data: pd.DataFrame) -> pd.Series:
            \"\"\"Формирует входной текст для трансформера.

            Args:
                data: Таблица с колонкой `text`.

            Returns:
                Серия подготовленных строк.
            \"\"\"
            return data["text"].map(clean_text)
        """
    ),
    code(
        """
        train_df = train_df.copy()
        test_df = test_df.copy()

        train_df["input"] = build_input_text(train_df)
        test_df["input"] = build_input_text(test_df)

        train_df[["keyword", "location", "text", "input", "target"]].head()
        """
    ),
    md(
        """
        ## Обучение модели
        """
    ),
    code(
        """
        train_data, valid_data = train_test_split(
            train_df[["input", "target"]].rename(columns={"input": "text", "target": "labels"}),
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=train_df["target"],
        )

        print("Train:", train_data.shape)
        print("Validation:", valid_data.shape)
        """
    ),
    code(
        """
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            normalization=True,
        )
        """
    ),
    code(
        """
        def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
            \"\"\"Токенизирует батч текстов.

            Args:
                batch: Батч записей HuggingFace Dataset.

            Returns:
                Токенизированный батч.
            \"\"\"
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=MAX_LENGTH,
            )
        """
    ),
    code(
        """
        train_ds = Dataset.from_pandas(train_data, preserve_index=False).map(tokenize_batch, batched=True)
        valid_ds = Dataset.from_pandas(valid_data, preserve_index=False).map(tokenize_batch, batched=True)
        test_ds = (
            Dataset.from_pandas(test_df[["input"]].rename(columns={"input": "text"}), preserve_index=False)
            .map(tokenize_batch, batched=True)
        )

        print(len(train_ds), len(valid_ds), len(test_ds))
        """
    ),
    code(
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )
        """
    ),
    md(
        """
        Для обучения используется готовая инфраструктура HuggingFace. `TrainingArguments` хранит параметры запуска: количество эпох, размер батча, learning rate, warmup, регуляризацию, стратегию оценки и сохранения checkpoints. Лучшая модель выбирается по `F1-score`, так как эта метрика используется в соревновании.

        `Trainer` объединяет модель, параметры обучения, датасеты, динамическое дополнение батчей через `DataCollatorWithPadding`, функцию расчета метрик и `EarlyStoppingCallback`. Поэтому основной цикл обучения не пишется вручную.
        """
    ),
    code(
        """
        def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
            \"\"\"Вычисляет метрики классификации.

            Args:
                eval_pred: Пара `logits`, `labels` от Trainer.

            Returns:
                Значения accuracy, binary F1 и macro F1.
            \"\"\"
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {
                "accuracy": float(accuracy_score(labels, predictions)),
                "f1": float(f1_score(labels, predictions)),
                "f1_macro": float(f1_score(labels, predictions, average="macro")),
            }
        """
    ),
    code(
        """
        steps_per_epoch = max(1, len(train_ds) // BATCH_SIZE)
        warmup_steps = int(steps_per_epoch * EPOCHS * WARMUP_RATIO)

        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            seed=RANDOM_STATE,
            report_to="none",
            use_cpu=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        """
    ),
    code(
        """
        trainer.train()
        """
    ),
    md(
        """
        ## Оценка качества

        Модель возвращает `logits` - ненормированные оценки классов. Для обычного выбора класса можно взять `argmax`, но для подбора собственного порога нужны вероятности класса `1`. Поэтому `logits` переводятся в вероятности через `softmax`.
        """
    ),
    code(
        """
        def softmax(logits: np.ndarray) -> np.ndarray:
            \"\"\"Преобразует logits в вероятности.

            Args:
                logits: Выходы модели до нормализации.

            Returns:
                Матрица вероятностей классов.
            \"\"\"
            shifted_logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(shifted_logits)
            return exp_logits / exp_logits.sum(axis=1, keepdims=True)
        """
    ),
    md(
        """
        Порог классификации определяет, с какого значения вероятности твит считается сообщением о катастрофе. Значение `0.5` является стандартным, но не обязательно дает лучший `F1-score`. Поэтому перебираются пороги от `0.25` до `0.75` с шагом `0.01`, и выбирается тот, при котором `F1-score` на validation-выборке максимален.
        """
    ),
    code(
        """
        def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
            \"\"\"Подбирает порог классификации по F1.

            Args:
                y_true: Истинные метки.
                probabilities: Вероятности класса `1`.

            Returns:
                Лучший порог и соответствующее значение F1.
            \"\"\"
            thresholds = np.arange(0.25, 0.76, 0.01)
            scores = [
                f1_score(y_true, (probabilities >= threshold).astype(int))
                for threshold in thresholds
            ]
            best_index = int(np.argmax(scores))
            return float(thresholds[best_index]), float(scores[best_index])
        """
    ),
    code(
        """
        valid_output = trainer.predict(valid_ds)
        valid_probabilities = softmax(valid_output.predictions)[:, 1]
        valid_labels = valid_data["labels"].to_numpy()

        best_threshold, best_f1 = find_best_threshold(valid_labels, valid_probabilities)
        valid_predictions = (valid_probabilities >= best_threshold).astype(int)

        print("Best threshold:", round(best_threshold, 3))
        print("Validation F1:", round(best_f1, 4))
        print(classification_report(valid_labels, valid_predictions, target_names=["Not Disaster", "Disaster"]))
        """
    ),
    md(
        """
        ## Формирование отправки
        """
    ),
    code(
        """
        def make_submission(probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
            \"\"\"Формирует таблицу для отправки на Kaggle.

            Args:
                probabilities: Вероятности класса `1` для тестовой выборки.
                threshold: Порог классификации.

            Returns:
                Таблица с колонками `id` и `target`.
            \"\"\"
            submission = sample_submission.copy()
            submission["target"] = (probabilities >= threshold).astype(int)
            return submission
        """
    ),
    code(
        """
        test_output = trainer.predict(test_ds)
        test_probabilities = softmax(test_output.predictions)[:, 1]

        submission = make_submission(test_probabilities, best_threshold)
        submission.to_csv(SUBMISSION_PATH, index=False)

        print(SUBMISSION_PATH)
        print(submission["target"].value_counts().to_string())
        submission.head()
        """
    ),
    md(
        """
        ## Вывод

        В работе использована модель `vinai/bertweet-base`, дообученная на данных соревнования Disaster Tweets. Для оценки применялся `F1-score`, соответствующий метрике соревнования. По результатам предсказаний сформирован файл `submission.csv`.
        """
    ),
]


def main() -> None:
    """Собирает notebook практической работы."""
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
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
    nbf.write(notebook, NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
