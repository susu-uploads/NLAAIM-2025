from __future__ import annotations

import argparse
import logging
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline


PRACTICE_DIR = Path.cwd()
DATA_DIR = PRACTICE_DIR / ".data"
DEFAULT_INDEX_DIR = DATA_DIR / "chroma_susu"
DEFAULT_HF_HOME = DATA_DIR / "hf_home"
DEFAULT_COLLECTION_NAME = "susu_rag"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_DEVICE = "auto"
DEFAULT_DTYPE = "auto"
DEFAULT_RETRIEVE_K = 20
DEFAULT_SEARCH_TYPE = "mmr"
DEFAULT_FETCH_K = 50
DEFAULT_MMR_LAMBDA = 0.5
DEFAULT_CONTEXT_TOP_K = 5
DEFAULT_RERANK_TOP_N = 5
DEFAULT_MAX_CONTEXT_CHARS = 12000
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_TRUST_REMOTE_CODE = False


@dataclass(frozen=True)
class ChatConfig:
    """Содержит настройки консольной RAG-системы.

    Attributes:
        index_dir: Директория Chroma-индекса.
        hf_home: Локальный кэш Hugging Face.
        collection_name: Имя коллекции Chroma.
        embedding_model: Модель embeddings.
        embedding_batch_size: Размер batch для embeddings.
        llm_model: Локальная LLM для генерации ответов.
        reranker_model: CrossEncoder-модель для reranker.
        device: Устройство для локальных моделей.
        dtype: Тип чисел для загрузки LLM.
        retrieve_k: Число чанков, извлекаемых из Chroma.
        search_type: Тип поиска в Chroma.
        fetch_k: Число кандидатов для MMR-поиска.
        mmr_lambda: Баланс релевантности и разнообразия в MMR.
        context_top_k: Число чанков в контексте без reranker.
        rerank_top_n: Число чанков после reranker.
        max_context_chars: Максимальная длина контекста.
        max_new_tokens: Максимальная длина ответа.
        temperature: Температура генерации.
        top_p: Параметр nucleus sampling.
        repetition_penalty: Штраф за повторы.
        question: Вопрос для одиночного запуска.
        rerank: Нужно ли применять reranker.
        no_sources: Нужно ли скрывать источники.
        verbose: Нужно ли оставить подробный вывод библиотек.
        trust_remote_code: Нужно ли разрешить remote code для моделей Hugging Face.
    """

    index_dir: Path
    hf_home: Path
    collection_name: str
    embedding_model: str
    embedding_batch_size: int
    llm_model: str
    reranker_model: str
    device: str
    dtype: str
    retrieve_k: int
    search_type: str
    fetch_k: int
    mmr_lambda: float
    context_top_k: int
    rerank_top_n: int
    max_context_chars: int
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    question: str
    rerank: bool
    no_sources: bool
    verbose: bool
    trust_remote_code: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ChatConfig:
        """Создает конфигурацию из аргументов командной строки.

        Args:
            args: Аргументы, полученные через `argparse`.

        Returns:
            Конфигурация консольной RAG-системы.
        """
        config = cls(
            index_dir=args.index_dir,
            hf_home=args.hf_home,
            collection_name=DEFAULT_COLLECTION_NAME,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
            llm_model=DEFAULT_LLM_MODEL,
            reranker_model=DEFAULT_RERANKER_MODEL,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE,
            retrieve_k=DEFAULT_RETRIEVE_K,
            search_type=DEFAULT_SEARCH_TYPE,
            fetch_k=DEFAULT_FETCH_K,
            mmr_lambda=DEFAULT_MMR_LAMBDA,
            context_top_k=DEFAULT_CONTEXT_TOP_K,
            rerank_top_n=DEFAULT_RERANK_TOP_N,
            max_context_chars=DEFAULT_MAX_CONTEXT_CHARS,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
            question=args.question,
            rerank=args.rerank,
            no_sources=args.no_sources,
            verbose=args.verbose,
            trust_remote_code=DEFAULT_TRUST_REMOTE_CODE,
        )
        config.validate()
        return config

    @property
    def show_sources(self) -> bool:
        """Проверяет, нужно ли печатать источники после ответа.

        Returns:
            `True`, если источники нужно показывать.
        """
        return not self.no_sources

    def validate(self) -> None:
        """Проверяет базовые ограничения конфигурации.

        Raises:
            SystemExit: Если числовые параметры заданы некорректно.
        """
        if self.embedding_batch_size <= 0:
            raise SystemExit("embedding_batch_size должен быть больше 0.")
        if self.retrieve_k <= 0:
            raise SystemExit("retrieve_k должен быть больше 0.")
        if self.fetch_k <= 0:
            raise SystemExit("fetch_k должен быть больше 0.")
        if self.context_top_k <= 0:
            raise SystemExit("context_top_k должен быть больше 0.")
        if self.rerank_top_n <= 0:
            raise SystemExit("rerank_top_n должен быть больше 0.")
        if self.max_context_chars <= 0:
            raise SystemExit("max_context_chars должен быть больше 0.")
        if self.max_new_tokens <= 0:
            raise SystemExit("max_new_tokens должен быть больше 0.")
        if self.temperature < 0:
            raise SystemExit("temperature не может быть отрицательной.")
        if not 0 < self.top_p <= 1:
            raise SystemExit("top_p должен быть в диапазоне (0, 1].")
        if self.repetition_penalty <= 0:
            raise SystemExit("repetition_penalty должен быть больше 0.")
        if not 0 <= self.mmr_lambda <= 1:
            raise SystemExit("mmr_lambda должен быть в диапазоне [0, 1].")


@dataclass
class PipelineLLM:
    """Содержит LangChain-обертку над локальной LLM.

    Attributes:
        tokenizer: Токенизатор Hugging Face.
        chain: LangChain-цепочка генерации и очистки ответа.
    """

    tokenizer: Any
    chain: Any


class AnswerCleaner:
    """Очищает ответ LLM от служебных продолжений."""

    @staticmethod
    def clean(text: str) -> str:
        """Обрезает служебные продолжения после ответа модели.

        Args:
            text: Сырой текст, сгенерированный моделью.

        Returns:
            Очищенный ответ без продолжения диалога и повторных
            блоков.
        """
        answer = text.strip()
        answer = re.sub(r"^(Ответ|Assistant|Ассистент)\s*:\s*", "", answer, flags=re.IGNORECASE)

        stop_patterns = [
            r"\n\s*Вопрос\s*:",
            r"\n\s*Ответ\s*:",
            r"\n\s*Источники\s*:",
            r"\n\s*Контекст\s*:",
            r"\n\s*Пример ответа\s*:",
            r"\n\s*User\s*:",
            r"\n\s*Assistant\s*:",
            r"<\|im_end\|>",
        ]

        for pattern in stop_patterns:
            match = re.search(pattern, answer, flags=re.IGNORECASE)
            if match:
                answer = answer[: match.start()].rstrip()

        return answer.strip()


class ModelRuntime:
    """Настраивает локальную среду и создает модели LangChain."""

    def __init__(self, config: ChatConfig) -> None:
        """Инициализирует runtime локальных моделей.

        Args:
            config: Конфигурация консольной RAG-системы.
        """
        self.config = config

    def configure_logging(self) -> None:
        """Настраивает уровень служебного вывода библиотек."""
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        if self.config.verbose:
            return

        warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
        warnings.filterwarnings("ignore", message=".*torch_dtype.*")
        warnings.filterwarnings("ignore", message=".*generation_config.*")
        warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)

        try:
            from huggingface_hub.utils import logging as hf_logging
            from transformers.utils import logging as transformers_logging

            hf_logging.set_verbosity_error()
            transformers_logging.set_verbosity_error()
            transformers_logging.disable_progress_bar()
        except Exception:
            pass

    def setup_cache(self) -> None:
        """Настраивает локальный кэш Hugging Face внутри директории
        практики.
        """
        self.config.hf_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.config.hf_home))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(self.config.hf_home / "sentence_transformers"))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def detect_device(self) -> str:
        """Выбирает устройство для локальных моделей.

        Returns:
            Строковый идентификатор устройства: `mps`, `cuda` или `cpu`.
        """
        if self.config.device != "auto":
            return self.config.device

        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def resolve_torch_dtype(self, device: str) -> Any | None:
        """Выбирает тип чисел для загрузки LLM.

        Args:
            device: Фактическое устройство запуска модели.

        Returns:
            Объект dtype из PyTorch или `None`, если тип должен выбрать Transformers.
        """
        import torch

        if self.config.dtype == "float16":
            return torch.float16
        if self.config.dtype == "bfloat16":
            return torch.bfloat16
        if self.config.dtype == "float32":
            return torch.float32
        if device in {"mps", "cuda"}:
            return torch.float16
        return None

    def make_embeddings(self) -> HuggingFaceEmbeddings:
        """Создает embedding-модель LangChain.

        Returns:
            Объект `HuggingFaceEmbeddings` для векторизации
            пользовательских вопросов.
        """
        device = self.detect_device()
        return HuggingFaceEmbeddings(
            model=self.config.embedding_model,
            cache_folder=str(self.config.hf_home),
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": self.config.embedding_batch_size,
            },
            show_progress=False,
        )

    def make_generation_config(self, tokenizer: Any) -> Any:
        """Создает конфигурацию генерации для Hugging Face pipeline.

        Args:
            tokenizer: Токенизатор локальной LLM.

        Returns:
            Объект `GenerationConfig` с параметрами ответа.
        """
        from transformers import GenerationConfig

        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            repetition_penalty=self.config.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        if self.config.temperature > 0:
            generation_config.do_sample = True
            generation_config.temperature = self.config.temperature
            generation_config.top_p = self.config.top_p
        else:
            generation_config.do_sample = False

        return generation_config

    def make_llm(self) -> PipelineLLM:
        """Загружает локальную LLM через Hugging Face pipeline.

        Returns:
            Токенизатор и LangChain-цепочка на основе `HuggingFacePipeline`.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        device = self.detect_device()
        dtype = self.resolve_torch_dtype(device)
        model_kwargs: dict[str, Any] = {
            "cache_dir": str(self.config.hf_home),
            "trust_remote_code": self.config.trust_remote_code,
        }

        if dtype is not None:
            model_kwargs["dtype"] = dtype

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_model,
            cache_dir=str(self.config.hf_home),
            trust_remote_code=self.config.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(self.config.llm_model, **model_kwargs)

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        if device != "cpu":
            model.to(torch.device(device))

        model.generation_config = self.make_generation_config(tokenizer)
        model.eval()
        text_generation = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1 if device == "cpu" else torch.device(device),
            return_full_text=False,
        )
        llm = HuggingFacePipeline(pipeline=text_generation)
        chain = llm | StrOutputParser() | RunnableLambda(AnswerCleaner.clean)
        return PipelineLLM(tokenizer=tokenizer, chain=chain)


class VectorStoreLoader:
    """Загружает сохраненный Chroma-индекс."""

    def __init__(self, config: ChatConfig) -> None:
        """Инициализирует loader Chroma-индекса.

        Args:
            config: Конфигурация консольной RAG-системы.
        """
        self.config = config

    def load(self, embeddings: HuggingFaceEmbeddings) -> Chroma:
        """Загружает сохраненную коллекцию Chroma.

        Args:
            embeddings: Embedding-модель, совместимая с индексом.

        Returns:
            Объект `Chroma`, подключенный к сохраненной коллекции.

        Raises:
            SystemExit: Если индекс отсутствует или коллекция пуста.
        """
        if not self.config.index_dir.exists() or not any(self.config.index_dir.iterdir()):
            raise SystemExit(
                f"Индекс не найден: {self.config.index_dir}. "
                "Сначала запустите скрипт 7.2."
            )

        vector_store = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.config.index_dir),
        )
        sample = vector_store.get(limit=1)

        if not sample.get("ids"):
            raise SystemExit(
                f"Коллекция Chroma пуста: {self.config.collection_name}. "
                "Пересоберите индекс скриптом 7.2."
            )

        return vector_store


class RetrieverFactory:
    """Создает retriever и optional reranker."""

    def __init__(self, config: ChatConfig, runtime: ModelRuntime) -> None:
        """Инициализирует фабрику retrieval-компонентов.

        Args:
            config: Конфигурация консольной RAG-системы.
            runtime: Runtime локальных моделей.
        """
        self.config = config
        self.runtime = runtime

    def make_search_kwargs(self) -> dict[str, Any]:
        """Формирует параметры поиска Chroma.

        Returns:
            Словарь параметров поиска.
        """
        search_kwargs: dict[str, Any] = {"k": self.config.retrieve_k}
        if self.config.search_type == "mmr":
            search_kwargs.update(
                {
                    "fetch_k": self.config.fetch_k,
                    "lambda_mult": self.config.mmr_lambda,
                }
            )

        return search_kwargs

    def make(self, vector_store: Chroma) -> Any:
        """Создает retriever и при необходимости добавляет reranker.

        Args:
            vector_store: Загруженный Chroma-индекс.

        Returns:
            Retriever LangChain, возвращающий документы по
            пользовательскому запросу.
        """
        base_retriever = vector_store.as_retriever(
            search_type=self.config.search_type,
            search_kwargs=self.make_search_kwargs(),
        )

        if not self.config.rerank:
            return base_retriever

        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

        cross_encoder = HuggingFaceCrossEncoder(
            model_name=self.config.reranker_model,
            model_kwargs={"device": self.runtime.detect_device()},
        )
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=self.config.rerank_top_n)

        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )


class PromptBuilder:
    """Формирует контекст и prompt для локальной LLM."""

    def __init__(self, config: ChatConfig) -> None:
        """Инициализирует builder prompt.

        Args:
            config: Конфигурация консольной RAG-системы.
        """
        self.config = config

    def format_context(self, documents: Sequence[Document]) -> str:
        """Формирует текстовый контекст из найденных документов.

        Args:
            documents: Документы, найденные retriever или reranker.

        Returns:
            Текст контекста с заголовками и ссылками на источники.
        """
        blocks: list[str] = []
        current_length = 0

        for index, document in enumerate(documents, start=1):
            title = document.metadata.get("title") or "Без названия"
            source = document.metadata.get("source") or ""
            text = document.page_content.strip()
            block = f"[{index}] {title}\nИсточник: {source}\n{text}"

            if current_length + len(block) > self.config.max_context_chars:
                remaining = self.config.max_context_chars - current_length
                if remaining <= 0:
                    break
                block = block[:remaining].rstrip()

            blocks.append(block)
            current_length += len(block)

        return "\n\n".join(blocks)

    def build_messages(self, question: str, documents: Sequence[Document]) -> list[dict[str, str]]:
        """Формирует сообщения для instruct-модели.

        Args:
            question: Пользовательский вопрос.
            documents: Документы, найденные в индексе.

        Returns:
            Список сообщений в формате chat template.
        """
        context = self.format_context(documents)
        system_prompt = (
            "Ты отвечаешь на вопросы по контексту из RAG-системы. "
            "Отвечай только на последний вопрос пользователя. "
            "Используй только факты из блока <context>. "
            "Не используй собственные знания и не добавляй факты, "
            "которых нет в контексте. "
            "Не придумывай новые вопросы, не продолжай диалог "
            "и не добавляй примеры ответа. "
            "Если ответа нет в контексте, напиши: "
            "В найденных материалах нет ответа на этот вопрос."
        )
        user_prompt = (
            f"<context>\n{context}\n</context>\n\n"
            f"Вопрос: {question}\n\n"
            "Дай краткий ответ на русском языке. "
            "Используй не больше трех предложений."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def build_prompt(self, tokenizer: Any, question: str, documents: Sequence[Document]) -> str:
        """Формирует prompt для ответа на вопрос по найденному контексту.

        Args:
            tokenizer: Токенизатор локальной LLM.
            question: Пользовательский вопрос.
            documents: Документы, найденные в индексе.

        Returns:
            Строка prompt для локальной LLM.
        """
        messages = self.build_messages(question, documents)
        if getattr(tokenizer, "chat_template", None):
            return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        context = self.format_context(documents)
        return (
            "Ответь на русском языке только на последний вопрос. "
            "Используй только факты из блока <context>. "
            "Не используй собственные знания и не добавляй факты, "
            "которых нет в контексте. "
            "Не добавляй новые вопросы, примеры ответа "
            "и продолжение диалога. "
            "Если ответа нет в контексте, напиши: "
            "В найденных материалах нет ответа на этот вопрос.\n\n"
            f"<context>\n{context}\n</context>\n\n"
            f"Вопрос: {question}\n\n"
            "Ответ:"
        )


class QuestionAnswerer:
    """Выполняет полный RAG-проход для пользовательского вопроса."""

    def __init__(self, config: ChatConfig, retriever: Any, llm: PipelineLLM, prompt_builder: PromptBuilder) -> None:
        """Инициализирует обработчик вопросов.

        Args:
            config: Конфигурация консольной RAG-системы.
            retriever: Retriever LangChain.
            llm: Локальная LLM.
            prompt_builder: Builder prompt для LLM.
        """
        self.config = config
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder

    def retrieve_documents(self, question: str) -> list[Document]:
        """Находит документы, релевантные пользовательскому вопросу.

        Args:
            question: Пользовательский вопрос.

        Returns:
            Список документов для передачи в LLM.
        """
        documents = list(self.retriever.invoke(question))
        if self.config.rerank:
            return documents
        return documents[: self.config.context_top_k]

    def generate_answer(self, prompt: str) -> str:
        """Генерирует ответ через LangChain HuggingFacePipeline.

        Args:
            prompt: Подготовленный prompt с контекстом и вопросом.

        Returns:
            Сгенерированный текст ответа.
        """
        return str(self.llm.chain.invoke(prompt)).strip()

    def answer(self, question: str) -> tuple[str, list[Document]]:
        """Отвечает на вопрос через retrieval и локальную LLM.

        Args:
            question: Пользовательский вопрос.

        Returns:
            Кортеж из ответа модели и документов, использованных
            как контекст.
        """
        documents = self.retrieve_documents(question)
        prompt = self.prompt_builder.build_prompt(self.llm.tokenizer, question, documents)
        answer = self.generate_answer(prompt)
        return answer, documents


class ConsolePrinter:
    """Печатает ответы RAG-системы в консоль."""

    @staticmethod
    def unique_sources(documents: Sequence[Document]) -> list[tuple[str, str]]:
        """Выделяет уникальные источники из списка документов.

        Args:
            documents: Документы, использованные как контекст.

        Returns:
            Список пар `(title, source)` без повторяющихся ссылок.
        """
        sources: list[tuple[str, str]] = []
        seen: set[str] = set()

        for document in documents:
            source = str(document.metadata.get("source") or "")
            title = str(document.metadata.get("title") or "Без названия")

            if not source or source in seen:
                continue

            sources.append((title, source))
            seen.add(source)

        return sources

    def print_answer(self, answer: str, documents: Sequence[Document], show_sources: bool) -> None:
        """Печатает ответ модели и список источников.

        Args:
            answer: Сгенерированный ответ.
            documents: Документы, использованные как контекст.
            show_sources: Нужно ли печатать источники после ответа.
        """
        print()
        print("Ответ:")
        print(answer)

        if not show_sources:
            return

        print()
        print("Источники:")
        for index, (title, source) in enumerate(self.unique_sources(documents), start=1):
            print(f"{index}. {title}")
            print(f"   {source}")


class ChatApplication:
    """Запускает консольную RAG-систему."""

    def __init__(self, config: ChatConfig) -> None:
        """Инициализирует приложение RAG-чата.

        Args:
            config: Конфигурация консольной RAG-системы.
        """
        self.config = config
        self.runtime = ModelRuntime(config)
        self.vector_store_loader = VectorStoreLoader(config)
        self.retriever_factory = RetrieverFactory(config, self.runtime)
        self.prompt_builder = PromptBuilder(config)
        self.printer = ConsolePrinter()

    def make_answerer(self) -> QuestionAnswerer:
        """Создает обработчик вопросов с загруженными моделями и
        индексом.

        Returns:
            Готовый обработчик вопросов.
        """
        self.runtime.configure_logging()
        self.runtime.setup_cache()

        embeddings = self.runtime.make_embeddings()
        vector_store = self.vector_store_loader.load(embeddings)
        retriever = self.retriever_factory.make(vector_store)
        llm = self.runtime.make_llm()

        return QuestionAnswerer(
            config=self.config,
            retriever=retriever,
            llm=llm,
            prompt_builder=self.prompt_builder,
        )

    def run_single_question(self, answerer: QuestionAnswerer) -> None:
        """Запускает RAG для одного вопроса из CLI.

        Args:
            answerer: Обработчик вопросов.
        """
        answer, documents = answerer.answer(self.config.question)
        self.printer.print_answer(answer, documents, show_sources=self.config.show_sources)

    def run_interactive_chat(self, answerer: QuestionAnswerer) -> None:
        """Запускает интерактивный консольный режим вопросов.

        Args:
            answerer: Обработчик вопросов.
        """
        print(
            "Введите вопрос. Для выхода используйте exit, quit "
            "или пустую строку."
        )

        while True:
            question = input("\nВопрос: ").strip()
            if question.lower() in {"", "exit", "quit", "q"}:
                break

            answer, documents = answerer.answer(question)
            self.printer.print_answer(answer, documents, show_sources=self.config.show_sources)

    def run(self) -> None:
        """Запускает одиночный или интерактивный режим RAG-чата."""
        answerer = self.make_answerer()
        if self.config.question:
            self.run_single_question(answerer)
        else:
            self.run_interactive_chat(answerer)


def build_arg_parser() -> argparse.ArgumentParser:
    """Создает CLI-парсер для RAG-чата.

    Returns:
        Настроенный парсер аргументов командной строки.
    """
    parser = argparse.ArgumentParser(
        description="Консольная RAG-система по материалам сайта ЮУрГУ.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Директория Chroma-индекса.",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=DEFAULT_HF_HOME,
        help="Локальный кэш Hugging Face.",
    )
    parser.add_argument(
        "--question",
        default="",
        help="Вопрос для одиночного запуска без интерактивного режима.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Применить reranker к найденным чанкам.",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Не печатать источники после ответа.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Показывать подробные сообщения Hugging Face.",
    )
    return parser


def main() -> None:
    """Создает конфигурацию и запускает приложение RAG-чата."""
    parser = build_arg_parser()
    config = ChatConfig.from_args(parser.parse_args())
    ChatApplication(config).run()


if __name__ == "__main__":
    main()
