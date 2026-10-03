from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PRACTICE_DIR = Path.cwd()
DATA_DIR = PRACTICE_DIR / ".data"
DEFAULT_CORPUS_PATH = DATA_DIR / "susu_pages.jsonl"
DEFAULT_INDEX_DIR = DATA_DIR / "chroma_susu"
DEFAULT_MANIFEST_PATH = DATA_DIR / "rag_index_manifest.json"
DEFAULT_HF_HOME = DATA_DIR / "hf_home"
DEFAULT_COLLECTION_NAME = "susu_rag"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_ADD_BATCH_SIZE = 2000
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_DEVICE = "auto"
DEFAULT_RETRIEVE_K = 20
DEFAULT_RERANK_TOP_N = 5


@dataclass(frozen=True)
class IndexConfig:
    """Содержит настройки построения индекса.

    Attributes:
        corpus: Путь к JSONL-файлу корпуса.
        index_dir: Директория Chroma-индекса.
        manifest: Путь к JSON-файлу манифеста.
        hf_home: Локальный кэш Hugging Face.
        collection_name: Имя коллекции Chroma.
        chunk_size: Размер чанка.
        chunk_overlap: Перекрытие соседних чанков.
        add_batch_size: Размер пакета добавления чанков в Chroma.
        embedding_model: Модель embeddings.
        embedding_batch_size: Размер batch для embeddings.
        reranker_model: Модель reranker.
        device: Устройство для локальных моделей.
        retrieve_k: Число чанков, извлекаемых до reranker.
        rerank_top_n: Число чанков после reranker.
        test_query: Тестовый запрос.
        rerank: Нужно ли применять reranker к тестовому запросу.
        rebuild: Нужно ли пересоздавать индекс.
    """

    corpus: Path
    index_dir: Path
    manifest: Path
    hf_home: Path
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    add_batch_size: int
    embedding_model: str
    embedding_batch_size: int
    reranker_model: str
    device: str
    retrieve_k: int
    rerank_top_n: int
    test_query: str
    rerank: bool
    rebuild: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> IndexConfig:
        """Создает конфигурацию из аргументов командной строки.

        Args:
            args: Аргументы, полученные через `argparse`.

        Returns:
            Конфигурация индексации.
        """
        config = cls(
            corpus=args.corpus,
            index_dir=args.index_dir,
            manifest=args.manifest,
            hf_home=args.hf_home,
            collection_name=DEFAULT_COLLECTION_NAME,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            add_batch_size=DEFAULT_ADD_BATCH_SIZE,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
            reranker_model=DEFAULT_RERANKER_MODEL,
            device=DEFAULT_DEVICE,
            retrieve_k=DEFAULT_RETRIEVE_K,
            rerank_top_n=DEFAULT_RERANK_TOP_N,
            test_query=args.test_query,
            rerank=args.rerank,
            rebuild=args.rebuild,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Проверяет базовые ограничения конфигурации.

        Raises:
            SystemExit: Если числовые параметры заданы некорректно.
        """
        if self.chunk_size <= 0:
            raise SystemExit("chunk_size должен быть больше 0.")
        if self.chunk_overlap < 0:
            raise SystemExit("chunk_overlap не может быть отрицательным.")
        if self.chunk_overlap >= self.chunk_size:
            raise SystemExit("chunk_overlap должен быть меньше chunk_size.")
        if self.add_batch_size <= 0:
            raise SystemExit("add_batch_size должен быть больше 0.")
        if self.embedding_batch_size <= 0:
            raise SystemExit("embedding_batch_size должен быть больше 0.")


class ModelRuntime:
    """Настраивает локальную среду и создает embedding-модель."""

    def __init__(self, config: IndexConfig) -> None:
        """Инициализирует runtime локальных моделей.

        Args:
            config: Конфигурация индексации.
        """
        self.config = config

    def setup_cache(self) -> None:
        """Настраивает локальный кэш Hugging Face внутри директории практики."""
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

    def make_embeddings(self) -> HuggingFaceEmbeddings:
        """Создает embedding-модель LangChain.

        Returns:
            Объект `HuggingFaceEmbeddings` для векторизации чанков и запросов.
        """
        device = self.detect_device()
        print(f"Embeddings model: {self.config.embedding_model}")
        print(f"Device: {device}")

        return HuggingFaceEmbeddings(
            model=self.config.embedding_model,
            cache_folder=str(self.config.hf_home),
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": self.config.embedding_batch_size},
            show_progress=True,
        )


class CorpusReader:
    """Читает корпус, созданный скриптом 7.1."""

    def __init__(self, config: IndexConfig) -> None:
        """Инициализирует читатель корпуса.

        Args:
            config: Конфигурация индексации.
        """
        self.config = config

    def read_records(self) -> list[dict[str, Any]]:
        """Читает JSONL-файл корпуса.

        Returns:
            Список словарей с данными страниц.

        Raises:
            SystemExit: Если файл корпуса отсутствует.
        """
        if not self.config.corpus.exists():
            raise SystemExit(f"Корпус не найден: {self.config.corpus}. Сначала запустите скрипт 7.1.")

        records: list[dict[str, Any]] = []
        with self.config.corpus.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        return records


class DocumentPreparer:
    """Преобразует записи корпуса в документы и чанки LangChain."""

    def __init__(self, config: IndexConfig) -> None:
        """Инициализирует подготовку документов.

        Args:
            config: Конфигурация индексации.
        """
        self.config = config

    @staticmethod
    def build_documents(records: list[dict[str, Any]]) -> list[Document]:
        """Преобразует записи корпуса в документы LangChain.

        Args:
            records: Записи страниц из JSONL-корпуса.

        Returns:
            Список документов LangChain с текстом и метаданными источника.
        """
        documents: list[Document] = []

        for page_id, record in enumerate(records):
            text = str(record.get("text") or "").strip()
            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page_id": page_id,
                        "source": record.get("url", ""),
                        "title": record.get("title", ""),
                        "char_count": record.get("char_count", len(text)),
                    },
                )
            )

        return documents

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Разбивает документы на чанки для индексации.

        Args:
            documents: Документы корпуса.

        Returns:
            Список чанков с добавленными служебными метаданными.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        for chunk_id, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["chunk_char_count"] = len(chunk.page_content)

        return chunks


class ChromaIndexBuilder:
    """Создает и наполняет Chroma-индекс."""

    def __init__(self, config: IndexConfig) -> None:
        """Инициализирует builder Chroma-индекса.

        Args:
            config: Конфигурация индексации.
        """
        self.config = config

    def prepare_index_dir(self) -> None:
        """Подготавливает директорию Chroma-индекса.

        Raises:
            SystemExit: Если индекс уже существует и не указан параметр `--rebuild`.
        """
        if self.config.rebuild and self.config.index_dir.exists():
            shutil.rmtree(self.config.index_dir)
        elif self.config.index_dir.exists() and any(self.config.index_dir.iterdir()):
            raise SystemExit(
                f"Индекс уже существует: {self.config.index_dir}. "
                "Для пересоздания запустите скрипт с параметром --rebuild."
            )

        self.config.index_dir.mkdir(parents=True, exist_ok=True)

    def make_store(self, embeddings: HuggingFaceEmbeddings) -> Chroma:
        """Создает объект Chroma.

        Args:
            embeddings: Модель embeddings для построения векторов.

        Returns:
            Инициализированное хранилище `Chroma`.
        """
        return Chroma(
            collection_name=self.config.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.config.index_dir),
        )

    def add_chunks(self, vector_store: Chroma, chunks: list[Document]) -> None:
        """Добавляет чанки в Chroma небольшими пакетами.

        Args:
            vector_store: Хранилище Chroma.
            chunks: Чанки документов для добавления.
        """
        ids = [f"chunk-{chunk.metadata['chunk_id']}" for chunk in chunks]
        print(f"Indexing chunks: {len(chunks)}")

        for start in range(0, len(chunks), self.config.add_batch_size):
            end = min(start + self.config.add_batch_size, len(chunks))
            print(f"    add batch: {start + 1}-{end}/{len(chunks)}")
            vector_store.add_documents(chunks[start:end], ids=ids[start:end])

    def build(self, chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> Chroma:
        """Строит и сохраняет Chroma-индекс.

        Args:
            chunks: Чанки документов для добавления.
            embeddings: Модель embeddings для построения векторов.

        Returns:
            Инициализированное хранилище `Chroma`.
        """
        self.prepare_index_dir()
        vector_store = self.make_store(embeddings)
        self.add_chunks(vector_store, chunks)
        return vector_store


class RetrievalProbe:
    """Проверяет retrieval и reranker на тестовом запросе."""

    def __init__(self, config: IndexConfig, runtime: ModelRuntime) -> None:
        """Инициализирует тестовый retrieval.

        Args:
            config: Конфигурация индексации.
            runtime: Runtime локальных моделей.
        """
        self.config = config
        self.runtime = runtime

    @staticmethod
    def print_documents(label: str, documents: list[Document]) -> None:
        """Печатает найденные документы и источники.

        Args:
            label: Заголовок блока вывода.
            documents: Документы, найденные retriever или reranker.
        """
        print()
        print(label)
        print("-" * len(label))

        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "")
            title = document.metadata.get("title", "")
            chunk_id = document.metadata.get("chunk_id", "")
            preview = document.page_content[:500].replace("\n", " ")

            print(f"{index}. {title}")
            print(f"   source: {source}")
            print(f"   chunk: {chunk_id}")
            print(f"   text: {preview}...")

    def rerank_documents(self, vector_store: Chroma) -> list[Document]:
        """Применяет reranker к тестовому запросу.

        Args:
            vector_store: Построенный Chroma-индекс.

        Returns:
            Документы после reranker.
        """
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

        retriever = vector_store.as_retriever(search_kwargs={"k": self.config.retrieve_k})
        cross_encoder = HuggingFaceCrossEncoder(
            model_name=self.config.reranker_model,
            model_kwargs={"device": self.runtime.detect_device()},
        )
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=self.config.rerank_top_n)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever,
        )

        return list(compression_retriever.invoke(self.config.test_query))

    def run(self, vector_store: Chroma) -> None:
        """Запускает тестовый retrieval, если задан тестовый запрос.

        Args:
            vector_store: Построенный Chroma-индекс.
        """
        if not self.config.test_query:
            return

        retriever = vector_store.as_retriever(search_kwargs={"k": self.config.retrieve_k})
        retrieved_docs = list(retriever.invoke(self.config.test_query))
        self.print_documents("Retriever results", retrieved_docs)

        if self.config.rerank:
            self.print_documents("Reranker results", self.rerank_documents(vector_store))


class IndexManifestWriter:
    """Сохраняет параметры построенного индекса."""

    def __init__(self, config: IndexConfig, runtime: ModelRuntime) -> None:
        """Инициализирует writer манифеста индекса.

        Args:
            config: Конфигурация индексации.
            runtime: Runtime локальных моделей.
        """
        self.config = config
        self.runtime = runtime

    def write(self, page_count: int, chunk_count: int) -> None:
        """Сохраняет параметры построенного индекса.

        Args:
            page_count: Количество страниц в корпусе.
            chunk_count: Количество чанков после разбиения.
        """
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "corpus_path": str(self.config.corpus),
            "index_dir": str(self.config.index_dir),
            "collection_name": self.config.collection_name,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "add_batch_size": self.config.add_batch_size,
            "embedding_model": self.config.embedding_model,
            "reranker_model": self.config.reranker_model,
            "device": self.runtime.detect_device(),
        }
        self.config.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.config.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


class IndexApplication:
    """Запускает полный пайплайн построения RAG-индекса."""

    def __init__(self, config: IndexConfig) -> None:
        """Инициализирует приложение индексации.

        Args:
            config: Конфигурация индексации.
        """
        self.config = config
        self.runtime = ModelRuntime(config)
        self.reader = CorpusReader(config)
        self.preparer = DocumentPreparer(config)
        self.builder = ChromaIndexBuilder(config)
        self.probe = RetrievalProbe(config, self.runtime)
        self.manifest = IndexManifestWriter(config, self.runtime)

    @staticmethod
    def validate_documents(documents: list[Document], chunks: list[Document]) -> None:
        """Проверяет, что корпус дал документы и чанки.

        Args:
            documents: Документы корпуса.
            chunks: Чанки документов.

        Raises:
            SystemExit: Если документов или чанков нет.
        """
        if not documents:
            raise SystemExit("В корпусе нет документов с текстом.")
        if not chunks:
            raise SystemExit("После разбиения не получилось ни одного чанка.")

    @staticmethod
    def print_summary(documents: list[Document], chunks: list[Document]) -> None:
        """Печатает краткую статистику подготовки корпуса.

        Args:
            documents: Документы корпуса.
            chunks: Чанки документов.
        """
        print(f"Pages: {len(documents)}")
        print(f"Chunks: {len(chunks)}")

    def run(self) -> None:
        """Выполняет чтение корпуса, построение Chroma-индекса и тестовый retrieval."""
        self.runtime.setup_cache()
        records = self.reader.read_records()
        documents = self.preparer.build_documents(records)
        chunks = self.preparer.split_documents(documents)
        self.validate_documents(documents, chunks)
        self.print_summary(documents, chunks)

        embeddings = self.runtime.make_embeddings()
        vector_store = self.builder.build(chunks, embeddings)
        self.manifest.write(len(documents), len(chunks))
        self.probe.run(vector_store)

        print()
        print(f"Index directory: {self.config.index_dir}")
        print(f"Manifest: {self.config.manifest}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Создает CLI-парсер для скрипта индексации.

    Returns:
        Настроенный парсер аргументов командной строки.
    """
    parser = argparse.ArgumentParser(
        description="Индексация корпуса ЮУрГУ для RAG через LangChain.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="JSONL-файл корпуса из 7.1.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Директория Chroma-индекса.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="JSON-файл статистики индекса.",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=DEFAULT_HF_HOME,
        help="Локальный кэш Hugging Face.",
    )
    parser.add_argument(
        "--test-query",
        default="",
        help="Тестовый запрос для проверки поиска.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Применить reranker к тестовому запросу.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Пересоздать Chroma-индекс с нуля.",
    )
    return parser


def main() -> None:
    """Создает конфигурацию и запускает приложение индексации."""
    parser = build_arg_parser()
    config = IndexConfig.from_args(parser.parse_args())
    IndexApplication(config).run()


if __name__ == "__main__":
    main()
