from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


PRACTICE_DIR = Path.cwd()
DATA_DIR = PRACTICE_DIR / ".data"
DEFAULT_OUTPUT_PATH = DATA_DIR / "susu_pages.jsonl"
DEFAULT_MANIFEST_PATH = DATA_DIR / "susu_pages_manifest.json"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)
DEFAULT_SEED_URLS = [
    "https://www.susu.ru/ru",
    "https://www.susu.ru/ru/university",
    "https://www.susu.ru/ru/university/structure",
    "https://www.susu.ru/ru/education",
    "https://www.susu.ru/ru/science",
    "https://www.susu.ru/ru/news",
    "https://www.susu.ru/ru/applicants",
    "https://www.susu.ru/en",
]
DEFAULT_ALLOWED_DOMAINS = ["www.susu.ru", "susu.ru"]
DEFAULT_EXCLUDE_PATTERNS = [
    r"^/(ru|en)?/?(user|admin|login|logout|search|comment|taxonomy|filter|system|batch)(/|$)",
    r"^/(ru|en)?/?node/add(/|$)",
    r"^/(ru|en)?/?(rss|print|sitemap)(/|$)",
    r"/(rss|print)(/|$)",
]
DEFAULT_CRAWL_PATTERN = r"^/(ru|en)?($|/)"
DEFAULT_CONTENT_PATTERN = r"^/(ru|en)?($|/)"
DEFAULT_MAX_PAGES = 2000
DEFAULT_MIN_PAGES = 1000
DEFAULT_MAX_FETCHES = 10000
DEFAULT_MAX_QUEUE = 50000
DEFAULT_MIN_CHARS = 600
DEFAULT_BATCH_SIZE = 25
DEFAULT_DELAY = 0.2
DEFAULT_TIMEOUT = 20.0
DEFAULT_REQUESTS_PER_SECOND = 2
DEFAULT_SAVE_QUERY_PAGES = False
DEFAULT_INCREMENTAL_SAVE = True
DEFAULT_TRUST_ENV = False

os.environ.setdefault("USER_AGENT", DEFAULT_USER_AGENT)

from langchain_community.document_loaders import WebBaseLoader


@dataclass(frozen=True)
class ParserConfig:
    """Содержит настройки запуска парсера.

    Attributes:
        seed_urls: Стартовые страницы обхода.
        allowed_domains: Домены, разрешенные для обхода и сохранения.
        crawl_pattern: Шаблон путей, которые можно добавлять в очередь обхода.
        content_pattern: Шаблон путей, которые можно сохранять в корпус.
        exclude_patterns: Шаблоны служебных страниц, исключаемых из обхода.
        max_pages: Максимальное число страниц в корпусе.
        min_pages: Минимально допустимое число страниц.
        max_fetches: Максимальное число HTML-страниц, загружаемых при поиске ссылок.
        max_queue: Максимальный размер очереди URL.
        min_chars: Минимальная длина очищенного текста страницы.
        batch_size: Размер пакета URL для WebBaseLoader.
        delay: Пауза между запросами при поиске ссылок.
        timeout: Таймаут HTTP-запроса.
        requests_per_second: Ограничение частоты запросов WebBaseLoader.
        save_query_pages: Нужно ли сохранять страницы с query-параметрами.
        resume: Нужно ли продолжать запись в существующий JSONL-файл.
        incremental_save: Нужно ли сохранять страницы по мере загрузки.
        trust_env: Нужно ли учитывать proxy-переменные окружения.
        output: Путь к JSONL-файлу корпуса.
        manifest: Путь к JSON-файлу манифеста.
        user_agent: User-Agent для HTTP-запросов.
    """

    seed_urls: list[str]
    allowed_domains: list[str]
    crawl_pattern: str
    content_pattern: str
    exclude_patterns: list[str]
    max_pages: int
    min_pages: int
    max_fetches: int
    max_queue: int
    min_chars: int
    batch_size: int
    delay: float
    timeout: float
    requests_per_second: int
    save_query_pages: bool
    resume: bool
    incremental_save: bool
    trust_env: bool
    output: Path
    manifest: Path
    user_agent: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ParserConfig:
        """Создает конфигурацию из аргументов командной строки.

        Args:
            args: Аргументы, полученные через `argparse`.

        Returns:
            Конфигурация парсера с примененными значениями по умолчанию.
        """
        return cls(
            seed_urls=DEFAULT_SEED_URLS,
            allowed_domains=DEFAULT_ALLOWED_DOMAINS,
            crawl_pattern=DEFAULT_CRAWL_PATTERN,
            content_pattern=DEFAULT_CONTENT_PATTERN,
            exclude_patterns=DEFAULT_EXCLUDE_PATTERNS,
            max_pages=DEFAULT_MAX_PAGES,
            min_pages=DEFAULT_MIN_PAGES,
            max_fetches=DEFAULT_MAX_FETCHES,
            max_queue=DEFAULT_MAX_QUEUE,
            min_chars=DEFAULT_MIN_CHARS,
            batch_size=DEFAULT_BATCH_SIZE,
            delay=DEFAULT_DELAY,
            timeout=DEFAULT_TIMEOUT,
            requests_per_second=DEFAULT_REQUESTS_PER_SECOND,
            save_query_pages=DEFAULT_SAVE_QUERY_PAGES,
            resume=args.resume,
            incremental_save=DEFAULT_INCREMENTAL_SAVE,
            trust_env=DEFAULT_TRUST_ENV,
            output=args.output,
            manifest=args.manifest,
            user_agent=DEFAULT_USER_AGENT,
        )

    @property
    def headers(self) -> dict[str, str]:
        """Возвращает HTTP-заголовки для запросов.

        Returns:
            Словарь заголовков, используемый requests и WebBaseLoader.
        """
        return {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru,en;q=0.8",
        }


@dataclass
class ParsedPage:
    """Содержит очищенный текст страницы, загруженной через LangChain.

    Attributes:
        url: URL исходной страницы.
        title: Заголовок страницы.
        text: Очищенный текст страницы.
        fetched_at: Время загрузки страницы в формате ISO 8601.
        loader: Название загрузчика, которым получена страница.
    """

    url: str
    title: str
    text: str
    fetched_at: str
    loader: str

    @property
    def char_count(self) -> int:
        """Возвращает длину очищенного текста.

        Returns:
            Количество символов в поле `text`.
        """
        return len(self.text)

    @classmethod
    def from_record(cls, record: dict) -> ParsedPage:
        """Создает страницу из записи JSONL.

        Args:
            record: Словарь, прочитанный из JSONL-файла корпуса.

        Returns:
            Объект `ParsedPage`.
        """
        return cls(
            url=str(record.get("url", "")),
            title=str(record.get("title", "")),
            text=str(record.get("text", "")),
            fetched_at=str(record.get("fetched_at", "")),
            loader=str(record.get("loader", "langchain_community.document_loaders.WebBaseLoader")),
        )

    def to_json(self) -> dict:
        """Преобразует страницу в JSON-совместимый словарь.

        Returns:
            Словарь с полями страницы и рассчитанной длиной текста.
        """
        data = asdict(self)
        data["char_count"] = self.char_count
        return data


class TextCleaner:
    """Очищает текст страниц перед сохранением в корпус."""

    @staticmethod
    def normalize(text: str) -> str:
        """Убирает HTML-сущности, повторяющиеся пробелы и пустые строки.

        Args:
            text: Исходный текст страницы.

        Returns:
            Очищенный текст без лишних пробелов и повторяющихся пустых строк.
        """
        text = unescape(text)
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r" *\n+ *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


@dataclass(frozen=True)
class UrlRules:
    """Инкапсулирует правила нормализации и фильтрации URL.

    Attributes:
        allowed_domains: Разрешенные домены сайта.
        crawl_pattern: Шаблон путей для обхода.
        content_pattern: Шаблон путей для сохранения в корпус.
        exclude_patterns: Шаблоны служебных страниц.
        save_query_pages: Нужно ли сохранять страницы с query-параметрами.
    """

    allowed_domains: set[str]
    crawl_pattern: re.Pattern[str]
    content_pattern: re.Pattern[str]
    exclude_patterns: list[re.Pattern[str]]
    save_query_pages: bool

    @classmethod
    def from_config(cls, config: ParserConfig) -> UrlRules:
        """Создает правила URL из конфигурации парсера.

        Args:
            config: Конфигурация парсера.

        Returns:
            Набор скомпилированных правил URL.
        """
        return cls(
            allowed_domains={domain.lower() for domain in config.allowed_domains},
            crawl_pattern=re.compile(config.crawl_pattern),
            content_pattern=re.compile(config.content_pattern),
            exclude_patterns=[re.compile(pattern) for pattern in config.exclude_patterns],
            save_query_pages=config.save_query_pages,
        )

    @staticmethod
    def normalize(url: str, base_url: str | None = None) -> str | None:
        """Приводит URL к каноническому виду.

        Args:
            url: Абсолютная или относительная ссылка.
            base_url: Базовый URL для относительной ссылки.

        Returns:
            Нормализованный абсолютный URL или `None`, если схема не поддерживается.
        """
        absolute_url = urljoin(base_url or "", url)
        parts = urlsplit(absolute_url)

        if parts.scheme not in {"http", "https"}:
            return None

        ignored_query_prefixes = ("utm_",)
        ignored_query_keys = {"fbclid", "gclid", "yclid", "ysclid", "from", "ref"}
        allowed_query_keys = {"page"}
        query_items = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith(ignored_query_prefixes)
            and key.lower() not in ignored_query_keys
            and key.lower() in allowed_query_keys
        ]
        query = urlencode(query_items, doseq=True)

        path = parts.path or "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")

        netloc = parts.netloc.lower()
        if netloc == "susu.ru":
            netloc = "www.susu.ru"

        return urlunsplit((parts.scheme, netloc, path, query, ""))

    @staticmethod
    def is_probably_html(url: str) -> bool:
        """Проверяет, похожа ли ссылка на HTML-страницу.

        Args:
            url: Проверяемый URL.

        Returns:
            `True`, если ссылка не указывает на типичный статический файл.
        """
        path = urlsplit(url).path.lower()
        blocked_suffixes = (
            ".7z",
            ".avi",
            ".css",
            ".csv",
            ".doc",
            ".docx",
            ".gif",
            ".jpeg",
            ".jpg",
            ".js",
            ".mp3",
            ".mp4",
            ".ods",
            ".odt",
            ".pdf",
            ".png",
            ".ppt",
            ".pptx",
            ".rar",
            ".rtf",
            ".svg",
            ".webp",
            ".xls",
            ".xlsx",
            ".zip",
        )
        return not path.endswith(blocked_suffixes)

    def is_excluded(self, url: str) -> bool:
        """Проверяет, относится ли URL к исключенным страницам.

        Args:
            url: Проверяемый URL.

        Returns:
            `True`, если URL подходит хотя бы под один паттерн исключения.
        """
        parts = urlsplit(url)
        target = parts.path
        if parts.query:
            target = f"{target}?{parts.query}"

        return any(pattern.search(target) for pattern in self.exclude_patterns)

    def can_crawl(self, url: str) -> bool:
        """Проверяет, можно ли добавлять URL в очередь обхода.

        Args:
            url: Проверяемый URL.

        Returns:
            `True`, если ссылку можно использовать для поиска новых страниц.
        """
        parts = urlsplit(url)
        return (
            parts.netloc.lower() in self.allowed_domains
            and self.is_probably_html(url)
            and bool(self.crawl_pattern.search(parts.path))
            and not self.is_excluded(url)
        )

    def can_save(self, url: str) -> bool:
        """Проверяет, можно ли сохранить URL в корпус.

        Args:
            url: Проверяемый URL.

        Returns:
            `True`, если URL разрешен для сохранения в корпус.
        """
        parts = urlsplit(url)
        is_content = (self.save_query_pages or not parts.query) and bool(self.content_pattern.search(parts.path))
        return (
            parts.netloc.lower() in self.allowed_domains
            and self.is_probably_html(url)
            and is_content
            and not self.is_excluded(url)
        )


class CorpusStore:
    """Читает и записывает файлы корпуса."""

    def __init__(self, config: ParserConfig) -> None:
        """Инициализирует хранилище корпуса.

        Args:
            config: Конфигурация парсера.
        """
        self.config = config

    def prepare_output(self) -> None:
        """Подготавливает выходной JSONL-файл к записи."""
        if self.config.output.exists() and not self.config.resume:
            self.config.output.unlink()

    def read_pages(self) -> list[ParsedPage]:
        """Читает уже сохраненные страницы для продолжения парсинга.

        Returns:
            Список страниц, которые уже есть в JSONL-файле.
        """
        if not self.config.output.exists():
            return []

        pages: list[ParsedPage] = []
        with self.config.output.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    pages.append(ParsedPage.from_record(json.loads(line)))
                except json.JSONDecodeError as error:
                    print(f"skip broken jsonl line: {error}")

        return pages

    def append_page(self, page: ParsedPage) -> None:
        """Добавляет одну страницу в JSONL-файл.

        Args:
            page: Страница для записи.
        """
        self.config.output.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output.open("a", encoding="utf-8") as file:
            file.write(json.dumps(page.to_json(), ensure_ascii=False) + "\n")

    def write_pages(self, pages: list[ParsedPage]) -> None:
        """Сохраняет полный список страниц в JSONL-файл.

        Args:
            pages: Страницы корпуса.
        """
        self.config.output.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output.open("w", encoding="utf-8") as file:
            for page in pages:
                file.write(json.dumps(page.to_json(), ensure_ascii=False) + "\n")

    def write_manifest(self, urls: list[str], pages: list[ParsedPage]) -> None:
        """Сохраняет краткое описание результата парсинга.

        Args:
            urls: Найденные URL страниц корпуса.
            pages: Сохраненные страницы корпуса.
        """
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed_urls": self.config.seed_urls,
            "allowed_domains": self.config.allowed_domains,
            "crawl_pattern": self.config.crawl_pattern,
            "content_pattern": self.config.content_pattern,
            "exclude_patterns": self.config.exclude_patterns,
            "resume": self.config.resume,
            "incremental_save": self.config.incremental_save,
            "discovered_url_count": len(urls),
            "page_count": len(pages),
            "total_char_count": sum(page.char_count for page in pages),
            "min_char_count": min((page.char_count for page in pages), default=0),
            "max_char_count": max((page.char_count for page in pages), default=0),
            "loader": "langchain_community.document_loaders.WebBaseLoader",
            "output_path": str(self.config.output),
            "pages": [
                {
                    "url": page.url,
                    "title": page.title,
                    "char_count": page.char_count,
                }
                for page in pages
            ],
        }

        self.config.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.config.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


class PageDiscoverer:
    """Ищет страницы сайта, подходящие для корпуса."""

    def __init__(self, config: ParserConfig, rules: UrlRules) -> None:
        """Инициализирует обходчик сайта.

        Args:
            config: Конфигурация парсера.
            rules: Правила нормализации и фильтрации URL.
        """
        self.config = config
        self.rules = rules

    def build_session(self) -> requests.Session:
        """Создает HTTP-сессию для поиска ссылок.

        Returns:
            Сессия requests с настроенными заголовками.
        """
        session = requests.Session()
        session.trust_env = self.config.trust_env
        session.headers.update(self.config.headers)
        return session

    def fetch_html(self, session: requests.Session, url: str) -> str:
        """Загружает HTML для поиска ссылок.

        Args:
            session: HTTP-сессия.
            url: URL страницы.

        Returns:
            HTML-код страницы.

        Raises:
            requests.HTTPError: Если сервер вернул ошибочный HTTP-статус.
            ValueError: Если ответ не похож на HTML-страницу или является редиректом.
        """
        response = session.get(url, timeout=self.config.timeout, allow_redirects=False)
        response.raise_for_status()

        if 300 <= response.status_code < 400:
            raise ValueError(f"redirect skipped: {response.headers.get('Location', '')}")

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            raise ValueError(f"unexpected content type: {content_type or 'unknown'}")

        return response.text

    def extract_links(self, html: str, base_url: str) -> list[str]:
        """Извлекает и нормализует ссылки со страницы.

        Args:
            html: HTML-код страницы.
            base_url: URL страницы, относительно которого раскрываются ссылки.

        Returns:
            Список нормализованных абсолютных ссылок.
        """
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []

        for anchor in soup.find_all("a", href=True):
            normalized = self.rules.normalize(anchor["href"], base_url)
            if normalized:
                links.append(normalized)

        return links

    def discover(self) -> list[str]:
        """Находит целевые URL страниц сайта университета.

        Returns:
            Список URL страниц, которые подходят для включения в корпус.
        """
        session = self.build_session()
        queue: deque[str] = deque()
        queued: set[str] = set()
        visited: set[str] = set()
        content_urls: list[str] = []
        content_seen: set[str] = set()

        for seed in self.config.seed_urls:
            normalized_seed = self.rules.normalize(seed)
            if normalized_seed and normalized_seed not in queued:
                queue.append(normalized_seed)
                queued.add(normalized_seed)

        while queue and len(content_urls) < self.config.max_pages and len(visited) < self.config.max_fetches:
            url = queue.popleft()
            queued.discard(url)
            if url in visited:
                continue

            visited.add(url)
            print(f"[{len(visited):03d}] discover {url}")

            if self.rules.can_save(url) and url not in content_seen:
                content_urls.append(url)
                content_seen.add(url)
                print(f"    candidate: {len(content_urls)}/{self.config.max_pages}")

            try:
                html = self.fetch_html(session, url)
            except Exception as error:
                print(f"    skip: {error}")
                continue

            for link in self.extract_links(html, url):
                if link in visited or link in queued:
                    continue
                if self.rules.can_crawl(link):
                    queue.append(link)
                    queued.add(link)
                    if len(queue) >= self.config.max_queue:
                        break

            if self.config.delay > 0:
                time.sleep(self.config.delay)

        return content_urls


class PageLoader:
    """Загружает и очищает страницы корпуса через WebBaseLoader."""

    def __init__(self, config: ParserConfig, rules: UrlRules, store: CorpusStore) -> None:
        """Инициализирует загрузчик страниц.

        Args:
            config: Конфигурация парсера.
            rules: Правила нормализации и фильтрации URL.
            store: Хранилище корпуса.
        """
        self.config = config
        self.rules = rules
        self.store = store

    def build_loader(self, urls: list[str]) -> WebBaseLoader:
        """Создает LangChain WebBaseLoader для списка страниц.

        Args:
            urls: Список URL для загрузки.

        Returns:
            Настроенный загрузчик `WebBaseLoader`.
        """
        return WebBaseLoader(
            web_paths=tuple(urls),
            header_template=self.config.headers,
            requests_per_second=self.config.requests_per_second,
            requests_kwargs={
                "timeout": self.config.timeout,
                "allow_redirects": False,
            },
            bs_get_text_kwargs={"separator": "\n", "strip": True},
            continue_on_failure=True,
            show_progress=True,
            trust_env=self.config.trust_env,
        )

    def load_documents(self, urls: list[str]) -> list[Document]:
        """Загружает документы с обработкой сетевых ошибок.

        Args:
            urls: Список URL для загрузки.

        Returns:
            Список документов, которые удалось загрузить.
        """
        if not urls:
            return []

        try:
            return self.build_loader(urls).load()
        except Exception as error:
            print(f"    batch load failed: {error}")

        documents: list[Document] = []
        for url in urls:
            try:
                documents.extend(self.build_loader([url]).load())
            except Exception as error:
                print(f"    skip load: {url} ({error})")

        return documents

    def parse_document(self, document: Document) -> ParsedPage | None:
        """Преобразует документ LangChain в страницу корпуса.

        Args:
            document: Документ, полученный из WebBaseLoader.

        Returns:
            Страница корпуса или `None`, если документ не прошел фильтры.
        """
        source = self.rules.normalize(str(document.metadata.get("source") or ""))
        if not source:
            return None
        if not self.rules.can_save(source):
            print(f"    skip redirected/excluded: {source}")
            return None

        title = TextCleaner.normalize(str(document.metadata.get("title") or ""))
        text = TextCleaner.normalize(document.page_content)
        if len(text) < self.config.min_chars:
            print(f"    skip short: {source}, chars={len(text)}")
            return None

        return ParsedPage(
            url=source,
            title=title,
            text=text,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            loader="langchain_community.document_loaders.WebBaseLoader",
        )

    def load_pages(self, urls: list[str]) -> list[ParsedPage]:
        """Загружает страницы корпуса.

        Args:
            urls: Список URL страниц корпуса.

        Returns:
            Список очищенных страниц, прошедших фильтры.
        """
        pages = self.store.read_pages() if self.config.resume else []
        seen_sources: set[str] = {page.url for page in pages if page.url}

        if pages:
            print(f"resume: loaded {len(pages)} existing pages from {self.config.output}")

        for start in range(0, len(urls), self.config.batch_size):
            batch_urls = [url for url in urls[start : start + self.config.batch_size] if url not in seen_sources]
            if not batch_urls:
                continue

            print(f"load batch {start // self.config.batch_size + 1}: {len(batch_urls)} pages")

            for document in self.load_documents(batch_urls):
                page = self.parse_document(document)
                if page is None or page.url in seen_sources:
                    continue

                pages.append(page)
                seen_sources.add(page.url)
                if self.config.incremental_save:
                    self.store.append_page(page)
                print(f"    saved: {len(pages)}, chars={page.char_count}")

        return pages


class ParserApplication:
    """Запускает полный пайплайн парсинга корпуса."""

    def __init__(self, config: ParserConfig) -> None:
        """Инициализирует приложение парсинга.

        Args:
            config: Конфигурация парсера.
        """
        self.config = config
        self.rules = UrlRules.from_config(config)
        self.store = CorpusStore(config)
        self.discoverer = PageDiscoverer(config, self.rules)
        self.loader = PageLoader(config, self.rules, self.store)

    def print_summary(self, urls: list[str], pages: list[ParsedPage]) -> None:
        """Печатает итоговую статистику парсинга.

        Args:
            urls: Найденные URL страниц корпуса.
            pages: Сохраненные страницы корпуса.
        """
        print()
        print(f"Discovered URLs: {len(urls)}")
        print(f"Saved pages: {len(pages)}")
        print(f"Corpus: {self.config.output}")
        print(f"Manifest: {self.config.manifest}")

    def validate(self, pages: list[ParsedPage]) -> None:
        """Проверяет, набрано ли минимальное число страниц.

        Args:
            pages: Сохраненные страницы корпуса.

        Raises:
            SystemExit: Если число страниц меньше требуемого минимума.
        """
        if len(pages) < self.config.min_pages:
            raise SystemExit(
                f"Недостаточно страниц: {len(pages)} из требуемых {self.config.min_pages}. "
                "Попробуйте увеличить --max-fetches или добавить другой --seed."
            )

    def run(self) -> None:
        """Выполняет поиск страниц, загрузку корпуса и сохранение манифеста."""
        self.store.prepare_output()
        urls = self.discoverer.discover()
        pages = self.loader.load_pages(urls)
        self.store.write_pages(pages)
        self.store.write_manifest(urls, pages)
        self.print_summary(urls, pages)
        self.validate(pages)


def build_arg_parser() -> argparse.ArgumentParser:
    """Создает CLI-парсер для скрипта.

    Returns:
        Настроенный парсер аргументов командной строки.
    """
    parser = argparse.ArgumentParser(
        description="Парсинг страниц сайта ЮУрГУ через LangChain WebBaseLoader.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить запись в существующий JSONL-файл корпуса.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Путь к JSONL-файлу корпуса.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Путь к JSON-файлу статистики.",
    )
    return parser


def main() -> None:
    """Создает конфигурацию и запускает приложение парсинга."""
    parser = build_arg_parser()
    config = ParserConfig.from_args(parser.parse_args())
    ParserApplication(config).run()


if __name__ == "__main__":
    main()
