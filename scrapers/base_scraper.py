"""Base scraper con retry exponencial y rate limiting."""
import time
import random
from abc import ABC, abstractmethod

import requests
from bs4 import BeautifulSoup


class ScraperError(Exception):
    pass


class BaseScraper(ABC):
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "es-UY,es;q=0.9,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def __init__(self, rate_limit: float = 2.0, max_retries: int = 3):
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._last_request = 0.0

    def _wait(self):
        elapsed = time.time() - self._last_request
        wait = self.rate_limit - elapsed + random.uniform(0.2, 0.8)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.time()

    def fetch(self, url: str, **kwargs) -> str:
        for attempt in range(1, self.max_retries + 1):
            self._wait()
            try:
                resp = self.session.get(url, timeout=30, **kwargs)
                if resp.status_code == 403:
                    raise ScraperError(f"403 Forbidden: {url}")
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise ScraperError(f"Fallo tras {attempt} intentos: {e}") from e
                backoff = 2 ** attempt + random.uniform(0, 1)
                print(f"  [reintento {attempt}/{self.max_retries}] espera {backoff:.1f}s — {e}")
                time.sleep(backoff)
        raise ScraperError("max_retries alcanzado")

    def fetch_soup(self, url: str, **kwargs) -> BeautifulSoup:
        html = self.fetch(url, **kwargs)
        return BeautifulSoup(html, "lxml")

    def fetch_with_playwright(self, url: str, wait_selector: str = "body") -> str:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers(self.HEADERS)
            page.goto(url, wait_until="networkidle", timeout=30_000)
            page.wait_for_selector(wait_selector, timeout=15_000)
            html = page.content()
            browser.close()
        return html

    @abstractmethod
    def run(self) -> list[dict]:
        """Ejecuta el scraper y retorna lista de sorteos como dicts."""
        ...
