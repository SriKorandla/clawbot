from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from playwright.async_api import BrowserContext, Page, async_playwright


DEFAULT_CHROME_MAC = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


@dataclass
class BrowserConfig:
    headless: bool = False
    use_chrome: bool = True
    chrome_executable_path: Optional[str] = None
    user_data_dir: Optional[str] = None
    viewport_width: int = 1280
    viewport_height: int = 800


class BrowserController:
    """
    Thin wrapper around Playwright that maintains a single persistent context.

    The agent tools call into this to create a new tab, navigate, search, read text, etc.
    """

    def __init__(self, config: BrowserConfig | None = None):
        self.config = config or BrowserConfig()
        self._pw = None
        self._context: BrowserContext | None = None
        self._active_page: Page | None = None

    async def start(self) -> None:
        if self._context:
            return

        self._pw = await async_playwright().start()

        user_data_dir = self._resolve_user_data_dir(self.config.user_data_dir)
        launch_kwargs = {
            "headless": self.config.headless,
            "viewport": {"width": self.config.viewport_width, "height": self.config.viewport_height},
        }

        if self.config.use_chrome:
            exec_path = self.config.chrome_executable_path or os.environ.get("CHROME_EXECUTABLE_PATH")
            if not exec_path and Path(DEFAULT_CHROME_MAC).exists():
                exec_path = DEFAULT_CHROME_MAC
            if exec_path:
                launch_kwargs["executable_path"] = exec_path
            launch_kwargs["channel"] = "chrome"

        # Persistent context is the easiest way to keep state across tool calls
        self._context = await self._pw.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            **launch_kwargs,
        )

        pages = self._context.pages
        self._active_page = pages[0] if pages else await self._context.new_page()

    async def close(self) -> None:
        if self._context:
            await self._context.close()
            self._context = None
        if self._pw:
            await self._pw.stop()
            self._pw = None

    def _require_page(self) -> Page:
        if not self._active_page:
            raise RuntimeError("Browser not started or no active page.")
        return self._active_page

    @staticmethod
    def _resolve_user_data_dir(user_data_dir: str | None) -> str:
        base = user_data_dir or os.environ.get("CLAWBOT_USER_DATA_DIR")
        if base:
            return str(Path(base).expanduser())
        # keep it local to project by default
        return str(Path(".clawbot_chrome_profile").absolute())

    async def new_tab(self, url: str | None = None) -> str:
        await self.start()
        assert self._context is not None
        self._active_page = await self._context.new_page()
        if url:
            await self.goto(url)
        return "Opened a new tab."

    async def goto(self, url: str) -> str:
        page = self._require_page()
        await page.goto(url, wait_until="domcontentloaded")
        return f"Navigated to {page.url}"

    async def search(self, query: str, engine: str = "duckduckgo") -> str:
        engine = (engine or "duckduckgo").lower().strip()
        if engine == "google":
            url = "https://www.google.com/search?q=" + _urlencode(query)
        elif engine in ("ddg", "duckduckgo"):
            url = "https://duckduckgo.com/?q=" + _urlencode(query)
        elif engine == "bing":
            url = "https://www.bing.com/search?q=" + _urlencode(query)
        elif engine == "startpage":
            url = "https://www.startpage.com/sp/search?query=" + _urlencode(query)
        elif engine == "brave":
            url = "https://search.brave.com/search?q=" + _urlencode(query)
        else:
            raise ValueError("engine must be one of: google, duckduckgo, bing, startpage, brave")
        await self.goto(url)
        return f"Searched {engine} for: {query}"

    async def has_captcha(self) -> bool:
        """Check if the current page shows a CAPTCHA."""
        page = self._require_page()
        await page.wait_for_load_state("domcontentloaded")
        # Check for common CAPTCHA indicators
        captcha_indicators = [
            "captcha",
            "verify you're human",
            "verify you are human",
            "prove you're not a robot",
            "prove you are not a robot",
            "challenge",
        ]
        text = (await self.page_text(max_chars=5000)).lower()
        title = (await page.title()).lower()
        url = page.url.lower()
        combined = f"{title} {text} {url}"
        return any(indicator in combined for indicator in captcha_indicators)

    async def page_text(self, max_chars: int = 12_000) -> str:
        page = self._require_page()
        await page.wait_for_load_state("domcontentloaded")
        text = await page.evaluate(
            """() => {
              const bad = new Set(["SCRIPT","STYLE","NOSCRIPT"]);
              const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
              let out = [];
              let n;
              while ((n = walker.nextNode())) {
                const p = n.parentElement;
                if (!p || bad.has(p.tagName)) continue;
                const s = n.textContent.replace(/\\s+/g, " ").trim();
                if (s.length >= 2) out.push(s);
              }
              return out.join("\\n");
            }"""
        )
        text = (text or "").strip()
        if len(text) > max_chars:
            return text[:max_chars] + "\n... (truncated)"
        return text

    async def screenshot(self, path: str | None = None, full_page: bool = True) -> str:
        page = self._require_page()
        out = Path(path or "clawbot_screenshot.png").expanduser().absolute()
        out.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(out), full_page=full_page)
        return f"Saved screenshot to {out}"


def _urlencode(s: str) -> str:
    from urllib.parse import quote_plus

    return quote_plus(s)

