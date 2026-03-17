from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, Field

from clawbot.browser import BrowserController
from clawbot.desktop import DesktopCapabilities


class NewTabArgs(BaseModel):
    url: Optional[str] = Field(default=None, description="Optional URL to open in the new tab")


class GotoArgs(BaseModel):
    url: str = Field(description="URL to navigate to")


class SearchArgs(BaseModel):
    query: str = Field(description="Search query")
    engine: str = Field(default="duckduckgo", description="Search engine: duckduckgo, google, bing, startpage, brave")


class PageTextArgs(BaseModel):
    max_chars: int = Field(default=12000, ge=500, le=50000, description="Max characters to return")


class ScreenshotArgs(BaseModel):
    path: Optional[str] = Field(default=None, description="Where to save screenshot (default: ./clawbot_screenshot.png)")
    full_page: bool = Field(default=True, description="Capture full scrollable page")


class SearchAndExtractArgs(BaseModel):
    query: str = Field(description="Search query")
    engine: str = Field(default="duckduckgo", description="Search engine: duckduckgo, google, bing, startpage, brave")
    max_chars: int = Field(default=12000, ge=500, le=50000, description="Max extracted characters to return")


class GotoKnownSiteArgs(BaseModel):
    site: str = Field(description="Known site name: alltrails, wikipedia, reddit, github, etc. Or provide a full URL.")


class NotifyArgs(BaseModel):
    title: str = Field(default="Clawbot", description="Notification title")
    message: str = Field(description="Notification message")
    sound: Optional[str] = Field(default=None, description="Optional macOS sound name (e.g. 'Glass')")


class ClipboardArgs(BaseModel):
    text: str = Field(description="Text to copy to clipboard")


class OpenTargetArgs(BaseModel):
    target: str = Field(description="URL or file path to open")

class SayArgs(BaseModel):
    text: str = Field(description="Text to speak aloud using macOS 'say'")
    voice: Optional[str] = Field(default=None, description="Optional voice name")


def build_browser_tools(browser: BrowserController):
    """
    Returns LangChain-compatible tools (callables) with Pydantic schemas.

    We keep this in a separate module so the agent can be framework-agnostic.
    """
    from langchain_core.tools import StructuredTool

    async def _new_tab(url: str | None = None) -> str:
        return await browser.new_tab(url=url)

    async def _goto(url: str) -> str:
        return await browser.goto(url=url)

    async def _search(query: str, engine: str = "duckduckgo") -> str:
        result = await browser.search(query=query, engine=engine)
        # Check for CAPTCHA and suggest fallback
        if await browser.has_captcha():
            return f"{result}\n\n⚠️ CAPTCHA detected. Try a different engine (bing, startpage, brave) or navigate directly to a specific website."
        return result

    async def _page_text(max_chars: int = 12000) -> str:
        return await browser.page_text(max_chars=max_chars)

    async def _screenshot(path: str | None = None, full_page: bool = True) -> str:
        return await browser.screenshot(path=path, full_page=full_page)

    async def _search_and_extract(query: str, engine: str = "duckduckgo", max_chars: int = 12000) -> str:
        # Single-step "browse" primitive: search then extract readable text from results page.
        # Try multiple engines if we hit a CAPTCHA.
        engines_to_try = [engine, "bing", "startpage", "brave"]
        last_error = None
        
        for eng in engines_to_try:
            try:
                await browser.search(query=query, engine=eng)
                # Check for CAPTCHA
                if await browser.has_captcha():
                    last_error = f"CAPTCHA detected on {eng}"
                    continue
                text = await browser.page_text(max_chars=max_chars)
                if text and len(text.strip()) > 100:  # Got meaningful content
                    return f"URL: (search results from {eng})\n\n{text}"
            except Exception as e:
                last_error = f"{eng}: {e}"
                continue
        
        return f"Error: Could not search (tried {', '.join(engines_to_try)}). {last_error or 'All engines failed'}. Try navigating directly to a specific website instead."

    async def _goto_known_site(site: str) -> str:
        """Navigate to a known site by name or URL."""
        known_sites = {
            "alltrails": "https://www.alltrails.com",
            "wikipedia": "https://www.wikipedia.org",
            "reddit": "https://www.reddit.com",
            "github": "https://github.com",
            "stackoverflow": "https://stackoverflow.com",
        }
        site_lower = site.lower().strip()
        if site_lower in known_sites:
            url = known_sites[site_lower]
        elif site.startswith("http://") or site.startswith("https://"):
            url = site
        else:
            # Try to construct a URL
            url = f"https://{site}"
        return await browser.goto(url)

    return [
        StructuredTool.from_function(
            name="browser_new_tab",
            description="Open a new browser tab. Optionally open a URL in that tab.",
            coroutine=_new_tab,
            args_schema=NewTabArgs,
        ),
        StructuredTool.from_function(
            name="browser_goto",
            description="Navigate the active tab to a URL.",
            coroutine=_goto,
            args_schema=GotoArgs,
        ),
        StructuredTool.from_function(
            name="browser_search",
            description="Search the web using a search engine in the active tab.",
            coroutine=_search,
            args_schema=SearchArgs,
        ),
        StructuredTool.from_function(
            name="browser_page_text",
            description="Extract visible page text from the active tab (for summarizing / answering).",
            coroutine=_page_text,
            args_schema=PageTextArgs,
        ),
        StructuredTool.from_function(
            name="browser_screenshot",
            description="Take a screenshot of the active tab and save it locally.",
            coroutine=_screenshot,
            args_schema=ScreenshotArgs,
        ),
        StructuredTool.from_function(
            name="browser_search_and_extract",
            description="Search the web and immediately extract page text from the results page. Use this first for quick research. Automatically tries fallback engines if CAPTCHA is detected.",
            coroutine=_search_and_extract,
            args_schema=SearchAndExtractArgs,
        ),
        StructuredTool.from_function(
            name="browser_goto_known_site",
            description="Navigate directly to a known website (alltrails, wikipedia, reddit, github, stackoverflow) or any URL. Use this to bypass search engines when they show CAPTCHAs.",
            coroutine=_goto_known_site,
            args_schema=GotoKnownSiteArgs,
        ),
    ]


def build_desktop_tools(desktop: DesktopCapabilities | None = None):
    """
    macOS-only desktop integration tools.
    """
    from langchain_core.tools import StructuredTool

    desk = desktop or DesktopCapabilities(enabled=True)

    def _notify(title: str = "Clawbot", message: str = "", sound: str | None = None) -> str:
        return desk.notify(title=title, message=message, sound=sound)

    def _clipboard_set(text: str) -> str:
        return desk.clipboard_set(text=text)

    def _open_target(target: str) -> str:
        return desk.open_target(target=target)

    def _say(text: str, voice: str | None = None) -> str:
        return desk.say(text=text, voice=voice)

    return [
        StructuredTool.from_function(
            name="desktop_notify",
            description="Send a macOS notification to the user.",
            func=_notify,
            args_schema=NotifyArgs,
        ),
        StructuredTool.from_function(
            name="desktop_clipboard_set",
            description="Copy text to the macOS clipboard.",
            func=_clipboard_set,
            args_schema=ClipboardArgs,
        ),
        StructuredTool.from_function(
            name="desktop_open",
            description="Open a URL or file path using the macOS default handler.",
            func=_open_target,
            args_schema=OpenTargetArgs,
        ),
        StructuredTool.from_function(
            name="desktop_say",
            description="Speak text aloud using macOS text-to-speech (useful when notifications are suppressed).",
            func=_say,
            args_schema=SayArgs,
        ),
    ]

