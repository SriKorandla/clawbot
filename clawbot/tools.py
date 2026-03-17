from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, Field

from clawbot.browser import BrowserController


class NewTabArgs(BaseModel):
    url: Optional[str] = Field(default=None, description="Optional URL to open in the new tab")


class GotoArgs(BaseModel):
    url: str = Field(description="URL to navigate to")


class SearchArgs(BaseModel):
    query: str = Field(description="Search query")
    engine: str = Field(default="duckduckgo", description="Search engine: duckduckgo, google, bing")


class PageTextArgs(BaseModel):
    max_chars: int = Field(default=12000, ge=500, le=50000, description="Max characters to return")


class ScreenshotArgs(BaseModel):
    path: Optional[str] = Field(default=None, description="Where to save screenshot (default: ./clawbot_screenshot.png)")
    full_page: bool = Field(default=True, description="Capture full scrollable page")


class SearchAndExtractArgs(BaseModel):
    query: str = Field(description="Search query")
    engine: str = Field(default="duckduckgo", description="Search engine: duckduckgo, google, bing")
    max_chars: int = Field(default=12000, ge=500, le=50000, description="Max extracted characters to return")


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
        return await browser.search(query=query, engine=engine)

    async def _page_text(max_chars: int = 12000) -> str:
        return await browser.page_text(max_chars=max_chars)

    async def _screenshot(path: str | None = None, full_page: bool = True) -> str:
        return await browser.screenshot(path=path, full_page=full_page)

    async def _search_and_extract(query: str, engine: str = "duckduckgo", max_chars: int = 12000) -> str:
        # Single-step “browse” primitive: search then extract readable text from results page.
        await browser.search(query=query, engine=engine)
        text = await browser.page_text(max_chars=max_chars)
        return f"URL: (search results)\n\n{text}"

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
            description="Search the web and immediately extract page text from the results page. Use this first for quick research.",
            coroutine=_search_and_extract,
            args_schema=SearchAndExtractArgs,
        ),
    ]

