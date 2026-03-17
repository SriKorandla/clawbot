## Clawbot (OpenClaw-like local browser agent)

`clawbot` is a small local agent you can chat with from your terminal. It can **open Chrome**, **create a new tab**, **search the web**, **read page text**, and **take screenshots** using Playwright.

It’s designed to use **Dartmouth Chat via `langchain_dartmouth`** (see the cookbook at `https://dartmouth.github.io/langchain-dartmouth-cookbook/00-intro.html`).

### What you get

- **Tool-calling agent**: the model decides when to use browser tools
- **Chrome control**: new tab, navigate, search, extract text, screenshot
- **CLI**: `clawbot "check something for me"`

### Setup

From your workspace root:

```bash
cd clawbot
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dartmouth]"
python -m playwright install
```

### Configure Dartmouth

Set your Dartmouth API key (name depends on `langchain_dartmouth` — adjust to whatever the cookbook uses):

```bash
export DARTMOUTH_API_KEY="..."
```

You can also create `clawbot/.env`:

```bash
DARTMOUTH_API_KEY=...
```

### Run

```bash
clawbot "Open a new tab, search for 'best hiking trails near Hanover NH', and summarize the top 3 results."
```

### Notes / Troubleshooting

- **Use a venv**: Playwright can conflict with other globally-installed packages (you hit this already). Using `clawbot/.venv` keeps it isolated.
- **Using your installed Google Chrome**: pass `--use-chrome` to launch Chrome (best effort) instead of Playwright’s bundled Chromium.
- **Using Playwright Chromium**: pass `--chromium`.
- **Search engine**: Clawbot uses DuckDuckGo by default (Google often blocks headless automation). You can still ask it to “search Google” explicitly.
- **Truncated answers**: increase model output length with `--max-tokens 1500` (Dartmouth defaults can be low).
- **First run**: Playwright needs browser binaries (`python -m playwright install`).

