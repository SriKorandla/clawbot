## Clawbot (OpenClaw-like local browser agent)

`clawbot` is a small local agent you can chat with from your terminal. It can **open Chrome**, **create a new tab**, **search the web**, **read page text**, and **take screenshots** using Playwright.

It’s designed to use **Dartmouth Chat via `langchain_dartmouth`** (see the cookbook at `https://dartmouth.github.io/langchain-dartmouth-cookbook/00-intro.html`).

### What you get

- **Tool-calling agent**: the model decides when to use browser tools
- **Chrome control**: new tab, navigate, search, extract text, screenshot
- **Desktop integration (macOS)**: notifications, clipboard copy, open a URL/file
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

**Optional (recommended for macOS notifications)**: Install `terminal-notifier` for more reliable desktop notifications:

```bash
brew install terminal-notifier
```

Clawbot will automatically use `terminal-notifier` if available, otherwise falls back to `osascript`.

### Configure Dartmouth

Set your Dartmouth Chat API key (see the [cookbook](https://dartmouth.github.io/langchain-dartmouth-cookbook/02-authentication.html)):

```bash
export DARTMOUTH_CHAT_API_KEY="..."
```

You can also create `clawbot/.env`:

```bash
DARTMOUTH_CHAT_API_KEY=...
```

**Default Model**: Clawbot uses `openai.gpt-oss-120b` by default (ChatDartmouth's default). To use a different model:

```bash
clawbot "your prompt" --model anthropic.claude-3-5-haiku-20241022
```

List all available models (including which support tool calling):

```bash
clawbot --list-models
```

See the [LLM cookbook page](https://dartmouth.github.io/langchain-dartmouth-cookbook/03-llms.html) for details on available models.

### Run

```bash
clawbot "Open a new tab, search for 'best hiking trails near Hanover NH', and summarize the top 3 results."
```

### Desktop integration examples (macOS)

```bash
clawbot "When you're done, send me a notification and copy the summary to my clipboard." --chromium --headless
```

### Notes / Troubleshooting

- **Use a venv**: Playwright can conflict with other globally-installed packages (you hit this already). Using `clawbot/.venv` keeps it isolated.
- **Using your installed Google Chrome**: pass `--use-chrome` to launch Chrome (best effort) instead of Playwright's bundled Chromium.
- **Using Playwright Chromium**: pass `--chromium`.
- **Search engine**: Clawbot uses DuckDuckGo by default (Google often blocks headless automation). You can still ask it to "search Google" explicitly.
- **Truncated answers**: increase model output length with `--max-tokens 1500` (Dartmouth defaults can be low).
- **First run**: Playwright needs browser binaries (`python -m playwright install`).
- **macOS notifications not showing**: Clawbot uses `terminal-notifier` by default (if installed via `brew install terminal-notifier`), which is more reliable than `osascript`. If `terminal-notifier` is not installed, it falls back to `osascript`, which may require notification permissions for Terminal/iTerm in System Settings → Notifications. Also check Do Not Disturb mode.

