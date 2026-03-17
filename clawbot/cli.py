from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

from clawbot.agent import AgentConfig, ClawbotAgent
from clawbot.browser import BrowserConfig
from clawbot.llm.dartmouth import DartmouthLLMConfig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clawbot", description="OpenClaw-like local browser agent (Playwright + Dartmouth).")
    p.add_argument("prompt", nargs="*", help="What you want the agent to do (use quotes).")
    p.add_argument("--headless", action="store_true", help="Run browser headless.")
    p.add_argument("--use-chrome", action="store_true", help="Use installed Google Chrome (best effort).")
    p.add_argument("--chromium", action="store_true", help="Use Playwright Chromium instead of Chrome.")
    p.add_argument("--chrome-path", default=None, help="Path to Chrome executable (overrides autodetect).")
    p.add_argument("--profile-dir", default=None, help="Persistent profile dir for browser context.")
    p.add_argument("--model", default=None, help="Dartmouth model name (default: openai.gpt-oss-120b). Use --list-models to see available models.")
    p.add_argument("--list-models", action="store_true", help="List available Dartmouth models and exit.")
    p.add_argument("--api-key-env", default="DARTMOUTH_CHAT_API_KEY", help="Env var holding Dartmouth API key.")
    p.add_argument("--max-tokens", type=int, default=2000, help="Max tokens for the Dartmouth model output (default: 2000).")
    p.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature.")
    p.add_argument("--max-iters", type=int, default=12, help="Max tool iterations.")
    p.add_argument("--verbose", action="store_true", help="Show debug output.")
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _build_parser().parse_args(argv)

    # Handle --list-models
    if args.list_models:
        try:
            from langchain_dartmouth.llms import ChatDartmouth
            models = ChatDartmouth.list()
            print("Available Dartmouth Chat Models:\n")
            print(f"{'Model ID':<50} {'Name':<40} {'Tool Calling':<15} {'Cost':<10}")
            print("-" * 115)
            for model in models:
                tool_calling = "✓" if "tool calling" in (model.capabilities or []) else ""
                cost = model.cost or "free" if model.is_local else (model.cost or "?")
                print(f"{model.id:<50} {model.name:<40} {tool_calling:<15} {cost:<10}")
            print("\nUse --model <model_id> to select a model.")
            print("See: https://dartmouth.github.io/langchain-dartmouth-cookbook/03-llms.html")
            return 0
        except ImportError:
            print("Error: langchain_dartmouth is not installed.", file=sys.stderr)
            print('Install with: pip install -e ".[dartmouth]"', file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error listing models: {e}", file=sys.stderr)
            return 1

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        print("Provide a prompt, e.g.:\n  clawbot \"Open a new tab and search for ...\"", file=sys.stderr)
        print("\nOr list available models:\n  clawbot --list-models", file=sys.stderr)
        return 2

    browser_cfg = BrowserConfig(
        headless=bool(args.headless),
        use_chrome=bool(args.use_chrome) and not bool(args.chromium),
        chrome_executable_path=args.chrome_path,
        user_data_dir=args.profile_dir,
    )
    llm_cfg = DartmouthLLMConfig(
        api_key_env=args.api_key_env,
        model_name=args.model,
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
    )
    cfg = AgentConfig(max_tool_iterations=int(args.max_iters), browser=browser_cfg, llm=llm_cfg, verbose=bool(args.verbose))

    try:
        out = asyncio.run(ClawbotAgent(cfg).run_once(prompt))
        print(out)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

