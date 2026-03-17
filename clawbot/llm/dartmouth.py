from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class DartmouthLLMConfig:
    """
    Minimal config wrapper.

    NOTE: Exact env var / model names may differ depending on `langchain_dartmouth`.
    We keep this small and fail with a helpful message if misconfigured.
    """

    # `langchain_dartmouth` uses `dartmouth_chat_api_key` parameter name.
    # We'll accept multiple env var names for convenience.
    api_key_env: str = "DARTMOUTH_CHAT_API_KEY"
    fallback_api_key_envs: tuple[str, ...] = ("DARTMOUTH_API_KEY",)
    model_name: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.3


def build_dartmouth_chat_model(config: DartmouthLLMConfig | None = None):
    """
    Build a LangChain chat model backed by Dartmouth.

    This uses `langchain_dartmouth` if installed. If not installed (or API key missing),
    we raise a clear error telling you what to do.
    """
    cfg = config or DartmouthLLMConfig()
    api_key = os.environ.get(cfg.api_key_env, "")
    if not api_key:
        for env in cfg.fallback_api_key_envs:
            api_key = os.environ.get(env, "")
            if api_key:
                break
    if not api_key:
        raise RuntimeError(
            f"Missing {cfg.api_key_env}. Set it in your environment (or in clawbot/.env). "
            "See the Dartmouth cookbook: https://dartmouth.github.io/langchain-dartmouth-cookbook/00-intro.html"
        )

    try:
        # `langchain_dartmouth` v0.3.x exposes ChatDartmouth under `langchain_dartmouth.llms`.
        from langchain_dartmouth.llms import ChatDartmouth  # type: ignore

        kwargs = {
            "dartmouth_chat_api_key": api_key,
            "max_tokens": int(cfg.max_tokens),
            "temperature": float(cfg.temperature),
        }
        if cfg.model_name:
            kwargs["model_name"] = cfg.model_name
        return ChatDartmouth(**kwargs)
    except ImportError as e:
        raise RuntimeError(
            "langchain_dartmouth is not installed. Install with:\n"
            '  pip install -e ".[dartmouth]"\n'
            "Then retry. Cookbook: https://dartmouth.github.io/langchain-dartmouth-cookbook/00-intro.html"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Dartmouth chat model: {e}") from e

