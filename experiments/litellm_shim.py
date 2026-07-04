"""
Shim that routes all localrouter traffic through the litellm proxy
(https://litellm.nielsrolf.com) using LITELLM_API_KEY.

Import this module BEFORE importing turingtest / running games:

    import litellm_shim  # noqa: F401

Model-name mapping:
- "anthropic/claude-fable-5" and other litellm-native anthropic ids pass through
- everything else with a provider prefix (e.g. "openai/gpt-5.2",
  "x-ai/grok-4.1-fast") is routed via OpenRouter: "openrouter/" + model
"""
import os
import openai
from localrouter import add_provider
from localrouter.llm import get_response_factory

BASE_URL = os.environ.get("LITELLM_BASE_URL", "https://litellm.nielsrolf.com")
API_KEY = os.environ["LITELLM_API_KEY"]

# Cloudflare blocks the default OpenAI SDK User-Agent
_client = openai.AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    default_headers={"User-Agent": "litellm-client/1.0"},
    timeout=600,
)
_oai_get_response = get_response_factory(_client)

# Models that litellm serves natively (don't reroute through openrouter)
NATIVE_PREFIXES = ("anthropic/claude-fable", "openrouter/", "local/", "gemini/")


REMAP = {
    # deprecated on openrouter -> native gemini route on litellm
    "google/gemini-3-pro-preview": "gemini/gemini-3-pro-preview",
}


def map_model(model: str) -> str:
    if model in REMAP:
        return REMAP[model]
    if model.startswith(NATIVE_PREFIXES):
        return model
    return "openrouter/" + model


async def litellm_get_response(messages, tools=None, response_format=None,
                               reasoning=None, **kwargs):
    kwargs["model"] = map_model(kwargs["model"])
    return await _oai_get_response(messages, tools,
                                   response_format=response_format,
                                   reasoning=reasoning, **kwargs)


import re as _re
add_provider(litellm_get_response, models=[_re.compile(r".+")], priority=1)
