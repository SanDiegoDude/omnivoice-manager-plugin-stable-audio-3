"""Stable Audio 3 prompt construction — ported from the reference comfy workflow.

The workflow drives a small LLM ("reprompt") to turn a short user description
into a detailed, model-ready prompt, steered by a category-specific system
prompt. This module owns that system (the four categories, their system prompts,
and the input template) so the host can run the reprompt step against OmniVoice's
configured Script-AI provider — no bundled LLM, no extra VRAM.

Pure standard library so it imports both in the host venv (to build the reprompt
call) and in the SA3 sidecar venv (which only needs the category list).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

# Canonical category order (mirrors the workflow's Music / Instrument / SFX /
# One-shot combo).
CATEGORIES: List[str] = ["Music", "Instrument", "SFX", "One-shot"]

# The input template from the workflow:
#   "{system}\n\nInput: {user}\nTarget audio length: {N} seconds.\nOutput:"
# The system prompt is sent as the chat system message; this is the user message.
USER_TEMPLATE = "Input: {user_input}\nTarget audio length: {duration} seconds.\nOutput:"

# Sensible default durations (seconds) per category if the UI doesn't supply one,
# matching the ranges the system prompts themselves recommend.
DEFAULT_DURATION: Dict[str, int] = {
    "Music": 180,
    "Instrument": 12,
    "SFX": 8,
    "One-shot": 2,
}

_PROMPTS_FILE = Path(__file__).resolve().parent / "category_prompts.json"
_cache: Dict[str, str] = {}


def load_category_prompts() -> Dict[str, str]:
    """The four system prompts, loaded (and cached) from category_prompts.json."""
    global _cache
    if not _cache:
        _cache = json.loads(_PROMPTS_FILE.read_text())
    return _cache


def normalize_category(name: str) -> str:
    """Map loose user/UI input to a canonical category key."""
    n = (name or "").strip().lower().replace("_", "-").replace(" ", "-")
    table = {
        "music": "Music",
        "instrument": "Instrument",
        "instruments": "Instrument",
        "sfx": "SFX",
        "sound-effect": "SFX",
        "sound-effects": "SFX",
        "foley": "SFX",
        "one-shot": "One-shot",
        "oneshot": "One-shot",
        "one-shots": "One-shot",
    }
    return table.get(n, "SFX")


def system_prompt(category: str) -> str:
    """The category-specific system prompt for the reprompt LLM."""
    cat = normalize_category(category)
    prompts = load_category_prompts()
    return prompts.get(cat) or prompts.get("SFX", "")


def user_message(user_input: str, duration: float) -> str:
    """The user-message half of the reprompt call (Input + target length)."""
    return USER_TEMPLATE.format(user_input=(user_input or "").strip(), duration=int(round(duration)))


def default_duration(category: str) -> int:
    return DEFAULT_DURATION.get(normalize_category(category), 8)
