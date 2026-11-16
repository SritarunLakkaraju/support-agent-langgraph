"""Optional Gemini access.

Every project has an offline (deterministic, rule-based) implementation of each "LLM step", so the
whole repo runs and is testable with no API key. When GEMINI_API_KEY (or GOOGLE_API_KEY) is set and the
google-genai package is installed, the projects can swap in Gemini for those steps via GeminiLLM.

Set GEMINI_MODEL to override the default model name.
"""
from __future__ import annotations

import json
import os


def gemini_available() -> bool:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return False
    try:
        from google import genai  # noqa: F401
        return True
    except ImportError:
        return False


class GeminiLLM:
    def __init__(self, model: str | None = None):
        from google import genai

        self.client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
        return self.client.models.generate_content(model=self.model, contents=prompt, config=cfg).text

    def generate_json(self, prompt: str, system: str | None = None) -> dict:
        """Ask for JSON only and parse it (strips ``` fences if the model adds them)."""
        from google.genai import types

        cfg = types.GenerateContentConfig(system_instruction=system, temperature=0.0,
                                          response_mime_type="application/json")
        text = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg).text
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        return json.loads(text)
