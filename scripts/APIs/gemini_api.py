#!/usr/bin/env python3
"""Minimal Gemini API call wrapper using the Google GenAI SDK.

Docs: https://ai.google.dev/gemini-api/docs/quickstart
Env:
- GEMINI_API_KEY
"""

from google import genai
import os
import sys
from prompts.prompt_builder import build_prompt


def load_env_file(path):
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("\"'")  # strip optional quotes
            if key and key not in os.environ:
                os.environ[key] = val


def call_gemini(prompt, model="gemini-3-flash-preview"):
    # The client picks up GEMINI_API_KEY from the environment.
    repo_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "API.env"))
    load_env_file(repo_env)
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text


if __name__ == "__main__":
    # Read source code from stdin and build prompt based on mode + optional dataset headers.
    mode = "original_telemetry"
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]

    source_code = sys.stdin.read().strip()
    dataset_headers = os.getenv("DATASET_HEADERS", "").strip()
    if "--headers" in sys.argv:
        idx = sys.argv.index("--headers")
        if idx + 1 < len(sys.argv):
            dataset_headers = sys.argv[idx + 1].strip()

    prompt_text = build_prompt(mode, source_code, dataset_headers)
    print(call_gemini(prompt_text))


def generate_code(mode, source_code, dataset_headers=""):
    prompt_text = build_prompt(mode, source_code, dataset_headers)
    return call_gemini(prompt_text)
