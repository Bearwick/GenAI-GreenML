#!/usr/bin/env python3
"""Minimal OpenAI API call wrapper using the OpenAI SDK."""

import os
from openai import OpenAI


def load_prompt_template(mode):
    prompt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "prompts"))
    prompt_path = os.path.join(prompt_dir, f"{mode}.prompt")
    if not os.path.isfile(prompt_path):
        raise RuntimeError(f"Missing prompt template: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_prompt(mode, source_code, dataset_headers):
    template = load_prompt_template(mode)
    parts = [template]
    if dataset_headers:
        parts.append("DATASET_HEADERS:")
        parts.append(dataset_headers)
    
    #if mode != "autonomous":
    parts.append("SOURCE_CODE:")
    parts.append(source_code)
    return "\n\n".join(parts)


def call_openai(prompt, model="gpt-5.2-codex"):
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=prompt,
        reasoning={ "effort": "high" }
    )
    return response.output_text


def generate_code(mode, source_code, dataset_headers=""):
    prompt_text = build_prompt(mode, source_code, dataset_headers)
    return call_openai(prompt_text)

