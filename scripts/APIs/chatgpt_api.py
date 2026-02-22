#!/usr/bin/env python3
"""Minimal OpenAI API call wrapper using the OpenAI SDK."""

from openai import OpenAI
from prompts.prompt_builder import build_prompt


def call_openai(prompt, model="gpt-5.2"):
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return response.output_text


def generate_code(mode, source_code, dataset_headers="", exampleRowDataset="", datasetPath="", projectContext=""):
    prompt_text = build_prompt(mode, source_code, dataset_headers, exampleRowDataset, datasetPath, projectContext)
    return call_openai(prompt_text)
