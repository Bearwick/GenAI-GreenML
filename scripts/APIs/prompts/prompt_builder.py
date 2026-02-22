#!/usr/bin/env python3
"""Shared prompt construction helpers for API wrappers."""

from __future__ import annotations

from pathlib import Path


def load_prompt_template(mode: str) -> str:
    prompt_dir = Path(__file__).resolve().parent
    prompt_path = prompt_dir / f"{mode}.prompt"
    if not prompt_path.is_file():
        raise RuntimeError(f"Missing prompt template: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def build_prompt(mode: str, source_code: str, dataset_headers: str, exampleRowDataset: str, datasetPath: str, projectContext: str) -> str:
    template = load_prompt_template(mode)
    parts = [template]

    if mode != "original_telemetry":
        if datasetPath:
            parts.append("DATASET_PATH:")
            parts.append(datasetPath)

        if dataset_headers:
            parts.append("DATASET_HEADERS:")
            parts.append(dataset_headers)

        if exampleRowDataset:
            parts.append("EXAMPLE_ROW_DATASET:")
            parts.append(exampleRowDataset)

        if projectContext:
            parts.append("ML_PROJECT_CONTEXT:")
            parts.append(projectContext)

    if mode != "autonomous":
        parts.append("SOURCE_CODE:")
        parts.append(source_code)


    return "\n\n".join(parts)
