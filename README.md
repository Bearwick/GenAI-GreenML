# GenAI-GreenML

**GenAI-GreenML** is an open-source dataset consisting of curated machine learning repositories collected for research on generative artificial intelligence and sustainable software development. The dataset includes 100 small-scale (<500 MB) GitHub projects focused on tabular and natural language processing (NLP) tasks. Each repository contains both source code and datasets that enable reproducible experiments on model performance, energy consumption, and green coding practices.

This dataset was developed as part of a master’s thesis at the **Norwegian University of Science and Technology (NTNU)**, investigating how large language models (LLMs) can assist in producing more energy-efficient machine learning code.

## Table of Contents

1. [Repository Information](#repository-information)
   1. [Repository Content](#repository-content)
   2. [Intended Use](#intended-use)
   3. [Citation](#citation)
2. [Adding ML Projects](#adding-ml-projects)
3. [Environment Variables Setup](#environment-variables-setup)
4. [Adding LLMs](#adding-llms)
5. [Generate LLM Code](#generate-llm-code)
   1. [Runability Check](#runability-check)
   2. [Running](#running)
6. [Run Projects and Capture Telemetry](#run-projects-and-capture-telemetry)

## Repository Information

### Repository Content

- 100 open-source ML repositories (tabular and NLP tasks)
- Metadata on repository size, task type, and primary programming language
- Experiment design and evaluation scripts for LLM-assisted code generation
- Benchmarking tools for energy consumption, FLOPS, and model performance

### Intended Use

The dataset supports reproducible research on:

- LLM-based code generation and refactoring
- Sustainable and energy-aware machine learning practices
- Comparative benchmarking of original vs. AI-generated implementations

### Citation

If you use this dataset in academic work, please cite it as:

> **Bjørnevik, E. (2025).** _GenAI-GreenML: A Dataset for Evaluating Generative AI in Green Machine Learning Code._ NTNU, Department of Computer Science.

## Adding ML Projects

To add a new ML project run the following from project root:

```
./scripts/import_repo.sh [GitHub url]
```

## Environment Variables Setup

Run from repo root to create `API.env`:

```bash
cat <<'EOF' > API.env
# Gemini
GEMINI_API_KEY=

# OpenAI
OPENAI_API_KEY=

# Claude
export ANTHROPIC_API_KEY=

# Groq
export GROQ_API_KEY=
EOF
```

Then source API.env:

```
source API.env
```

## Adding LLMs

Add LLM APIs in scripts/APIs and make sure the module expose:<br>generate_code(mode, source_code, headers) -> str

Then in [generate_llm_code.py](./scripts/generate_llm_code.py):

1. Update DEFAULT_ALL_LLMS on line 75 with the LLM name.

2. Register the module in load_llm_clients.<br>
   Add a line in the registry section:
   registry["newllm"] = \_try_load("newllm", "newllm_api")
   The module must expose:
   generate_code(mode, source_code, headers) -> str

Finally, add API key to API.env and source from project root:

```
source API.env
```

## Generate LLM Code

```
cd scripts
source venv/bin/activate
./generate_llm_code.sh
```

### Runability Check (might delete)

```
./scripts/runability_check.sh
```

#### If no requirements.txt

```
cd ./repos/[project path]
python3 -m venv venv
pip freeze > requirements.txt
```

## Run Projects and Capture Telemetry

From project root:

```
python scripts/run_ml_projects.py
```

See output in the results folder.
