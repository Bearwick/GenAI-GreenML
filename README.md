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
   1. [Free LLM API](#free-llm-groq)
5. [Generate LLM Code](#generate-llm-code)
   1. [Delete Old Generated Code](#delete-old-generated-code)
   2. [Runability Check](#runability-check)
6. [Run Projects and Capture Telemetry](#run-projects-and-capture-telemetry)
7. [Analyse Results](#analyse-results)
8. [Generate Code Iterations (Failed Scripts)](#generate-code-iterations-failed-scripts)
9. [Analyse Failed Code](#analyse-failed-code)

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

If the repository includes multiple python files, rename the correct file to `original.py`

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

### Free LLM Groq

Groq is a free LLM API but is currenlty commented out because of performance. If wanted, uncomment where 'groq' appears in `generate_llm_code.py`

## Generate LLM Code

```
cd scripts
source venv/bin/activate
python ./generate_llm_code.py
```

### Delete Old Generated Code

To delete old generated code, i.e., files with prefix `GENAIGREENML`, run the following from project root:

```
./scripts/delete_generated_ml.sh
```

### Runability Check

This scripts verifies that all ML projects runs. You can skip this check for now, but it exists if there are any problems with the telemetry script below.
Note: it only runs the original python code in each ML projects, i.e., it skips if there are multiple python files. That also means, any plots or visuals will be shown.

```
./scripts/runability_check.sh
```

## Run Projects and Capture Telemetry

From project root:

```
python scripts/run_ml_projects.py
```

See output in the results folder.

## Analyse Results

The analysis compares assisted and autonomous LLM code against the original code on accuracy, execution time and energy consumption, and prints if they have increased, decreased or are equal.

From project root:

```
python ./scripts/analyse_results.py
```

### CLI Flag

Select a specific CSV file from the results/ folder.

```
python3 ./scripts/analyse_results.py --results-file results_20260220_135839.csv
```

## Generate Code Iterations (Failed Scripts)

The script `move_failed_scripts.py` moves ML projects with failed scripts, except for GENAIGREENML\* files that ran correctly. The ML projects are copied to a folder `failed_generated_code_iteration_N` and then the failed scripts are removed from the ML projects in `repos`.

```
python ./scripts/move_failed_scripts.py
```

## Analyse Failed Code

To analyse the errors occured in generated files. To analyse all iterations use:

```
python ./scripts/analyse_failed_scripts.py
```

### CLI Flag

To analyse a specific iteration:

```
python ./scripts/analyse_failed_scripts.py --iteration failed_generated_code_iteration_1
```
