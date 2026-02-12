# Environment Variables Setup

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

# Adding LLMs

Add LLM API under the APIs folder and make sure the module exposes:<br>generate_code(mode, source_code, headers) -> str

## generate_llm_code.py

1. Update DEFAULT_ALL_LLMS on line 75 with the LLM name.

2. Register the module in load_llm_clients.<br>
   Add a line in the registry section:
   registry["newllm"] = \_try_load("newllm", "newllm_api")
   The module must expose:
   generate_code(mode, source_code, headers) -> str
3. Add API key to API.env and source from project root:
   ```
   source API.env
   ```
