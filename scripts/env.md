# Environment file setup

Copy/paste this from the repo root to create `API.env`:

```bash
cat <<'EOF' > API.env
# Gemini
GEMINI_API_KEY=

# OpenAI
OPENAI_API_KEY=
EOF
```

Then, insert API keys to your desired LLMs.
To add additional LLMs:

1. Add LLM name to the ALL_LLMS list on line 11 in generate_llm_code.sh
2. TODO
