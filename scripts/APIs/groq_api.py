
from openai import OpenAI
import os


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
    parts.append("SOURCE_CODE:")
    parts.append(source_code)
    return "\n\n".join(parts)


def call_groq(prompt, model="openai/gpt-oss-20b"):
    api=os.environ.get("GROQ_API_KEY")
    print(api)
    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return response.output_text


def generate_code(mode, source_code, dataset_headers=""):
    prompt_text = build_prompt(mode, source_code, dataset_headers)
    return call_groq(prompt_text)

