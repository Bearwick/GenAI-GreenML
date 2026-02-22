
from openai import OpenAI
import os
from prompts.prompt_builder import build_prompt


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
