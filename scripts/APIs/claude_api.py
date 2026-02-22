import anthropic
from prompts.prompt_builder import build_prompt


def call_claude(prompt, model="claude-opus-4-6"):
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1000,
         messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
    )
    # Anthropic Messages API returns blocks in response.content, not output_text.
    blocks = getattr(response, "content", None) or []
    text_parts = []
    for block in blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", "")
            if text:
                text_parts.append(text)
    return "\n".join(text_parts).strip()

def generate_code(mode, source_code, dataset_headers=""):
    prompt_text = build_prompt(mode, source_code, dataset_headers)
    return call_claude(prompt_text)
