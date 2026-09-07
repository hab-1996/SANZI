"""Manual FP16 smoke test for both local SANZI models."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Use project-relative model locations without depending on the current directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLAMA_PATH = PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct"
QWEN_PATH = PROJECT_ROOT / "models" / "Qwen1.5-0.5B-Chat"


# Load one model and generate a short FP16 response
def chat(model_path: Path, prompt: str) -> None:
    """Load one local model and print a short FP16 response."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float16,
        device_map="auto",
    )


    # Format a single user message with the model's own chat template.
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        do_sample=True,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)


# Run the FP16 smoke test for both models
def main() -> None:
    print("\n--- Llama response ---")
    chat(LLAMA_PATH, "Explain artificial intelligence in simple words.")

    print("\n--- Qwen response ---")
    chat(QWEN_PATH, "Explain artificial intelligence in simple words.")


if __name__ == "__main__":
    main()
