import gc
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MetalConfig


# Project paths and model options
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "1": {
        "name": "Llama-3.2-1B-Instruct",
        "path": PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct",
    },
    "2": {
        "name": "Qwen1.5-0.5B-Chat",
        "path": PROJECT_ROOT / "models" / "Qwen1.5-0.5B-Chat",
    },
}

# Map menu choices to the supported precision modes.
PRECISIONS = {
    "1": {"id": "fp16", "name": "FP16"},
    "2": {"id": "int8", "name": "8-bit Metal"},
    "3": {"id": "int4", "name": "4-bit Metal"},
}

# Limit input tokens to reduce memory use during generation.
SAFE_TOKEN_LIMIT = 2500
MODEL_LOAD_OOM_MESSAGE = (
    "Model loading failed because there is not enough memory. "
    "Try Qwen instead of Llama, use 8-bit or 4-bit Metal when available, "
    "or close other memory-heavy applications."
)


class ModelLoadOOMError(RuntimeError):
    """Report an out-of-memory failure while loading model weights."""


# Load a local model in FP16, 8-bit Metal, or 4-bit Metal
def load_model(model_path: Path, quantized: bool = False, bits: int = 4):
    """Load one local model with the selected precision."""
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        # Disable legacy whitespace cleanup for BPE tokenizers.
        tokenizer.clean_up_tokenization_spaces = False

        if quantized:
            # Quantized weights require an available Apple MPS/Metal device.
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    f"{bits}-bit Metal requires an Apple "
                    "Silicon Mac with an available MPS backend."
                )
            # Use the same Metal weight quantization as the backend.
            quantization_config = MetalConfig(bits=bits, group_size=64)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                device_map={"": "mps"} if bits == 8 else "mps",
                quantization_config=quantization_config,
            )
        else:
            # Automatic placement selects MPS on the tested Apple Silicon system.
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=torch.float16,
                device_map="auto",
            )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if tokenizer is not None:
                del tokenizer
            clear_model_memory()
            raise ModelLoadOOMError(MODEL_LOAD_OOM_MESSAGE) from exc
        raise

    return tokenizer, model


# Clear model memory after switching models
def clear_model_memory() -> None:
    """Release Python and MPS allocations after switching models."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# Generate one chatbot response
def generate_reply(tokenizer, model, conversation, max_tokens, temperature, top_p):
    """Generate one reply and return its input and output token counts."""
    # Apply the selected model's chat template to the complete saved history.
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_tokens = inputs["input_ids"].shape[1]

    # Stop before generation when the conversation exceeds the safe limit.
    if input_tokens > SAFE_TOKEN_LIMIT:
        return None, input_tokens, 0

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    except RuntimeError as exc:
        # Recover from device OOM without discarding the saved conversation.
        if "out of memory" in str(exc).lower():
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return "OOM_ERROR", input_tokens, 0
        raise

    output_tokens = outputs.shape[1] - input_tokens

    # Keep BPE whitespace unchanged while decoding.
    full_response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Both bundled chat templates mark the assistant portion in decoded output.
    if "assistant" in full_response:
        reply = full_response.split("assistant")[-1].strip()
    else:
        reply = full_response.strip()
    return reply, input_tokens, output_tokens


# Read and validate a numeric setting
def ask_number(prompt, default, min_value, max_value, number_type=float):
    """Prompt until the user enters a number inside the allowed range."""
    while True:
        value = input(
            f"{prompt} ({min_value}-{max_value}, default {default}): "
        ).strip()
        if value == "":
            return default

        try:
            value = number_type(value)
            if min_value <= value <= max_value:
                return value
            print(f"Please enter a value between {min_value} and {max_value}.")
        except ValueError:
            print("Please enter a valid number.")


# Run the command-line chatbot
def main() -> None:
    # Collect the model, precision, and generation settings before loading weights.
    print("Choose a model:")
    print("1. Llama-3.2-1B-Instruct")
    print("2. Qwen1.5-0.5B-Chat")
    choice = input("Enter 1 or 2: ").strip()

    if choice not in MODELS:
        print("Invalid choice.")
        return
    selected = MODELS[choice]

    # Choose between normal FP16 and Metal weight quantization.
    print("\nChoose model precision:")
    print("1. FP16 (normal)")
    print("2. 8-bit Metal")
    print("3. 4-bit Metal")
    precision_choice = input("Enter 1, 2, or 3: ").strip()

    if precision_choice not in PRECISIONS:
        print("Invalid choice.")
        return
    precision = PRECISIONS[precision_choice]

    # Let the user configure response length and sampling.
    print("\nChoose generation settings:")
    max_tokens = ask_number("Max tokens", 150, 20, 300, int)
    temperature = ask_number("Temperature", 0.7, 0.1, 1.5, float)
    top_p = ask_number("Top-p", 0.9, 0.1, 1.0, float)
    seed = ask_number("Seed", 42, 0, 1_000_000, int)

    print("\nSelected settings:")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Top-p: {top_p}")
    print(f"Seed: {seed}")
    print(f"Precision: {precision['name']}")
    print(f"\nLoading {selected['name']}...")

    try:
        tokenizer, model = load_model(
            selected["path"],
            quantized=precision["id"] != "fp16",
            bits=8 if precision["id"] == "int8" else 4,
        )
    except ModelLoadOOMError as exc:
        print(f"\n{exc}")
        return
    

    print(f"{selected['name']} is ready. Type 'exit' to stop.\n")
    conversation = []

    # Keep the conversation in memory until the user clears or trims it.
    while True:
        prompt = input("You: ")

        if prompt.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        # Clear every saved conversation round.
        if prompt.lower() == "/clear":
            conversation.clear()
            print("Conversation cleared.\n")
            continue
        # Show the saved user and assistant messages.
        if prompt.lower() == "/history":
            if not conversation:
                print("No conversation history.\n")
                continue

            print("\nConversation History:")
            for index in range(0, len(conversation), 2):
                round_number = index // 2 + 1
                user_message = conversation[index]["content"]
                assistant_message = conversation[index + 1]["content"]
                print(f"\nRound {round_number}")
                print(f"You: {user_message}")
                print(f"Bot: {assistant_message}")
            print()
            continue

        # Switch models without losing conversation history.
        if prompt.lower().startswith("/model"):
            parts = prompt.split()
            if len(parts) != 2 or parts[1] not in MODELS:
                print("Usage: /model 1 or /model 2\n")
                continue

            new_choice = parts[1]
            if new_choice == choice:
                print(f"{selected['name']} is already active.\n")
                continue

            print(
                f"\nSwitching from {selected['name']} "
                f"to {MODELS[new_choice]['name']}..."
            )
            # Release the old weights before loading the new model.
            previous_choice = choice
            previous_selected = selected
            del model
            del tokenizer
            clear_model_memory()

            # Reuse the selected precision for the new model.
            try:
                tokenizer, model = load_model(
                    MODELS[new_choice]["path"],
                    quantized=precision["id"] != "fp16",
                    bits=8 if precision["id"] == "int8" else 4,
                )
            except ModelLoadOOMError as exc:
                print(f"\n{exc}")
                print(f"Restoring {previous_selected['name']}...")
                try:
                    tokenizer, model = load_model(
                        previous_selected["path"],
                        quantized=precision["id"] != "fp16",
                        bits=8 if precision["id"] == "int8" else 4,
                    )
                except ModelLoadOOMError as restore_exc:
                    print(f"Could not restore the previous model. {restore_exc}")
                    print("Goodbye!")
                    return
                choice = previous_choice
                selected = previous_selected
                print("Previous model restored.")
                print("Conversation history kept.\n")
                continue
            choice = new_choice
            selected = MODELS[choice]
            print("Previous model unloaded.")
            print("Conversation history kept.")
            print(f"{selected['name']} is ready.\n")
            continue

        # Remove the oldest rounds to reduce conversation token use.
        if prompt.lower().startswith("/remove"):
            parts = prompt.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("Usage: /remove n\n")
                continue

            rounds_to_remove = int(parts[1])
            messages_to_remove = 2 * rounds_to_remove
            if messages_to_remove > len(conversation):
                conversation.clear()
                print("Removed all available conversation rounds.\n")
            else:
                del conversation[:messages_to_remove]
                print(f"Removed first {rounds_to_remove} conversation round(s).\n")
            continue

        conversation.append({"role": "user", "content": prompt})

        # Reset the seed before each generation
        random.seed(seed)
        torch.manual_seed(seed)        
        reply, input_tokens, output_tokens = generate_reply(
            tokenizer,
            model,
            conversation,
            max_tokens,
            temperature,
            top_p,
        )

        # Reject oversized input without saving the failed user message.
        if reply is None:
            print("\nConversation too large.")
            print(f"Input tokens: {input_tokens}")
            print(f"Safe limit: {SAFE_TOKEN_LIMIT}")
            print("Use /remove n or /clear to reduce the conversation history.\n")
            conversation.pop()
            continue

        # Keep earlier history when the latest generation runs out of memory.
        if reply == "OOM_ERROR":
            print("\nOut of memory detected.")
            print("The model could not generate the response.")
            print("Try /remove n, /clear, or use a smaller Max tokens.\n")
            conversation.pop()
            continue

        conversation.append({"role": "assistant", "content": reply})
        print(f"\nBot: {reply}\n")
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Total tokens: {input_tokens + output_tokens}\n")


if __name__ == "__main__":
    main()
