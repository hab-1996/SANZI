from __future__ import annotations

import argparse
import gc
import json
import mimetypes
import random
import re
import threading
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

# Runtime safety limits
SAFE_INPUT_TOKEN_LIMIT = 2500
CONTEXT_WARNING_THRESHOLD = 2000
MAX_GENERATION_SECONDS = 120

# Model configuration
MODELS: dict[str, dict[str, Any]] = {
    "llama": {
        "id": "llama",
        "name": "Llama 3.2 1B",
        "full_name": "Llama-3.2-1B-Instruct",
        "description": "More capable reasoning and instruction following.",
        "parameters": "1B",
        "context_capacity": "128K tokens",
        "path": PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct",
    },
    "qwen": {
        "id": "qwen",
        "name": "Qwen 1.5 0.5B",
        "full_name": "Qwen1.5-0.5B-Chat",
        "description": "Faster, lighter responses with the smallest memory footprint.",
        "parameters": "0.5B",
        "context_capacity": "32K tokens",
        "path": PROJECT_ROOT / "models" / "Qwen1.5-0.5B-Chat",
    },
}

# Precision configuration
PRECISIONS: dict[str, dict[str, str]] = {
    "fp16": {
        "id": "fp16",
        "name": "FP16",
        "description": "Standard half-precision model loading.",
    },
    "int8": {
        "id": "int8",
        "name": "8-bit Metal",
        "description": "Reduced memory use on Apple Silicon.",
    },
    "int4": {
        "id": "int4",
        "name": "4-bit Metal",
        "description": "Lower memory use on Apple Silicon.",
    },
}

# Response completeness detection
INCOMPLETE_ENDING_PATTERN = re.compile(
    r"\b(?:such\s+as|the|a|an|and|or|but|because|with|without|into|to|of|in|for|"
    r"from|by|including|that|which|where|when|while|if|although|as|at|on|is|are|"
    r"was|were|has|have|had)\s*$",
    re.IGNORECASE,
)
LIST_ITEM_PATTERN = re.compile(r"^(?:[-*•]|\d+[.)])\s+")


# Check whether a response appears incomplete
def response_appears_incomplete(text: str) -> bool:
    """Conservatively flag prose that appears to stop mid-sentence."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith((".", "!", "?", "…", '"', "'", "”", "’", ")", "]", "}")):
        return False
    if "```" in stripped or re.search(r"(?m)^(?: {4}|\t)\S", text):
        return False

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 2:
        list_lines = sum(bool(LIST_ITEM_PATTERN.match(line)) for line in lines)
        if list_lines >= max(2, len(lines) - 1):
            return False

    words = re.findall(r"\b[\w'-]+\b", stripped)
    if not words:
        return False
    if len(lines) == 1 and "=" in stripped and len(words) <= 10:
        return False
    if INCOMPLETE_ENDING_PATTERN.search(stripped):
        return True
    if len(words) < 12:
        return False

    title_words = [word for word in words if len(word) > 2]
    if len(words) <= 14 and title_words:
        capitalized = sum(word[0].isupper() for word in title_words)
        if capitalized / len(title_words) >= 0.7:
            return False
    return stripped[-1].isalnum() or stripped.endswith((",", ";", ":", "-", "—"))


# Application errors and generation settings

class AppError(Exception):
    status = HTTPStatus.BAD_REQUEST


class ModelNotLoadedError(AppError):
    status = HTTPStatus.CONFLICT


class ConversationTooLongError(AppError):
    status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE


class OutOfMemoryAppError(AppError):
    status = HTTPStatus.INSUFFICIENT_STORAGE


@dataclass
class GenerationSettings:
    max_length: int = 256
    seed: int = 42
    temperature: float = 0.7
    top_p: float = 0.9

    # Validate generation settings from an API request
    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "GenerationSettings":
        payload = payload or {}
        try:
            settings = cls(
                max_length=int(payload.get("max_length", 256)),
                seed=int(payload.get("seed", 42)),
                temperature=float(payload.get("temperature", 0.7)),
                top_p=float(payload.get("top_p", 0.9)),
            )
        except (TypeError, ValueError) as exc:
            raise AppError("Generation settings must be numbers.") from exc

        limits = {
            "max_length": (settings.max_length, 20, 768),
            "seed": (settings.seed, 0, 1_000_000),
            "temperature": (settings.temperature, 0.1, 1.5),
            "top_p": (settings.top_p, 0.1, 1.0),
        }
        for name, (value, low, high) in limits.items():
            if not low <= value <= high:
                raise AppError(f"{name} must be between {low} and {high}.")
        return settings


# Model engine

class LocalLLMEngine:
    """Loads one local Transformers model at a time and performs inference."""

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.model_id: str | None = None
        self.precision: str | None = None
        self.device_name = "not loaded"
        self.load_time_seconds: float | None = None
        self._context_tokenizer = None
        self._context_tokenizer_model_id: str | None = None

    # Check whether Metal quantization is available
    @staticmethod
    def metal_quantization_available() -> bool:
        try:
            import kernels  # noqa: F401
            import torch

            return bool(torch.backends.mps.is_available())
        except (ImportError, RuntimeError):
            return False

    # Clear unused CPU and device memory
    @staticmethod
    def _clear_memory() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, RuntimeError):
            pass

    # Unload the current model and release memory
    def unload(self) -> None:
        # Keep the tokenizer for context counting while powered off.
        if self.tokenizer is not None and self.model_id is not None:
            self._context_tokenizer = self.tokenizer
            self._context_tokenizer_model_id = self.model_id
        self.model = None
        self.tokenizer = None
        self.model_id = None
        self.precision = None
        self.device_name = "not loaded"
        self.load_time_seconds = None
        self._clear_memory()

    # Load the selected model with FP16, 8-bit, or 4-bit precision
    def load(self, model_id: str, precision: str = "fp16") -> None:
        if model_id not in MODELS:
            raise AppError("Unknown model.")
        if precision not in PRECISIONS:
            raise AppError("Unknown precision setting.")
        if self.model_id == model_id and self.precision == precision and self.model is not None:
            return
        if precision in {"int8", "int4"} and not self.metal_quantization_available():
            raise AppError(
                f"{PRECISIONS[precision]['name']} requires an Apple Silicon Mac with "
                "MPS and kernels 0.16.0."
            )

        model_path = MODELS[model_id]["path"]
        if not model_path.exists():
            raise AppError(f"Local model directory is missing: {model_path}")

        self.unload()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, MetalConfig

            # Keep the whole model on one device. Automatic disk offloading makes token
            # generation extremely slow and leaves less control over memory use.
            dtype = torch.float16
            if precision in {"int8", "int4"}:
                # Configure weight-only Metal quantization.
                device_map = {"": "mps"}
                bits = 8 if precision == "int8" else 4
                quantization_config = MetalConfig(bits=bits, group_size=64)
            elif torch.backends.mps.is_available():
                device_map = {"": "mps"}
                quantization_config = None
            elif torch.cuda.is_available():
                device_map = {"": "cuda"}
                quantization_config = None
            else:
                device_map = {"": "cpu"}
                quantization_config = None

            load_started_at = time.perf_counter()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True
            )
            self._context_tokenizer = self.tokenizer
            self._context_tokenizer_model_id = model_id
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=dtype,
                device_map=device_map,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
            )
            self.model.eval()
            self.model_id = model_id
            self.precision = precision
            self.device_name = str(self.model.device)
            self.load_time_seconds = time.perf_counter() - load_started_at
        except RuntimeError as exc:
            self.unload()
            if "out of memory" in str(exc).lower():
                raise OutOfMemoryAppError(
                    "The model could not be loaded because memory is full. "
                    "Close other applications or select Qwen 0.5B."
                ) from exc
            raise
        except Exception:
            self.unload()
            raise

    # Get the tokenizer for a selected model
    def _tokenizer_for_model(self, model_id: str):
        if model_id not in MODELS:
            raise AppError("Unknown model.")
        if self.tokenizer is not None and self.model_id == model_id:
            return self.tokenizer
        if (
            self._context_tokenizer is not None
            and self._context_tokenizer_model_id == model_id
        ):
            return self._context_tokenizer

        # Cache the tokenizer without loading model weights.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MODELS[model_id]["path"], local_files_only=True
        )
        self._context_tokenizer = tokenizer
        self._context_tokenizer_model_id = model_id
        return tokenizer

    # Format conversation history with the model chat template
    @staticmethod
    def _render_conversation(
        tokenizer,
        conversation: list[dict[str, str]],
        *,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
    ) -> str:
        template_options: dict[str, Any] = {"tokenize": False}
        if continue_final_message:
            template_options["continue_final_message"] = True
        else:
            template_options["add_generation_prompt"] = add_generation_prompt
        return tokenizer.apply_chat_template(conversation, **template_options)

    # Count tokens in one message
    def count_text_tokens(self, text: str, model_id: str) -> int:
        """Count only message content tokens, without role/template overhead."""
        tokenizer = self._tokenizer_for_model(model_id)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return int(encoded["input_ids"].shape[1])

    # Count tokens in the complete conversation
    def count_conversation_tokens(
        self,
        conversation: list[dict[str, str]],
        model_id: str,
        *,
        continue_final_message: bool = False,
    ) -> int:
        """Re-tokenize the current role/content history with its chat template."""
        if not conversation:
            return 0
        tokenizer = self._tokenizer_for_model(model_id)
        text = self._render_conversation(
            tokenizer,
            conversation,
            continue_final_message=continue_final_message,
        )
        encoded = tokenizer(text, return_tensors="pt")
        return int(encoded["input_ids"].shape[1])

    # Build the safe token limit error message
    @staticmethod
    def _safe_limit_error(
        input_tokens: int,
        safe_limit: int,
        *,
        continue_final_message: bool = False,
    ) -> str:
        excess = input_tokens - safe_limit
        excess_unit = "token" if excess == 1 else "tokens"
        if continue_final_message:
            return (
                f"Continuing this response would require {input_tokens} input tokens, "
                f"exceeding SANZI’s {safe_limit}-token safe limit by {excess} "
                f"{excess_unit}. "
                "Remove earlier rounds or clear the conversation and try again."
            )
        return (
            f"This message would require {input_tokens} input tokens, exceeding "
            f"SANZI’s {safe_limit}-token safe limit by {excess} {excess_unit}. "
            "Shorten the "
            "message, remove earlier rounds, or clear the conversation and try again."
        )

    # Generate a response from the current conversation
    def generate(
        self,
        conversation: list[dict[str, str]],
        settings: GenerationSettings,
        continue_final_message: bool = False,
    ) -> tuple[str, int, int, str | None]:
        if self.model is None or self.tokenizer is None:
            raise ModelNotLoadedError("Turn on the chatbot before sending a message.")

        import torch

        random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings.seed)

        text = self._render_conversation(
            self.tokenizer,
            conversation,
            add_generation_prompt=True,
            continue_final_message=continue_final_message,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[1])

        # Keep input and output within the safe token budget.
        model_context = int(
            getattr(self.model.config, "max_position_embeddings", 4096)
        )
        safe_limit = min(SAFE_INPUT_TOKEN_LIMIT, model_context - settings.max_length)
        if input_tokens > safe_limit:
            del inputs
            raise ConversationTooLongError(
                self._safe_limit_error(
                    input_tokens,
                    safe_limit,
                    continue_final_message=continue_final_message,
                )
            )

        try:
            started_at = time.monotonic()
            with torch.inference_mode():
                # Stop generation after the runtime safety limit.
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=settings.max_length,
                    max_time=MAX_GENERATION_SECONDS,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = outputs[0, input_tokens:]
            decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
            reply = decoded.rstrip() if continue_final_message else decoded.strip()
            output_tokens = int(generated.shape[0])
            elapsed_seconds = time.monotonic() - started_at
            del outputs, generated, inputs
            if elapsed_seconds >= MAX_GENERATION_SECONDS:
                stop_reason = "time_limit"
            elif output_tokens == settings.max_length:
                stop_reason = "output_limit"
            else:
                stop_reason = None
            if not continue_final_message and not reply:
                reply = "I could not produce a response."
            return reply, input_tokens, output_tokens, stop_reason
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self._clear_memory()
                raise OutOfMemoryAppError(
                    "Generation ran out of memory. Reduce Max length, remove early "
                    "rounds, or switch to Qwen 0.5B."
                ) from exc
            raise


# Mock engine

class MockLLMEngine(LocalLLMEngine):
    """Small deterministic engine used only for UI development and tests."""

    @staticmethod
    def metal_quantization_available() -> bool:
        return True

    def load(self, model_id: str, precision: str = "fp16") -> None:
        if model_id not in MODELS:
            raise AppError("Unknown model.")
        if precision not in PRECISIONS:
            raise AppError("Unknown precision setting.")
        load_started_at = time.perf_counter()
        self.model_id = model_id
        self.precision = precision
        self.model = True
        self.tokenizer = True
        self.device_name = "mock"
        self.load_time_seconds = time.perf_counter() - load_started_at

    def count_text_tokens(self, text: str, model_id: str) -> int:
        if model_id not in MODELS:
            raise AppError("Unknown model.")
        return max(1, len(text.split())) if text else 0

    def count_conversation_tokens(
        self,
        conversation: list[dict[str, str]],
        model_id: str,
        *,
        continue_final_message: bool = False,
    ) -> int:
        if model_id not in MODELS:
            raise AppError("Unknown model.")
        if not conversation:
            return 0
        user_messages = {
            message["content"]
            for message in conversation
            if message.get("role") == "user"
        }
        if "__mock_blocked__" in user_messages:
            return SAFE_INPUT_TOKEN_LIMIT
        if "__mock_warning__" in user_messages:
            return 2100
        content_tokens = sum(
            self.count_text_tokens(message["content"], model_id)
            for message in conversation
        )
        template_overhead = len(conversation) * (3 if model_id == "qwen" else 4)
        if not continue_final_message:
            template_overhead += 1
        return content_tokens + template_overhead

    def generate(
        self,
        conversation: list[dict[str, str]],
        settings: GenerationSettings,
        continue_final_message: bool = False,
    ) -> tuple[str, int, int, str | None]:
        if not self.model_id:
            raise ModelNotLoadedError("Turn on the chatbot before sending a message.")
        if continue_final_message:
            current = conversation[-1]["content"]
            if current.endswith("Germany was founded in"):
                return "1871.", max(8, len(current.split()) * 2), 2, None
            if current.endswith("The calculation was still running"):
                return "when it paused.", max(8, len(current.split()) * 2), 4, None
            return "Here is the rest of the response.", max(8, len(current.split()) * 2), 7, None
        prompt = conversation[-1]["content"]
        if prompt == "__mock_oversized__":
            input_tokens = 2639
            raise ConversationTooLongError(
                self._safe_limit_error(input_tokens, SAFE_INPUT_TOKEN_LIMIT)
            )
        if prompt == "__mock_incomplete__":
            return "Germany was founded in", 8, 5, None
        if prompt == "__mock_output_limit__":
            return "Germany was founded in", 8, settings.max_length, "output_limit"
        if prompt == "__mock_time_limit__":
            return "The calculation was still running", 8, 6, "time_limit"
        reply = (
            f"This is a local preview response to “{prompt}”. The real app will use "
            f"{MODELS[self.model_id]['full_name']} with your selected generation settings."
        )
        return reply, max(8, len(prompt.split()) * 2), len(reply.split()), None


# Conversation and chatbot state management

class ChatService:
    """Manages conversation state and model operations for the HTTP API."""

    def __init__(self, engine: LocalLLMEngine | None = None) -> None:
        self.engine = engine or LocalLLMEngine()
        self.conversation: list[dict[str, Any]] = []
        self.selected_model = "qwen"
        self.selected_precision = "fp16"
        self.settings = GenerationSettings()
        # Keep model operations separate across HTTP worker threads.
        self.lock = threading.RLock()

    # Build role and content history for the model
    def _model_conversation(self) -> list[dict[str, str]]:
        # Keep UI token metadata out of the model prompt.
        return [
            {
                "role": str(item.get("role", "")),
                "content": str(item.get("content", "")),
            }
            for item in self.conversation
        ]

    # Calculate current conversation token usage
    def _context_status(self) -> dict[str, Any]:
        # Chat templates add model-specific tokens.
        model_id = self.engine.model_id or self.selected_model
        continue_final_message = bool(
            self.conversation
            and self.conversation[-1].get("role") == "assistant"
            and self.conversation[-1].get("continuation_reason")
        )
        context_tokens = self.engine.count_conversation_tokens(
            self._model_conversation(),
            model_id,
            continue_final_message=continue_final_message,
        )
        if context_tokens >= SAFE_INPUT_TOKEN_LIMIT:
            status = "blocked"
        elif context_tokens >= CONTEXT_WARNING_THRESHOLD:
            status = "approaching"
        else:
            status = "normal"
        return {
            "context_tokens": context_tokens,
            "remaining": max(0, SAFE_INPUT_TOKEN_LIMIT - context_tokens),
            "safe_limit": SAFE_INPUT_TOKEN_LIMIT,
            "model_capacity": MODELS[model_id]["context_capacity"],
            "status": status,
        }

    # Return the state used by the frontend
    def public_state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "powered_on": self.engine.model_id is not None,
                "selected_model": self.selected_model,
                "active_model": self.engine.model_id,
                "selected_precision": self.selected_precision,
                "active_precision": self.engine.precision,
                "device": self.engine.device_name,
                "load_time_seconds": self.engine.load_time_seconds,
                "settings": asdict(self.settings),
                "conversation": self.conversation,
                "context_status": self._context_status(),
                "rounds": len(self.conversation) // 2,
                "models": [
                    {key: value for key, value in model.items() if key != "path"}
                    for model in MODELS.values()
                ],
                "precisions": [
                    {
                        **precision,
                        "available": precision["id"] not in {"int8", "int4"}
                        or self.engine.metal_quantization_available(),
                    }
                    for precision in PRECISIONS.values()
                ],
            }

    # Turn the selected model on or off
    def power(
        self,
        on: bool,
        model_id: str | None = None,
        precision: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            # Loading a new selection unloads the previous model.
            if on:
                target = model_id or self.selected_model
                target_precision = precision or self.selected_precision
                self.engine.load(target, target_precision)
                self.selected_model = target
                self.selected_precision = target_precision
            else:
                self.engine.unload()
            return self.public_state()

    # Switch between Llama and Qwen
    def switch_model(self, model_id: str) -> dict[str, Any]:
        with self.lock:
            if model_id not in MODELS:
                raise AppError("Unknown model.")
            if self.engine.model_id is not None:
                self.engine.load(model_id, self.selected_precision)
            self.selected_model = model_id
            return self.public_state()

    # Switch between FP16, 8-bit, and 4-bit precision
    def switch_precision(self, precision: str) -> dict[str, Any]:
        with self.lock:
            if precision not in PRECISIONS:
                raise AppError("Unknown precision setting.")
            if (
                precision in {"int8", "int4"}
                and not self.engine.metal_quantization_available()
            ):
                raise AppError(
                    f"{PRECISIONS[precision]['name']} requires an Apple Silicon Mac "
                    "with MPS and kernels 0.16.0."
                )
            if self.engine.model_id is not None:
                self.engine.load(self.selected_model, precision)
            self.selected_precision = precision
            return self.public_state()

    # Generate and save one conversation round
    def chat(self, message: str, raw_settings: dict[str, Any] | None) -> dict[str, Any]:
        message = message.strip()
        if not message:
            raise AppError("Message cannot be empty.")
        if len(message) > 8_000:
            raise AppError("Message is too long (maximum 8,000 characters).")

        with self.lock:
            self.settings = GenerationSettings.from_payload(raw_settings)
            saved_context_status = self._context_status()
            if saved_context_status["context_tokens"] >= SAFE_INPUT_TOKEN_LIMIT:
                raise ConversationTooLongError(
                    f"The saved conversation uses {saved_context_status['context_tokens']} "
                    f"tokens and has reached SANZI’s {SAFE_INPUT_TOKEN_LIMIT}-token safe "
                    "limit. Remove earlier rounds or clear the conversation before "
                    "sending another message."
                )
            generated_model_id = self.engine.model_id or self.selected_model
            user_tokens = self.engine.count_text_tokens(message, generated_model_id)
            previous_continuation_reason = None
            if self.conversation and self.conversation[-1].get("role") == "assistant":
                previous_continuation_reason = self.conversation[-1].pop(
                    "continuation_reason", None
                )
            self.conversation.append({"role": "user", "content": message})
            model_conversation = self._model_conversation()
            try:
                reply, _, output_tokens, stop_reason = self.engine.generate(
                    model_conversation, self.settings
                )
            except Exception:
                # Remove the tentative message when generation fails.
                self.conversation.pop()
                if previous_continuation_reason is not None:
                    self.conversation[-1][
                        "continuation_reason"
                    ] = previous_continuation_reason
                raise
            if stop_reason is None and response_appears_incomplete(reply):
                stop_reason = "incomplete"
            round_total = user_tokens + output_tokens
            token_usage = {
                "user": user_tokens,
                "output": output_tokens,
                "round_total": round_total,
            }
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": reply,
                "token_usage": token_usage,
            }
            if stop_reason:
                assistant_message["continuation_reason"] = stop_reason
            self.conversation.append(assistant_message)
            context_status = self._context_status()
            return {
                "reply": reply,
                "conversation": self.conversation,
                "token_usage": token_usage,
                "context_status": context_status,
            }

    # Join a continuation to the current response
    @staticmethod
    def _append_continuation(existing: str, continuation: str) -> str:
        if not continuation:
            return existing
        if not existing:
            return continuation.lstrip()
        if existing[-1].isspace() or continuation[0].isspace():
            return existing + continuation
        if continuation[0] in ".,!?;:)]}":
            return existing + continuation
        return existing + " " + continuation

    # Continue an incomplete assistant response
    def continue_response(
        self, raw_settings: dict[str, Any] | None
    ) -> dict[str, Any]:
        with self.lock:
            if not self.conversation or self.conversation[-1].get("role") != "assistant":
                raise AppError("There is no assistant response to continue.")
            assistant_message = self.conversation[-1]
            if assistant_message.get("continuation_reason") not in {
                "output_limit",
                "time_limit",
                "incomplete",
            }:
                raise AppError("The latest assistant response does not need continuation.")
            existing_usage = assistant_message.get("token_usage")
            if not isinstance(existing_usage, dict):
                raise AppError("The latest assistant response has no token metadata.")

            # Continue the final assistant message in the same round.
            self.settings = GenerationSettings.from_payload(raw_settings)
            model_conversation = self._model_conversation()
            continuation, _, new_output_tokens, stop_reason = self.engine.generate(
                model_conversation,
                self.settings,
                continue_final_message=True,
            )
            assistant_message["content"] = self._append_continuation(
                str(assistant_message.get("content", "")), continuation
            )
            if stop_reason is None and response_appears_incomplete(
                str(assistant_message["content"])
            ):
                stop_reason = "incomplete"

            cumulative_output = int(existing_usage.get("output", 0)) + new_output_tokens
            user_tokens = int(existing_usage.get("user", 0))
            round_total = user_tokens + cumulative_output
            existing_usage.update(
                {
                    "output": cumulative_output,
                    "round_total": round_total,
                }
            )
            if stop_reason:
                assistant_message["continuation_reason"] = stop_reason
            else:
                assistant_message.pop("continuation_reason", None)

            context_status = self._context_status()
            return {
                "reply": assistant_message["content"],
                "conversation": self.conversation,
                "token_usage": existing_usage,
                "context_status": context_status,
            }

    # Clear the complete conversation
    def clear(self) -> dict[str, Any]:
        with self.lock:
            self.conversation.clear()
            return self.public_state()

    # Remove conversation rounds from the beginning
    def remove_rounds(self, rounds: int) -> dict[str, Any]:
        if rounds < 1:
            raise AppError("Rounds must be at least 1.")
        with self.lock:
            del self.conversation[: min(len(self.conversation), rounds * 2)]
            return self.public_state()

    # Remove one selected conversation round
    def remove_round(self, round_index: int) -> dict[str, Any]:
        if round_index < 0:
            raise AppError("Round index must be zero or greater.")
        with self.lock:
            start = round_index * 2
            if start >= len(self.conversation):
                raise AppError("Conversation round does not exist.")
            end = start + 1
            if (
                end < len(self.conversation)
                and self.conversation[end].get("role") == "assistant"
            ):
                end += 1
            del self.conversation[start:end]
            return self.public_state()


# HTTP API and frontend server

class AppRequestHandler(BaseHTTPRequestHandler):
    """Expose the SANZI JSON API and serve the compiled Vue application."""

    service = ChatService()
    server_version = "SANZI/1.0"

    # Write one HTTP log line
    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[http] {self.address_string()} - {fmt % args}")

    # Send a JSON response
    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "http://localhost:5173")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.end_headers()
        self.wfile.write(body)


    # Read and validate a JSON request body
    def _read_json(self) -> dict[str, Any]:
        # Limit request body size before reading it.
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                raise AppError("Request body is too large.")
            return json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            raise AppError("Invalid JSON request.") from exc


    # Handle expected and unexpected API errors
    def _handle_error(self, exc: Exception) -> None:
        if isinstance(exc, AppError):
            self._send_json({"error": str(exc)}, exc.status)
            return
        print(f"[error] {type(exc).__name__}: {exc}")
        self._send_json(
            {"error": "Unexpected backend error. Check the terminal for details."},
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )


    # Handle browser CORS preflight requests
    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json({}, HTTPStatus.NO_CONTENT)


    # Handle API state requests and frontend files
    def do_GET(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            if path in {"/api/state", "/api/health"}:
                self._send_json(self.service.public_state())
            elif path.startswith("/api/"):
                self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            else:
                self._serve_frontend(path)
        except Exception as exc:
            self._handle_error(exc)

    # Handle state-changing API requests
    def do_POST(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            payload = self._read_json()
            if path == "/api/power":
                result = self.service.power(
                    bool(payload.get("on")),
                    payload.get("model_id"),
                    payload.get("precision"),
                )
            elif path == "/api/model":
                result = self.service.switch_model(str(payload.get("model_id", "")))
            elif path == "/api/precision":
                result = self.service.switch_precision(str(payload.get("precision", "")))
            elif path == "/api/chat":
                result = self.service.chat(
                    str(payload.get("message", "")), payload.get("settings")
                )
            elif path == "/api/chat/continue":
                result = self.service.continue_response(payload.get("settings"))
            elif path == "/api/conversation/remove":
                try:
                    rounds = int(payload.get("rounds", 0))
                except (TypeError, ValueError) as exc:
                    raise AppError("Rounds must be a number.") from exc
                result = self.service.remove_rounds(rounds)
            elif path == "/api/conversation/remove-round":
                try:
                    round_index = int(payload.get("round_index", -1))
                except (TypeError, ValueError) as exc:
                    raise AppError("Round index must be a number.") from exc
                result = self.service.remove_round(round_index)
            else:
                self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(result)
        except Exception as exc:
            self._handle_error(exc)

    # Clear conversation history through the API
    def do_DELETE(self) -> None:  # noqa: N802
        try:
            if urlparse(self.path).path == "/api/conversation":
                self._send_json(self.service.clear())
            else:
                self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._handle_error(exc)

    # Serve the compiled Vue frontend
    def _serve_frontend(self, request_path: str) -> None:
        if not FRONTEND_DIST.exists():
            self._send_json(
                {"error": "Frontend is not built. Run `pnpm install && pnpm build` in frontend/."},
                HTTPStatus.NOT_FOUND,
            )
            return

        # Fall back to index.html for Vue client routes.
        relative = unquote(request_path).lstrip("/") or "index.html"
        candidate = (FRONTEND_DIST / relative).resolve()
        try:
            candidate.relative_to(FRONTEND_DIST.resolve())
        except ValueError:
            self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            candidate = FRONTEND_DIST / "index.html"
        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Cache-Control", "no-cache" if candidate.name == "index.html" else "public, max-age=31536000"
        )
        self.end_headers()
        self.wfile.write(body)


# Application startup

# Start the SANZI HTTP server
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SANZI local chat backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--mock", action="store_true", help="Use lightweight preview responses instead of loading a model."
    )
    args = parser.parse_args()
    if args.mock:
        AppRequestHandler.service = ChatService(MockLLMEngine())

    server = ThreadingHTTPServer((args.host, args.port), AppRequestHandler)
    print(f"SANZI is available at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping SANZI...")
    finally:
        AppRequestHandler.service.engine.unload()
        server.server_close()


if __name__ == "__main__":
    main()
