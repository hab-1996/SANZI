"""Focused tests for CLI model-loading failures."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from CLI import chat_terminal


class ModelLoadingOOMTests(unittest.TestCase):
    def test_load_model_cleans_memory_and_raises_friendly_oom_error(self) -> None:
        with (
            patch.object(
                chat_terminal.AutoTokenizer,
                "from_pretrained",
                return_value=Mock(),
            ),
            patch.object(
                chat_terminal.AutoModelForCausalLM,
                "from_pretrained",
                side_effect=RuntimeError("MPS backend out of memory"),
            ),
            patch.object(chat_terminal, "clear_model_memory") as clear_memory,
        ):
            with self.assertRaises(chat_terminal.ModelLoadOOMError) as raised:
                chat_terminal.load_model(Path("unused"))

        clear_memory.assert_called_once_with()
        message = str(raised.exception)
        self.assertIn("Qwen instead of Llama", message)
        self.assertIn("8-bit or 4-bit Metal", message)
        self.assertIn("close other memory-heavy applications", message)

    def test_initial_model_load_oom_exits_without_a_traceback(self) -> None:
        answers = iter(["1", "1", "", "", "", ""])
        with (
            patch("builtins.input", side_effect=lambda *args: next(answers)),
            patch.object(
                chat_terminal,
                "load_model",
                side_effect=chat_terminal.ModelLoadOOMError(
                    chat_terminal.MODEL_LOAD_OOM_MESSAGE
                ),
            ),
            patch("builtins.print") as print_message,
        ):
            chat_terminal.main()

        printed = " ".join(str(call.args[0]) for call in print_message.call_args_list)
        self.assertIn("Model loading failed", printed)

    def test_failed_model_switch_restores_previous_model_and_history(self) -> None:
        answers = iter(["1", "1", "", "", "", "", "/model 2", "exit"])
        first_model = (Mock(), Mock())
        restored_model = (Mock(), Mock())
        with (
            patch("builtins.input", side_effect=lambda *args: next(answers)),
            patch.object(
                chat_terminal,
                "load_model",
                side_effect=[
                    first_model,
                    chat_terminal.ModelLoadOOMError(
                        chat_terminal.MODEL_LOAD_OOM_MESSAGE
                    ),
                    restored_model,
                ],
            ) as load_model,
            patch.object(chat_terminal, "clear_model_memory"),
            patch("builtins.print") as print_message,
        ):
            chat_terminal.main()

        self.assertEqual(
            [call.args[0] for call in load_model.call_args_list],
            [
                chat_terminal.MODELS["1"]["path"],
                chat_terminal.MODELS["2"]["path"],
                chat_terminal.MODELS["1"]["path"],
            ],
        )
        printed = " ".join(str(call.args[0]) for call in print_message.call_args_list)
        self.assertIn("Previous model restored", printed)
        self.assertIn("Conversation history kept", printed)


if __name__ == "__main__":
    unittest.main()
