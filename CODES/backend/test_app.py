"""Unit tests for SANZI backend state, safeguards, and conversation behavior."""

import unittest
from copy import deepcopy
from unittest.mock import patch

from backend.app import (
    AppError,
    ChatService,
    ConversationTooLongError,
    GenerationSettings,
    LocalLLMEngine,
    MockLLMEngine,
    SAFE_INPUT_TOKEN_LIMIT,
    response_appears_incomplete,
)


# Mock engines used to simulate backend behavior during tests
class RecordingMockLLMEngine(MockLLMEngine):
    def __init__(self) -> None:
        super().__init__()
        self.last_context: list[dict[str, str]] = []
        self.continuation_calls = 0
        self.context_count_calls: list[tuple[list[dict[str, str]], str, bool]] = []
        self.load_calls: list[tuple[str, str]] = []

    def load(self, model_id, precision="fp16"):
        super().load(model_id, precision)
        self.load_calls.append((model_id, precision))

    def generate(self, conversation, settings, continue_final_message=False):
        self.last_context = [message.copy() for message in conversation]
        if continue_final_message:
            self.continuation_calls += 1
        return super().generate(conversation, settings, continue_final_message)

    def count_conversation_tokens(
        self,
        conversation,
        model_id,
        *,
        continue_final_message=False,
    ):
        self.context_count_calls.append(
            (
                [message.copy() for message in conversation],
                model_id,
                continue_final_message,
            )
        )
        return super().count_conversation_tokens(
            conversation,
            model_id,
            continue_final_message=continue_final_message,
        )


class HighUsageMockLLMEngine(MockLLMEngine):
    def generate(self, conversation, settings, continue_final_message=False):
        return "Large response", 2490, 20, None

    def count_conversation_tokens(
        self, conversation, model_id, *, continue_final_message=False
    ):
        return 2600 if conversation else 0


class GrowingContextMockLLMEngine(MockLLMEngine):
    def generate(self, conversation, settings, continue_final_message=False):
        input_tokens = len(conversation) * 20
        return "Context-aware response", input_tokens, 5, None

    def count_conversation_tokens(
        self, conversation, model_id, *, continue_final_message=False
    ):
        return len(conversation) * 20


class RepeatingInterruptionMockLLMEngine(MockLLMEngine):
    def generate(self, conversation, settings, continue_final_message=False):
        if continue_final_message:
            return "and more", 30, settings.max_length, "output_limit"
        return "An unfinished response", 8, settings.max_length, "output_limit"


class FailingModelSwitchEngine(MockLLMEngine):
    def load(self, model_id, precision="fp16"):
        if model_id == "llama":
            self.unload()
            raise AppError("Simulated model loading failure.")
        super().load(model_id, precision)


# Minimal fake tokenizer and model objects for token-limit tests
class FakeInputIds:
    shape = (1, 2501)


class FakeBatch(dict):
    def to(self, device):
        return self


class TooLongTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.template_options = {}

    def apply_chat_template(self, conversation, **options):
        self.template_options = options
        return "oversized prompt"

    def __call__(self, text, return_tensors=None, **kwargs):
        return FakeBatch(input_ids=FakeInputIds())


class FakeModelConfig:
    max_position_embeddings = 32768


class FakeModel:
    device = "cpu"
    config = FakeModelConfig()


# Generation settings and safe token-limit tests
class GenerationSettingsTests(unittest.TestCase):
    def test_accepts_valid_settings(self) -> None:
        settings = GenerationSettings.from_payload(
            {"max_length": 768, "seed": 7, "temperature": 0.5, "top_p": 0.85}
        )
        self.assertEqual(settings.max_length, 768)
        self.assertEqual(settings.seed, 7)

    def test_default_max_length_is_256(self) -> None:
        settings = GenerationSettings.from_payload({})
        self.assertEqual(settings.max_length, 256)

    def test_safe_input_token_limit_remains_2500(self) -> None:
        self.assertEqual(SAFE_INPUT_TOKEN_LIMIT, 2500)

    def test_oversized_context_still_uses_2500_token_warning(self) -> None:
        engine = LocalLLMEngine()
        engine.model = FakeModel()
        engine.tokenizer = TooLongTokenizer()
        with self.assertRaises(ConversationTooLongError) as context:
            engine.generate([{"role": "user", "content": "large"}], GenerationSettings())
        self.assertEqual(
            str(context.exception),
            "This message would require 2501 input tokens, exceeding SANZI’s "
            "2500-token safe limit by 1 token. Shorten the message, remove earlier "
            "rounds, or clear the conversation and try again.",
        )

    def test_continuation_uses_continue_final_message_chat_template(self) -> None:
        engine = LocalLLMEngine()
        engine.model = FakeModel()
        tokenizer = TooLongTokenizer()
        engine.tokenizer = tokenizer
        with self.assertRaises(ConversationTooLongError):
            engine.generate(
                [{"role": "assistant", "content": "unfinished"}],
                GenerationSettings(),
                continue_final_message=True,
            )
        self.assertTrue(tokenizer.template_options["continue_final_message"])
        self.assertNotIn("add_generation_prompt", tokenizer.template_options)

    def test_live_context_count_uses_chat_template_without_generation_prompt(self) -> None:
        engine = LocalLLMEngine()
        tokenizer = TooLongTokenizer()
        engine.model_id = "qwen"
        engine.tokenizer = tokenizer
        count = engine.count_conversation_tokens(
            [{"role": "user", "content": "hello"}], "qwen"
        )
        self.assertEqual(count, 2501)
        self.assertFalse(tokenizer.template_options["add_generation_prompt"])
        self.assertFalse(tokenizer.template_options["tokenize"])

    def test_rejects_out_of_range_settings(self) -> None:
        with self.assertRaises(AppError):
            GenerationSettings.from_payload({"max_length": 769})


# Response completeness detection tests
class ResponseCompletenessTests(unittest.TestCase):
    def test_obviously_unfinished_prose_is_detected(self) -> None:
        self.assertTrue(response_appears_incomplete("Germany was founded in"))
        self.assertTrue(
            response_appears_incomplete(
                "The experiment produced several useful measurements because"
            )
        )

    def test_complete_sentence_is_not_detected(self) -> None:
        self.assertFalse(response_appears_incomplete("Germany was founded in 1871."))

    def test_code_list_equation_and_short_answer_are_not_detected(self) -> None:
        examples = (
            "```python\nprint('hello')\n```",
            "- apples\n- oranges\n- pears",
            "E = mc^2",
            "All good",
            "Release Notes",
        )
        for example in examples:
            with self.subTest(example=example):
                self.assertFalse(response_appears_incomplete(example))


# Conversation, model, precision, and context-management tests
class ChatServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RecordingMockLLMEngine()
        self.service = ChatService(self.engine)
        self.service.power(True, "qwen")

    def tearDown(self) -> None:
        self.service.power(False)

    def test_chat_and_clear(self) -> None:
        result = self.service.chat("Hello", None)
        self.assertTrue(result["reply"])
        self.assertEqual(result["token_usage"]["user"], 1)
        self.assertEqual(
            result["token_usage"]["user"] + result["token_usage"]["output"],
            result["token_usage"]["round_total"],
        )
        self.assertEqual(
            result["context_status"]["remaining"],
            2500 - result["context_status"]["context_tokens"],
        )
        self.assertEqual(result["context_status"]["safe_limit"], 2500)
        self.assertEqual(result["context_status"]["status"], "normal")
        self.assertEqual(self.service.public_state()["rounds"], 1)
        cleared = self.service.clear()
        self.assertEqual(cleared["rounds"], 0)
        self.assertEqual(cleared["context_status"]["context_tokens"], 0)
        self.assertEqual(cleared["context_status"]["remaining"], 2500)

    def test_remaining_safe_context_is_clamped_at_zero(self) -> None:
        service = ChatService(HighUsageMockLLMEngine())
        service.power(True, "qwen")
        try:
            result = service.chat("Hello", None)
            self.assertEqual(result["context_status"]["context_tokens"], 2600)
            self.assertEqual(result["context_status"]["remaining"], 0)
        finally:
            service.power(False)

    def test_remaining_decreases_as_context_grows(self) -> None:
        service = ChatService(GrowingContextMockLLMEngine())
        service.power(True, "qwen")
        try:
            first = service.chat("First", None)["context_status"]
            second = service.chat("Second", None)["context_status"]
            self.assertGreater(second["context_tokens"], first["context_tokens"])
            self.assertLess(second["remaining"], first["remaining"])
        finally:
            service.power(False)

    def test_approaching_context_warns_without_blocking_generation(self) -> None:
        first = self.service.chat("__mock_warning__", None)
        self.assertEqual(first["context_status"]["context_tokens"], 2100)
        self.assertEqual(first["context_status"]["remaining"], 400)
        self.assertEqual(first["context_status"]["status"], "approaching")

        second = self.service.chat("warning does not block", None)
        self.assertEqual(second["context_status"]["status"], "approaching")
        self.assertEqual(self.service.public_state()["rounds"], 2)

    def test_oversized_new_message_keeps_saved_conversation_unchanged(self) -> None:
        self.service.chat("saved round", None)
        before = deepcopy(self.service.public_state())

        with self.assertRaises(ConversationTooLongError) as context:
            self.service.chat("__mock_oversized__", None)

        self.assertEqual(
            str(context.exception),
            "This message would require 2639 input tokens, exceeding SANZI’s "
            "2500-token safe limit by 139 tokens. Shorten the message, remove earlier "
            "rounds, or clear the conversation and try again.",
        )
        after = self.service.public_state()
        self.assertEqual(after["conversation"], before["conversation"])
        self.assertEqual(after["rounds"], before["rounds"])
        self.assertEqual(after["context_status"], before["context_status"])

    def test_removing_round_clears_approaching_warning_and_restores_remaining(self) -> None:
        self.service.chat("__mock_warning__", None)
        state = self.service.remove_round(0)
        self.assertEqual(state["context_status"]["context_tokens"], 0)
        self.assertEqual(state["context_status"]["remaining"], 2500)
        self.assertEqual(state["context_status"]["status"], "normal")

    def test_saved_context_at_safe_limit_blocks_new_normal_generation(self) -> None:
        result = self.service.chat("__mock_blocked__", None)
        self.assertEqual(result["context_status"]["context_tokens"], 2500)
        self.assertEqual(result["context_status"]["remaining"], 0)
        self.assertEqual(result["context_status"]["status"], "blocked")

        with self.assertRaises(ConversationTooLongError) as context:
            self.service.chat("must not be generated", None)
        self.assertIn("The saved conversation uses 2500 tokens", str(context.exception))
        self.assertEqual(self.service.public_state()["rounds"], 1)

    def test_model_context_capacities_are_exposed(self) -> None:
        models = {
            model["id"]: model for model in self.service.public_state()["models"]
        }
        self.assertEqual(models["qwen"]["context_capacity"], "32K tokens")
        self.assertEqual(models["llama"]["context_capacity"], "128K tokens")
        self.assertEqual(
            self.service.public_state()["context_status"]["model_capacity"],
            "32K tokens",
        )
        llama_state = self.service.switch_model("llama")
        self.assertEqual(
            llama_state["context_status"]["model_capacity"], "128K tokens"
        )

    def test_user_tokens_only_count_the_current_user_message(self) -> None:
        first = self.service.chat("one two", None)["token_usage"].copy()
        second = self.service.chat("three four five", None)["token_usage"].copy()
        self.assertEqual(first["user"], 2)
        self.assertEqual(second["user"], 3)
        self.assertEqual(first["round_total"], first["user"] + first["output"])
        self.assertEqual(second["round_total"], second["user"] + second["output"])
        self.assertEqual(
            self.service.public_state()["conversation"][1]["token_usage"], first
        )

    def test_live_context_includes_chat_template_overhead(self) -> None:
        result = self.service.chat("Hello", None)
        self.assertGreater(
            result["context_status"]["context_tokens"],
            result["token_usage"]["round_total"],
        )
        last_call = self.engine.context_count_calls[-1]
        self.assertEqual(last_call[1], "qwen")
        self.assertEqual(
            [set(message) for message in last_call[0]],
            [{"role", "content"}, {"role", "content"}],
        )

    def test_power_cycle_preserves_assistant_token_metadata(self) -> None:
        result = self.service.chat("Keep this metadata", None)
        assistant_before = result["conversation"][-1]
        usage_before = assistant_before["token_usage"].copy()
        reply_before = assistant_before["content"]

        off_state = self.service.power(False)
        self.assertEqual(off_state["conversation"][-1]["content"], reply_before)
        self.assertEqual(off_state["conversation"][-1]["token_usage"], usage_before)

        on_state = self.service.power(True, "qwen")
        self.assertEqual(on_state["conversation"][-1]["content"], reply_before)
        self.assertEqual(on_state["conversation"][-1]["token_usage"], usage_before)
        self.assertEqual(on_state["context_status"]["model_capacity"], "32K tokens")

    def test_successful_load_time_is_exposed_and_cleared_when_off(self) -> None:
        state = self.service.public_state()
        self.assertIsInstance(state["load_time_seconds"], float)
        self.assertGreaterEqual(state["load_time_seconds"], 0.0)

        off_state = self.service.power(False)
        self.assertIsNone(off_state["load_time_seconds"])

    def test_model_and_precision_reloads_replace_load_time(self) -> None:
        engine = MockLLMEngine()
        service = ChatService(engine)
        with patch(
            "backend.app.time.perf_counter",
            side_effect=[10.0, 12.0, 20.0, 23.5, 30.0, 35.25],
        ):
            first = service.power(True, "qwen", "fp16")
            second = service.switch_precision("int8")
            third = service.switch_model("llama")

        self.assertEqual(first["load_time_seconds"], 2.0)
        self.assertEqual(second["load_time_seconds"], 3.5)
        self.assertEqual(third["load_time_seconds"], 5.25)

    def test_failed_reload_does_not_report_a_successful_load_time(self) -> None:
        service = ChatService(FailingModelSwitchEngine())
        loaded = service.power(True, "qwen", "fp16")
        self.assertIsNotNone(loaded["load_time_seconds"])

        with self.assertRaises(AppError):
            service.switch_model("llama")

        state = service.public_state()
        self.assertFalse(state["powered_on"])
        self.assertIsNone(state["load_time_seconds"])

    def test_round_removal_removes_stored_token_metadata(self) -> None:
        result = self.service.chat("Remove this metadata", None)
        self.assertIn("token_usage", result["conversation"][-1])
        state = self.service.remove_round(0)
        self.assertEqual(state["conversation"], [])

    def test_token_metadata_is_not_sent_to_the_model(self) -> None:
        self.service.chat("First", None)
        self.service.chat("Second", None)
        self.assertTrue(self.engine.last_context)
        for message in self.engine.last_context:
            self.assertEqual(set(message), {"role", "content"})

    def test_continues_output_limited_response_in_same_round(self) -> None:
        initial = self.service.chat("__mock_output_limit__", None)
        original_usage = initial["token_usage"].copy()
        self.assertEqual(initial["conversation"][-1]["continuation_reason"], "output_limit")
        self.assertTrue(self.engine.context_count_calls[-1][2])

        continued = self.service.continue_response(None)
        state = self.service.public_state()
        self.assertEqual(state["rounds"], 1)
        self.assertEqual(len(state["conversation"]), 2)
        self.assertEqual(
            state["conversation"][-1]["content"],
            "Germany was founded in 1871.",
        )
        self.assertNotIn("continuation_reason", state["conversation"][-1])
        self.assertEqual(
            continued["token_usage"]["output"], original_usage["output"] + 2
        )
        self.assertEqual(
            continued["token_usage"]["round_total"],
            continued["token_usage"]["user"] + continued["token_usage"]["output"],
        )
        self.assertFalse(self.engine.context_count_calls[-1][2])
        self.assertFalse(
            any(
                message["role"] == "user" and message["content"].lower() == "continue"
                for message in state["conversation"]
            )
        )

        self.service.chat("Use the completed answer", None)
        context_text = " ".join(message["content"] for message in self.engine.last_context)
        self.assertIn("Germany was founded in 1871.", context_text)

    def test_continues_time_limited_response_in_same_round(self) -> None:
        initial = self.service.chat("__mock_time_limit__", None)
        self.assertEqual(initial["conversation"][-1]["continuation_reason"], "time_limit")

        self.service.continue_response(None)
        state = self.service.public_state()
        self.assertEqual(state["rounds"], 1)
        self.assertEqual(len(state["conversation"]), 2)
        self.assertEqual(
            state["conversation"][-1]["content"],
            "The calculation was still running when it paused.",
        )
        self.assertNotIn("continuation_reason", state["conversation"][-1])
        self.assertEqual(self.engine.continuation_calls, 1)

    def test_naturally_incomplete_response_continues_in_same_round(self) -> None:
        initial = self.service.chat("__mock_incomplete__", None)
        self.assertEqual(
            initial["conversation"][-1]["continuation_reason"], "incomplete"
        )

        self.service.continue_response(None)
        state = self.service.public_state()
        self.assertEqual(state["rounds"], 1)
        self.assertEqual(len(state["conversation"]), 2)
        self.assertEqual(
            state["conversation"][-1]["content"],
            "Germany was founded in 1871.",
        )
        self.assertNotIn("continuation_reason", state["conversation"][-1])
        self.assertFalse(
            any(
                message["role"] == "user"
                and message["content"].lower() == "continue"
                for message in state["conversation"]
            )
        )

    def test_naturally_complete_response_has_no_continuation_reason(self) -> None:
        result = self.service.chat("Hello", None)
        self.assertNotIn("continuation_reason", result["conversation"][-1])

    def test_continue_button_can_reappear_after_another_interruption(self) -> None:
        service = ChatService(RepeatingInterruptionMockLLMEngine())
        service.power(True, "qwen")
        try:
            service.chat("Start", None)
            state = service.continue_response(None)
            self.assertEqual(state["conversation"][-1]["continuation_reason"], "output_limit")
            self.assertEqual(service.public_state()["rounds"], 1)
        finally:
            service.power(False)

    def test_removes_rounds_from_the_beginning(self) -> None:
        self.service.chat("first", None)
        self.service.chat("second", None)
        self.service.remove_rounds(1)
        state = self.service.public_state()
        self.assertEqual(state["rounds"], 1)
        self.assertEqual(state["conversation"][0]["content"], "second")

    def test_removes_first_specific_round(self) -> None:
        for prompt in ("first", "second", "third"):
            self.service.chat(prompt, None)
        self.service.remove_round(0)
        user_messages = [
            message["content"]
            for message in self.service.public_state()["conversation"]
            if message["role"] == "user"
        ]
        self.assertEqual(user_messages, ["second", "third"])

    def test_removes_middle_specific_round(self) -> None:
        for prompt in ("first", "second", "third"):
            self.service.chat(prompt, None)
        self.service.remove_round(1)
        user_messages = [
            message["content"]
            for message in self.service.public_state()["conversation"]
            if message["role"] == "user"
        ]
        self.assertEqual(user_messages, ["first", "third"])

    def test_round_removal_recalculates_context_without_changing_history_metadata(self) -> None:
        self.service.chat("first user message", None)
        second_result = self.service.chat("second", None)
        second_usage = second_result["conversation"][-1]["token_usage"].copy()
        remaining_before = second_result["context_status"]["remaining"]

        state = self.service.remove_round(0)
        self.assertGreater(state["context_status"]["remaining"], remaining_before)
        self.assertEqual(state["conversation"][-1]["token_usage"], second_usage)

    def test_removes_latest_round_and_excludes_it_from_future_context(self) -> None:
        for prompt in ("first", "second", "deleted latest"):
            self.service.chat(prompt, None)
        self.service.remove_round(2)
        self.assertEqual(self.service.public_state()["rounds"], 2)

        self.service.chat("new message", None)
        context_text = " ".join(message["content"] for message in self.engine.last_context)
        self.assertNotIn("deleted latest", context_text)
        self.assertIn("first", context_text)
        self.assertIn("second", context_text)
        self.assertIn("new message", context_text)

    def test_removes_incomplete_user_only_round_safely(self) -> None:
        self.service.chat("complete", None)
        self.service.conversation.append({"role": "user", "content": "unfinished"})
        self.service.remove_round(1)
        state = self.service.public_state()
        self.assertEqual(state["rounds"], 1)
        self.assertEqual(len(state["conversation"]), 2)
        self.assertEqual(state["conversation"][0]["content"], "complete")

    def test_model_switch_keeps_history(self) -> None:
        self.service.chat("Remember this", None)
        self.service.switch_model("llama")
        state = self.service.public_state()
        self.assertEqual(state["active_model"], "llama")
        self.assertEqual(state["rounds"], 1)

    def test_model_switch_while_off_recalculates_live_context(self) -> None:
        result = self.service.chat("Remember this while off", None)
        usage_before = result["conversation"][-1]["token_usage"].copy()
        self.service.power(False)
        state = self.service.switch_model("llama")
        self.assertFalse(state["powered_on"])
        self.assertEqual(state["context_status"]["model_capacity"], "128K tokens")
        self.assertEqual(state["conversation"][-1]["token_usage"], usage_before)

    def test_precision_switch_keeps_history(self) -> None:
        self.service.chat("Remember this too", None)
        self.service.switch_precision("int4")
        state = self.service.public_state()
        self.assertEqual(state["selected_precision"], "int4")
        self.assertEqual(state["active_precision"], "int4")
        self.assertEqual(state["rounds"], 1)

    def test_int8_precision_is_accepted(self) -> None:
        state = self.service.switch_precision("int8")
        self.assertEqual(state["selected_precision"], "int8")
        self.assertEqual(state["active_precision"], "int8")

    def test_qwen_loads_in_int8(self) -> None:
        self.service.power(False)
        state = self.service.power(True, "qwen", "int8")
        self.assertEqual(state["active_model"], "qwen")
        self.assertEqual(state["active_precision"], "int8")
        self.assertIn(("qwen", "int8"), self.engine.load_calls)

    def test_llama_loads_in_int8(self) -> None:
        self.service.power(False)
        state = self.service.power(True, "llama", "int8")
        self.assertEqual(state["active_model"], "llama")
        self.assertEqual(state["active_precision"], "int8")
        self.assertIn(("llama", "int8"), self.engine.load_calls)

    def test_switches_fp16_to_int8_to_int4(self) -> None:
        self.assertEqual(self.service.public_state()["active_precision"], "fp16")
        self.assertEqual(
            self.service.switch_precision("int8")["active_precision"], "int8"
        )
        self.assertEqual(
            self.service.switch_precision("int4")["active_precision"], "int4"
        )
        self.assertEqual(
            self.engine.load_calls[-3:],
            [("qwen", "fp16"), ("qwen", "int8"), ("qwen", "int4")],
        )

    def test_int8_precision_switch_preserves_history(self) -> None:
        self.service.chat("Keep this while switching to int8", None)
        conversation_before = deepcopy(self.service.public_state()["conversation"])
        state = self.service.switch_precision("int8")
        self.assertEqual(state["conversation"], conversation_before)
        self.assertEqual(state["rounds"], 1)
        self.assertEqual(state["active_precision"], "int8")

    def test_rejects_unknown_precision(self) -> None:
        with self.assertRaises(AppError):
            self.service.switch_precision("int3")


# Run the test suite when this file is executed directly
if __name__ == "__main__":
    unittest.main()