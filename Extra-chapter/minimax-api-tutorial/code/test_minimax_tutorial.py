"""
Unit tests for MiniMax API tutorial examples.
Tests use mocked OpenAI client to avoid real API calls.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

CODE_DIR = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_mock_response(content: str = "Mock response text"):
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    usage = MagicMock()
    usage.prompt_tokens = 50
    usage.completion_tokens = 100
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_mock_stream_chunk(content: str):
    delta = MagicMock()
    delta.content = content
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


# ---------------------------------------------------------------------------
# Tests for 01_basic_chat.py
# ---------------------------------------------------------------------------
class TestBasicChat(unittest.TestCase):
    @patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key-123"})
    @patch("openai.OpenAI")
    def test_basic_chat_success(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "Self-Attention 的核心思想是让序列中每个位置都能关注到其他所有位置。"
        )

        mod = _load_module("basic_chat", CODE_DIR / "01_basic_chat.py")
        mod.basic_chat()

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M3")
        self.assertAlmostEqual(call_kwargs["temperature"], 0.7)
        # temperature must be in (0.0, 1.0]
        self.assertGreater(call_kwargs["temperature"], 0.0)
        self.assertLessEqual(call_kwargs["temperature"], 1.0)

    @patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key-123"})
    @patch("openai.OpenAI")
    def test_system_prompt_included(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response()

        mod = _load_module("basic_chat", CODE_DIR / "01_basic_chat.py")
        mod.basic_chat()

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        roles = [m["role"] for m in messages]
        self.assertIn("system", roles)
        self.assertIn("user", roles)

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self):
        mod = _load_module("basic_chat2", CODE_DIR / "01_basic_chat.py")
        with self.assertRaises(ValueError, msg="应在缺少 API Key 时抛出 ValueError"):
            mod.basic_chat()

    @patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key-123"})
    @patch("openai.OpenAI")
    def test_base_url_is_minimax(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response()

        mod = _load_module("basic_chat3", CODE_DIR / "01_basic_chat.py")
        mod.basic_chat()

        init_kwargs = mock_openai_cls.call_args[1]
        self.assertIn("minimax.io", init_kwargs.get("base_url", ""))


# ---------------------------------------------------------------------------
# Tests for 02_streaming.py
# ---------------------------------------------------------------------------
class TestStreaming(unittest.TestCase):
    @patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key-123"})
    @patch("openai.OpenAI")
    def test_streaming_output(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        chunks = [
            _make_mock_stream_chunk("BERT"),
            _make_mock_stream_chunk(" 使用 MLM"),
            _make_mock_stream_chunk("，GPT 使用 CLM"),
        ]
        mock_client.chat.completions.create.return_value = iter(chunks)

        mod = _load_module("streaming", CODE_DIR / "02_streaming.py")
        mod.streaming_chat()

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertTrue(call_kwargs.get("stream"), "stream 参数应为 True")

    @patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key-123"})
    @patch("openai.OpenAI")
    def test_streaming_uses_minimax_model(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter([])

        mod = _load_module("streaming2", CODE_DIR / "02_streaming.py")
        mod.streaming_chat()

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertIn("MiniMax", call_kwargs["model"])

    @patch.dict("os.environ", {}, clear=True)
    def test_streaming_missing_key_raises(self):
        mod = _load_module("streaming3", CODE_DIR / "02_streaming.py")
        with self.assertRaises(ValueError):
            mod.streaming_chat()


# ---------------------------------------------------------------------------
# Tests for 03_multi_turn.py
# ---------------------------------------------------------------------------
class TestMultiTurn(unittest.TestCase):
    @patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key-123"})
    @patch("openai.OpenAI")
    @patch("builtins.input", side_effect=["你好", "quit"])
    def test_multi_turn_adds_history(self, mock_input, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("你好！有什么可以帮助你的？")

        mod = _load_module("multi_turn", CODE_DIR / "03_multi_turn.py")
        mod.multi_turn_chat()

        # Should have been called once for the "你好" message
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)
        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_messages = [m for m in messages if m["role"] == "user"]
        self.assertEqual(len(user_messages), 1)
        self.assertEqual(user_messages[0]["content"], "你好")

    @patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key-123"})
    @patch("openai.OpenAI")
    @patch("builtins.input", side_effect=["quit"])
    def test_multi_turn_quit_immediately(self, mock_input, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mod = _load_module("multi_turn2", CODE_DIR / "03_multi_turn.py")
        mod.multi_turn_chat()

        mock_client.chat.completions.create.assert_not_called()

    @patch.dict("os.environ", {}, clear=True)
    def test_multi_turn_missing_key_raises(self):
        mod = _load_module("multi_turn3", CODE_DIR / "03_multi_turn.py")
        with self.assertRaises(ValueError):
            mod.multi_turn_chat()


# ---------------------------------------------------------------------------
# Tests for 04_long_context.py
# ---------------------------------------------------------------------------
class TestLongContext(unittest.TestCase):
    @patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key-123"})
    @patch("openai.OpenAI")
    def test_long_context_analysis(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "摘要：Transformer 是一种完全基于注意力机制的序列模型。\n"
            "创新点：1. 抛弃循环结构 2. 多头注意力 3. 可并行训练"
        )

        mod = _load_module("long_ctx", CODE_DIR / "04_long_context.py")
        mod.analyze_long_document("这是一段测试文档内容。")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M3")
        self.assertGreater(call_kwargs["temperature"], 0.0)

    @patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key-123"})
    @patch("openai.OpenAI")
    def test_document_included_in_message(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response()

        test_doc = "这是专属测试文档 XYZ123"
        mod = _load_module("long_ctx2", CODE_DIR / "04_long_context.py")
        mod.analyze_long_document(test_doc)

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        combined_content = " ".join(m.get("content", "") for m in messages)
        self.assertIn("XYZ123", combined_content)

    @patch.dict("os.environ", {}, clear=True)
    def test_long_context_missing_key_raises(self):
        mod = _load_module("long_ctx3", CODE_DIR / "04_long_context.py")
        with self.assertRaises(ValueError):
            mod.analyze_long_document("test")


# ---------------------------------------------------------------------------
# Tests for temperature constraint (MiniMax-specific)
# ---------------------------------------------------------------------------
class TestTemperatureConstraint(unittest.TestCase):
    """Verify all examples use temperature in (0.0, 1.0]"""

    def _get_temperature(self, module_path: Path, mock_fn: str, extra_patches=None):
        with patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_openai_cls:
                mock_client = MagicMock()
                mock_openai_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = _make_mock_response()
                mock_client.chat.completions.create.return_value = iter([]) if "stream" in str(module_path) else _make_mock_response()

                mod = _load_module(f"temp_{module_path.stem}", module_path)
                try:
                    patches = extra_patches or []
                    func = getattr(mod, mock_fn)
                    for p in patches:
                        p.start()
                    try:
                        func()
                    except Exception:
                        pass
                    finally:
                        for p in patches:
                            p.stop()
                except Exception:
                    pass

                if mock_client.chat.completions.create.called:
                    return mock_client.chat.completions.create.call_args[1].get("temperature")
        return None

    def test_all_temps_in_valid_range(self):
        files_and_funcs = [
            (CODE_DIR / "01_basic_chat.py", "basic_chat"),
            (CODE_DIR / "02_streaming.py", "streaming_chat"),
            (CODE_DIR / "04_long_context.py", "analyze_long_document"),
        ]
        for path, fn_name in files_and_funcs:
            temp = self._get_temperature(path, fn_name)
            if temp is not None:
                self.assertGreater(temp, 0.0, f"{path.name}: temperature 必须 > 0.0")
                self.assertLessEqual(temp, 1.0, f"{path.name}: temperature 必须 <= 1.0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
