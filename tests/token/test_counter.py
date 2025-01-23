from unittest.mock import Mock, patch

import pytest

from src.config import LogConfiguration
from src.token.counter import TokenCounter, TokenizerType, DEFAULT_IMAGE_TOKEN_COUNT, _process_messages


class TestTokenCounter:
    @pytest.fixture
    def mock_logger(self):
        logger = Mock()
        return logger

    @pytest.fixture
    def mock_token_counter(self, mock_logger):
        logger_cfg = LogConfiguration()
        counter = TokenCounter(logger_cfg)
        counter.logger = mock_logger
        return counter

    @pytest.fixture
    def mock_text(self):
        return "Hello"

    @pytest.fixture
    def mock_messages(self, mock_text):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": mock_text},
                    {"type": "image_url", "image_url": {"url": "test.jpg"}}
                ]
            }
        ]

    def test_select_llama_tokenizer(self, mock_token_counter):
        with patch("src.token.counter.Tokenizer.from_pretrained") as mock_from_pretrained:
            result = mock_token_counter._select_tokenizer_helper("llama-3-8b")
            mock_from_pretrained.assert_called_once_with("Xenova/llama-3-tokenizer")
            assert result.tokenizer_type == TokenizerType.HuggingFace

    def test_select_default_tokenizer(self, mock_token_counter):
        result = mock_token_counter._select_tokenizer_helper("gpt-4")
        assert result.tokenizer_type == TokenizerType.OpenAI

    def test_process_messages_with_images(self, mock_token_counter, mock_messages):
        text, is_tool_call = _process_messages(mock_messages)
        assert is_tool_call == False
        assert "Hello" in text

    def test_openai_token_counting_with_tools(self, mock_token_counter):
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        with patch("src.token.counter._format_function_definitions") as mock_format:
            mock_format.return_value = "formatted_tools"
            num_tokens = mock_token_counter._openai_token_counter(
                model="gpt-4",
                messages=[{"role": "system", "content": "test"}],
                tools=tools
            )
            assert mock_format.called
            assert num_tokens > 9  # Verify tool tokens added

    def test_token_counter_with_huggingface(self, mock_token_counter):
        with patch("src.token.counter.Tokenizer") as mock_tokenizer:
            mock_encoder = Mock()
            mock_encoder.encode.return_value.ids = [1, 2, 3]
            mock_token_counter._select_tokenizer_helper = Mock(return_value=Mock(
                tokenizer_type=TokenizerType.HuggingFace,
                tokenizer=mock_encoder
            ))
            result = mock_token_counter.token_counter(
                model="llama-3",
                text="test"
            )
            assert result == 3 + 0  # 3 text tokens + 0 image tokens

    def test_error_when_no_input(self, mock_token_counter):
        with pytest.raises(ValueError):
            mock_token_counter.token_counter(model="gpt-4")

    def test_tool_choice_handling(self, mock_token_counter):
        num_tokens = mock_token_counter._openai_token_counter(
            model="gpt-4",
            tool_choice={"function": {"name": "test_tool"}}
        )
        assert num_tokens >= 7  # Minimum tokens for tool choice

    def test_caching_mechanism(self, mock_token_counter):
        with patch("src.token.counter.Tokenizer.from_pretrained") as mock_from_pretrained:
            mock_token_counter._select_tokenizer_helper("llama-3-8b")
            mock_token_counter._select_tokenizer_helper("llama-3-8b")
            assert mock_from_pretrained.call_count == 1

    def test_fallback_to_default_encoding(self, mock_token_counter, mock_logger):
        with patch("tiktoken.encoding_for_model") as mock_encoding:
            mock_encoding.side_effect = KeyError
            mock_token_counter._get_encoding("unknown-model")
            mock_logger.error.assert_called()

    def test_openai_token_counter_simple(self, mock_token_counter):
        model = "gpt-4o"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": "Launching pytest with arguments test_counter.py::TestTokenCounter::test_openai_token_with_image_and_text ",
                        "type": "text",
                    }
                ],
            }
        ]
        actual = mock_token_counter.token_counter(model=model, messages=messages)
        assert actual == 28

    def test_openai_token_counter_without_model(self, mock_token_counter):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": "Launching pytest with arguments test_counter.py::TestTokenCounter::test_openai_token_with_image_and_text ",
                        "type": "text",
                    }
                ],
            }
        ]
        actual = mock_token_counter.token_counter(messages=messages)
        assert actual == 21

    def test_openai_token_counter_with_image(self, mock_token_counter):
        model = "gpt-4o"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": "Launching pytest with arguments test_counter.py::TestTokenCounter::test_openai_token_with_image_and_text ",
                        "type": "text",
                    },
                    {
                        "type": "image_url",
                        "image_url": "http://a.jpg",
                    },
                ],
            }
        ]
        actual = mock_token_counter.token_counter(model=model, messages=messages)
        assert actual == 28 + DEFAULT_IMAGE_TOKEN_COUNT

    def test_openai_token_counter_with_tool(self, mock_token_counter):
        model = "gpt-4o"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": "Launching pytest with arguments test_counter.py::TestTokenCounter::test_openai_token_with_image_and_text ",
                        "type": "text",
                    },
                ],
            }
        ]
        tool_name = "tool A"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "tool A",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "foo_prop": {
                                "type": "string",
                                "description": "tool A ",
                            }
                        },
                        "required": ["foo_prop"],
                    },
                },
            }
        ]
        actual = mock_token_counter.token_counter(model=model, messages=messages, tools=tools, tool_choice="none")
        assert actual == 67

    def test_openai_token_counter_with_tool_system_msg(self, mock_token_counter):
        model = "gpt-4o"
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "text": "Launching pytest with arguments test_counter.py::TestTokenCounter::test_openai_token_with_image_and_text ",
                        "type": "text",
                    },
                ],
            }
        ]
        tool_name = "tool A"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "tool A",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "foo_prop": {
                                "type": "string",
                                "description": "tool A ",
                            }
                        },
                        "required": ["foo_prop"],
                    },
                },
            }
        ]
        actual = mock_token_counter.token_counter(model=model, messages=messages, tools=tools, tool_choice="none")
        assert actual == 63
