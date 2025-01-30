import pytest

from src.model.message import (
    ChatCompletionTextObject,
    ChatCompletionAudioObject,
    ChatCompletionImageObject,
    ChatCompletionToolMessage,
    ChatCompletionUserMessage,
    ChatCompletionSystemMessage,
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionToolCallFunctionChunk,
)


@pytest.mark.asyncio
async def test_user_message_creation():
    mock_content = "test content"
    message = ChatCompletionUserMessage(role="user", content=mock_content)
    assert message["role"] == "user"
    assert message["content"] == mock_content


@pytest.mark.asyncio
async def test_assistant_message_with_tool_calls():
    mock_tool_call = ChatCompletionAssistantToolCall(
        id="test_id", type="function", function={"name": "test_fn", "arguments": "{}"}
    )
    message = ChatCompletionAssistantMessage(role="assistant", tool_calls=[mock_tool_call])
    assert message["role"] == "assistant"
    assert message["tool_calls"][0]["id"] == "test_id"


@pytest.mark.asyncio
async def test_image_url_object():
    mock_image = ChatCompletionImageObject(type="image_url", image_url={"url": "test.jpg", "detail": "high"})
    assert mock_image["image_url"]["url"] == "test.jpg"
    assert mock_image["type"] == "image_url"


@pytest.mark.asyncio
async def test_mixed_content_message():
    mock_content = [
        ChatCompletionTextObject(type="text", text="hello"),
        ChatCompletionImageObject(type="image_url", image_url="test.jpg"),
    ]
    message = ChatCompletionUserMessage(role="user", content=mock_content)
    assert len(message["content"]) == 2
    assert message["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_tool_message_validation():
    message = ChatCompletionToolMessage(role="tool", content="test result", tool_call_id="call_123")
    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call_123"


@pytest.mark.asyncio
async def test_function_call_chunk():
    mock_chunk = ChatCompletionToolCallFunctionChunk(name="test_function", arguments='{"param": 1}')
    assert mock_chunk["name"] == "test_function"
    assert "arguments" in mock_chunk


@pytest.mark.asyncio
async def test_cache_control_field():
    message = ChatCompletionUserMessage(role="user", content="test", cache_control={"type": "ephemeral"})
    assert message["cache_control"]["type"] == "ephemeral"


@pytest.mark.asyncio
async def test_audio_content_part():
    mock_audio = ChatCompletionAudioObject(audio_url="audio.mp3", transcript="test transcript")
    assert "audio_url" in mock_audio
    assert "transcript" in mock_audio


@pytest.mark.asyncio
async def test_system_message_with_list_content():
    message = ChatCompletionSystemMessage(role="system", content=[{"type": "text", "text": "system prompt"}])
    assert isinstance(message["content"], list)
    assert message["content"][0]["type"] == "text"
