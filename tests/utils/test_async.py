import asyncio
from unittest.mock import patch, MagicMock

import pytest

from src.utils.asyncy import run_async_function


async def sample_async_function(x):
    await asyncio.sleep(0.1)
    return x * 2


async def failing_async_function():
    await asyncio.sleep(0.1)
    raise ValueError("Test error")


def test_run_async_function_without_running_loop():
    result = run_async_function(sample_async_function, 5)
    assert result == 10


def test_run_async_function_with_running_loop():
    async def wrapper():
        return run_async_function(sample_async_function, 5)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(wrapper())
        assert result == 10
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def test_run_async_function_handles_exceptions():
    with pytest.raises(ValueError, match="Test error"):
        run_async_function(failing_async_function)


@patch('concurrent.futures.ThreadPoolExecutor')
def test_run_async_function_uses_thread_pool_when_in_loop(mock_executor):
    mock_pool = MagicMock()
    mock_future = MagicMock()
    mock_future.result.return_value = 10
    mock_pool.submit.return_value = mock_future
    mock_executor.return_value.__enter__.return_value = mock_pool

    async def wrapper():
        return run_async_function(sample_async_function, 5)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(wrapper())
        assert result == 10
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@pytest.mark.asyncio
async def test_nested_calls():
    async def nested_async_function():
        return run_async_function(sample_async_function, 5)

    result = await nested_async_function()
    assert result == 10


def test_multiple_concurrent_calls():
    async def run_multiple():
        tasks = [
            asyncio.create_task(sample_async_function(i))
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)
        return results

    results = run_async_function(run_multiple)
    assert results == [0, 2, 4, 6, 8]
