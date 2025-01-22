import asyncio
from concurrent.futures import ThreadPoolExecutor


def run_async_function(async_fn, *args, **kwargs):
    def run_in_new_loop():
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(async_fn(*args, **kwargs))
        finally:
            new_loop.close()
            asyncio.set_event_loop(None)

    try:
        # Try to get the current event loop
        _ = asyncio.get_running_loop()
        # If we're already in an event loop, run in a separate thread to avoid nested event loop issues
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_new_loop)
            return future.result()
    except RuntimeError:
        # No running event loop, we can safely run in this thread
        return run_in_new_loop()
