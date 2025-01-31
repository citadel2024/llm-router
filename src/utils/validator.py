from typing import Any


def validate_integer(obj: Any, filed: str):
    if hasattr(obj, filed):
        v = getattr(obj, filed)
        if isinstance(v, int) and v < 0:
            raise ValueError(f"Invalid {filed} value: {v}")
    else:
        raise ValueError(f"Object does not have {filed} attribute")
