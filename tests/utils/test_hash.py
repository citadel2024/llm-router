import json

from src.utils.hash import generate_unique_id


def test_generate_unique_id_equal():
    json_a = {"a": 1, "b": 2}
    json_b = {"b": 2, "a": 1}
    str_a = json.dumps(json_a, sort_keys=True)
    str_b = json.dumps(json_b, sort_keys=True)

    id_a = generate_unique_id(str_a)
    id_b = generate_unique_id(str_b)

    assert id_a == id_b


def test_generate_unique_id_not_equal():
    json_a = {"a": 1, "b": 2}
    json_b = {"b": 2, "a": 2}
    str_a = json.dumps(json_a, sort_keys=True)
    str_b = json.dumps(json_b, sort_keys=True)

    id_a = generate_unique_id(str_a)
    id_b = generate_unique_id(str_b)
    assert id_a != id_b


def test_generate_unique_id_exception():
    id_a = generate_unique_id(1)
    assert 32 == len(id_a)
