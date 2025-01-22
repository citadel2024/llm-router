import hashlib
import uuid


def generate_unique_id(ordered_json: str):
    """
    Generate a unique id from the ordered json string
    :param ordered_json:
    :return:
    """
    try:
        hash_object = hashlib.sha256(ordered_json.encode('utf-8'))
        return hash_object.hexdigest()
    except Exception:
        return str(uuid.uuid4()).replace('-', '')
