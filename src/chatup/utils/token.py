import tiktoken
from functools import lru_cache
from chatup.constants import EMBEDDING_SETTINGS


@lru_cache()
def get_encoder(model_name: str = EMBEDDING_SETTINGS.EMBED_MODEL):
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding(EMBEDDING_SETTINGS.TOKEN_EMBEDDING_SCHEME)


def count_tokens(text: str, enc=None) -> int:
    if enc is None:
        enc = get_encoder()
    return len(enc.encode(text))