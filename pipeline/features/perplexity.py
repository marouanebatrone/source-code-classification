import zlib

def get_perplexity(code: str) -> float:

    if not code or len(code) < 20:
        return 0.0
    encoded = code.encode("utf-8")
    compressed = zlib.compress(encoded, level=9)
    return len(compressed) / len(encoded)