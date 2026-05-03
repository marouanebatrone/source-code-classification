from pipeline.features.perplexity  import get_perplexity
from pipeline.features.linguistic  import get_comment_ratio, get_avg_identifier_length
from pipeline.features.structural  import get_ast_features

def extract(code: str) -> list[float]:
    """
    Returns a 5-dimensional feature vector:
      [perplexity, comment_ratio, avg_identifier_len, ast_nodes, ast_depth]
    """
    nodes, depth = get_ast_features(code)
    return [
        get_perplexity(code),
        get_comment_ratio(code),
        get_avg_identifier_length(code),
        float(nodes),
        float(depth),
    ]