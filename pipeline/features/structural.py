import ast
from utils.logger import get_logger

log = get_logger("structural")

def _max_depth(node: ast.AST, level: int = 1) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return level
    return max(_max_depth(c, level + 1) for c in children)

def get_ast_features(code: str) -> tuple[int, int]:
    """Returns (node_count, max_depth). Both 0 if code is unparseable."""
    try:
        tree = ast.parse(code)
        return len(list(ast.walk(tree))), _max_depth(tree)
    except Exception as e:
        log.warning(f"AST parse failed: {e}")
        return 0, 0