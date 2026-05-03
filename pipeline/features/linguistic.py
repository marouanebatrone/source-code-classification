import re

def get_comment_ratio(code: str) -> float:
    lines = code.splitlines()
    if not lines:
        return 0.0
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    return comment_lines / len(lines)

def get_avg_identifier_length(code: str) -> float:
    identifiers = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", code)
    if not identifiers:
        return 0.0
    return sum(len(i) for i in identifiers) / len(identifiers)