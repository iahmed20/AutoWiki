import ast
import json
from typing import List


def parse_string_list(raw: str) -> List[str]:

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            continue
    return [line.strip("-• ").strip() for line in raw.splitlines() if line.strip()]
