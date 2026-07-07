"""Filter candidate chunks with generated regular expressions."""

import re


class RegexFilter:
    """Apply configured regex patterns to candidate chunks."""

    def keep_matching(self, candidates, patterns: list[str]):
        if not patterns:
            return list(candidates)

        compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        kept = []
        for candidate in candidates:
            text = getattr(candidate, "text", str(candidate))
            if any(pattern.search(text) for pattern in compiled):
                kept.append(candidate)
        return kept

