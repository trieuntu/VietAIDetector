"""VietAIDetector — Vietnamese Text Utilities"""

import re

# Compiled Regex Patterns

DEHYPHEN_RE = re.compile(
    r'([a-zA-ZÀ-ỹà-ỹ])-\s*\n\s*([a-zA-ZÀ-ỹà-ỹ])',
    re.UNICODE
)
"""De-hyphenation: fix word breaks from line wrapping."""

LINEBREAK_RE = re.compile(r'(?<![.!?…])\n(?!\n)')
"""Remove mid-sentence line breaks."""

WHITESPACE_RE = re.compile(r'[ \t\r\u00a0\ufeff]+')
"""Collapse multiple whitespace characters (spaces, tabs, NBSP, BOM) into one space."""

NON_SEMANTIC_RE = re.compile(
    r'[^\w\sÀ-ỹà-ỹ.,!?;:()\-\'\"""\u2018\u2019\u201c\u201d]',
    re.UNICODE
)
"""Detect special/non-semantic characters."""


def count_words(text: str) -> int:
    """Count the number of words in a string."""
    return len(text.split())


def special_char_ratio(text: str) -> float:
    """Calculate the ratio of non-semantic characters to total characters."""
    if not text:
        return 0.0
    return len(NON_SEMANTIC_RE.findall(text)) / len(text)
