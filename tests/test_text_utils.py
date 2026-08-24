"""Unit tests for preprocessing/text_utils.py"""

import pytest
from preprocessing.text_utils import (
    DEHYPHEN_RE,
    LINEBREAK_RE,
    WHITESPACE_RE,
    NON_SEMANTIC_RE,
    count_words,
    special_char_ratio,
)


# DEHYPHEN_RE Tests

class TestDehyphenation:
    """Test de-hyphenation regex for fixing line-wrapped Vietnamese words."""

    def test_basic_dehyphen(self):
        text = "phát tri-\nển"
        result = DEHYPHEN_RE.sub(r'\1\2', text)
        assert "phát triển" in result

    def test_dehyphen_with_spaces(self):
        text = "công nghệ thông tin hiện đ-  \n  ại"
        result = DEHYPHEN_RE.sub(r'\1\2', text)
        assert "đại" in result

    def test_no_dehyphen_for_real_hyphens(self):
        text = "Hà Nội - thành phố"
        result = DEHYPHEN_RE.sub(r'\1\2', text)
        # Should not change: hyphen not followed by newline
        assert "Hà Nội - thành phố" == result

    def test_dehyphen_with_diacritics(self):
        text = "giáo dụ-\nc"
        result = DEHYPHEN_RE.sub(r'\1\2', text)
        assert "giáo dục" in result


# LINEBREAK_RE Tests

class TestLinebreakRemoval:
    """Test mid-sentence line break removal."""

    def test_keep_linebreak_after_period(self):
        text = "Câu một.\nCâu hai"
        result = LINEBREAK_RE.sub(' ', text)
        # Should keep newline after period
        assert "Câu một.\nCâu hai" == result

    def test_remove_mid_sentence_linebreak(self):
        text = "Đây là\nmột câu"
        result = LINEBREAK_RE.sub(' ', text)
        assert "Đây là một câu" == result

    def test_keep_linebreak_after_question(self):
        text = "Tại sao?\nVì rằng"
        result = LINEBREAK_RE.sub(' ', text)
        assert "Tại sao?\nVì rằng" == result

    def test_keep_linebreak_after_exclamation(self):
        text = "Tuyệt vời!\nTiếp tục"
        result = LINEBREAK_RE.sub(' ', text)
        assert "Tuyệt vời!\nTiếp tục" == result


# WHITESPACE_RE Tests

class TestWhitespaceCollapse:
    """Test whitespace normalization."""

    def test_collapse_multiple_spaces(self):
        text = "hello    world"
        result = WHITESPACE_RE.sub(' ', text)
        assert result == "hello world"

    def test_collapse_tabs(self):
        text = "hello\t\tworld"
        result = WHITESPACE_RE.sub(' ', text)
        assert result == "hello world"

    def test_collapse_nbsp(self):
        text = "hello\u00a0\u00a0world"
        result = WHITESPACE_RE.sub(' ', text)
        assert result == "hello world"

    def test_single_space_unchanged(self):
        text = "hello world"
        result = WHITESPACE_RE.sub(' ', text)
        assert result == "hello world"


# Helper Functions Tests

class TestCountWords:
    """Test word counting function."""

    def test_normal_sentence(self):
        assert count_words("Đây là một câu tiếng Việt") == 6

    def test_empty_string(self):
        assert count_words("") == 0

    def test_single_word(self):
        assert count_words("xin") == 1

    def test_extra_whitespace(self):
        assert count_words("  hello   world  ") == 2


class TestSpecialCharRatio:
    """Test special character ratio calculation."""

    def test_normal_text(self):
        ratio = special_char_ratio("Đây là văn bản tiếng Việt bình thường.")
        assert ratio < 0.1

    def test_table_data(self):
        ratio = special_char_ratio("|||===###$$$%%%^^^&&&")
        assert ratio > 0.5

    def test_empty_string(self):
        assert special_char_ratio("") == 0.0

    def test_pure_vietnamese(self):
        ratio = special_char_ratio("Xin chào thế giới")
        assert ratio == 0.0
