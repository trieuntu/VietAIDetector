"""Unit tests for preprocessing/normalizer.py"""

import pytest
from preprocessing.normalizer import TextNormalizer


normalizer = TextNormalizer()


# normalize() Tests

class TestNormalize:
    """Test the normalize method (de-hyphen + linebreak + whitespace)."""

    def test_dehyphen_and_join(self):
        text = "phát tri-\nển kinh tế"
        result = normalizer.normalize(text)
        assert "phát triển" in result

    def test_remove_mid_sentence_linebreak(self):
        text = "Đây là\nmột câu"
        result = normalizer.normalize(text)
        assert "Đây là một câu" == result

    def test_collapse_whitespace(self):
        text = "Nhiều   khoảng    trắng"
        result = normalizer.normalize(text)
        assert result == "Nhiều khoảng trắng"

    def test_strip_result(self):
        text = "  hello world  "
        result = normalizer.normalize(text)
        assert result == "hello world"

    def test_combined_normalization(self):
        text = "Đây là   một\nvăn bản\t\tcó nhiều    lỗi"
        result = normalizer.normalize(text)
        assert "  " not in result
        assert "\t" not in result


# filter_paragraphs() Tests

class TestFilterParagraphs:
    """Test paragraph filtering (short paragraphs and high special char ratio)."""

    def test_remove_short_paragraphs(self):
        text = "Đây là đoạn bình thường có đủ số từ.\nHai từ\nĐoạn khác cũng đủ dài để giữ lại."
        result = normalizer.filter_paragraphs(text)
        # "Hai từ" should be filtered (< 5 words)
        assert len(result) == 2
        assert all("Hai từ" not in p for p in result)

    def test_remove_special_char_paragraphs(self):
        text = "Đoạn văn bình thường tiếng Việt.\n|||===###$$$%%%^^^&&&***"
        result = normalizer.filter_paragraphs(text)
        assert len(result) == 1

    def test_empty_paragraphs_filtered(self):
        text = "Đoạn một có đủ năm từ.\n\n\nĐoạn hai cũng có đủ số từ."
        result = normalizer.filter_paragraphs(text)
        assert len(result) == 2

    def test_all_filtered(self):
        text = "hi\nlo\nok"
        result = normalizer.filter_paragraphs(text)
        assert len(result) == 0

    def test_whitespace_normalization_in_filter(self):
        text = "Đoạn   có   nhiều   khoảng   trắng   thừa   bên   trong."
        result = normalizer.filter_paragraphs(text)
        assert len(result) == 1
        assert "  " not in result[0]


# preprocess() Tests (plain text only — no DOCX/PDF)

class TestPreprocess:
    """Test the full preprocessing pipeline with plain text input."""

    def test_success_plain_text(self):
        text = "Đây là một văn bản tiếng Việt bình thường dùng để kiểm tra."
        result = normalizer.preprocess("test.txt", text)
        assert result.extraction_status == "success"
        assert result.source_format == "plain"
        assert len(result.cleaned_text) > 0

    def test_empty_after_filtering(self):
        text = "hi\nlo"
        result = normalizer.preprocess("test.txt", text)
        assert result.extraction_status == "success"
        assert result.cleaned_text == ""

    def test_document_name_preserved(self):
        result = normalizer.preprocess("my_doc.txt", "Văn bản có đủ số từ để giữ lại.")
        assert result.document_name == "my_doc.txt"
