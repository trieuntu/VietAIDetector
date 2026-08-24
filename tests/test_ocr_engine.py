"""Unit tests for the VinternOCR engine."""

import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image


class TestLoadImage(unittest.TestCase):
    """Test image preprocessing utilities."""

    def test_load_image_returns_tensor(self):
        """load_image should return a 4D tensor (tiles, 3, 448, 448)."""
        from preprocessing.ocr_engine import load_image

        img = Image.new("RGB", (800, 600), color="white")
        tensor = load_image(img, max_num=2)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dim(), 4)
        self.assertEqual(tensor.shape[1], 3)
        self.assertEqual(tensor.shape[2], 448)
        self.assertEqual(tensor.shape[3], 448)

    def test_load_image_single_tile_for_small_image(self):
        """A 448x448 image should produce exactly 1 tile (+ thumbnail = 1 since only 1 tile no thumbnail)."""
        from preprocessing.ocr_engine import load_image

        img = Image.new("RGB", (448, 448), color="white")
        tensor = load_image(img, max_num=12)

        # Single tile + no thumbnail (thumbnail only added when >1 tile)
        self.assertEqual(tensor.shape[0], 1)

    def test_load_image_grayscale_conversion(self):
        """Grayscale images should be converted to RGB."""
        from preprocessing.ocr_engine import load_image

        img = Image.new("L", (448, 448), color=128)
        tensor = load_image(img, max_num=2)

        self.assertEqual(tensor.shape[1], 3)  # Should be 3 channels (RGB)


class TestDynamicPreprocess(unittest.TestCase):
    """Test the dynamic high-resolution tiling strategy."""

    def test_large_image_produces_multiple_tiles(self):
        """A large image should produce >1 tile."""
        from preprocessing.ocr_engine import _dynamic_preprocess

        img = Image.new("RGB", (1600, 1200), color="white")
        tiles = _dynamic_preprocess(img, max_num=6)

        self.assertGreater(len(tiles), 1)

    def test_small_image_produces_one_tile(self):
        """An image smaller than tile size should produce 1 tile."""
        from preprocessing.ocr_engine import _dynamic_preprocess

        img = Image.new("RGB", (200, 200), color="white")
        tiles = _dynamic_preprocess(img, min_num=1, max_num=4)

        self.assertEqual(len(tiles), 1)


class TestVinternOCR(unittest.TestCase):
    """Test VinternOCR engine with mocked model."""

    @patch("preprocessing.ocr_engine.VinternOCR._ensure_loaded")
    def test_extract_page_returns_text(self, mock_ensure):
        """extract_page should return model's chat output."""
        from preprocessing.ocr_engine import VinternOCR

        ocr = VinternOCR(device="cpu")
        ocr._loaded = True

        # Mock model.chat to return sample text
        mock_model = MagicMock()
        mock_model.chat.return_value = "Đây là văn bản trích xuất từ ảnh."
        ocr.model = mock_model
        ocr.tokenizer = MagicMock()

        img = Image.new("RGB", (800, 600), color="white")
        result = ocr.extract_page(img)

        self.assertEqual(result, "Đây là văn bản trích xuất từ ảnh.")
        mock_model.chat.assert_called_once()

    @patch("preprocessing.ocr_engine.VinternOCR._ensure_loaded")
    def test_extract_page_skips_short_output(self, mock_ensure):
        """Pages with output shorter than OCR_MIN_PAGE_CHARS should return empty."""
        from preprocessing.ocr_engine import VinternOCR

        ocr = VinternOCR(device="cpu")
        ocr._loaded = True

        mock_model = MagicMock()
        mock_model.chat.return_value = "abc"  # Too short
        ocr.model = mock_model
        ocr.tokenizer = MagicMock()

        img = Image.new("RGB", (800, 600), color="white")
        result = ocr.extract_page(img)

        self.assertEqual(result, "")

    @patch("preprocessing.ocr_engine.VinternOCR._ensure_loaded")
    def test_extract_page_antihallucination_config(self, mock_ensure):
        """Verify anti-hallucination generation config is correctly applied."""
        from preprocessing.ocr_engine import VinternOCR

        ocr = VinternOCR(device="cpu")
        ocr._loaded = True

        mock_model = MagicMock()
        mock_model.chat.return_value = "Văn bản dài hơn mười ký tự cho kiểm thử."
        ocr.model = mock_model
        ocr.tokenizer = MagicMock()

        img = Image.new("RGB", (800, 600), color="white")
        ocr.extract_page(img)

        # Capture the generation_config passed to model.chat
        call_args = mock_model.chat.call_args
        gen_config = call_args[0][3]  # 4th positional arg

        self.assertFalse(gen_config["do_sample"])
        self.assertEqual(gen_config["temperature"], 0.0)
        self.assertEqual(gen_config["num_beams"], 1)
        self.assertEqual(gen_config["repetition_penalty"], 1.0)

    @patch("preprocessing.ocr_engine.VinternOCR._ensure_loaded")
    def test_extract_page_empty_response(self, mock_ensure):
        """Empty model response should return empty string."""
        from preprocessing.ocr_engine import VinternOCR

        ocr = VinternOCR(device="cpu")
        ocr._loaded = True

        mock_model = MagicMock()
        mock_model.chat.return_value = ""
        ocr.model = mock_model
        ocr.tokenizer = MagicMock()

        img = Image.new("RGB", (800, 600), color="white")
        result = ocr.extract_page(img)

        self.assertEqual(result, "")


class TestDocumentReaderScannedPDF(unittest.TestCase):
    """Test that document_reader handles pdf_scanned format correctly."""

    def test_extract_from_pdf_returns_scanned_for_empty_pdf(self):
        """A PDF with no text layer should return ('', 'pdf_scanned')."""
        from preprocessing.document_reader import DocumentReader

        reader = DocumentReader()

        # Create a minimal blank PDF using PyMuPDF
        import fitz
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        pdf_bytes = doc.tobytes()
        doc.close()

        text, fmt = reader.extract_from_pdf(pdf_bytes)
        self.assertEqual(text, "")
        self.assertEqual(fmt, "pdf_scanned")

    def test_read_raises_without_ocr_engine(self):
        """read() should raise UnsupportedFormatError for scanned PDF without OCR engine."""
        from preprocessing.document_reader import DocumentReader, UnsupportedFormatError

        reader = DocumentReader()

        import fitz
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        pdf_bytes = doc.tobytes()
        doc.close()

        with self.assertRaises(UnsupportedFormatError):
            reader.read("scanned.pdf", pdf_bytes, ocr_engine=None)

    def test_read_delegates_to_ocr_engine(self):
        """read() should delegate to ocr_engine for scanned PDFs."""
        from preprocessing.document_reader import DocumentReader

        reader = DocumentReader()

        import fitz
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        pdf_bytes = doc.tobytes()
        doc.close()

        mock_ocr = MagicMock()
        mock_ocr.extract_from_pdf.return_value = "OCR extracted text"

        text, fmt = reader.read("scanned.pdf", pdf_bytes, ocr_engine=mock_ocr)
        self.assertEqual(text, "OCR extracted text")
        self.assertEqual(fmt, "pdf_scanned")
        mock_ocr.extract_from_pdf.assert_called_once_with(pdf_bytes)


if __name__ == "__main__":
    unittest.main()
