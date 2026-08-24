"""VietAIDetector — OCR Engine (Vintern-1B-v2)"""

import logging
import math

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from config.settings import (
    HF_TOKEN,
    OCR_DEVICE,
    OCR_MAX_IMAGE_TILES,
    OCR_MAX_NEW_TOKENS,
    OCR_MIN_PAGE_CHARS,
    OCR_MODEL,
    OCR_PROMPT,
    USE_BFLOAT16,
)

logger = logging.getLogger(__name__)

# Image Preprocessing (from InternVL2 / Vintern model repo)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int = 448):
    """Build the image transform pipeline for Vintern-1B-v2."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize(
            (input_size, input_size),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Find the closest aspect ratio from the target set."""
    best_ratio = (1, 1)
    best_diff = float("inf")
    area = width * height

    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
        elif diff == best_diff:
            # Prefer ratio whose total pixel area is closer to original
            if abs(area - ratio[0] * ratio[1] * image_size * image_size) < abs(
                area - best_ratio[0] * best_ratio[1] * image_size * image_size
            ):
                best_ratio = ratio

    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = OCR_MAX_IMAGE_TILES,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    """Split an image into tiles using dynamic high-resolution strategy."""
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    # Generate all valid tile arrangements
    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if min_num <= i * j <= max_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_w, orig_h, image_size
    )

    target_w = best_ratio[0] * image_size
    target_h = best_ratio[1] * image_size
    blocks = best_ratio[0] * best_ratio[1]

    resized = image.resize((target_w, target_h))
    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_w // image_size)) * image_size,
            (i // (target_w // image_size)) * image_size,
            ((i % (target_w // image_size)) + 1) * image_size,
            ((i // (target_w // image_size)) + 1) * image_size,
        )
        processed_images.append(resized.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        thumbnail = image.resize((image_size, image_size))
        processed_images.append(thumbnail)

    return processed_images


def load_image(
    image: Image.Image,
    max_num: int = OCR_MAX_IMAGE_TILES,
) -> torch.Tensor:
    """Preprocess a PIL Image into a tensor for Vintern-1B-v2 inference."""
    transform = _build_transform(input_size=448)
    images = _dynamic_preprocess(image, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


# VinternOCR Engine


class VinternOCR:
    """OCR engine using Vintern-1B-v2 for Vietnamese text extraction."""

    def __init__(self, device: str = OCR_DEVICE):
        """Initialize the OCR engine (model is loaded lazily)."""
        self.model = None
        self.tokenizer = None
        self.device = device
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load the Vintern-1B-v2 model and tokenizer."""
        if self._loaded:
            return

        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading Vintern-1B-v2 OCR model onto %s...", self.device)

        # Bypass transformers check_imports for flash_attn
        import transformers.dynamic_module_utils as dmu
        if not hasattr(dmu, '_original_check_imports'):
            dmu._original_check_imports = dmu.check_imports
            def _custom_check_imports(filename):
                try:
                    return dmu._original_check_imports(filename)
                except Exception as e:
                    if "flash_attn" in str(e):
                        logger.debug("Bypassed flash_attn check in check_imports")
                        return dmu.get_relative_imports(filename)
                    else:
                        raise e
            dmu.check_imports = _custom_check_imports

        dtype = torch.bfloat16 if USE_BFLOAT16 else torch.float32
        hf_kwargs = dict(
            trust_remote_code=True, 
            token=HF_TOKEN,
            attn_implementation="eager"
        )

        self.model = AutoModel.from_pretrained(
            OCR_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": self.device},
            **hf_kwargs,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            OCR_MODEL,
            use_fast=False,
            **hf_kwargs,
        )

        self._loaded = True
        logger.info("Vintern-1B-v2 loaded successfully on %s.", self.device)

    @torch.inference_mode()
    def extract_page(self, image: Image.Image) -> str:
        """Extract text from a single page image using Vintern-1B-v2."""
        self._ensure_loaded()

        dtype = torch.bfloat16 if USE_BFLOAT16 else torch.float32
        pixel_values = load_image(image).to(dtype).to(self.device)

        # Hard-coded anti-hallucination generation config
        generation_config = dict(
            max_new_tokens=OCR_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            repetition_penalty=1.0,
        )

        question = f"<image>\n{OCR_PROMPT}"
        # Force the default cuda device to match the model's device
        # because the remote modeling code hardcodes `.cuda()` calls.
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
        )

        # Post-processing: validate output length
        text = response.strip() if response else ""
        if len(text) < OCR_MIN_PAGE_CHARS:
            logger.warning(
                "OCR output too short (%d chars), skipping page.", len(text)
            )
            return ""

        return text

    def extract_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from all pages of a scanned PDF."""
        import fitz  # PyMuPDF

        all_pages_text: list[str] = []

        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            logger.info("Starting OCR on %d pages...", len(pdf))

            for page_num, page in enumerate(pdf, start=1):
                # Render page to image at 300 DPI for high-quality OCR
                mat = fitz.Matrix(300 / 72, 300 / 72)  # 300 DPI
                pix = page.get_pixmap(matrix=mat)

                # Convert to PIL Image
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                page_text = self.extract_page(img)
                if page_text:
                    all_pages_text.append(page_text)
                    logger.info(
                        "Page %d/%d: extracted %d characters.",
                        page_num, len(pdf), len(page_text),
                    )
                else:
                    logger.info("Page %d/%d: skipped (blank or too short).", page_num, len(pdf))

        return "\n".join(all_pages_text)
