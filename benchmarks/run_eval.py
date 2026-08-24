"""
Benchmark Evaluation Script.
Evaluates JSON datasets using the VietBinoculars pipeline and exports to CSV.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_WINDOW,
    DEFAULT_MODE,
    OBSERVER_MODEL,
    PERFORMER_MODEL,
    THRESHOLD_MODES,
)
from core.scorer import VietBinocularsScorer
from preprocessing.normalizer import TextNormalizer
from processing.aggregator import ScoreAggregator
from processing.chunker import TextChunker

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_eval")

# CSV column order
CSV_COLUMNS = [
    "document_id",
    "ai_percentage",
    "final_decision",
    "applied_threshold",
    "applied_mode",
    "chunk_window",
    "chunk_overlap",
    "total_chunks",
    "ai_chunk_count",
    # Ground-truth metadata from the dataset (if present)
    "is_ai_generated",
    "model",
    "topic",
    "processing_time_seconds",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run VietBinoculars benchmark evaluation on a JSON dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available threshold modes:\n"
            + "\n".join(f"  - {m}" for m in THRESHOLD_MODES)
        ),
    )
    parser.add_argument(
        "input",
        type=str,
        help=(
            "Path to a JSON dataset file or a directory containing JSON files. "
            "Each JSON file must be a list of objects with at least 'id' and 'text' fields."
        ),
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help=(
            "Output CSV file path. Defaults to "
            "'benchmarks/results/<input_stem>_<mode_short>_w<W>_o<O>.csv'."
        ),
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=list(THRESHOLD_MODES.keys()),
        help=f"Detection threshold mode (default: '{DEFAULT_MODE}').",
    )
    parser.add_argument(
        "-w", "--window",
        type=int,
        default=CHUNK_WINDOW,
        help=f"Chunk window size in tokens (default: {CHUNK_WINDOW}).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Chunk overlap in tokens (default: {CHUNK_OVERLAP}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a previous run by skipping document IDs already in the "
            "output CSV."
        ),
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict]:
    """Load a single JSON dataset file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path.name}: expected a JSON array, got {type(data).__name__}")
    log.info("Loaded %d documents from %s", len(data), path.name)
    return data


def collect_input_files(input_path: str) -> list[Path]:
    """Resolve the input argument to a list of JSON file paths."""
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON files found in {p}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {p}")


def build_output_path(input_path: Path, mode: str, window: int, overlap: int) -> Path:
    """Generate a default output CSV path based on input filename and params."""
    mode_short = mode.split("(")[0].strip().replace(" ", "_").lower()
    stem = input_path.stem
    out_dir = PROJECT_ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_{mode_short}_w{window}_o{overlap}.csv"


def load_existing_ids(csv_path: Path) -> set[str]:
    """Read already-processed document IDs from an existing CSV (for --resume)."""
    if not csv_path.exists():
        return set()
    ids = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row["document_id"])
    log.info("Resume mode: found %d existing results in %s", len(ids), csv_path.name)
    return ids


def run_pipeline(
    doc: dict,
    normalizer: TextNormalizer,
    chunker: TextChunker,
    scorer: VietBinocularsScorer,
    aggregator: ScoreAggregator,
    threshold: float,
    mode_name: str,
    window: int,
    overlap: int,
) -> dict:
    """Run the full detection pipeline on a single document and return a result dict."""
    doc_id = doc.get("id", str(uuid.uuid4())[:8])
    text = doc["text"]

    t0 = time.perf_counter()

    # Step 1: Preprocess
    prep = normalizer.preprocess(f"{doc_id}.txt", text)
    if prep.extraction_status == "error":
        log.warning("  Preprocessing failed for %s: %s", doc_id, prep.error_message)
        return None

    # Step 2: Chunk
    chunk_output = chunker.chunk(
        prep.cleaned_text,
        document_id=doc_id,
        window=window,
        overlap=overlap,
    )
    if chunk_output.total_chunks == 0:
        log.warning("  No chunks produced for %s (text too short?)", doc_id)
        return None

    # Step 3: Score
    scored_chunks = scorer.score_chunks(chunk_output.chunks)

    # Step 4: Aggregate
    agg = aggregator.aggregate(scored_chunks, threshold=threshold)

    elapsed = time.perf_counter() - t0

    return {
        "document_id": doc_id,
        "ai_percentage": agg.ai_percentage,
        "final_decision": agg.final_decision,
        "applied_threshold": round(threshold, 10),
        "applied_mode": mode_name,
        "chunk_window": window,
        "chunk_overlap": overlap,
        "total_chunks": agg.total_chunks,
        "ai_chunk_count": agg.ai_chunk_count,
        # Ground-truth metadata (may be absent)
        "is_ai_generated": doc.get("is_ai_generated", ""),
        "model": doc.get("model", ""),
        "topic": doc.get("topic", ""),
        "processing_time_seconds": round(elapsed, 3),
    }


def main():
    args = parse_args()

    # Resolve inputs
    input_files = collect_input_files(args.input)
    threshold = THRESHOLD_MODES[args.mode]

    log.info("=" * 60)
    log.info("VietAIDetector Benchmark Evaluation")
    log.info("=" * 60)
    log.info("Mode       : %s (threshold=%.10f)", args.mode, threshold)
    log.info("Chunk      : window=%d, overlap=%d", args.window, args.overlap)
    log.info("Input files: %d", len(input_files))
    log.info("=" * 60)

    # Initialize models (once)
    log.info("Loading models...")
    scorer = VietBinocularsScorer(OBSERVER_MODEL, PERFORMER_MODEL)
    normalizer = TextNormalizer()
    chunker = TextChunker(tokenizer_name_or_obj=scorer.tokenizer)
    aggregator = ScoreAggregator()
    log.info("Models loaded successfully.")

    # Process each dataset file
    for input_file in input_files:
        dataset = load_dataset(input_file)

        # Determine output path
        if args.output and len(input_files) == 1:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = build_output_path(
                input_file, args.mode, args.window, args.overlap
            )

        # Resume support
        skip_ids = load_existing_ids(output_path) if args.resume else set()

        # Open CSV in append mode for resume, write mode otherwise
        file_exists = output_path.exists() and args.resume
        mode = "a" if file_exists else "w"

        with open(output_path, mode, newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()

            total = len(dataset)
            skipped = 0
            failed = 0

            for i, doc in enumerate(dataset, start=1):
                doc_id = doc.get("id", "unknown")

                if doc_id in skip_ids:
                    skipped += 1
                    log.info("[%d/%d] SKIP %s (already processed)", i, total, doc_id)
                    continue

                log.info("[%d/%d] Processing %s ...", i, total, doc_id)

                result = run_pipeline(
                    doc, normalizer, chunker, scorer, aggregator,
                    threshold, args.mode, args.window, args.overlap,
                )

                if result is None:
                    failed += 1
                    continue

                writer.writerow(result)
                csvfile.flush()  # flush after each row for crash safety

                log.info(
                    "  → %s | AI=%.1f%% | chunks=%d (AI=%d) | %.1fs",
                    result["final_decision"],
                    result["ai_percentage"],
                    result["total_chunks"],
                    result["ai_chunk_count"],
                    result["processing_time_seconds"],
                )

        log.info("-" * 60)
        log.info(
            "Done: %s → %s  (processed=%d, skipped=%d, failed=%d)",
            input_file.name,
            output_path.name,
            total - skipped - failed,
            skipped,
            failed,
        )

    log.info("=" * 60)
    log.info("All evaluations complete.")


if __name__ == "__main__":
    main()
