"""Independently reconstruct exported documents and audit their provenance."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import PreTrainedTokenizerFast


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--check-reference", action="store_true",
                        help="Require byte-identical datasets and manifests from the manuscript benchmark.")
    parser.add_argument("--source", type=Path,
                        help="Override the recorded JSONL path after relocating input files.")
    parser.add_argument("--software-root", type=Path,
                        help="Override the recorded VietAIDetector path.")
    parser.add_argument("--exclude-csv", type=Path, action="append",
                        help="Override recorded calibration paths, in the same order.")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Run verification without Python's -O or -OO options.")
    root = args.benchmark
    report = json.loads((root / "audit/report.json").read_text(encoding="utf-8"))
    source = args.source or Path(report["source"]["path"])
    assert sha(source.read_bytes()) == report["source"]["sha256"]
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == report["source"]["rows"]
    software_root = args.software_root or Path(report["preprocessing"][0]["path"]).parent.parent
    for item in report["preprocessing"]:
        path = software_root / "preprocessing" / Path(item["path"]).name
        assert sha(path.read_bytes()) == item["sha256"]
    sys.path.insert(0, str(software_root))
    from preprocessing.normalizer import TextNormalizer

    class PlainTextNormalizer(TextNormalizer):
        def _get_ocr_engine(self):
            return None

    normalizer = PlainTextNormalizer()
    asset = root / "assets/phogpt-tokenizer.json"
    assert sha(asset.read_bytes()) == report["tokenizer"]["sha256"]
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(asset))
    excluded = set()
    calibration_inputs = report["calibration_exclusion_inputs"]
    calibration_paths = args.exclude_csv or [Path(entry["path"]) for entry in calibration_inputs]
    if len(calibration_paths) != len(calibration_inputs):
        parser.error("Provide one --exclude-csv for each recorded calibration input.")
    for path, entry in zip(calibration_paths, calibration_inputs):
        assert sha(path.read_bytes()) == entry["sha256"]
        with path.open(encoding="utf-8") as stream:
            excluded.update(r["y_id"] for r in csv.DictReader(stream))
    with (root / "audit/source_manifest.csv").open(encoding="utf-8") as stream:
        manifest = {int(r["source_line"]): r for r in csv.DictReader(stream)}
    assert len(manifest) == len(rows)
    assert Counter(r["status"] for r in manifest.values()) == report["source_status_counts"]
    all_doc_ids, all_hashes = set(), set()
    pair_sources, component_pairs = {}, {}
    seen_sources = {0: set(), 1: set()}
    seen_source_ids = {0: set(), 1: set()}
    pair_members = defaultdict(list)
    split_counts, lengths = {}, {}
    for split in ("development", "test"):
        path = root / "data" / f"{split}.json"
        documents = json.loads(path.read_text(encoding="utf-8"))
        classes = Counter()
        split_lengths = {0: [], 1: []}
        for doc in documents:
            assert doc["id"] not in all_doc_ids
            all_doc_ids.add(doc["id"])
            assert doc["split"] == split
            label = doc["label"]
            assert label in (0, 1) and doc["is_ai_generated"] == bool(label)
            assert doc["model"] == ("google/gemma-3-12b-it" if label else "human")
            classes[label] += 1
            indices = doc["source_lines"]
            assert len(indices) == doc["source_count"] and len(set(indices)) == len(indices)
            assert len(indices) >= report["configuration"]["minimum_parts"]
            assert not set(indices) & seen_sources[label]
            seen_sources[label].update(indices)
            source_ids = [rows[i - 1]["id"] for i in indices]
            assert source_ids == doc["source_ids"]
            assert not set(source_ids) & seen_source_ids[label]
            seen_source_ids[label].update(source_ids)
            assert not set(source_ids) & excluded
            field = "google-gemma-3-12b-it-generated_text" if label else "text"
            reconstructed = "\n\n".join(rows[i - 1][field].strip() for i in indices)
            assert reconstructed == doc["text"]
            assert sha(reconstructed.encode("utf-8")) == doc["text_sha256"]
            prep = normalizer.preprocess(doc["id"] + ".txt", reconstructed)
            assert prep.extraction_status == "success"
            clean_hash = sha(prep.cleaned_text.encode("utf-8"))
            assert clean_hash == doc["preprocessed_sha256"]
            assert clean_hash not in all_hashes
            all_hashes.add(clean_hash)
            n = len(tokenizer.encode(prep.cleaned_text, add_special_tokens=False, truncation=False))
            raw_n = len(tokenizer.encode(reconstructed, add_special_tokens=False, truncation=False))
            assert n == doc["num_tokens"] and raw_n == doc["raw_num_tokens"]
            assert n >= report["configuration"]["minimum_preprocessed_tokens"]
            split_lengths[label].append(n)
            pair_id = doc["pair_id"]
            if pair_id in pair_sources:
                assert pair_sources[pair_id] == (split, indices)
            pair_sources[pair_id] = (split, indices)
            pair_members[pair_id].append(label)
            for line in indices:
                entry = manifest[line]
                assert entry["status"] == "used" and entry["split"] == split and entry["pair_id"] == pair_id
                component = entry["component_first_line"]
                if component in component_pairs:
                    assert component_pairs[component] == (pair_id, line)
                component_pairs[component] = (pair_id, line)
        assert classes[0] == classes[1] == report["splits"][split]["pairs"]
        assert len(documents) == report["splits"][split]["documents"]
        split_counts[split] = len(documents)
        lengths[split] = {"human_min": min(split_lengths[0]), "human_max": max(split_lengths[0]),
                          "ai_min": min(split_lengths[1]), "ai_max": max(split_lengths[1])}
    assert seen_sources[0] == seen_sources[1]
    assert seen_sources[0] == {i for i, r in manifest.items() if r["status"] == "used"}
    assert all(sorted(labels) == [0, 1] for labels in pair_members.values())
    for edge in json.loads((root / "audit/near_duplicate_matches.json").read_text(encoding="utf-8")):
        a, b = manifest[edge["source_line_a"]], manifest[edge["source_line_b"]]
        assert a["component_first_line"] == b["component_first_line"]
    result = {"passed": True, "documents_verified": len(all_doc_ids), "pairs_verified": len(pair_members),
              "source_rows_used": len(seen_sources[0]), "split_document_counts": split_counts,
              "token_ranges_after_actual_pipeline_preprocessing": lengths,
              "checks": ["source and calibration file hashes", "preprocessing code hashes", "tokenizer hash",
                         "exact document reconstruction from source lines", "actual TextNormalizer.preprocess output",
                         "token counts through Transformers tokenizer wrapper", "class balance and paired sources",
                         "source ID and source component isolation", "calibration ID exclusion",
                         "unique document hashes", "near-duplicate component consistency"],
              "file_sha256": {str(p.relative_to(root)): sha(p.read_bytes())
                              for p in sorted(root.rglob("*")) if p.is_file()
                              and p.suffix in (".json", ".csv", ".py", ".md")
                              and p.name != "validation.json" and "__pycache__" not in p.parts}}
    if args.check_reference:
        expected = json.loads(Path(__file__).with_name("expected_sha256.json").read_text(encoding="utf-8"))
        for name, digest in expected.items():
            if sha((root / name).read_bytes()) != digest:
                raise ValueError(f"Manuscript benchmark SHA-256 mismatch: {name}")
        result["reference_files_verified"] = sorted(expected)
    if args.write_report:
        (root / "audit/validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
