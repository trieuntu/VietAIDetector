"""Build paired, source-disjoint long documents without detector-score selection."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import random
import re
import shutil
import statistics
import sys
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer


AI_FIELD = "google-gemma-3-12b-it-generated_text"
MODEL = "google/gemma-3-12b-it"
SEPARATOR = "\n\n"


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Components:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, i):
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, i, j):
        i, j = self.find(i), self.find(j)
        self.parent[max(i, j)] = min(i, j)


def shingles(text):
    words = re.findall(r"\w+", canonical(text))
    return {
        int.from_bytes(hashlib.blake2b(" ".join(words[i:i + 5]).encode("utf-8"),
                                     digest_size=8).digest(), "big")
        for i in range(len(words) - 4)
    }


def find_related(rows, cleaned, threshold):
    """Use a complete prefix filter for Jaccard joins, then verify candidates."""
    components = Components(len(rows))
    seen = {}
    text_owners = defaultdict(list)
    exact_edges = 0
    for i, row in enumerate(rows):
        keys = [("id", str(row["id"]))]
        prompt = canonical(row.get("user_prompt", ""))
        if prompt:
            keys.append(("prompt", prompt))
        for label in (0, 1):
            value = canonical(cleaned[i][label])
            if value:
                keys.append(("text", value))
                text_owners[value].append((i, label))
        for key in keys:
            if key in seen:
                components.join(i, seen[key])
                exact_edges += 1
            else:
                seen[key] = i

    unique = list(text_owners)
    sets = [shingles(text) for text in unique]
    frequencies = Counter(s for item in sets for s in item)
    inverted = defaultdict(list)
    conflicts = set()
    for owners in text_owners.values():
        if len({label for _, label in owners}) > 1:
            conflicts.update(i for i, _ in owners)
    near_edges = []
    for position, i in enumerate(sorted(range(len(sets)), key=lambda i: (len(sets[i]), i))):
        item = sets[i]
        if not item:
            continue
        prefix_length = len(item) - math.ceil(threshold * len(item)) + 1
        prefix = sorted(item, key=lambda s: (frequencies[s], s))[:prefix_length]
        candidates = {j for s in prefix for j in inverted[s]
                      if len(sets[j]) >= threshold * len(item)}
        for j in sorted(candidates):
            intersection = len(item & sets[j])
            similarity = intersection / (len(item) + len(sets[j]) - intersection)
            if similarity + 1e-12 < threshold:
                continue
            oi, oj = text_owners[unique[i]], text_owners[unique[j]]
            components.join(oi[0][0], oj[0][0])
            near_edges.append({"source_line_a": oi[0][0] + 1,
                               "source_line_b": oj[0][0] + 1,
                               "jaccard": round(similarity, 8)})
            if len({label for _, label in oi + oj}) > 1:
                conflicts.update(k for k, _ in oi + oj)
        for s in prefix:
            inverted[s].append(i)
        if position % 2000 == 0:
            print(f"Near-duplicate audit {position}/{len(sets)}", flush=True)
    return components, conflicts, exact_edges, near_edges


def describe(values):
    values = sorted(values)
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "mean": round(statistics.mean(values), 3), "max": max(values),
            "p25": values[int((len(values) - 1) * .25)],
            "p75": values[int((len(values) - 1) * .75)]}


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, metavar="FILE",
                        help="UTF-8 JSONL containing paired Human passages and Gemma continuations.")
    parser.add_argument("--software-root", type=Path, default=here.parents[1],
                        help="VietAIDetector checkout containing preprocessing/.")
    parser.add_argument("--tokenizer", type=Path, default=here / "inputs/phogpt-tokenizer.json")
    parser.add_argument("--exclude-csv", type=Path, action="append", metavar="CSV",
                        help="Calibration CSV with y_id. Repeat to replace the bundled ID list.")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory. Existing directories are never overwritten.")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--development-fraction", type=float, default=.2)
    parser.add_argument("--min-tokens", type=int, default=2048)
    parser.add_argument("--min-parts", type=int, default=3)
    parser.add_argument("--near-jaccard", type=float, default=.8)
    args = parser.parse_args()
    if args.exclude_csv is None:
        args.exclude_csv = [here / "inputs/calibration_ids.csv"]
    for path in [args.source, args.tokenizer, *args.exclude_csv]:
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    if not (args.software_root / "preprocessing/normalizer.py").is_file():
        parser.error("--software-root must point to a VietAIDetector checkout.")
    if args.output.exists():
        parser.error(f"Output already exists. Choose a new directory: {args.output}")
    if not (0 < args.development_fraction < 1 and 0 < args.near_jaccard <= 1):
        parser.error("Fractions must be between zero and one.")
    if args.min_parts < 1 or args.min_tokens < 1:
        parser.error("Minimum sizes must be positive.")

    software = args.software_root.resolve()
    sys.path.insert(0, str(software))
    from preprocessing.normalizer import TextNormalizer, DEHYPHEN_RE
    normalizer = TextNormalizer()

    def clean(text):
        parts = normalizer.filter_paragraphs(DEHYPHEN_RE.sub(r"\1\2", text))
        return "\n".join(normalizer.normalize(part) for part in parts)

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_truncation()
    tokenizer.no_padding()

    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        parser.error("The source JSONL contains no passage pairs.")
    for row in rows:
        if not all(isinstance(row.get(k), str) for k in ("id", "text", AI_FIELD)):
            raise ValueError("Missing ID or text in source data")
        if row.get("model_name") != MODEL:
            raise ValueError("Unexpected generator model")
    cleaned = [(clean(r["text"]), clean(r[AI_FIELD])) for r in rows]
    counts = [(count(h), count(a)) for h, a in cleaned]
    excluded_ids = set()
    calibration = []
    for path in args.exclude_csv:
        with path.open(encoding="utf-8") as stream:
            ids = {r["y_id"] for r in csv.DictReader(stream)}
        excluded_ids.update(ids)
        calibration.append({"path": str(path.resolve()), "sha256": file_digest(path),
                            "unique_ids": len(ids),
                            "source_overlap_ids": len(ids & {r["id"] for r in rows})})

    components, conflicts, exact_edges, near_edges = find_related(rows, cleaned, args.near_jaccard)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[components.find(i)].append(i)
    status = {}
    eligible = []
    for root, members in sorted(groups.items()):
        if any(rows[i]["id"] in excluded_ids for i in members):
            for i in members:
                status[i] = "calibration_id_or_related_source"
        elif any(i in conflicts for i in members):
            for i in members:
                status[i] = "cross_label_exact_or_near_duplicate"
        else:
            usable = [i for i in members if min(counts[i]) >= 64]
            if not usable:
                for i in members:
                    status[i] = "fewer_than_64_tokens_after_preprocessing"
                continue
            representative = usable[0]
            eligible.append(representative)
            for i in members:
                status[i] = "eligible" if i == representative else "duplicate_component_nonrepresentative"

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    cut = round(len(eligible) * args.development_fraction)
    if cut == 0 or cut == len(eligible):
        parser.error("The eligible source count and development fraction must produce two nonempty partitions.")
    split_sources = {"development": eligible[:cut], "test": eligible[cut:]}
    outputs = {}
    source_split, source_pair = {}, {}
    pair_groups = {}
    for split, indices in split_sources.items():
        documents = []
        pending = []
        number = 0
        for i in indices:
            source_split[i] = split
            pending.append(i)
            if len(pending) < args.min_parts or min(sum(counts[j][c] for j in pending) for c in (0, 1)) < args.min_tokens:
                continue
            raw = [SEPARATOR.join(rows[j][field].strip() for j in pending) for field in ("text", AI_FIELD)]
            actual_counts = [count(clean(text)) for text in raw]
            if min(actual_counts) < args.min_tokens:
                continue
            number += 1
            pair_id = f"gemma-news-concat-{split}-{number:05d}"
            pair_groups[pair_id] = list(pending)
            for j in pending:
                status[j] = "used"
                source_pair[j] = pair_id
            for label in (0, 1):
                text = raw[label]
                documents.append({
                    "id": pair_id + ("-human" if label == 0 else "-ai"),
                    "pair_id": pair_id, "split": split, "text": text,
                    "is_ai_generated": bool(label), "label": label,
                    "model": "human" if label == 0 else MODEL,
                    "topic": "news", "source_topics": [rows[j].get("topic", "") for j in pending],
                    "source_ids": [rows[j]["id"] for j in pending],
                    "source_lines": [j + 1 for j in pending],
                    "source_count": len(pending), "num_tokens": actual_counts[label],
                    "raw_num_tokens": count(text), "text_sha256": digest(text),
                    "preprocessed_sha256": digest(clean(text)),
                    "generation_metadata": [rows[j].get("gen_meta", {}) for j in pending] if label else [],
                })
            pending = []
        for j in pending:
            status[j] = "unused_split_tail_below_target"
        if not documents:
            raise ValueError(f"No complete pairs in {split}")
        random.Random(args.seed + (1 if split == "development" else 2)).shuffle(documents)
        outputs[split] = documents

    # Every retained source is used once per class, always in the same paired group.
    used = [j for group in pair_groups.values() for j in group]
    assert len(used) == len(set(used))
    assert not ({rows[j]["id"] for j in used} & excluded_ids)
    assert len({components.find(j) for j in used}) == len(used)
    doc_hashes = set()
    summary = {}
    for split, documents in outputs.items():
        by_pair = defaultdict(list)
        for doc in documents:
            by_pair[doc["pair_id"]].append(doc)
            assert doc["num_tokens"] >= args.min_tokens
            assert doc["preprocessed_sha256"] not in doc_hashes
            doc_hashes.add(doc["preprocessed_sha256"])
            assert all(source_split[line - 1] == split for line in doc["source_lines"])
        for pair in by_pair.values():
            assert sorted(d["label"] for d in pair) == [0, 1]
            assert pair[0]["source_lines"] == pair[1]["source_lines"]
        summary[split] = {
            "documents": len(documents), "pairs": len(by_pair),
            "human_documents": sum(d["label"] == 0 for d in documents),
            "ai_documents": sum(d["label"] == 1 for d in documents),
            "allocated_source_rows": len(split_sources[split]),
            "used_source_rows": sum(d["source_count"] for d in documents if d["label"] == 0),
            "human_tokens": describe([d["num_tokens"] for d in documents if d["label"] == 0]),
            "ai_tokens": describe([d["num_tokens"] for d in documents if d["label"] == 1]),
            "parts_per_pair": dict(sorted(Counter(d["source_count"] for d in documents if d["label"] == 0).items())),
        }

    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".benchmark-", dir=destination.parent) as staging:
        output = Path(staging) / "complete"
        for directory in (output / "data", output / "audit", output / "assets"):
            directory.mkdir(parents=True, exist_ok=True)
        asset = output / "assets" / "phogpt-tokenizer.json"
        if asset != args.tokenizer.resolve():
            shutil.copyfile(args.tokenizer, asset)
        for split, documents in outputs.items():
            save_json(output / "data" / f"{split}.json", documents)
        manifest = [{"source_line": i + 1, "source_id": row["id"],
                     "component_first_line": components.find(i) + 1, "status": status[i],
                     "split": source_split.get(i, ""), "pair_id": source_pair.get(i, ""),
                     "human_tokens": counts[i][0], "ai_tokens": counts[i][1],
                     "human_preprocessed_sha256": digest(cleaned[i][0]),
                     "ai_preprocessed_sha256": digest(cleaned[i][1]),
                     "prompt_sha256": digest(canonical(row.get("user_prompt", "")))}
                    for i, row in enumerate(rows)]
        save_csv(output / "audit" / "source_manifest.csv", manifest)
        save_json(output / "audit" / "near_duplicate_matches.json", near_edges)
        save_csv(output / "audit" / "document_manifest.csv", [
            {key: d[key] for key in ("id", "pair_id", "split", "label", "source_count", "num_tokens", "raw_num_tokens", "text_sha256")}
            for documents in outputs.values() for d in documents])
        preprocessing_files = [software / "preprocessing" / name for name in ("normalizer.py", "text_utils.py")]
        report = {
            "benchmark": "Gemma-3-12B-News concatenated long-document benchmark v1",
            "source": {"path": str(args.source.resolve()), "sha256": file_digest(args.source), "rows": len(rows)},
            "configuration": {"seed": args.seed, "development_fraction_of_eligible_sources": args.development_fraction,
                              "minimum_preprocessed_tokens": args.min_tokens, "minimum_parts": args.min_parts,
                              "separator": SEPARATOR, "near_duplicate_word_shingle_size": 5,
                              "near_duplicate_jaccard": args.near_jaccard,
                              "minimum_source_tokens_per_class": 64,
                              "selection_uses_detector_scores": False},
            "tokenizer": {"model": "vinai/PhoGPT-4B", "url": "https://huggingface.co/vinai/PhoGPT-4B/resolve/main/tokenizer.json",
                          "sha256": file_digest(asset), "add_special_tokens": False,
                          "truncation": False, "tokenizers_version": importlib.metadata.version("tokenizers")},
            "preprocessing": [{"path": str(p), "sha256": file_digest(p)} for p in preprocessing_files],
            "calibration_exclusion_inputs": calibration,
            "eligible_representatives": len(eligible), "exact_link_events": exact_edges,
            "near_duplicate_matches": len(near_edges), "source_status_counts": dict(sorted(Counter(status.values()).items())),
            "splits": summary,
            "verification": {"unique_source_per_pair": True, "source_components_disjoint": True,
                             "no_used_id_in_calibration_inputs": True, "balanced_classes": True,
                             "all_documents_meet_token_minimum": True, "no_duplicate_preprocessed_documents": True},
            "limitations": [
                "Calibration exclusion is ID-based because the supplied calibration tables contain scores and IDs, not text.",
                "Text duplicate checks cover this Gemma corpus. External articles with different IDs cannot be excluded by these checks.",
                "Near duplicates use five-word shingle Jaccard >= the configured threshold, not semantic equivalence.",
                "The concatenated documents combine independently generated news continuations and are not naturally authored long documents.",
                "The generator uses greedy decoding in this corpus. This benchmark does not establish robustness across generators or decoding methods.",
                "Human and AI lengths are not matched by truncation. Report length distributions and account for their difference in analysis.",
                "This corpus was previously evaluated in VietBinoculars. The new split is source-disjoint within this protocol, not a historically unseen corpus.",
            ],
        }
        save_json(output / "audit" / "report.json", report)
        os.rename(output, destination)
    print(json.dumps({"splits": summary, "source_status_counts": report["source_status_counts"]}, indent=2))


if __name__ == "__main__":
    main()
