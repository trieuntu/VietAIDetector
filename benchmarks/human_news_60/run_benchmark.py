"""Evaluate the frozen 120-document dataset using a VietAIDetector checkout."""

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path

from common import (CODE_FILES, CONFIGURATIONS, GENERATORS, THRESHOLD, TOPICS,
                    file_hash, metrics, sha256, text_normalizer, write_csv, write_json)


def validate_dataset(dataset, repo, source=None):
    protocol = json.loads((dataset / "protocol.json").read_text(encoding="utf-8"))
    for filename, key in (("evaluation.json", "evaluation_sha256"),
                          ("human_manifest.csv", "human_manifest_sha256")):
        if file_hash(dataset / filename) != protocol[key]:
            raise ValueError(f"Dataset file changed: {filename}")
    for name in CODE_FILES:
        if file_hash(repo / name) != protocol["repository_code"][name]:
            raise ValueError(f"Repository code differs from the frozen build: {name}")
    if protocol["threshold"] != THRESHOLD or protocol["configurations"] != [list(x) for x in CONFIGURATIONS]:
        raise ValueError("Protocol settings differ from the fixed evaluation settings")
    documents = json.loads((dataset / "evaluation.json").read_text(encoding="utf-8"))
    if len(documents) != 120 or len({r["id"] for r in documents}) != 120:
        raise ValueError("Expected 120 unique documents")
    for cohort in GENERATORS:
        for topic in TOPICS:
            for label in (0, 1):
                if sum(r["cohort"] == cohort and r["topic"] == topic and r["label"] == label
                       for r in documents) != 4:
                    raise ValueError("Unexpected class/topic/cohort counts")
    clean = text_normalizer(repo)
    from config import settings
    expected_settings = dict(OBSERVER_MODEL="vinai/PhoGPT-4B", PERFORMER_MODEL="vinai/PhoGPT-4B-Chat",
                             MAX_MODEL_TOKENS=768, CHUNK_MIN_TOKENS=50)
    if any(getattr(settings, key) != value for key, value in expected_settings.items()):
        raise ValueError("Repository model/chunk settings differ from the evaluation protocol")
    from tokenizers import Tokenizer
    tokenizer_path = repo / "benchmarks/concatenated_news/inputs/phogpt-tokenizer.json"
    if file_hash(tokenizer_path) != protocol["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from the frozen build")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    for row in documents:
        if len(tokenizer.encode(clean(row["text"]), add_special_tokens=False).ids) != row["num_tokens"]:
            raise ValueError(f"Token count changed: {row['id']}")
    with (dataset / "human_manifest.csv").open(encoding="utf-8") as stream:
        manifest = {r["id"]: r for r in csv.DictReader(stream)}
    if len(manifest) != 60:
        raise ValueError("Expected 60 distinct Human manifest rows")
    for row in documents:
        if row["label"] == 0:
            entry = manifest[row["id"]]
            if (sha256(row["text"]) != entry["text_sha256"] or
                    sha256(clean(row["text"])) != entry["preprocessed_sha256"]):
                raise ValueError(f"Manifest text mismatch: {row['id']}")
            if any(str(row[k]) != entry[k] for k in ("cohort", "topic", "num_tokens", "length_bin")):
                raise ValueError(f"Manifest metadata mismatch: {row['id']}")
    for cohort in GENERATORS:
        path = repo / "benchmarks/datasets" / f"{cohort}.json"
        original = {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
        for row in documents:
            if row["label"] == 1 and row["cohort"] == cohort:
                if row["text"] != original[row["original_id"]]["text"]:
                    raise ValueError(f"AI text differs from Section 3.3 input: {row['id']}")
    if source is not None:
        if file_hash(source) != protocol["inputs"][0]["sha256"]:
            raise ValueError("Original news source hash differs")
        originals = json.loads(source.read_text(encoding="utf-8"))
        for row in documents:
            if row["label"] == 0:
                entry = manifest[row["id"]]
                original = originals[int(entry["source_index"])]
                if str(original["id"]) != entry["source_id"] or original["content"] != row["text"]:
                    raise ValueError(f"Human document differs from its intact source article: {row['id']}")
    return protocol, documents, clean, tokenizer


def summarize(records):
    output = []
    for window, overlap in CONFIGURATIONS:
        subset = [r for r in records if (r["window"], r["overlap"]) == (window, overlap)]
        groups = [("overall", "all", subset),
                  ("sensitivity", "common_length_2048_2559", [r for r in subset if r["num_tokens"] < 2560])]
        for field in ("cohort", "topic", "length_bin"):
            for value in sorted({r[field] for r in subset}):
                groups.append((field, value, [r for r in subset if r[field] == value]))
        for field, value, rows in groups:
            output.append(dict(window=window, overlap=overlap, group=field, value=value, **metrics(rows)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path, help="Optional original news JSON for full Human provenance validation.")
    parser.add_argument("--precision", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    protocol, documents, clean, reference = validate_dataset(args.dataset, repo, args.source)
    if args.validate_only:
        print("Validated all 120 documents, topic/class counts, token counts, hashes and repository code.")
        return
    if args.output is None:
        parser.error("--output is required for inference")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU inference is required; use --validate-only for CPU validation")
    if args.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support the archived bfloat16 setting. Use a compatible GPU or explicitly select --precision float16 and report the changed precision.")
    from config.settings import OBSERVER_MODEL, PERFORMER_MODEL, MAX_MODEL_TOKENS
    from core.scorer import VietBinocularsScorer
    from processing.chunker import TextChunker

    git_head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    environment = dict(
        dataset_sha256=protocol["evaluation_sha256"], protocol_sha256=file_hash(args.dataset / "protocol.json"),
        runner_sha256=file_hash(Path(__file__)), common_sha256=file_hash(Path(__file__).with_name("common.py")),
        repository_commit=git_head, precision=args.precision,
        python=platform.python_version(), cuda=torch.version.cuda,
        packages={name: importlib.metadata.version(name) for name in
                  ("torch", "transformers", "tokenizers", "numpy", "scikit-learn")},
        devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        threshold=THRESHOLD, configurations=[list(c) for c in CONFIGURATIONS], chunk_batch_size=1,
    )
    if args.output.exists() and not args.resume:
        raise FileExistsError("Output exists; choose a new directory or use --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    run_path = args.output / "run.json"
    if args.resume and not run_path.exists():
        raise ValueError("Cannot resume without run.json")
    if run_path.exists() and json.loads(run_path.read_text()) != environment:
        raise ValueError("Resume environment or inputs differ from the archived run")
    write_json(run_path, environment)

    scorer = VietBinocularsScorer(OBSERVER_MODEL, PERFORMER_MODEL,
                                 device1="cuda:0", device2="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0",
                                 use_bf16=args.precision != "float32")
    if args.precision == "float16":
        scorer.observer.to(dtype=torch.float16)
        scorer.performer.to(dtype=torch.float16)
    models = {name: getattr(model.config, "_commit_hash", None)
              for name, model in (("observer", scorer.observer), ("performer", scorer.performer))}
    models_path = args.output / "model_revisions.json"
    if args.resume and models_path.exists() and json.loads(models_path.read_text()) != models:
        raise ValueError("Model revisions changed since the previous run")
    write_json(models_path, models)
    chunker = TextChunker(scorer.tokenizer)
    records, completed = [], set()
    checkpoint = args.output / "predictions.jsonl"
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["id"], row["window"], row["overlap"])
            if key in completed:
                raise ValueError("Duplicate prediction in checkpoint")
            completed.add(key)
            records.append(row)
    expected = {(d["id"], w, o) for d in documents for w, o in CONFIGURATIONS}
    if not completed <= expected:
        raise ValueError("Checkpoint contains unknown documents or configurations")

    with checkpoint.open("a", encoding="utf-8") as stream:
        for document in documents:
            text = clean(document["text"])
            actual_ids = scorer.tokenizer.encode(text, add_special_tokens=False)
            if actual_ids != reference.encode(text, add_special_tokens=False).ids:
                raise ValueError(f"Scoring tokenizer differs: {document['id']}")
            for window, overlap in CONFIGURATIONS:
                if (document["id"], window, overlap) in completed:
                    continue
                chunks = chunker.chunk(text, document_id=document["id"], window=window, overlap=overlap).chunks
                if not chunks:
                    raise ValueError(f"No chunks for {document['id']}")
                if any(len(scorer.tokenizer.encode(c.text)) > MAX_MODEL_TOKENS for c in chunks):
                    raise ValueError("A chunk would be truncated by the scorer")
                for chunk in chunks:
                    chunk.score = round(float(scorer.compute_score(chunk.text)), 6)
                scored = chunks
                scores = [float(c.score) for c in scored]
                if not all(math.isfinite(s) for s in scores):
                    raise ValueError("Non-finite detector score")
                votes = sum(s < THRESHOLD for s in scores)
                row = {k: document[k] for k in ("id", "label", "cohort", "topic", "length_bin", "num_tokens")}
                row.update(window=window, overlap=overlap, threshold=THRESHOLD,
                           prediction=int(2 * votes > len(scores)), score_vote=votes / len(scores),
                           score_mean=-sum(scores) / len(scores), chunk_scores=scores,
                           chunk_token_counts=[c.token_count for c in scored], text_sha256=sha256(text))
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                records.append(row)
                print(f"{len(records)}/240 {document['id']} W={window} O={overlap}", flush=True)
    if len(records) != 240:
        raise ValueError("Incomplete predictions; summaries were not written")
    write_csv(args.output / "metrics.csv", summarize(records))
    print(f"Completed fixed-configuration evaluation: {args.output / 'metrics.csv'}")


if __name__ == "__main__":
    main()
