"""Select 60 intact news articles without consulting detector scores."""

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from common import (BOILERPLATE, CODE_FILES, CONFIGURATIONS, GENERATORS, LENGTH_BINS,
                    SEED, THRESHOLD, TOPICS, OverlapIndex, canonical, file_hash,
                    length_bin, read_jsonl, reference_texts, sha256, text_normalizer,
                    write_csv, write_json)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True,
                        help="Directory containing the archived VietBinoculars datasets.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def inputs(args):
    history = args.history_root
    references = [
        history / "train_news_dataset/train_sailor_gen/train_full.json-Sailor2-8B-Chat.jsonl",
        history / "val_news_dataset/val_sailor_gen/val_full.json-Sailor2-8B-Chat.jsonl",
        history / "test_news_dataset/test_gemma_gen/test_full_cleaned.json-gemma-3-12b-it.jsonl",
        args.repo / "benchmarks/train_datasets/gemini-3.7-flash-vietnamese_news_ai_dataset_5000.jsonl",
        args.repo / "benchmarks/train_datasets/gpt-5.6-luna-vietnamese_news_ai_dataset_5000.jsonl",
    ]
    historical = [history / f"{split}_news_dataset.json" for split in ("train", "val", "test")]
    calibration = args.repo / "benchmarks/concatenated_news/inputs/calibration_ids.csv"
    tokenizer = args.repo / "benchmarks/concatenated_news/inputs/phogpt-tokenizer.json"
    ai = [args.repo / "benchmarks/datasets" / f"{name}.json" for name in GENERATORS]
    paths = [args.source, *references, *historical, calibration, tokenizer, *ai]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return references, historical, calibration, tokenizer, ai, paths


def allocate(candidates, slots, rejected):
    chosen, publishers, used, urls, titles = [], Counter(), set(), set(), set()
    ordered = sorted(candidates, key=lambda r: sha256(f"{SEED}:{r['source_id']}"))
    for generator, topic, target_length in slots:
        ranked = sorted(ordered, key=lambda r: abs(math.log(r["num_tokens"] / target_length)))
        for row in ranked:
            if row["source_id"] in rejected or row["source_id"] in used:
                continue
            if row["topic"] != topic:
                continue
            if publishers[row["publisher"]] >= 10:
                continue
            if row["url"] in urls or canonical(row["title"]) in titles:
                continue
            chosen.append(dict(row, cohort=generator))
            used.add(row["source_id"])
            urls.add(row["url"])
            titles.add(canonical(row["title"]))
            publishers[row["publisher"]] += 1
            break
        else:
            raise ValueError(f"Insufficient eligible articles for {(generator, topic, target_length)}")
    return chosen


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output must be a new directory: {args.output}")
    references, historical, calibration, token_path, ai_paths, paths = inputs(args)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(token_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    clean = text_normalizer(args.repo)

    def length(text):
        return len(tokenizer.encode(clean(text), add_special_tokens=False).ids)

    with calibration.open(encoding="utf-8") as stream:
        excluded_ids = {r["y_id"] for r in csv.DictReader(stream)}
    for path in references:
        excluded_ids.update(str(r["id"]) for r in read_jsonl(path) if "id" in r)
    numeric_exclusions = set()
    for path in historical:
        for r in read_jsonl(path):
            if sha256(r["content"].strip())[:16] in excluded_ids:
                numeric_exclusions.add(str(r["id"]))
    print(f"Historical/calibration exclusions mapped to {len(numeric_exclusions)} original article IDs", flush=True)

    ai_documents, slots = [], []
    for generator, path in zip(GENERATORS, ai_paths):
        records = json.loads(path.read_text(encoding="utf-8"))
        if len(records) != 20 or Counter(r["topic"] for r in records) != Counter({t: 4 for t in TOPICS}):
            raise ValueError(f"Unexpected AI topic counts in {path}")
        for r in records:
            n = length(r["text"])
            bin_name = length_bin(n)
            ai_documents.append(dict(id=f"{generator}/{r['id']}", original_id=r["id"],
                                     label=1, is_ai_generated=True, model=r["model"],
                                     cohort=generator, topic=r["topic"],
                                     sub_topic=r.get("sub_topic", ""), text=r["text"],
                                     num_tokens=n, length_bin=bin_name))
            slots.append((generator, r["topic"], n))
    slots.sort(key=lambda s: (s[1], s[2], s[0]))

    lookup = {canonical(v): k for k, values in TOPICS.items() for v in values}
    source = json.loads(args.source.read_text(encoding="utf-8"))
    counts, candidates = Counter(), []
    for index, r in enumerate(source):
        counts["input_articles"] += 1
        topic = lookup.get(canonical(str(r.get("topic", ""))))
        reason = None
        text = r.get("content", "")
        if not topic:
            reason = "unmapped_topic"
        elif str(r["id"]) in numeric_exclusions:
            reason = "historical_or_calibration_article"
        elif not all(r.get(k) for k in ("content", "title", "source", "url")):
            reason = "missing_provenance_or_text"
        elif re.search(r"lịch thi đấu|tử vi|kết quả xổ số|soi kèo", r["title"], re.IGNORECASE):
            reason = "non_article_listing"
        elif BOILERPLATE.search(text):
            reason = "page_boilerplate_marker"
        if reason:
            counts[reason] += 1
            continue
        n = length(text)
        if not 2048 <= n < 8192:
            counts["outside_length_range"] += 1
            continue
        candidates.append(dict(source_index=index, source_id=str(r["id"]), topic=topic,
                               original_topic=r["topic"], publisher=r["source"], url=r["url"],
                               title=r["title"], author=r.get("author", ""),
                               crawled_at=r.get("crawled_at", ""), num_tokens=n,
                               length_bin=length_bin(n), text=text))
    counts["eligible_before_text_overlap_checks"] = len(candidates)
    del source
    print(f"Length/topic candidates: {dict(Counter(r['topic'] for r in candidates))}", flush=True)

    rejected, overlap_log = set(), []
    index = OverlapIndex([clean(r["text"]) for r in candidates])

    def reject_matches(text, reference, identifier, field):
        for i, jaccard, containment in index.matches(clean(text)):
            row = candidates[i]
            rejected.add(row["source_id"])
            overlap_log.append(dict(source_id=row["source_id"], reference=reference,
                                    reference_id=str(identifier), field=field,
                                    jaccard=jaccard, containment=containment))

    for row in ai_documents:
        reject_matches(row["text"], "section_3_3_ai", row["id"], "text")
    for path in references:
        print(f"Candidate overlap audit: {path.name}", flush=True)
        for r in read_jsonl(path):
            for field, text in reference_texts(r):
                reject_matches(text, path.name, r.get("id", ""), field)
    rounds = 0
    while True:
        rounds += 1
        chosen = allocate(candidates, slots, rejected)
        selected_index = OverlapIndex([clean(r["text"]) for r in chosen])
        new_rejections = set()
        for i, row in enumerate(chosen):
            for j, jaccard, containment in selected_index.matches(clean(row["text"])):
                if j <= i:
                    continue
                other = chosen[j]
                new_rejections.add(other["source_id"])
                overlap_log.append(dict(source_id=other["source_id"], reference="selected_human",
                                        reference_id=row["source_id"], field="text",
                                        jaccard=jaccard, containment=containment))
        if not new_rejections:
            break
        rejected.update(new_rejections)
    print(f"Excluded {len(rejected)} candidates by lexical overlap checks", flush=True)

    human_documents, manifest = [], []
    for row in chosen:
        identifier = f"human-news-{row['source_id']}"
        human_documents.append(dict(id=identifier, label=0, is_ai_generated=False,
                                    model="Human", cohort=row["cohort"], topic=row["topic"],
                                    text=row["text"], num_tokens=row["num_tokens"],
                                    length_bin=row["length_bin"]))
        manifest.append(dict(id=identifier, label=0, cohort=row["cohort"], topic=row["topic"],
                             length_bin=row["length_bin"], num_tokens=row["num_tokens"],
                             source_id=row["source_id"], source_index=row["source_index"],
                             original_topic=row["original_topic"], publisher=row["publisher"],
                             title=row["title"], url=row["url"], author=row["author"],
                             crawled_at=row["crawled_at"], text_sha256=sha256(row["text"]),
                             preprocessed_sha256=sha256(clean(row["text"]))))
    documents = sorted(human_documents + ai_documents, key=lambda r: r["id"])
    if len({r["id"] for r in documents}) != 120:
        raise ValueError("Expected 120 unique documents")
    for cohort in GENERATORS:
        for label in (0, 1):
            subset = [r for r in documents if r["cohort"] == cohort and r["label"] == label]
            if len(subset) != 20:
                raise ValueError("Incorrect cohort balance")
    args.output.mkdir(parents=True)
    write_json(args.output / "evaluation.json", documents)
    write_csv(args.output / "human_manifest.csv", manifest)
    protocol = dict(
        name="Human-News-60-v1", seed=SEED, human_count=60, ai_count=60,
        selection="Nearest log token length within topic, with hash-seeded tie breaking and distinct Human articles across generator cohorts",
        topic_mapping=TOPICS, length_bins=LENGTH_BINS, maximum_articles_per_publisher=10,
        human_text="Original content field, without concatenation or truncation",
        preprocessing="Repository text normalizer methods, without invoking OCR",
        ai_prompt_provenance="Author confirmed on 2026-11-08 that AI prompts used topics only, without Human source articles. Original prompts were not inspected.",
        evaluation="Supplementary unpaired article-disjoint evaluation within the existing news corpus",
        threshold=THRESHOLD, configurations=CONFIGURATIONS,
        decision="AI if strictly more than half of chunks have score < threshold; ties are Human",
        auroc="AI-oriented vote fraction and negative unweighted mean chunk score",
        model_training="No retraining, threshold recalibration or configuration search",
        source_independence="Article exclusions and lexical checks do not establish independence from the historical corpus, publishers, semantic content or pretrained-model exposure.",
        inference_status="Not run", numerical_precision="Use the archived benchmark precision; record any change separately",
        inputs=[dict(name=str(p.relative_to(args.repo)) if p.is_relative_to(args.repo) else p.name,
                     sha256=file_hash(p)) for p in paths],
        repository_code={name: file_hash(args.repo / name) for name in CODE_FILES},
        tokenizer_sha256=file_hash(token_path),
        evaluation_sha256=file_hash(args.output / "evaluation.json"),
        human_manifest_sha256=file_hash(args.output / "human_manifest.csv"),
    )
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "audit.json", dict(
        counts=dict(counts), excluded_historical_article_ids=len(numeric_exclusions),
        candidate_topic_counts=dict(Counter(r["topic"] for r in candidates)),
        overlap_rule="Five-word Jaccard >= 0.8 OR intersection/min(set sizes) >= 0.8, with at least 25 distinct shingles in the shorter text",
        reference_audit_passes=1, selection_rounds=rounds, overlap_exclusions=overlap_log,
        final_overlap_matches=0, human_publishers=dict(Counter(r["publisher"] for r in chosen)),
        human_topic_counts=dict(Counter(r["topic"] for r in chosen)),
        human_length_range=[min(r["num_tokens"] for r in chosen), max(r["num_tokens"] for r in chosen)],
        limitations=["The source corpus was previously used in VietBinoculars.",
                     "Topic matching uses original category metadata, not AI subtopic annotations.",
                     "Human lengths are not exactly matched to AI lengths; intact articles are retained and the common length range is reported separately.",
                     "Lexical overlap checks do not establish semantic independence or verified human authorship.",
                     "AI prompt provenance is an author declaration.",
                     "Reference coverage is limited to the archived inputs listed in protocol.json."]))
    print(f"Saved 60 Human and 60 AI documents to {args.output}", flush=True)


if __name__ == "__main__":
    main()
