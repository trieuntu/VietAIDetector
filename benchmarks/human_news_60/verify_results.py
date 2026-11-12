"""Verify archived predictions and recompute all reported metric rows."""

import csv
import json
import math
from pathlib import Path

from common import CONFIGURATIONS, THRESHOLD, file_hash
from run_benchmark import summarize


def main():
    here = Path(__file__).resolve().parent
    run = json.loads((here / 'results/run.json').read_text())
    for name, key in [('dataset/evaluation.json', 'dataset_sha256'),
                      ('dataset/protocol.json', 'protocol_sha256'),
                      ('run_benchmark.py', 'runner_sha256'), ('common.py', 'common_sha256')]:
        if file_hash(here / name) != run[key]:
            raise ValueError(f'Hash mismatch: {name}')
    documents = {r['id']: r for r in json.loads((here / 'dataset/evaluation.json').read_text())}
    rows = [json.loads(line) for line in (here / 'results/predictions.jsonl').read_text().splitlines()]
    expected = {(i, w, o) for i in documents for w, o in CONFIGURATIONS}
    if len(rows) != 240 or {(r['id'], r['window'], r['overlap']) for r in rows} != expected:
        raise ValueError('Missing, duplicate or unexpected predictions')
    for r in rows:
        if any(r[k] != documents[r['id']][k] for k in
               ('label', 'topic', 'cohort', 'length_bin', 'num_tokens')):
            raise ValueError(f"Document metadata mismatch: {r['id']}")
        scores = r['chunk_scores']
        if not scores or not all(math.isfinite(x) for x in scores) or r['threshold'] != THRESHOLD:
            raise ValueError('Invalid scores or threshold')
        vote = sum(x < THRESHOLD for x in scores) / len(scores)
        if (r['prediction'] != int(vote > 0.5)
                or not math.isclose(r['score_vote'], vote, abs_tol=1e-12)
                or not math.isclose(r['score_mean'], -sum(scores) / len(scores), abs_tol=1e-12)):
            raise ValueError(f"Aggregation mismatch: {r['id']}")
    with (here / 'results/metrics.csv').open() as stream:
        reported = list(csv.DictReader(stream))
    calculated = summarize(rows)
    if len(reported) != len(calculated):
        raise ValueError('Metric row count mismatch')
    for old, new in zip(reported, calculated):
        for key, value in new.items():
            if value is None:
                valid = old[key] == ''
            elif isinstance(value, (int, float)):
                valid = math.isclose(float(old[key]), value, abs_tol=1e-12)
            else:
                valid = old[key] == value
            if not valid:
                raise ValueError(f'Metric mismatch: {key}')
    print(f'Verified {len(rows)} predictions and {len(calculated)} metric rows.')


if __name__ == '__main__':
    main()
