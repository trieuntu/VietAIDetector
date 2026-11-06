# Gemma-3-12B-News concatenated benchmark

Reproduce **Gemma-3-12B-News-Concat-Dev** and **Gemma-3-12B-News-Concat-Test**, including source allocation, exclusions and the ordered passage membership of each long-document pair.

## Reproduce

From the repository root, use Python 3.11 and the repository dependencies. The original build used `tokenizers==0.15.2` and `transformers==4.39.3`.

```bash
python -m pip install -r requirements.txt
python -m pip install tokenizers==0.15.2
python benchmarks/concatenated_news/build_benchmark.py \
  --source benchmarks/concatenated_news/inputs/source.jsonl \
  --output ../gemma-news-rebuilt
python benchmarks/concatenated_news/verify_benchmark.py \
  --benchmark ../gemma-news-rebuilt --check-reference
```

`--source` accepts a relative or absolute JSONL path. Keep the original row order to reproduce the manuscript benchmark. `--output` must name a new directory. Construction uses the repository's text preprocessing and the bundled tokenizer, runs on the CPU, and selects passages using source relationships and length criteria. It does not load detector weights. Once dependencies are installed, both commands run offline.

The output contains `data/development.json`, `data/test.json`, `audit/source_manifest.csv`, `audit/document_manifest.csv`, `audit/near_duplicate_matches.json` and `audit/report.json`. The dataset JSON files contain ordered `source_ids` and one-based `source_lines` for every document. The verifier reconstructs every document from its source passages, checks preprocessing and token counts, and validates class balance, source separation and calibration exclusions. `--check-reference` additionally requires both datasets, both manifests and the near-duplicate matches to match the archived SHA-256 values in `expected_sha256.json`.

The generated report records the local input paths. After moving inputs or a checkout, use the verifier's `--source`, `--software-root` and `--exclude-csv` options to provide their new locations. Their hashes must still match the recorded inputs. Both scripts provide `--help`.

## Construction and counts

The source contains 7,222 pairs of Human news passages and Gemma-3-12B-it continuations. Grouping by source article ID, normalized prompt, exact text or five-word Jaccard similarity of at least 0.8 produces 7,033 groups. Groups linked to threshold-calibration IDs are excluded first (141 pairs in 121 groups). The subsequent cross-label duplicate check excludes zero additional pairs. Thirteen groups (14 pairs) have no pair with at least 64 preprocessed tokens in each passage. Retaining the first eligible pair in every remaining group excludes 168 other pairs, leaving 6,899 eligible pairs.

| Stage | Development | Test | Total |
| --- | ---: | ---: | ---: |
| Passage pairs allocated with seed 20260929 | 1,380 | 5,519 | 6,899 |
| Passage pairs unused at the partition end | 2 | 1 | 3 |
| Passage pairs used in concatenation | 1,378 | 5,518 | 6,896 |
| Completed long-document pairs | 296 | 1,179 | 1,475 |
| Documents, balanced between Human and AI | 592 | 2,358 | 2,950 |

The target development fraction is 20% before concatenation. A long-document pair is completed once it contains at least three passage pairs and both documents reach 2,048 PhoGPT BPE tokens after preprocessing. Human and AI use the same sources in the same order. Completed pairs contain three to seven passage pairs in this benchmark. Each source pair contributes to one completed pair at most.

## Included inputs and manifests

- `inputs/source.jsonl` is the original `test_full_cleaned.json-gemma-3-12b-it.jsonl`, with SHA-256 `87a8636718caf17ff437e81a3e1a54023b3eaea3802cbb08880145a3edfaee8c`.
- `inputs/phogpt-tokenizer.json` is the PhoGPT-4B tokenizer used for the benchmark, with SHA-256 `e5d0981153b0fa9014832a2395bd81f4c6036e9df9ce64ee79241e53e2326130`. Its upstream source is [vinai/PhoGPT-4B](https://huggingface.co/vinai/PhoGPT-4B).
- `inputs/calibration_ids.csv` contains the sorted union of 50,991 unique `y_id` values from the original and updated calibration tables. The original table SHA-256 is `9f762d88fc31be5c891ba7b567a21a49cd1acf6680c13f6f1d23dac8e72abe8d` and the updated table SHA-256 is `ab7a806d0b125257ebdb5bbd324def6f49f288c72370c7eccc5167521770f593`. The historical build supplied two byte-identical copies of the original table. Taking the union gives the same exclusion IDs.
- `manifests/source_manifest.csv` records all 7,222 source rows. `source_line` is the unique one-based row key, `component_first_line` identifies a source group, and `status` records use or exclusion. Blank `split` and `pair_id` fields indicate exclusion before partitioning. The three `unused_split_tail_below_target` rows have partition assignments and blank final pair IDs.
- `manifests/document_manifest.csv` records the 2,950 completed documents. Each `pair_id` occurs once with `label=0` (Human) and once with `label=1` (AI).

The manifests are byte-identical to those used for the manuscript. The construction procedure is retained from the archived builder, whose SHA-256 is `16b7066aa6ebbc383b63d7f449fdb2fa17c9b47945ac95013c6578546512bf7e`. This repository version adds portable command-line inputs, output validation and checks against the reference files.

Calibration exclusions use source IDs and related groups in this corpus. The duplicate checks cover exact and lexical similarity at the stated threshold. The concatenated news benchmark uses one generator and was derived from a corpus previously evaluated in VietBinoculars. These checks establish the documented source separation within this construction procedure.
