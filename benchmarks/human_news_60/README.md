# Human-News-60 supplementary benchmark

This exploratory evaluation combines 60 naturally authored Vietnamese news articles with the 60 AI documents used in Section 3.3. Human articles are retained without concatenation or truncation. Each AI source dataset contributes 20 documents and is evaluated with a distinct subset of 20 Human articles. Each class contains 12 documents per news topic. Labels are 0 for Human and 1 for AI.

## Protocol and scope

The configurations `(650,150)` and `(600,0)` were selected on Gemma-3-12B-News-Concat-Dev. Both configurations and the Youden threshold `0.9279661178588868` remain fixed. All 120 documents were scored again in one environment. Majority voting predicts AI when more than half of the chunk scores are below the threshold. Ties are classified as Human. Majority AUROC uses vote fractions, and mean-score AUROC uses negative mean chunk scores.

Human selection uses topic and token-length criteria without detector scores, with seed 20261108 for tie breaking and a maximum of ten articles per publisher label. The manifest contains 16 publisher labels. Source identifiers and five-word overlap checks exclude matches against the available historical and calibration records, the AI documents and other selected Human articles. See `build_dataset.py`, `dataset/protocol.json` and `dataset/audit.json` for inputs and exclusions.

The authors confirm that AI generation used topic-only prompts without Human article content. The original prompts were not independently audited. Human articles come from the news corpus previously used in VietBinoculars. This evaluation therefore removes paired Human source prompts but does not establish corpus-level independence or exclude pretraining exposure. Small subsets and unequal length distributions limit generalization. The five topics belong to the news domain.

## Archived results

Results use bfloat16 on two Tesla T4 GPUs. Counts below refer to each two-class subset. Each document contributes once to the pooled result per configuration.

| Configuration | AI source dataset | Human | AI | Majority macro-F1 | Majority AUROC | Mean-score AUROC | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| (650,150) | Claude-Sonnet-4.6-News | 20 | 20 | 0.9750 | 0.9700 | 0.9950 | 1 | 0 |
| (650,150) | Gemini-3.7-Flash-News | 20 | 20 | 0.7995 | 0.8450 | 0.8075 | 5 | 3 |
| (650,150) | GPT-5.6-Luna-News | 20 | 20 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| (650,150) | Pooled | 60 | 60 | 0.9250 | 0.9567 | 0.9528 | 6 | 3 |
| (600,0) | Claude-Sonnet-4.6-News | 20 | 20 | 0.9750 | 0.9975 | 0.9875 | 1 | 0 |
| (600,0) | Gemini-3.7-Flash-News | 20 | 20 | 0.8500 | 0.9213 | 0.9400 | 3 | 3 |
| (600,0) | GPT-5.6-Luna-News | 20 | 20 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| (600,0) | Pooled | 60 | 60 | 0.9417 | 0.9832 | 0.9842 | 4 | 3 |

Pooled AI recall is 95.0% with a 95% Wilson confidence interval of [86.3%, 98.3%] for both configurations. Human FPR is 10.0% [4.7%, 20.1%] for (650,150) and 6.7% [2.6%, 15.9%] for (600,0). Human FPR differences between generator subsets refer to different Human articles and cannot be attributed to the generator.

`results/metrics.csv` also reports results by news topic and PhoGPT BPE token count after preprocessing. Each grouping describes the same dataset. Macro-F1 and AUROC are omitted for categories containing only one class. The 240 records in `results/predictions.jsonl` retain document labels, chunk scores and decisions. These supplementary results do not replace the archived Section 3.3 predictions.

## Reproduce scoring

Run the commands from the repository root. Use Python 3.13 and two CUDA GPUs for the recorded configuration. The tested runtime used Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0, Transformers 4.39.3 and Accelerate 0.29.3. Each T4 had 15,360 MiB of memory. This is device capacity, not measured peak memory.

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r benchmarks/human_news_60/environment/requirements.txt
python -B benchmarks/human_news_60/run_benchmark.py --dataset benchmarks/human_news_60/dataset --repo . --validate-only
python -B benchmarks/human_news_60/run_benchmark.py --dataset benchmarks/human_news_60/dataset --repo . --precision bfloat16 --output /path/to/new-results
```

The runner downloads `vinai/PhoGPT-4B` and `vinai/PhoGPT-4B-Chat` from Hugging Face and places them on separate GPUs. Internet access and model cache space are required. Model revisions are recorded in `results/model_revisions.json`. One chunk is scored per inference call. A prior float16 attempt produced a non-finite score. Use the tested bfloat16 setting rather than substituting float16. Hardware and software changes may change individual predictions.

The recorded scoring checkout is `5eea659f3edd06b3fcc6be938ef037a206557652`. Validation checks the relevant repository files against their frozen hashes. Keep this benchmark package available if using a separate checkout of that commit. Output must be a new directory outside the repository. Use `--resume` only with the same inputs and runtime.

`environment/requirements.txt` pins supporting packages for text scoring. It is not a full application lock. The archived Kaggle environment included Gradio 6.30.0, whereas the application requires Gradio below version 5. Web UI, OCR and PDF generation were not exercised by this benchmark. The raw runtime metadata is retained in `environment/environment.json` and `results/run.json`.

## Validate and rebuild

```bash
python -B benchmarks/human_news_60/verify_results.py
python -B benchmarks/human_news_60/build_dataset.py --source /path/to/news_dataset.json --history-root /path/to/VietBinoculars/datasets --repo . --output /path/to/new-dataset
```

Rebuilding requires the original news corpus and historical inputs listed in `build_dataset.py`. They are not duplicated here. `dataset/human_manifest.csv` preserves source identifiers, URLs, metadata, token counts and text hashes. `dataset/protocol.json` records input and implementation hashes. To verify Human text against the original corpus, add `--source /path/to/news_dataset.json` to the validation command. CPU validation does not rerun model inference.
