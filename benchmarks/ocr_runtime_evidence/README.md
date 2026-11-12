# Archived OCR runtime evidence

These records document the historical OCR stage and subsequent detector evaluation for the 454-document clean-scan experiment. They are runtime metadata, not a complete dependency lock or an installation test.

Vintern-1B-v2 transcription ran on NVIDIA L4 with Python 3.12.11, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, Transformers 4.39.3 and CUDA 12.8, using bfloat16. The model revision is recorded in `ocr_environment.json`.

The later detector evaluation reused the archived OCR text unchanged. Original and OCR texts were scored on NVIDIA A100 with Python 3.10.12, PyTorch 2.14.1, Transformers 4.39.3 and CUDA 13.0, using bfloat16 and batch size 8. Versions are transcribed from archived records. No full package snapshot or Accelerate version was recorded for these stages.

`ocr_execution_summary.csv` records 2,589 pages processed sequentially in the historical OCR stage. Timing values are historical costs, not newly measured end-to-end latency. `ocr_environment_comparison.json` records matching hashes for the reused OCR environment. Its hashes use sorted JSON serialization and differ from the file-byte checksums below. These records do not establish compatibility of the Gradio interface or PDF report generation.

## Provenance

The four evidence files are unchanged copies from the archived `ocr_clean454_v4` results.

| File | Archive location | File SHA-256 |
|---|---|---|
| `ocr_environment.json` | `ocr_clean454_v4/ocr_environment.json` | `1a029dab8dc91541473d286822c42f06b787a3cf2bc0edb6199a308115dcc8ee` |
| `detector_environment.json` | `ocr_clean454_v4/detector_environment.json` | `cdcc7a6c59f7818ac6ef3e61fa170576b727ee712f30245f2cb053bd3f79630a` |
| `ocr_environment_comparison.json` | `ocr_clean454_v4/provenance/ocr_environment_comparison.json` | `e9307b0c605930c38777c6fa3708cd01091b48f6c2a08cefdbe5cb7452107cdd` |
| `ocr_execution_summary.csv` | `ocr_clean454_v4/report/ocr_execution_summary.csv` | `1a0234896b573a18b2f40192f444829f7906bcca65216adc022332f84ebd8a33` |
