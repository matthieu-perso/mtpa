# MTPA: MultiTask Personalization Assessment (EMNLP Findings 2025)

MTPA is a comprehensive framework that takes real-world survey data (e.g. World Values Survey, European Social Survey and General Social Survey), transforms it into standardized user personas, and provides tools to evaluate how well large language models can personalize their responses across different tasks and user types.

It comprises both a dataset builder of real world personas, as well as an evaluation of persona conditioning on downstream datasets. 


## Quick start (TL;DR)

- Install:
```bash
pip install -r requirements.txt
```
- Set API keys (if running API-based evaluation):
```bash
export OPENAI_API_KEY=... GEMINI_API_KEY=... ANTHROPIC_API_KEY=... HF_API_KEY=...
```
- 1) Build dataset (from raw to merged):
```bash
python dataset/dataset_construction/prepare.py --raw-dataset-path dataset/raw || true

python dataset/dataset_construction/transform.py

python dataset/dataset_construction/merge.py
```
- 2) Evaluate models on downstream datasets:
```bash
python evaluation/benchmark.py
```





## Building the Dataset

### 1. Acquire raw data
- Place files under `dataset/raw/` (filenames can differ; update args accordingly):
  - `dataset/raw/EVS_WVS_Joint_Csv_v5_0.csv` (or the official WV7 CSV from WVSA)
  - `dataset/dataset_construction/gss_transformed_qa_format.json`

### 2. Prepare
- Script: `dataset/dataset_construction/prepare.py`
- Runs filtering, conversion and splits using:
  - `question_metadata.json`, `answer_adjustment.json`, `codebook.json`
- Command:
```bash
python dataset/dataset_construction/prepare.py \
  --raw-dataset-path dataset/raw/EVS_WVS_Joint_Csv_v5_0.csv
```
- Output: TSV of


### 3. Create respondent-level file
- Script: `dataset/dataset_construction/transform_qa_data_sample.py`
- Merges value+demographic TSVs on `D_INTERVIEW`, drops rows missing then stratifies by country/age/sex.
- Command:
```bash
python dataset/dataset_construction/transform.py
```
- Output: JSON list of respondents, each with `data` entries

### 4. Merge WVS with GSS
```bash
python dataset/dataset_construction/merge.py
```
- Output: Merger json list

### 5) Evaluate Models


## Citation
```bibtex
@inproceedings{tehenan2025mtpa,
  title = {MTPA: MultiTask Personalization Assessment},
  author = {Tehenan, Matthieu},
  booktitle = {EMNLP Findings},
  year = {2025},
  month = {November}
}
```