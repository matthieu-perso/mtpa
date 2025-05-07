# General Personalized Multitask Evaluation (GPME)

This repository accompanies the paper submitted to NeurIPS 2025, focusing on a unified benchmark for evaluating large language models (LLMs) across multiple tasks. The benchmark is designed to be modular, extensible, and maintainable, facilitating a comprehensive evaluation of LLMs on tasks that were previously considered separately.

## Project Structure

```
my-benchmark/
├── datasets/                  # Raw and processed data
├── src/                      # Core transformation and processing code
├── configs/                  # YAML or JSON config files per dataset
├── unified_schema/           # Definition of the unified data format
├── examples/                 # Sample usage scripts or notebooks
├── scripts/                  # CLI tools for automation
├── evaluation/              # Evaluation pipeline and metrics
│   ├── models/              # Model wrappers (HuggingFace, OpenAI, etc.)
│   ├── metrics.py           # Evaluation metrics
│   ├── evaluate.py          # Evaluation orchestrator
│   └── results/             # Saved evaluation results
└── tests/                   # Unit tests
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd my-benchmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Processing

1. Place your raw datasets in the `datasets/` directory
2. Configure your dataset processing in `configs/`
3. Run the pipeline:
```bash
python scripts/run_all.py
```

### Model Evaluation

The evaluation pipeline supports both open-source models (HuggingFace) and API-based models (OpenAI). To evaluate models on your datasets:

1. Register models in the evaluation pipeline:
```python
from evaluation.models.model_registry import registry

# Register a HuggingFace model
registry.register_model(
    "my-model",
    "huggingface",
    {
        "task_type": "text-classification",
        "device": "cuda"
    }
)

# Register an OpenAI model
registry.register_model(
    "gpt-3.5-turbo",
    "openai",
    {
        "task_type": "text-classification",
        "temperature": 0
    }
)
```

2. Run evaluation:
```bash
python scripts/eval_all.py \
    --dataset datasets/processed/dataset_a.json \
    --models distilbert-base-uncased gpt-3.5-turbo \
    --task-type text-classification \
    --detailed \
    --output results/evaluation
```

The evaluation pipeline provides:
- Support for multiple model types (HuggingFace, OpenAI)
- Standard metrics (accuracy, F1, precision, recall)
- Detailed per-class metrics
- Human-readable reports
- JSON result storage

## Development

- Add new datasets by creating corresponding config files in `configs/`
- Implement dataset-specific loaders in `src/ingestion/`
- Add transformation logic in `src/transformation/`
- Add new models by implementing the `BaseModel` interface
- Write tests for new components in `tests/`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 