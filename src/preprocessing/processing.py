from src.ingestion.loader import load_dataset_hf
from src.preprocessing.normalization import normalize


def process_merged_dataset(config):
    datasets = {}
    for src in config["merge_sources"]:
        data = load_dataset_hf(src)
        keep = src["keep_columns"]
        datasets[src["name"]] = {
            row["user_id"]: {k: row[k] for k in keep}
            for row in data
        }

    # Merge on key
    user_ids = set.intersection(*[set(d.keys()) for d in datasets.values()])
    merged = []
    for user_id in user_ids:
        merged_row = {}
        for src in config["merge_sources"]:
            merged_row[src["name"]] = datasets[src["name"]][user_id]
        merged_row["user_id"] = user_id
        normalized = normalize(merged_row, config)
        if normalized:
            merged.append(normalized)
    return merged
