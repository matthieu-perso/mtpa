from src.ingestion.loader import load_dataset_hf
from src.preprocessing.normalization import normalize


from datasets import load_dataset
from src.preprocessing.normalization import normalize

def process_merged_dataset(config):
    datasets = {}

    for src_cfg in config["merge_sources"]:
        # ✅ Pass source-specific config to loader
        data = load_dataset_hf(src_cfg)

        # ✅ Filter only relevant columns
        keep = src_cfg["keep_columns"]
        filtered = data.map(lambda x: {k: x.get(k) for k in keep})

        # ✅ Index by merge_key (e.g., user_id)
        datasets[src_cfg["name"]] = {
            row[config["merge_key"]]: row for row in filtered
        }

    # ✅ Intersect keys (only users present in both)
    merge_key = config["merge_key"]
    shared_keys = set.intersection(*(set(d.keys()) for d in datasets.values()))

    final = []
    for user_id in shared_keys:
        merged_row = {src_name: datasets[src_name][user_id] for src_name in datasets}
        merged_row[merge_key] = user_id  # include user_id at top level
        try:
            normalized = normalize(merged_row, config)
            if normalized:
                final.append(normalized)
        except Exception as e:
            print(f"Skipping user_id={user_id} due to error: {e}")
    return final
