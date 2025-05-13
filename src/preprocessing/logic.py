def get_nested(d, path):
    if "." not in path:
        return d.get(path)
        
    source, field = path.split(".", 1)
    if source not in d:
        return None
        
    source_data = d[source]
    if not isinstance(source_data, dict):
        return None
        
    return source_data.get(field)

def resolve_merged(field, example):
    """Resolve field for merged datasets that use dotted notation (e.g. survey.field)"""
    if isinstance(field, str):
        if field.startswith("merge("):
            fields = field[6:-1].split(",")
            return " ".join(str(get_nested(example, f.strip())) for f in fields if get_nested(example, f.strip()))
        elif field.startswith("merge_metadata("):
            keys = field[15:-1].split(",")
            return {k.strip(): get_nested(example, k.strip()) for k in keys}
        elif "." in field:  # Try to resolve as dotted path
            return get_nested(example, field)
        else:
            return field  # Return literal string value for non-dotted paths
    elif isinstance(field, dict):
        if "input" in field and "output" in field:
            # Handle behavior input-output pairs
            input_val = resolve_merged(field["input"], example)
            output_val = resolve_merged(field["output"], example)
            if input_val and output_val:
                return {"input": input_val, "output": output_val}
            return None
        return {k: resolve_merged(v, example) for k, v in field.items()}
    elif isinstance(field, list):
        resolved = [resolve_merged(f, example) for f in field]
        return [r for r in resolved if r is not None]  # Filter out None values
    return field

def resolve_single(field, example):
    """Resolve field for single datasets that use direct field access"""
    if isinstance(field, str):
        if isinstance(example, dict):
            return example.get(field, field)
        return field
    elif isinstance(field, dict):
        if "item" in field and "score" in field:
            # Handle scored preferences
            return {
                "item": field["item"],
                "score": example.get(field["score"], 0)
            }
        elif "input" in field and "output" in field:
            # Handle behavior input-output pairs
            input_val = resolve_single(field["input"], example)
            output_val = resolve_single(field["output"], example)
            if input_val and output_val:
                return {"input": input_val, "output": output_val}
            return None
        return {k: resolve_single(v, example) for k, v in field.items()}
    elif isinstance(field, list):
        resolved = [resolve_single(f, example) for f in field]
        return [r for r in resolved if r is not None]  # Filter out None values
    return field

def resolve(field, example, is_merged=False):
    """Main resolve function that delegates to appropriate resolver"""
    if is_merged:
        return resolve_merged(field, example)
    else:
        return resolve_single(field, example)
