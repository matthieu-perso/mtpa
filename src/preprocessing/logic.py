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

def resolve(field, example):
    if isinstance(field, str):
        if field.startswith("merge("):
            fields = field[6:-1].split(",")
            return " ".join(str(get_nested(example, f.strip())) for f in fields if get_nested(example, f.strip()))
        elif field.startswith("merge_metadata("):
            keys = field[15:-1].split(",")
            return {k.strip(): get_nested(example, k.strip()) for k in keys}
        elif "." in field:  # Only try to resolve as path if it contains a dot
            return get_nested(example, field)
        else:
            return field  # Return literal string value
    elif isinstance(field, list):
        return [resolve(f, example) for f in field]
    return field
