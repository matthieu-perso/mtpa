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

def format_template(template, example):
    """Format a template string with values from example."""
    result = template
    # Find all fields in {field} format
    import re
    fields = re.findall(r'{([^}]+)}', template)
    for field in fields:
        value = get_nested(example, field)
        if value is not None:
            result = result.replace('{' + field + '}', str(value))
    return result

def process_profile_behavior(behavior_config, profile_list):
    """Process profile behavior based on config."""
    results = []
    
    # Handle template and input/output field cases
    for profile in profile_list:
        if "template" in behavior_config:
            # Handle product review case with template
            input_str = format_template(behavior_config["template"], {"profile": profile})
            output_str = get_nested({"profile": profile}, behavior_config["output_field"])
            if output_str:
                results.append({"input": input_str, "output": output_str})
        elif "input_field" in behavior_config and "output_field" in behavior_config:
            # Handle simple input/output field case
            input_str = get_nested({"profile": profile}, behavior_config["input_field"])
            output_str = get_nested({"profile": profile}, behavior_config["output_field"])
            if input_str and output_str:
                results.append({"input": input_str, "output": output_str})
        elif "field" in behavior_config:
            # Handle single field case
            value = get_nested({"profile": profile}, behavior_config["field"])
            if value:
                results.append(value)
    
    return results

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
        if "template" in field or ("input_field" in field and "output_field" in field):
            # Handle profile behavior
            profile_list = example.get("profile", [])
            return process_profile_behavior(field, profile_list)
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
        elif "template" in field or ("input_field" in field and "output_field" in field) or "field" in field:
            # Handle profile behavior
            profile_list = example.get("profile", [])
            return process_profile_behavior(field, profile_list)
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
