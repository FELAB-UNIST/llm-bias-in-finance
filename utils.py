import re
import json
from typing import Optional, Dict

def parse_json_from_text(text: str) -> Optional[Dict]:
    """Extracts a JSON object from a string."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

def get_short_model_prefix(model_id: str) -> str:
    """Creates a short, file-safe prefix from the model ID."""
    model_name_part = model_id.split('/')[-1]
    parts = model_name_part.split('-')
    # Take up to the first 3 parts of the model name
    prefix = "-".join(parts[:3])
    return prefix