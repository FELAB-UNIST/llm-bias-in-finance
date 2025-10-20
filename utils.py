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
    
    # Check if we have at least 4 parts and the 4th part contains digits
    if len(parts) >= 4 and any(char.isdigit() for char in parts[3]):
        # Take up to 4 parts if the 4th part has numbers
        prefix = "-".join(parts[:4])
    else:
        # Otherwise take up to 3 parts
        prefix = "-".join(parts[:3])
    
    return prefix