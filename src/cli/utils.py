from typing import Any, Dict

def filter_config_kwargs(config_class: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to only include valid parameters for the given config class."""
    return {k: v for k, v in kwargs.items() if k in config_class.__annotations__.keys()}
