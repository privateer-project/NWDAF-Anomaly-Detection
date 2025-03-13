from typing import Any, Dict

def set_config(config_class: Any, kwargs: Dict[str, Any]) -> Any:
    """Filter kwargs to only include valid parameters for the given config class."""
    return config_class(**{k: v for k, v in kwargs.items() if k in config_class.__annotations__.keys()})
