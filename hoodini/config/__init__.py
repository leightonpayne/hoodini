from importlib.resources import files
import tomli

def load_default_config() -> dict:
    defaults_path = files("hoodini.config").joinpath("defaults.toml")
    with open(defaults_path, "rb") as f:
        grouped = tomli.load(f)

    flat = {}
    for section, values in grouped.items():
        if isinstance(values, dict):
            flat.update(values)   # merge keys inside section
        else:
            flat[section] = values  # handle top-level keys if any
    return flat
