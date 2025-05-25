import yaml
import os

def read_params(config_path: str = "params.yaml") -> dict:
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config
