"""
Copied from https://github.com/cvg/SOLD2/blob/main/sold2/experiment.py
"""
import yaml
import os


def load_config(config_path):
    """ Load configurations from a given yaml file. """
    # Check file exists
    if not os.path.exists(config_path):
        raise ValueError("The provided config path is not valid.")

    # Load the configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
