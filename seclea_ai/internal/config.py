import re
from typing import Dict, List

import yaml


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def human_2_numeric_bytes(value: str) -> int:
    units = {"k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12, "p": 1e15, "e": 1e18, "z": 1e21, "y": 1e24}
    match = re.search(r"([0-9]+)([A-Za-z]+)", value)
    if match is None:
        raise ValueError(
            "Byte values must be specified in the format <n><unit> where n is the numerical value and "
            "unit is a unit from [K, M, G, T, P, E, Z, Y, KB, MB, GB, TB, PB, EB, ZB, YB] ignoring case."
        )
    value, unit = match.groups()
    return int(value) * units[unit.lower().rstrip("b")]


def convert_human_readable_values(config_dict: Dict, keys: List) -> Dict:
    """
    Convert certain keys in the config dict from human readable to numeric.
    :param config_dict:
    :param keys:
    :return:
    """
    new_config = {**config_dict}
    for key in keys:
        if config_dict.get(key) is not None:
            new_config[key] = human_2_numeric_bytes(config_dict[key])
    return new_config


def read_config(file_path) -> Dict:
    """
    Reads a specified config file and returns the config dict.
    Processes certain keys:
        max_storage_space: converts human readable byte values into integer values.
    :param file_path:
    :return:
    """
    try:
        # TODO ensure that the spec always returns a dict, not a list - adjust otherwise.
        config = read_yaml(file_path)
    except FileNotFoundError:
        # no file so return an empty config dict.
        return dict()
    else:
        convert_keys = ["max_storage_size"]
        config = convert_human_readable_values(config_dict=config, keys=convert_keys)
        return config
