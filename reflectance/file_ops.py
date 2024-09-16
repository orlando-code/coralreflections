# general
import yaml
from pathlib import Path
from itertools import product

# custom
from reflectance import optimisation_pipeline as optpip

# define immutable path structure
BASE_DIR_FP = Path(__file__).resolve().parent.parent
MODULE_DIR_FP = BASE_DIR_FP / "reflectance"
RESOURCES_DIR_FP = MODULE_DIR_FP / "resources"
DATA_DIR_FP = BASE_DIR_FP / "data"
RESULTS_DIR_FP = BASE_DIR_FP / "results"
TMP_DIR_FP = BASE_DIR_FP / "tmp"
CONFIG_DIR_FP = BASE_DIR_FP / "configs"


def get_dir(dir_fp: str | Path) -> Path:
    dir_fp = Path(dir_fp)
    if not dir_fp.exists():
        dir_fp.mkdir()
    return dir_fp


def get_f(fp: str | Path) -> Path:
    file_path = Path(fp).resolve()
    if not file_path.exists():
        file_path.touch()
    return file_path


def resolve_paths(config, base_dir):
    """
    Resolve relative paths in the configuration to absolute paths.
    """
    for key, value in config.items():
        if isinstance(value, str) and (
            value.startswith("data/") or value.startswith("reflectance/")
        ):
            config[key] = (base_dir / value).resolve()
    return config


def instantiate_single_configs_instance(run_ind: int = 0):
    run_cfgs = read_yaml(CONFIG_DIR_FP / "run_cfgs.yaml")
    glob_cfg = read_yaml(CONFIG_DIR_FP / "glob_cfg.yaml")
    # resolve relative paths
    glob_cfg = optpip.GlobalOptPipeConfig(resolve_paths(glob_cfg, BASE_DIR_FP))
    # select run configuration
    run_cfgs = optpip.RunOptPipeConfig(run_cfgs[run_ind])

    return glob_cfg, run_cfgs


def read_yaml(yaml_path: str | Path):
    with open(yaml_path, "r") as file:
        yaml_info = yaml.safe_load(file)
    return yaml_info


def edit_yaml(yaml_path: str | Path, info: dict):
    yaml_info = read_yaml(yaml_path)
    yaml_info.update(info)
    save_yaml(yaml_path, yaml_info)


def save_yaml(yaml_path: str | Path, info: dict):
    with open(yaml_path, "w") as file:
        yaml.dump(info, file)


def generate_config_dicts(nested_dict):
    """
    Generate a list of configuration dictionaries from a nested dictionary of lists of parameters.
    """

    def recursive_product(d):
        if isinstance(d, dict):
            keys, values = zip(*d.items())
            if all(isinstance(v, list) for v in values):
                for combination in product(*values):
                    yield dict(zip(keys, combination))
            else:
                for combination in product(
                    *[
                        recursive_product(v) if isinstance(v, dict) else [(k, v)]
                        for k, v in d.items()
                    ]
                ):
                    yield {k: v for d in combination for k, v in d.items()}
        else:
            yield d

    def combine_dicts(dicts):
        combined = {}
        for d in dicts:
            for k, v in d.items():
                if k not in combined:
                    combined[k] = v
                else:
                    if isinstance(combined[k], dict) and isinstance(v, dict):
                        combined[k] = combine_dicts([combined[k], v])
                    else:
                        combined[k] = v
        return combined

    # Generate all combinations for the nested dictionaries
    nested_combinations = []
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            nested_combinations.append([{key: v} for v in recursive_product(value)])
        else:
            nested_combinations.append([{key: value}])

    # Combine the nested combinations to preserve the structure
    final_config_dicts = []
    for combination in product(*nested_combinations):
        combined_dict = combine_dicts(combination)
        final_config_dicts.append(combined_dict)

    return final_config_dicts
