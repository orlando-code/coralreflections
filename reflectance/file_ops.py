# general
import yaml
from pathlib import Path
from itertools import product
from dataclasses import dataclass

# profiling
import cProfile
import pstats
import io

# define immutable path structure
BASE_DIR_FP = Path(__file__).resolve().parent.parent
MODULE_DIR_FP = BASE_DIR_FP / "reflectance"
RESOURCES_DIR_FP = MODULE_DIR_FP / "resources"
DATA_DIR_FP = BASE_DIR_FP / "data"
RESULTS_DIR_FP = BASE_DIR_FP / "results"
TMP_DIR_FP = BASE_DIR_FP / "tmp"
CONFIG_DIR_FP = BASE_DIR_FP / "configs"


@dataclass
class GlobalOptPipeConfig:
    spectra_source: str
    spectra_fp: str
    spectral_library_fp: str
    validation_data_fp: str
    save_fits: bool
    endmember_map: dict
    endmember_schema: dict

    def __init__(self, conf: dict):
        self.spectra_source = conf["spectra_source"]
        self.spectra_fp = conf["spectra_fp"]
        self.spectral_library_fp = conf["spectral_library_fp"]
        self.validation_data_fp = conf["validation_data_fp"]
        self.save_fits = conf["save_fits"]
        self.endmember_map = conf["endmember_map"]
        self.endmember_schema = conf["endmember_schema"]


@dataclass
class RunOptPipeConfig:
    aop_group_num: int
    nir_wavelengths: tuple[float]
    sensor_range: tuple[float]
    endmember_type: str
    endmember_normalisation: str | bool
    endmember_class_schema: int
    spectra_normalisation: str | bool
    objective_fn: str
    bb_bounds: tuple[float]
    Kd_bounds: tuple[float]
    H_bounds: tuple[float]
    simulation: dict
    solver: str
    tol: float

    def __init__(self, conf: dict):
        self.aop_group_num = conf["processing"]["aop_group_num"]
        self.nir_wavelengths = conf["processing"]["nir_wavelengths"]
        self.sensor_range = conf["processing"]["sensor_range"]
        self.endmember_type = conf["processing"]["endmember_type"]
        self.endmember_normalisation = conf["processing"]["endmember_normalisation"]
        self.endmember_class_schema = conf["processing"]["endmember_class_schema"]
        self.spectra_normalisation = conf["processing"]["spectra_normalisation"]
        self.objective_fn = conf["fitting"]["objective_fn"]
        self.bb_bounds = conf["fitting"]["bb_bounds"]
        self.Kd_bounds = conf["fitting"]["Kd_bounds"]
        self.H_bounds = conf["fitting"]["H_bounds"]
        self.simulation = conf["simulation"]
        self.solver = conf["fitting"]["solver"]
        self.tol = conf["fitting"]["tol"]


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
    glob_cfg = GlobalOptPipeConfig(resolve_paths(glob_cfg, BASE_DIR_FP))
    # select run configuration
    run_cfgs = RunOptPipeConfig(run_cfgs[run_ind])

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


def profile_step(step_name, step_method):
    profiler = cProfile.Profile()
    profiler.enable()
    step_method()
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
    ps.print_stats(10)  # Print top 10 slowest functions
    print(f"Profile stats for {step_name}:\n{s.getvalue()}")
