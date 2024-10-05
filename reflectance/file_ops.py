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
MODELS_DIR_FP = RESOURCES_DIR_FP / "models"
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
    processing: dict
    fitting: dict
    simulation: dict
    aop_group_num: int
    nir_wavelengths: tuple[float]
    sensor_range: tuple[float]
    endmember_source: str
    endmember_dimensionality_reduction: str
    endmember_normalisation: str | bool
    endmember_class_schema: int
    spectra_normalisation: str | bool
    objective_fn: str
    bb_bounds: tuple[float]
    Kd_bounds: tuple[float]
    H_bounds: tuple[float]
    solver: str
    tol: float
    """TODO: currently having cake and eating it here, since duplicating attributes from the nested dictionaries.
    If I unpacked the dictionaries in the RunOpt I abstract away from the dataclass definition
    """

    def __init__(self, conf: dict):
        self.processing = conf["processing"]
        self.fitting = conf["fitting"]
        self.simulation = conf["simulation"]
        # data processing
        self.aop_group_num = self.processing["aop_group_num"]
        self.nir_wavelengths = self.processing["nir_wavelengths"]
        self.sensor_range = self.processing["sensor_range"]
        self.endmember_source = self.processing["endmember_source"]
        self.endmember_dimensionality_reduction = self.processing[
            "endmember_dimensionality_reduction"
        ]
        self.endmember_normalisation = self.processing["endmember_normalisation"]
        self.endmember_class_schema = self.processing["endmember_class_schema"]
        self.spectra_normalisation = self.processing["spectra_normalisation"]
        # fitting
        self.objective_fn = self.fitting["objective_fn"]
        self.Rb_init = self.fitting["Rb_init"]
        self.bb_bounds = self.fitting["bb_bounds"]
        self.Kd_bounds = self.fitting["Kd_bounds"]
        self.H_bounds = self.fitting["H_bounds"]
        self.endmember_bounds = self.fitting["endmember_bounds"]
        self.solver = self.fitting["solver"]
        self.tol = self.fitting["tol"]

    def get_dict_values(self):
        """Return dictionary containing only keys whose values are dictionaries."""
        return {k: v for k, v in self.__dict__.items() if isinstance(v, dict)}

    def get_config_summaries(self):
        """Return dictionaries to unpack for config (these to be processed for results summary)"""
        return unpack_nested_dict({k: v for k, v in self.get_dict_values().items()})


def unpack_nested_dict(d, parent_key=()):
    items = []
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            items.extend(unpack_nested_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def skip_textfile_rows(fp, start_str):
    with open(fp, "r") as f:
        start_found = False
        skiprows = 0
        while not start_found:
            line = f.readline()
            if line.startswith(start_str):
                start_found = True
            else:
                skiprows += 1
    return skiprows


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


def resolve_path(fp, base_dir=BASE_DIR_FP):
    return (base_dir / Path(fp)).resolve()


def resolve_paths(config: dict, base_dir: Path):
    """
    Resolve relative paths in the configuration dictionary to absolute paths.
    """
    for key, value in config.items():
        if isinstance(value, str) and (
            value.startswith("data/") or value.startswith("reflectance/")
        ):
            value = resolve_path(value, base_dir)
        config[key] = value
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


def process_grd_dataset(
    ds: xa.Dataset, src_crs: str = "EPSG:XXXXX", dst_crs: str = "EPSG:4326"
) -> xa.Dataset:
    x_range = ds.x_range.values
    y_range = ds.y_range.values
    spacing = ds.spacing.values

    x_coords = np.arange(min(x_range), max(x_range) + spacing[0], spacing[0])
    y_coords = np.arange(min(y_range), max(y_range) + spacing[1], spacing[1])[::-1]

    # reproject x, y coordinates to lat/lon (EPSG:4326)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    lon_coords, lat_coords = np.meshgrid(x_coords, y_coords)
    lon_coords, lat_coords = transformer.transform(lon_coords, lat_coords)

    # reshape z data to match new (lat, lon) grid
    z_data = ds["z"].values.reshape(len(y_coords), len(x_coords))

    new_ds = xa.Dataset(
        {"lidar_depth": (["lat", "lon"], z_data)},
        coords={
            "lon": (["lon"], lon_coords[0, :]),  # Longitude grid
            "lat": (["lat"], lat_coords[:, 0]),  # Latitude grid
        },
        attrs=ds.attrs,  # Keep original attributes
    )
    new_ds.where(new_ds.apply(np.isfinite)).fillna(np.nan)
    # rioxarray formatting
    new_ds.rio.write_crs(dst_crs, inplace=True)
    new_ds.rio.set_spatial_dims("lon", "lat", inplace=True)
    # replace inf with nan
    return new_ds


def load_grd_to_xarray(grd_file: str) -> xa.Dataset:
    # Load the .grd file directly with xarray (assuming NetCDF format)
    ds = xa.open_dataset(grd_file)
    return ds
