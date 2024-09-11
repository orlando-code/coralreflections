import yaml
from pathlib import Path

DATA_DIR_FP = Path(__file__).resolve().parent.parent / 'data'
RESULTS_DIR_FP = Path(__file__).resolve().parent.parent / 'results'
RESOURCES_DIR_FP = Path(__file__).resolve().parent / 'resources'
TMP_DIR_FP = Path(__file__).resolve().parent.parent / 'tmp'
CONFIG_FP = Path(__file__).resolve().parent.parent / 'config.yaml'



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