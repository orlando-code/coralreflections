# general
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt

# stats
import scipy.stats as stats
from sklearn.metrics import r2_score

# ML
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

# custom
from reflectance import spectrum_utils, file_ops


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


class MLDataPipe:

    def __init__(
        self,
        endmember_class_schema: str = "three_endmember",
        gcfg: dict = file_ops.read_yaml(file_ops.CONFIG_DIR_FP / "glob_cfg.yaml"),
        train_ratio: float = 0.8,
        normalise: bool = True,
        scaler_type: str = "minmax",
    ):
        self.endmember_class_schema = endmember_class_schema
        self.train_ratio = train_ratio
        self.normalise = normalise
        self.scaler_type = scaler_type
        self.gcfg = gcfg

    def load_prism_spectra(self):
        raw_spectra = spectrum_utils.load_spectra()
        self.spectra = spectrum_utils.preprocess_prism_spectra(
            raw_spectra, spectrum_utils.NIR_WAVELENGTHS, spectrum_utils.SENSOR_RANGE
        )

    def normalise_data(self):
        scaler = spectrum_utils.instantiate_scaler(self.scaler_type)
        self.X_train = pd.DataFrame(
            scaler.fit_transform(self.X_train), columns=self.X_train.columns
        )
        self.X_test = pd.DataFrame(
            scaler.transform(self.X_test), columns=self.X_test.columns
        )

        self.y_train = pd.DataFrame(
            scaler.fit_transform(self.y_train), columns=self.y_train.columns
        )
        self.y_test = pd.DataFrame(
            scaler.transform(self.y_test), columns=self.y_test.columns
        )

    def load_validation_data(self):
        validation_data = pd.read_csv(
            file_ops.DATA_DIR_FP / "CORAL_validation_data.csv"
        )
        # process to correct class
        self.validation_data = spectrum_utils.map_validation(
            validation_data, self.gcfg["endmember_map"]
        )

    def characterise_endmembers(self):
        endmember_schema_map = self.gcfg["endmember_schema"][
            self.endmember_class_schema
        ]

        grouped_val_data = pd.DataFrame()
        # group validation data by endmember categories in endmember_schema_map
        for (
            endmember_dimensionality_reduction,
            validation_fields,
        ) in endmember_schema_map.items():
            # fill in validation data with sum of all fields in the category
            grouped_val_data[endmember_dimensionality_reduction] = self.validation_data[
                validation_fields
            ].sum(axis=1)
        self.labels = grouped_val_data

    def do_train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.spectra, self.labels, test_size=1 - self.train_ratio
        )

    def generate_data(self):
        self.load_prism_spectra()
        self.load_validation_data()
        self.characterise_endmembers()
        self.do_train_test_split()
        self.normalise_data()

        return (self.X_train, self.X_test), (self.y_train, self.y_test), self.labels


class sklModels:
    def __init__(
        self,
        model_type: str = "random_forest",
        n_iter_search: int = 50,
        n_jobs: int = -5,
        n_report: int = 3,
    ):
        self.model_type = model_type
        self.n_iter_search = n_iter_search
        self.n_jobs = n_jobs
        self.n_report = n_report

    def instantiate_grid_search(self):
        match self.model_type:
            case "random_forest":
                self.grid = {
                    "bootstrap": [True, False],
                    "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
                    "max_features": [None, "log2", "sqrt"],
                    "min_samples_leaf": [1, 2, 4],
                    "min_samples_split": [2, 5, 10],
                    "n_estimators": [130, 180, 230],
                }
            case "gradient_boosting":
                self.grid = {
                    "estimator__max_depth": [
                        10,
                        20,
                        30,
                        40,
                        50,
                        60,
                        70,
                        80,
                        90,
                        100,
                        110,
                        None,
                    ],
                    "estimator__max_features": [None, "log2", "sqrt"],
                    "estimator__min_samples_leaf": [1, 2, 4],
                    "estimator__min_samples_split": [2, 5, 10],
                }
            case "mlp":
                self.grid = {
                    "hidden_layer_sizes": [(50, 50, 50), (50, 100, 50), (100,)],
                    "activation": ["tanh"],
                }
            case _:
                raise ValueError("Model type not recognised")

    def instantiate_model(self):
        match self.model_type:
            case "random_forest":
                self.model = RandomForestRegressor()
            case "gradient_boosting":
                self.model = MultiOutputRegressor(GradientBoostingRegressor())
            case "mlp":
                self.model = MLPRegressor()
            case _:
                raise ValueError("Model type not recognised")

    def parameter_search(self, X_train, y_train):
        self.instantiate_grid_search()
        self.instantiate_model()

        random_search = RandomizedSearchCV(
            self.model,
            param_distributions=self.grid,
            n_iter=self.n_iter_search,
            n_jobs=self.n_jobs,
        )

        start = time()
        random_search.fit(X_train, y_train)
        print(
            f"RandomizedSearchCV took {(time() - start):.2f} seconds for {self.n_iter_search} candidates parameter settings."
        )
        report(random_search.cv_results_, n_top=self.n_report)
        self.random_search = random_search

    def return_fitted_model(self, X_train, y_train):
        self.parameter_search(X_train, y_train)
        return self.random_search.best_estimator_
