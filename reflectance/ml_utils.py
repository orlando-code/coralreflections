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


class MLDataPipe:

    def __init__(
        self,
        endmember_class_schema: str = "three_endmember",
        gcfg: dict = file_ops.read_yaml(file_ops.CONFIG_DIR_FP / "glob_cfg.yaml"),
        train_ratio: float = 0.8,
        target: str = "endmember",
        normalise: bool = True,
        scaler_type: str = "minmax",
        random_seed: int = 42,
    ):
        self.endmember_class_schema = endmember_class_schema
        self.gcfg = gcfg
        self.train_ratio = train_ratio
        self.target = target.lower()
        self.normalise = normalise
        self.scaler_type = scaler_type
        self.random_seed = random_seed

    def load_prism_spectra(self):
        raw_spectra = spectrum_utils.load_spectra()
        self.spectra = spectrum_utils.preprocess_prism_spectra(
            raw_spectra, spectrum_utils.NIR_WAVELENGTHS, spectrum_utils.SENSOR_RANGE
        )

    # def load_simulation_spectra(self):

    def normalise_data(self):
        scaler = spectrum_utils.instantiate_scaler(self.scaler_type)
        self.X_train = pd.DataFrame(
            scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index,
        )
        self.X_test = pd.DataFrame(
            scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index,
        )

        self.y_train = pd.DataFrame(
            scaler.fit_transform(
                self.y_train
                if not self.target == "depth"
                else self.y_train.values.reshape(-1, 1)
            ),
            columns=self.y_train.columns if not self.target == "depth" else ["Depth"],
            index=self.y_train.index,
        )
        self.y_test = pd.DataFrame(
            scaler.fit_transform(
                self.y_test
                if not self.target == "depth"
                else self.y_test.values.reshape(-1, 1)
            ),
            columns=self.y_test.columns if not self.target == "depth" else ["Depth"],
            index=self.y_test.index,
        )

    def load_validation_data(self):
        self.validation_data = pd.read_csv(
            file_ops.DATA_DIR_FP / "CORAL_validation_data.csv"
        )

    def generate_endmember_labels(self):
        # process to correct class
        self.validation_data = spectrum_utils.map_validation(
            self.validation_data, self.gcfg["endmember_map"]
        )

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

    def generate_depth_labels(self):
        self.labels = pd.DataFrame(
            self.validation_data["Depth"],
            columns=["Depth"],
            index=self.validation_data.index,
        )

    def do_train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.spectra,
            self.labels,
            test_size=1 - self.train_ratio,
            random_state=self.random_seed,
        )

    def generate_data(self):
        match self.data_source:
            case "prism":
                self.load_prism_spectra()
            case "simulation":
                self.load_simulation_spectra()
        self.load_validation_data()
        match self.target:
            case "endmember":
                self.generate_endmember_labels()
            case "depth":
                self.generate_depth_labels()
            case _:
                raise ValueError(f"Target '{self.target}' not recognised")

        self.do_train_test_split()
        self.normalise_data()

        return (
            (self.X_train, self.X_test),
            (self.y_train, self.y_test),
            self.labels,
        )


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
                    "bootstrap": [True, False],  # comment out to allow OOB score
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
                    "hidden_layer_sizes": [
                        (10, 30, 10),
                        (20,),
                        (50, 50, 50),
                        (50, 100, 50),
                        (100,),
                        (100, 100, 100),
                    ],
                    "activation": ["identity", "logistic", "tanh", "relu"],
                    "solver": ["sgd", "adam"],
                    "alpha": [0.0001, 0.05],
                    "tol": [1e-4, 1e-9],
                    "learning_rate": ["constant", "adaptive"],
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
                self.model = MLPRegressor(
                    max_iter=10000, early_stopping=True, n_iter_no_change=50
                )
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
        random_search.fit(
            X_train, y_train.values.ravel() if y_train.shape[1] == 1 else y_train
        )  # adjust if only one target label (single regression)
        print(
            f"RandomizedSearchCV took {(time() - start):.2f} seconds for {self.n_iter_search} candidates parameter settings."
        )
        report(random_search.cv_results_, n_top=self.n_report)
        self.random_search = random_search

    def return_fitted_model(self, X_train, y_train):
        self.parameter_search(X_train, y_train)
        return self.random_search.best_estimator_

    def search_fit_predict(model, X_train, X_test, y_train):
        self.return_fitted_model(X_train, y_train)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("R2 score:", r2_score(y_test, y_pred))
