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
from sklearn.preprocessing import MinMaxScaler

# spatial
import xarray as xa
import xesmf as xe
import rioxarray
from pyproj import Transformer
import rasterio

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
        data_source: str = "prism",
        endmember_class_schema: str = "three_endmember",
        gcfg: dict = file_ops.read_yaml(file_ops.CONFIG_DIR_FP / "glob_cfg.yaml"),
        train_ratio: float = 0.8,
        target: str = "endmember",
        normalise: bool = True,
        scaler_type: str = "minmax",
        random_seed: int = 42,
        emulation_source: str = None,
    ):
        self.data_source = data_source
        self.endmember_class_schema = endmember_class_schema
        self.gcfg = gcfg
        self.train_ratio = train_ratio
        self.target = target.lower()
        self.normalise = normalise
        self.scaler_type = scaler_type
        self.random_seed = random_seed
        self.emulation_source = emulation_source

    def load_prism_spectra(self):
        self.raw_spectra = spectrum_utils.load_spectra()
        if self.emulation_source:
            match self.emulation_source.lower():
                case "s2":
                    self.response_fns = spectrum_utils.load_s2_response_fns()
                    self.bois = ["B2", "B3", "B4", "B8A"]

                case "planet":
                    self.response_fns = spectrum_utils.load_planet_response_fns()
                    self.bois = ["Blue", "Green", "Red", "NIR"]
                case _:
                    raise ValueError(
                        f"Emulation source '{self.emulation_source}' not recognised"
                    )
            self.spectra = spectrum_utils.visualise_satellite_from_prism(
                self.raw_spectra, self.response_fns, self.bois
            )
        else:
            self.spectra = spectrum_utils.preprocess_prism_spectra(
                self.raw_spectra,
                spectrum_utils.NIR_WAVELENGTHS,
                spectrum_utils.SENSOR_RANGE,
            )

    def load_fitted_spectra(self):
        # TODO: not hardcoded
        fit_fp = "/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/coralreflections/results/fits/fit_results_1.csv"
        fits = pd.read_csv(fit_fp, header=[0, 1])

        fitted_spectra = fits.fitted_spectra
        fitted_spectra.columns = fitted_spectra.columns.astype(float)
        return fitted_spectra

    def load_simulation_spectra(self):
        #     self.spectra = optimisation_pipeline.SimulateSpectra(
        #         self.cfg, self.gcfg
        #     ).generate_simulated_spectra()
        # loading from file for now
        import pickle

        self.spectra = pickle.load(
            open(file_ops.TMP_DIR_FP / "sims_df.pkl", "rb")
        ).iloc[:, 4:]
        self.labels = pickle.load(open(file_ops.TMP_DIR_FP / "labels_df.pkl", "rb"))

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
        match self.data_source:
            # if not simulation
            case "prism" | "prism_fits" | "fits":
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
                    grouped_val_data[endmember_dimensionality_reduction] = (
                        self.validation_data[validation_fields].sum(axis=1)
                    )
                self.labels = grouped_val_data
            case "simulation":
                pass

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
        match self.data_source:
            case "prism_fits":
                fitted_spectra = self.load_fitted_spectra()

                if self.emulation_source:
                    # add more bands (0 value) to fitted spectra to match raw_prism
                    fitted_spectra = spectrum_utils.expand_df_with_empty_columns(
                        self.raw_spectra, fitted_spectra
                    )
                    fitted_spectra = spectrum_utils.visualise_satellite_from_prism(
                        fitted_spectra, self.response_fns, self.bois
                    )

                train_fitted_inds = self.X_train.index.intersection(
                    fitted_spectra.index
                )
                self.X_train = pd.concat(
                    [self.X_train, fitted_spectra.loc[train_fitted_inds]]
                )
                self.y_train = pd.concat(
                    [self.y_train, self.y_train.loc[train_fitted_inds]]
                )
                train_val_inds = self.X_train.index.intersection(
                    self.validation_data.index
                )
                remaining_inds = fitted_spectra.index.difference(train_val_inds)
                self.labels = pd.concat(
                    [
                        self.labels.loc[train_val_inds],
                        self.labels.loc[train_val_inds],
                        self.labels.loc[remaining_inds],
                    ]
                )

    def generate_data(self):
        self.load_validation_data()
        # match self.data_source:
        #     case "prism":
        #         self.load_prism_spectra()
        #     case "fits" | "fitted":
        #         self.load_fitted_spectra()
        #     case "fits_og":
        #         self.load_fitted_spectra()
        #         start_fits = self.spectra
        #         self.load_prism_spectra()
        #         self.spectra = pd.concat([start_fits, self.spectra])
        #         self.validation_data = pd.concat(
        #             [self.validation_data, self.validation_data]
        #         )
        #     case "simulation":
        #         self.load_simulation_spectra()
        #     case _:
        #         raise ValueError(f"Data source '{self.data_source}' not recognised")
        match self.data_source:
            case "prism" | "prism_fits":
                self.load_prism_spectra()
            case "fits":
                self.spectra = self.load_fitted_spectra()
            case "simulation":
                self.load_simulation_spectra()

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
            verbose=2,
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


def infer_on_spatial(
    trained_model, spectra_xa: xa.Dataset, prediction_classes: list[str]
) -> xa.Dataset:
    spectra_df = spectral_xa_to_processed_spectral_df(spectra_xa)
    no_nans_spectra_df_scaled = process_df_for_inference(spectra_df)
    # infer on dataframe
    predictions = trained_model.predict(no_nans_spectra_df_scaled)
    # reassamble dataframe
    predictions = pd.DataFrame(
        predictions, columns=prediction_classes, index=no_nans_spectra_df_scaled.index
    )
    full_index = pd.RangeIndex(
        start=spectra_df.index.min(), stop=spectra_df.index.max() + 1
    )
    return append_inferred_values_to_xa(predictions, full_index, spectra_xa)


def spectral_xa_to_processed_spectral_df(
    spectra_xa: xa.Dataset, sensor_range: tuple[float] = spectrum_utils.SENSOR_RANGE
) -> pd.DataFrame:
    spectra_vals = spectra_xa.values.reshape(spectra_xa.shape[0], -1)
    wvs = spectra_xa.coords["band"].values
    return spectrum_utils.preprocess_prism_spectra(
        pd.DataFrame(spectra_vals.T, columns=wvs), sensor_range=sensor_range
    )


def process_df_for_inference(spectra_df: pd.DataFrame) -> pd.DataFrame:
    no_nans_spectra_df = spectra_df.dropna()
    scaler = MinMaxScaler()
    scaler = scaler.fit(no_nans_spectra_df)
    return pd.DataFrame(
        scaler.transform(no_nans_spectra_df),
        index=no_nans_spectra_df.index,
        columns=no_nans_spectra_df.columns,
    )


def append_inferred_values_to_xa(
    inferred_values: pd.DataFrame, full_index: pd.Index, spectra_xa: xa.Dataset
) -> xa.Dataset:
    filled_prediction_vals = inferred_values.reindex(full_index)

    # cast to dataset and return
    prediction_array = filled_prediction_vals.values.reshape(
        spectra_xa.shape[1], spectra_xa.shape[2], len(inferred_values.columns)
    )

    if isinstance(spectra_xa, xa.DataArray):
        spectra_xa.name = "spectra"
        spectra_xa = spectra_xa.to_dataset()

    for i, class_name in enumerate(inferred_values.columns):
        spectra_xa[class_name + "_pred"] = (["lat", "lon"], prediction_array[:, :, i])

    return spectra_xa


def regrid_with_xesmf(ds: xa.Dataset | xa.DataArray) -> xa.Dataset:
    # Reproject the dataset first, keeping the original shape and filling nodata with NaN
    ds = ds.rio.reproject(
        "EPSG:4326", shape=(ds.sizes["y"], ds.sizes["x"]), nodata=np.nan
    )

    # Create new latitude and longitude arrays for the target grid
    lat_new = np.linspace(ds.lat.min(), ds.lat.max(), ds.sizes["y"])
    lon_new = np.linspace(ds.lon.min(), ds.lon.max(), ds.sizes["x"])

    # Define the target grid with the new latitude and longitude coordinates
    target_grid = xa.Dataset({"lat": (["lat"], lat_new), "lon": (["lon"], lon_new)})

    # Create the regridder object (bilinear interpolation)
    regridder = xe.Regridder(ds, target_grid, "bilinear", unmapped_to_nan=True)

    if isinstance(ds, xa.Dataset):
        regridded_data = xa.Dataset()
        for var in ds.data_vars:
            # Only regrid variables that have dimensions compatible with lat/lon
            if "y" in ds[var].dims and "x" in ds[var].dims:
                regridded_data[var] = regridder(ds[var])
                regridded_data[var].attrs = ds[var].attrs
            else:
                # Skip variables without spatial dimensions or handle them differently
                regridded_data[var] = ds[var]
    else:
        # If ds is a DataArray, directly regrid it
        regridded_data = regridder(ds)
        regridded_data.attrs = ds.attrs

    # Ensure the new dataset has the correct CRS after regridding
    regridded_data.rio.write_crs("EPSG:4326", inplace=True)

    return regridded_data


def envi_to_xarray_with_latlon(envi_fp, band_vals: list[float] = None):
    # Open the ENVI file using rasterio
    with rasterio.open(envi_fp) as src:
        data = src.read()
        transform = src.transform
        height, width = src.shape

        # generate UTM coordinates from pixel indices using the affine transform
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        x_coords, y_coords = rasterio.transform.xy(transform, y_coords, x_coords)

        # Convert the UTM coordinates to numpy arrays
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)

        # Convert UTM to lat/lon using pyproj or rasterio's CRS info
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        lon_coords, lat_coords = transformer.transform(x_coords, y_coords)

        # Reshape lat and lon arrays to match the image dimensions
        lat_coords = np.reshape(lat_coords, (height, width))
        lon_coords = np.reshape(lon_coords, (height, width))

        # Create the xarray Dataset with lat/lon as coordinates
        dataset = xa.DataArray(
            data=data,  # The raster data
            dims=("band", "y", "x"),  # Dimensions of the data (e.g., bands, rows, cols)
            coords={
                "lat": (["y", "x"], lat_coords),  # Latitude coordinates (reshaped)
                "lon": (["y", "x"], lon_coords),  # Longitude coordinates (reshaped)
                "band": (
                    band_vals if band_vals else np.arange(1, data.shape[0] + 1)
                ),  # Band indices
            },
            attrs=src.meta,  # Include the ENVI metadata
        )

    return dataset


def get_model_std_dev(model, X_test, y_test: pd.DataFrame) -> pd.DataFrame:

    predictions = []
    for i, tree in enumerate(model.estimators_):
        predictions.append(tree.predict(X_test))

    np.array(predictions).shape
    stds = np.std(predictions, axis=0)

    return stds


def generate_model_metadata(validation_data, model, X_test, y_test):
    std_dev_predictions = get_model_std_dev(model, X_test, y_test)
    predictions_std_dev = pd.DataFrame(
        std_dev_predictions,
        index=y_test.index,
        columns=[col + "_std_dev" for col in y_test.columns],
    )
    return pd.concat(
        [validation_data.loc[y_test.index, ["Locale", "Depth"]], predictions_std_dev],
        axis=1,
    )
