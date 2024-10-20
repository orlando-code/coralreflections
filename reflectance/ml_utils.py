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
        gcfg: dict = file_ops.instantiate_single_configs_instance()[0],
        train_ratio: float = 0.8,
        target: str = "endmember",
        normalise: bool = True,
        scaler_type: str = "minmax",
        random_seed: int = 42,
        emulation_source: str = None,
        label_noise_level: float = None,
        cfg: dict = file_ops.instantiate_single_configs_instance()[1],
    ):
        self.data_source = data_source
        self.endmember_class_schema = endmember_class_schema
        self.cfg = cfg
        self.gcfg = gcfg
        self.train_ratio = train_ratio
        self.target = target.lower()
        self.normalise = normalise
        self.scaler_type = scaler_type
        self.random_seed = random_seed
        self.emulation_source = emulation_source
        self.label_noise_level = label_noise_level
        self.n_plus = 0

    def load_prism_spectra(self):
        if "kaneohe" in self.data_source:
            self.raw_spectra = spectrum_utils.load_spectra(
                file_ops.KANEOHE_VAL_SPECTRA_FP
            )
        else:
            self.raw_spectra = spectrum_utils.load_spectra()
        if self.emulation_source:  # if emulating a different sensor
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
        fit_dfs = []

        # count number of "+" in self.data_source
        self.n_plus = self.data_source.count("+")
        for i in range(self.n_plus + 1):
            fit_fp = file_ops.RESULTS_DIR_FP / f"fits/fit_results_{i+1}.csv"
            fits = pd.read_csv(fit_fp, header=[0, 1])
            fit_dfs.append(fits.fitted_spectra)
        fitted_spectra = pd.concat(fit_dfs)
        # fit_fp = file_ops.RESULTS_DIR_FP / "/fits/fit_results_1.csv"
        # fits = pd.read_csv(fit_fp, header=[0, 1])

        fitted_spectra.columns = fitted_spectra.columns.astype(float)
        return fitted_spectra

    def load_simulation_spectra(self):
        from reflectance import optimisation_pipeline

        # TODO: add depth info as metadata

        Rb_array = np.random.dirichlet(np.ones(3), 100)
        Rb_df = pd.DataFrame(Rb_array, columns=["algae", "coral", "sand"])

        sims = []
        for Rb_vals in Rb_df.values:
            self.cfg.simulation["Rb_vals"] = Rb_vals
            sims.append(
                optimisation_pipeline.SimulateSpectra(
                    self.gcfg, self.cfg
                ).generate_simulated_spectra()
            )

        sims = pd.concat(sims)
        # tile Rb_df to match the number of rows in sims
        Rb_df = Rb_df.loc[Rb_df.index.repeat(self.cfg.simulation["N"])].reset_index(
            drop=True
        )
        self.labels = Rb_df
        self.spectra = sims.iloc[:, 4:]

        # loading from file for now
        # import pickle

        # self.spectra = pickle.load(
        #     open(file_ops.TMP_DIR_FP / "sims_df.pkl", "rb")
        # ).iloc[:, 4:]
        # self.labels = pickle.load(open(file_ops.TMP_DIR_FP / "labels_df.pkl", "rb"))

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
        if self.data_source != "simulation":
            # if not simulation
            if self.data_source == "kaneohe" or self.data_source == "kaneohe_fits":
                kbay_inds = self.validation_data.index[
                    self.validation_data["Locale"] == "Kaneohe Bay"
                ]
                self.validation_data = self.validation_data.loc[kbay_inds]
                self.index_subset = kbay_inds

            # process to correct class
            self.validation_data = spectrum_utils.map_validation(
                self.validation_data, self.gcfg.endmember_map
            )

            endmember_schema_map = self.gcfg.endmember_schema[
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
            self.labels.reset_index(drop=True, inplace=True)
            # add gaussian noise to labels
            np.random.seed(self.random_seed)
            if self.label_noise_level:
                self.labels = self.labels + np.random.normal(
                    0, self.label_noise_level, self.labels.shape
                )
        elif self.data_source == "simulation":
            pass  # already handled in load_simulation_spectra
        else:
            raise ValueError(f"Data source '{self.data_source}' not recognised")

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
        if "fits" in self.data_source:
            fitted_spectra = self.load_fitted_spectra()

            if self.emulation_source:
                # add more bands (0 value) to fitted spectra to match raw_prism
                fitted_spectra = spectrum_utils.expand_df_with_empty_columns(
                    self.raw_spectra, fitted_spectra
                )
                fitted_spectra = spectrum_utils.visualise_satellite_from_prism(
                    fitted_spectra, self.response_fns, self.bois
                )
            if self.data_source == "kaneohe_fits":
                fitted_spectra = fitted_spectra.loc[self.index_subset]
                fitted_spectra.reset_index(drop=True, inplace=True)
                # only use kaneohe bay data
                # fitted_spectra = fitted_spectra.loc[
                #     fitted_spectra.index.intersection(
                #         self.validation_data.loc[self.index_subset].index
                #     )
                # ]
                if np.any(fitted_spectra.columns != self.spectra.columns):
                    # TODO: potentially interpolate fitted_spectra to match spectra
                    fitted_spectra = fitted_spectra.reindex(
                        columns=self.spectra.columns
                    )

            train_fitted_inds = self.X_train.index.intersection(fitted_spectra.index)
            self.X_train = pd.concat(
                [self.X_train, fitted_spectra.loc[train_fitted_inds]]
            )
            # double up on labels.
            self.y_train = pd.concat(
                [
                    self.y_train,
                    pd.concat(
                        [self.y_train.loc[train_fitted_inds]] * (self.n_plus + 1),
                        ignore_index=True,
                    ),
                ]
            )
            dropped_validation_inds = self.validation_data.reset_index(inplace=False)
            train_val_inds = self.X_train.index.intersection(
                # self.validation_data.index
                dropped_validation_inds.index
            )
            remaining_inds = fitted_spectra.index.difference(train_val_inds)
            self.labels = pd.concat(
                [
                    pd.concat(
                        [self.labels.loc[train_val_inds]] * (self.n_plus + 1),
                        ignore_index=True,
                    ),
                    # self.labels.loc[train_val_inds] * (self.n_plus + 1),
                    self.labels.loc[remaining_inds],
                ]
            )

    def generate_data(self):
        self.load_validation_data()

        if "fits" in self.data_source and "+" in self.data_source:
            self.load_prism_spectra()
        elif self.data_source == "fits":
            self.spectra = self.load_fitted_spectra()
        elif self.data_source == "simulation":
            self.load_simulation_spectra()
        elif "prism" in self.data_source or "kaneohe" in self.data_source:
            self.load_prism_spectra()

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
                    "min_samples_leaf": [1, 2, 4, 8, 16],
                    "min_samples_split": [2, 5, 10, 15, 20],
                    "n_estimators": [130, 180, 230, 300],
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
            verbose=1,
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
    trained_model,
    spectra_xa: xa.Dataset,
    prediction_classes: list[str],
    dim_red: int = None,
) -> xa.Dataset:
    spectra_df = spectral_xa_to_processed_spectral_df(spectra_xa)
    no_nans_spectra_df_scaled = process_df_for_inference(spectra_df, dim_red=dim_red)
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


def process_df_for_inference(
    spectra_df: pd.DataFrame, dim_red: int = 5
) -> pd.DataFrame:
    no_nans_spectra_df = spectra_df.dropna()
    if dim_red:
        from sklearn.decomposition import PCA

        print("Doing PCA...")
        pca = PCA(n_components=dim_red)
        no_nans_spectra_df = pd.DataFrame(
            pca.fit_transform(no_nans_spectra_df),
            index=no_nans_spectra_df.index,
            columns=[i for i in range(5)],
        )

    scaler = MinMaxScaler()
    scaler = scaler.fit(no_nans_spectra_df)
    return pd.DataFrame(
        scaler.transform(no_nans_spectra_df),
        index=no_nans_spectra_df.index,
        columns=no_nans_spectra_df.columns,
    )


def reform_spatial_from_df(
    df, full_index, og_ds: xa.DataArray | xa.Dataset, new_dim_name: str = "new"
) -> xa.Dataset:
    filled_df = df.reindex(full_index)

    data = filled_df.values.reshape(og_ds.shape[1], og_ds.shape[2], len(df.columns))
    # create new dataset with same spatial extent as the original

    # Create a new DataArray with the new data and the original spatial coordinates
    ds = xa.DataArray(
        data,
        coords={"lat": og_ds.lat, "lon": og_ds.lon, new_dim_name: df.columns},
        dims=["lat", "lon", new_dim_name],
    )
    return ds


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
        # replace -9999 with NaN
        dataset = dataset.where(dataset != -9999)

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
