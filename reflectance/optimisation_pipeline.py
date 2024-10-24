# general
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path
import xarray as xa
import logging
import traceback

# profiling
import cProfile
import pstats

# fitting
# import multiprocess as mp
from functools import partial
from joblib import Parallel, delayed

# custom
from reflectance import spectrum_utils, file_ops


class GenerateEndmembers:
    """
    Generate endmembers
    """

    def __init__(
        self,
        endmember_class_map: dict,
        endmember_dimensionality_reduction: str = "mean",
        endmember_normalisation: bool = False,
        spectral_library_fp: Path = file_ops.RESOURCES_DIR_FP
        / "spectral_library_clean_v3_PRISM_wavebands.csv",
    ):
        self.spectral_library_fp = spectral_library_fp
        self.endmember_class_map = endmember_class_map
        self.endmember_dimensionality_reduction = endmember_dimensionality_reduction
        self.endmember_normalisation = endmember_normalisation

    def load_spectral_library(self):
        """Load spectral library"""
        self.spectral_library = spectrum_utils.load_spectral_library(
            self.spectral_library_fp
        )

    def characterise_endmembers(self):
        """Characterise endmembers using specified method"""
        # remap classes to desired endmember schema
        self.spectral_library = spectrum_utils.group_classes(
            self.spectral_library, self.endmember_class_map
        )

        if isinstance(self.endmember_dimensionality_reduction, str):
            reduction_methods = {
                "mean": spectrum_utils.mean_endmembers,
                "median": spectrum_utils.median_endmembers,
            }
            try:
                self.endmembers = reduction_methods[
                    self.endmember_dimensionality_reduction
                ](self.spectral_library)
            except KeyError:
                raise ValueError(
                    f"Endmember type {self.endmember_dimensionality_reduction} not recognised\n"
                    "Have you forgotten a dimension number?"
                )
        elif isinstance(self.endmember_dimensionality_reduction, (tuple, list)):
            self.endmembers = spectrum_utils.calculate_endmembers(
                self.spectral_library,
                self.endmember_dimensionality_reduction[0],
                self.endmember_dimensionality_reduction[1],
            )
        else:
            raise ValueError(
                f"Endmember type {self.endmember_dimensionality_reduction} not recognised"
            )

        # if specified, normalise endmembers
        if self.endmember_normalisation:
            self.endmembers = spectrum_utils.normalise_spectra(
                self.endmembers, self.endmember_normalisation
            )

    def generate_endmembers(self):
        """Generate endmembers"""
        self.load_spectral_library()
        self.characterise_endmembers()
        return self.endmembers


class SimulateSpectra(GenerateEndmembers):

    def __init__(
        self,
        cfg: file_ops.RunOptPipeConfig,
        gcfg: file_ops.GlobalOptPipeConfig,
    ):
        self.cfg = cfg
        self.gcfg = gcfg

    def spread_spectra(self):
        self.raw_spectra, self.spectra_metadata = (
            spectrum_utils.spread_simulate_spectra(
                wvs=self.wvs,
                endmember_array=self.endmembers,
                AOP_args=self.aop_args,
                Rb_vals=self.cfg.simulation["Rb_vals"],
                N=self.cfg.simulation["N"],
                depth_lims=self.cfg.simulation["depth_lims"],
                k_lims=self.cfg.simulation["k_lims"],
                bb_lims=self.cfg.simulation["bb_lims"],
            )
        )

    def regular_spectra(self):
        self.raw_spectra, self.spectra_metadata = spectrum_utils.simulate_spectra(
            endmember_array=self.endmembers,
            wvs=self.wvs,
            AOP_args=self.aop_args,
            Rb_vals=self.cfg.simulation["Rb_vals"],
            N=self.cfg.simulation["N"],
            n_depths=self.cfg.simulation["n_depths"],
            depth_lims=self.cfg.simulation["depth_lims"],
            n_ks=self.cfg.simulation["n_ks"],
            k_lims=self.cfg.simulation["k_lims"],
            n_bbs=self.cfg.simulation["n_bbs"],
            bb_lims=self.cfg.simulation["bb_lims"],
            n_noise_levels=self.cfg.simulation["n_noise_levels"],
            noise_lims=self.cfg.simulation["noise_lims"],
        )

    def handle_metadata(self):
        # concatenate metadata to spectra
        self.sim_spectra = pd.concat([self.spectra_metadata, self.raw_spectra], axis=1)

    def load_aop_model(self):
        """Load the AOP model dependent on the specified group"""
        self.aop_model = spectrum_utils.load_aop_model(self.cfg.aop_group_num)
        self.aop_args = spectrum_utils.process_aop_model(
            self.aop_model, self.cfg.sensor_range
        )

    def generate_simulated_spectra(self):
        self.load_aop_model()
        self.wvs = self.aop_model.loc[
            min(self.cfg.sensor_range) : max(self.cfg.sensor_range)
        ].index

        self.endmembers = GenerateEndmembers(
            self.gcfg.endmember_schema[self.cfg.endmember_class_schema],
            self.cfg.endmember_dimensionality_reduction,
            self.cfg.endmember_normalisation,
            self.gcfg.spectral_library_fp,
        ).generate_endmembers()
        self.endmembers = spectrum_utils.crop_spectra_to_range(
            self.endmembers, self.cfg.sensor_range
        )

        match self.cfg.simulation["type"]:
            case "spread":
                self.spread_spectra()
            case "regular":
                self.regular_spectra()
            case _:
                raise ValueError(
                    f"Simulation type {self.cfg.simulation['type']} not recognised"
                )
        self.handle_metadata()
        return self.sim_spectra


class OptPipe:
    """
    Class to run the optimisation pipeline for a range of parameterisation schema
    """

    def __init__(
        self,
        glob_cfg: file_ops.GlobalOptPipeConfig,
        run_cfg: file_ops.RunOptPipeConfig,
        # exec_kwargs: dict,
    ):
        """Initialises class attributes"""
        self.gcfg = glob_cfg
        self.endmember_map = self.gcfg.endmember_map
        self.cfg = run_cfg

        # self.exec_kwargs = exec_kwargs

    def preprocess_spectra(self):
        """Correct for glint, calculate subsurface reflectance, crop to sensor range"""
        self.spectra = spectrum_utils.preprocess_prism_spectra(
            self.raw_spectra,
            nir_wavelengths=self.cfg.nir_wavelengths,
            sensor_range=self.cfg.sensor_range,
        ).loc[:, :]
        # normalise spectra if specified
        if self.cfg.spectra_normalisation:
            self.spectra = spectrum_utils.normalise_spectra(
                self.spectra, self.cfg.spectra_normalisation
            )

    def preprocess_endmembers(self):
        """Go from raw spectral library to analysis-ready endmembers i.e. crop to relevant wavelengths"""
        self.endmembers = spectrum_utils.crop_spectra_to_range(
            self.endmembers, self.cfg.sensor_range
        )

    # def convert_classes(self):
    #     """Map validation data classes to endmember classes"""
    #     # TODO: implement
    #     endmember_map = self.gcfg.endmember_map
    #     mapped_df = pd.DataFrame(index=target_df.index, columns=endmember_map)

    #     for end_member_type, validation_fields in endmember_map.items():
    #         # fill in validation data with sum of all fields in the category
    #         mapped_df.loc[:, end_member_type] = target_df.loc[:, validation_fields].sum(axis=1)

    #     self.validation_df = mapped_df

    def retrieve_class_map(self):
        dict_object = next(
            (
                d
                for d in self.gcfg.endmember_schema
                if self.cfg.endmember_class_schema in d
            ),
            None,
        )
        return list(dict_object.values())[0]

    def load_spectra(self):
        """Load spectra"""
        if self.gcfg.spectra_source == "simulation":
            sim_params = self.cfg.simulation
            # select wvs within sensor_range
            wvs = self.aop_model.loc[
                min(self.cfg.sensor_range) : max(self.cfg.sensor_range)
            ].index
            if sim_params["type"] == "spread":
                raw_spectra, spectra_metadata = spectrum_utils.spread_simulate_spectra(
                    wvs=wvs,
                    endmember_array=self.endmembers,
                    AOP_args=self.aop_args,
                    Rb_vals=sim_params["Rb_vals"],
                    N=sim_params["N"],
                    noise_lims=sim_params["noise_lims"],
                    depth_lims=sim_params["depth_lims"],
                    k_lims=sim_params["k_lims"],
                    bb_lims=sim_params["bb_lims"],
                )
            elif sim_params["type"] == "regular":
                raw_spectra, spectra_metadata = spectrum_utils.simulate_spectra(
                    endmember_array=self.endmembers,
                    AOP_args=self.aop_args,
                    Rb_vals=sim_params["Rb_vals"],
                    N=sim_params["N"],
                    n_depths=sim_params["n_depths"],
                    depth_lims=sim_params["depth_lims"],
                    n_ks=sim_params["n_ks"],
                    k_lims=sim_params["k_lims"],
                    n_bbs=sim_params["n_bbs"],
                    bb_lims=sim_params["bb_lims"],
                    n_noise_levels=sim_params["n_noise_levels"],
                    noise_lims=sim_params["noise_lims"],
                )  # TODO: handle metadata
            # reshape array and metadata to two dataframes
            self.raw_spectra = pd.DataFrame(
                raw_spectra.reshape(-1, raw_spectra.shape[-1]),
                columns=wvs,
            )
            self.spectra_metadata = spectra_metadata
            # concatenate metadata to spectra
            sim_spectra = pd.concat([self.spectra_metadata, self.raw_spectra], axis=1)
            # save spectra and metadata to file
            fits_dir = file_ops.get_dir(
                file_ops.get_dir(file_ops.RESULTS_DIR_FP) / "fits"
            )
            self.get_run_id()
            fits_fp = file_ops.get_f(fits_dir / f"sim_spectra_{self.run_id}.csv")
            sim_spectra.to_csv(fits_fp, index=False)
        elif self.gcfg.spectra_source == "kaneohe":
            xa_ds = xa.open_dataset(
                file_ops.KANEOHE_HS_FP
            ).spectra  # TODO: make not janky
            raw_spectra_with_nans = xa_ds.values.reshape(xa_ds.sizes["band"], -1)
            wvs = xa_ds.coords["band"].values
            self.raw_spectra = pd.DataFrame(
                raw_spectra_with_nans.T, columns=wvs
            ).dropna(axis=0)
        else:
            self.raw_spectra = spectrum_utils.load_spectra(self.gcfg.spectra_fp)

    def load_aop_model(self):
        """Load the AOP model dependent on the specified group"""
        self.aop_model = spectrum_utils.load_aop_model(self.cfg.aop_group_num)
        self.aop_args = spectrum_utils.process_aop_model(
            self.aop_model, self.cfg.sensor_range
        )

    def return_objective_fn(self):

        match self.cfg.objective_fn.lower():
            case "r2":
                return spectrum_utils.r2_objective_fn
            case "og_r2":
                return spectrum_utils.og_rg_of
            case "euclidean_distance":
                return spectrum_utils.euclidean_distance
            case "mahalanobis_distance":
                return spectrum_utils.mahalanobis_distance
            case "spectral_similarity_gradient":
                return spectrum_utils.spectral_similarity_gradient_of
            case "spectral_information_divergence":
                return spectrum_utils.spectral_information_divergence
            case "sidsam":
                return spectrum_utils.sidsam
            case "jmsam":
                return spectrum_utils.jmsam
            case "spectral_angle":
                return spectrum_utils.spectral_angle_objective_fn
            case "spectral_angle_w1":
                return spectrum_utils.spectral_angle_objective_fn_w1
            case _:
                raise ValueError(
                    f"Objective function {self.cfg.objective_fn} not recognised"
                )

    def fit_spectra(self):
        # create wrapper for function to allow parallel processing
        of = self.return_objective_fn()
        partial_wrapper = partial(
            spectrum_utils._wrapper,
            of=of,
            obs_spectra=self.spectra,
            AOP_args=self.aop_args,
            endmember_array=self.endmembers.values,
            Rb_init=self.cfg.Rb_init,
            bb_bounds=self.cfg.bb_bounds,
            Kd_bounds=self.cfg.Kd_bounds,
            H_bounds=self.cfg.H_bounds,
            endmember_bounds=self.cfg.endmember_bounds,
            solver=self.cfg.solver,
            tol=self.cfg.tol,
        )

        # if self.exec_kwargs["tqdm"]:
        fitted_params = Parallel(n_jobs=128)(
            delayed(partial_wrapper)(index)
            for index in tqdm(self.spectra.index, miniters=10, desc="Fitting spectra")
        )

        self.fitted_params = pd.DataFrame(
            fitted_params,
            index=self.spectra.index,
            columns=["bb", "K", "H"] + list(list(self.endmembers.index)),
        )

    def generate_results_df(self):
        """Generate a dataframe with a multiindex: run parameters, true spectra, fit results, error metrics"""
        self.results = spectrum_utils.generate_results_df(
            self.cfg, self.spectra, self.fitted_params, self.metrics
        )

    def generate_spectra_from_fits(self):
        self.fitted_spectra = spectrum_utils.generate_spectra_from_fits(
            self.fitted_params,
            self.spectra.columns,
            self.endmembers.values,
            self.aop_args,
        )

    def calculate_error_metrics(self):
        self.error_metrics = spectrum_utils.calculate_metrics(
            self.spectra, self.fitted_spectra
        )

    def generate_fit_results(self):
        # combine fitted_params with fitted_spectra and metrics
        self.fit_results = spectrum_utils.generate_fit_results(
            self.fitted_params, self.fitted_spectra, self.error_metrics
        )
        if self.gcfg.save_fits:
            # generate filename from summary_results index
            fits_dir_fp = file_ops.get_dir(file_ops.RESULTS_DIR_FP / "fits")
            # find maximum index in results_summary and create fp
            self.get_run_id()
            fits_fp = file_ops.get_f(fits_dir_fp / f"fit_results_{self.run_id}.csv")
            # save to csv
            self.fit_results.to_csv(fits_fp, index=False)

    def generate_results_summary(self):
        # TODO: arrangement of method?
        # calculate results summary
        metrics_summary = spectrum_utils.generate_results_summary(self.error_metrics)

        # generate metadata
        self.generate_run_metadata()
        self.generate_glob_summary()
        self.generate_cfg_summary()
        if hasattr(self, "e"):
            self.generate_error_summary()
        else:
            self.error_df = None
        # concatenate run summaries
        self.results_summary = pd.concat(
            [
                self.metadata_df,
                self.glob_summary,
                self.cfg_df,
                metrics_summary,
                self.error_df,
            ],
            axis=1,
        )
        self.save_results_summary()

    def generate_cfg_summary(self):
        config_summary_dict = self.cfg.get_config_summaries()
        multiindex = pd.MultiIndex.from_tuples(config_summary_dict.keys())
        self.cfg_df = pd.DataFrame([config_summary_dict.values()], columns=multiindex)

    def generate_run_metadata(self):
        # return date and time as multiindex df with metadata on first, header on second
        datetime = pd.Timestamp.now("UTC")
        # overkill for now, but may add further metadata in future
        multiindex = pd.MultiIndex.from_product([["metadata"], ["datetime (UTC)"]])
        self.metadata_df = pd.DataFrame([[datetime]], columns=multiindex)

    def generate_glob_summary(self):
        # select keys of glob_cfg to include in summary
        glob_summary = {
            k: v
            for k, v in self.gcfg.__dict__.items()
            if k not in ["endmember_map", "endmember_schema"]
        }
        if self.gcfg.spectra_source == "simulation":
            glob_summary["spectra_fp"] = None
        # create dataframe
        glob_summary_df = pd.DataFrame([glob_summary])
        # create multiindex columns
        multiindex_columns = pd.MultiIndex.from_product(
            [["global_configuration"], glob_summary_df.columns]
        )
        glob_summary_df.columns = multiindex_columns
        self.glob_summary = glob_summary_df

    def get_run_id(self):
        # get run id (maximum index in results_summary.csv)
        try:
            # try to read csv
            summary_csv = pd.read_csv(file_ops.RESULTS_DIR_FP / "results_summary.csv")
            self.run_id = summary_csv.index.max()
        except Exception:
            self.run_id = 1

    def save_results_summary(self):
        # save resulting fits to file
        file_ops.results_dir_fp = file_ops.get_dir(file_ops.RESULTS_DIR_FP)
        # create results csv if doesn't already exist
        results_fp = file_ops.results_dir_fp / "results_summary.csv"
        # write header if new file
        if not results_fp.exists():
            self.results_summary.to_csv(results_fp, index=False)
        else:
            # append results_summary to results_csv file
            self.results_summary.to_csv(results_fp, mode="a", header=False, index=False)

    def generate_error_summary(self):
        error_message = (
            f"Error in {self.e_step}: {str(self.e)}\n{traceback.format_exc()}"
        )
        logging.error(error_message)
        self.get_run_id()
        # Append the error message to the results summary
        error_df = pd.DataFrame(
            [[self.run_id, self.e_step, error_message]],
            columns=["run_id", "step", "error_message"],
        )
        error_df.columns = pd.MultiIndex.from_product([["error"], error_df.columns])
        self.error_df = error_df

    def profile_fit_spectra(self):
        profiler = cProfile.Profile()
        profiler.enable()
        self.fit_spectra()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(10)  # Print top 10 slowest functions

    def generate_endmembers(self):
        if not self.cfg.endmember_source == "spectral_library":
            # resolve path
            self.cfg.endmember_source = Path(self.cfg.endmember_source)
            self.endmembers = spectrum_utils.load_Rb_model(self.cfg.endmember_source)
        else:
            self.endmembers = GenerateEndmembers(
                self.gcfg.endmember_schema[self.cfg.endmember_class_schema],
                self.cfg.endmember_dimensionality_reduction,
                self.cfg.endmember_normalisation,
                self.gcfg.spectral_library_fp,
            ).generate_endmembers()

    def check_for_repeat_run(self):
        """Check results csv for identical run configuration"""
        results_fp = file_ops.RESULTS_DIR_FP / "results_summary.csv"
        # write header if new file
        if not results_fp.exists():
            pass
        else:
            runs = pd.read_csv(results_fp)
            irrelevant_sub_columns = [
                "datetime (UTC)",
                "save_fits",
                "count",
                "mean",
                "std",
                "min",
            ]
            irrelevant_columns = [
                "spectral_angle",
                "r2",
                "rmse",
                "mean_abs_dev",
                "median_abs_dev",
            ]
            # drop any columns which contain strings included in "irrelevant_columns"
            runs = runs.drop(
                columns=[
                    col
                    for col in runs.columns
                    if any([sub in col for sub in irrelevant_columns])
                ]
            )
            # drop any columns for which value in first row is in "irrelevant_sub_columns"
            runs = runs.drop(
                columns=[
                    col
                    for col in runs.columns
                    if runs[col].iloc[0] in irrelevant_sub_columns
                ]
            )
            runs = runs.fillna(str(-9999))
            gcfg_dict = self.gcfg.__dict__.copy()  # prevent overwriting gcfg
            cfg_dict = self.cfg.get_config_summaries()
            # drop endmember_map, endmember_schema from gcfg_dict
            gcfg_dict.pop("endmember_map")
            gcfg_dict.pop("endmember_schema")
            gcfg_dict.pop("save_fits")

            config_values = list(gcfg_dict.values()) + list(cfg_dict.values())
            # convert all values to strings, and cast any PosixPath objects to strings
            config_values = [
                str(val) if not isinstance(val, Path) else str(val)
                for val in config_values
            ]
            config_values = [
                val if val != "None" else str(-9999) for val in config_values
            ]
            # Check for a match in any row
            for _, row in runs.iloc[1:].iterrows():
                if all(row.values == config_values):
                    return True
            return False

    def run(self):
        """
        Runs the optimisation pipeline
        """
        # if this would be a repeat run, skip the pipeline
        if self.check_for_repeat_run():
            print(f"Repeat run arrested for {self.cfg}")
            return

        pipeline_steps = [
            ("load_aop_model", self.load_aop_model),
            ("generate_endmembers", self.generate_endmembers),
            ("preprocess_endmembers", self.preprocess_endmembers),
            ("load_spectra", self.load_spectra),
            ("preprocess_spectra", self.preprocess_spectra),
            ("fit_spectra", self.fit_spectra),
            ("generate_spectra_from_fits", self.generate_spectra_from_fits),
            ("calculate_error_metrics", self.calculate_error_metrics),
        ]
        print("\n")
        # for step_name, step_method in pipeline_steps:
        #     # print(step_name)
        #     step_method()
        #     # profile_step(step_name, step_method)

        # try:
        #     # generate results (these steps not guarded by error catcher intentionally)
        #     self.generate_results_summary()
        #     self.generate_fit_results()
        # except:
        #     pass
        for step_name, step_method in pipeline_steps:
            try:
                step_method()
                # profile_step(step_name, step_method)
            except Exception as e:
                print(
                    "e:", e
                )  # useful for debugging since logging otherwise hides until end
                print(
                    "step_name", step_name
                )  # useful for debugging since logging otherwise hides until end
                self.e = e
                self.e_step = step_name

        try:
            # generate results (these steps not guarded by error catcher intentionally)
            self.generate_results_summary()
            self.generate_fit_results()
            print(self.cfg)
        except Exception as e:
            print("e", e)

        return self.fit_results


def run_pipeline(glob_cfg: dict, run_cfgs: dict):
    """
    Run the optimisation pipeline for a range of parameterisation schema
    """
    # resolve relative paths in the configuration to absolute paths
    glob_cfg = file_ops.resolve_paths(glob_cfg, file_ops.BASE_DIR_FP)
    glob_cfg = file_ops.GlobalOptPipeConfig(glob_cfg)
    for cfg in tqdm(run_cfgs, total=len(run_cfgs), desc="Running pipeline for configs"):
        run_cfg = file_ops.RunOptPipeConfig(cfg)
        opt_pipe = OptPipe(glob_cfg, run_cfg)
        opt_pipe.run()


# optionally run as script
if __name__ == "__main__":
    # load config_dict
    glob_cfg = file_ops.read_yaml(file_ops.CONFIG_DIR_FP / "glob_cfg.yaml")
    run_cfgs = file_ops.read_yaml(file_ops.CONFIG_DIR_FP / "run_cfgs.yaml")
    run_pipeline(glob_cfg, run_cfgs)


"""DEPRECATED

    # def generate_spectra_from_endmembers(self):
    #     for i, row in tqdm(self.fitted_params.iterrows(), total=len(self.fitted_params)):
    #         bb_m, bb_c, Kd_m, Kd_c = self.aop_args
    #         bb, K, H = row.values[:3]
    #         pred = spectrum_utils.sub_surface_reflectance_Rb(
        # prism_spectra.columns, endmember_array, bb, K, H, AOP_args, *row.values[3:-2])

# been replaced by error-protected pipeline
    # def run(self):
    #     # load spectra
    #     self.load_spectra()
    #     # preprocess observed spectra, normalising if specified
    #     self.preprocess_spectra()
    #     # load AOP model parameters
    #     self.load_aop_model()
    #     # load endmembers
    #     self.load_spectral_library()
    #     # preprocess endmembers
    #     self.preprocess_endmembers()
    #     # generate specified endmember parameterisation, normalising if specified
    #     self.characterise_endmembers()
    #     # fit normalised spectra using specified objective function
    #     self.fit_spectra()
    #     # calculate resulting spectra
    #     self.generate_spectra_from_fits()
    #     # calculate error metrics and append to results
    #     self.calculate_error_metrics()
    #     # calculate and save summary of results
    #     self.generate_results_summary()
    #     # calculate and save fits
    #     self.generate_fit_results()
"""
