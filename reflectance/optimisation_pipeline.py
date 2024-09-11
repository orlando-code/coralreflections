# general
from dataclasses import dataclass
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import logging
import traceback

# fitting
import multiprocess as mp
from functools import partial

# custom
from reflectance import spectrum_utils, file_ops

@dataclass
class GlobalOptPipeConfig:
    spectra_fp: str
    spectral_library_fp: str
    validation_data_fp: str
    save_fits: bool
    endmember_map: dict
    endmember_schema: dict
    
    def __init__(self, conf: dict):
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
        self.solver = conf["fitting"]["solver"]
        self.tol = conf["fitting"]["tol"]
    
    
class OptPipe():
    """
    Class to run the optimisation pipeline for a range of parameterisation schema
    """

    def __init__(self, glob_cfg: GlobalOptPipeConfig, run_cfg: RunOptPipeConfig):
        """Initialises class attributes"""
        self.gcfg = glob_cfg
        self.endmember_map = self.gcfg.endmember_map
        self.cfg = run_cfg

    def preprocess_spectra(self):
        """Correct for glint, calculate subsurface reflectance, crop to sensor range"""
        # fetch indices within NIR wavelength range
        spectra_deglinted = spectrum_utils.deglint_spectra(self.raw_spectra, self.cfg.nir_wavelengths)
        # subsurface reflectance
        spectra_subsurface = spectrum_utils.retrieve_subsurface_reflectance(spectra_deglinted)
        # crop to sensor range (with # TEMP RESTRICTION FOR TESTING)
        self.spectra = spectrum_utils.crop_spectra_to_range(spectra_subsurface, self.cfg.sensor_range).loc[:10,:]
        # normalise spectra if specified        
        if self.cfg.spectra_normalisation:
            self.spectra = spectrum_utils.normalise_spectra(self.spectra, self.cfg.spectra_normalisation)

    def preprocess_spectral_library(self):
        """Go from raw spectral library to analysis-ready endmembers i.e. crop to relevant wavelengths"""
        self.spectral_library = spectrum_utils.crop_spectra_to_range(self.spectral_library, self.cfg.sensor_range)
    
    def convert_classes(self):
        """Map validation data classes to endmember classes"""
        endmember_map = self.gcfg.endmember_map
        mapped_df = pd.DataFrame(index=target_df.index, columns=endmember_map)

        for end_member_type, validation_fields in endmember_map.items():
            # fill in validation data with sum of all fields in the category
            mapped_df.loc[:, end_member_type] = target_df.loc[:, validation_fields].sum(axis=1)
                
        self.validation_df = mapped_df
        
    def retrieve_class_map(self):
        dict_object = next((d for d in self.gcfg.endmember_schema if self.cfg.endmember_class_schema in d), None)
        return list(dict_object.values())[0]
    
    def characterise_endmembers(self):
        """Characterise endmembers using specified method"""
        # remap classes if necessary
        if self.cfg.endmember_class_schema:
            endmember_schema = self.retrieve_class_map()
            # rewrite spectral_library in classes in terms of class_map
            self.spectral_library = spectrum_utils.group_classes(self.spectral_library, endmember_schema)

        match self.cfg.endmember_type:
            case "mean":
                self.endmembers = spectrum_utils.mean_endmembers(self.spectral_library)
            case "median":
                self.endmembers = spectrum_utils.median_endmembers(self.spectral_library)
            case "pca" | "nmf" | "ica" | "svd":
                self.endmembers = spectrum_utils.calculate_endmembers(self.spectral_library, "pca")
            case _:
                raise ValueError(f"Endmember type {self.cfg.endmember_type} not recognised")

        # if specified, normalise endmembers
        if self.cfg.endmember_normalisation:
            self.endmembers = spectrum_utils.normalise_spectra(self.endmembers, self.cfg.endmember_normalisation)

    def load_spectra(self):
        """Load spectra"""
        self.raw_spectra = spectrum_utils.load_spectra(self.gcfg.spectra_fp)
    
    def load_aop_model(self):
        """Load the AOP model dependent on the specified group"""
        self.aop_model = spectrum_utils.load_aop_model(self.cfg.aop_group_num)
        aop_sub = self.aop_model.loc[self.spectra.columns]
        self.aop_args = (aop_sub.bb_m.values, aop_sub.bb_c.values, aop_sub.Kd_m.values, aop_sub.Kd_c.values)
        
    def load_spectral_library(self):
        """Load spectral library"""
        self.spectral_library = spectrum_utils.load_spectral_library(self.gcfg.spectral_library_fp)
            
    def return_objective_fn(self):
        
        match self.cfg.objective_fn.lower():
            case "r2":
                return spectrum_utils.r2_objective_fn
            case "euclidean_distance":
                return spectrum_utils.euclidean_distance
            case "mahalanobis_distance":
                return spectrum_utils.mahalanobis_distance
            case "spectral_similarity_gradient":
                return spectrum_utils.spectral_similarity_gradient
            case "spectral_information_divergence":
                return spectrum_utils.spectral_information_divergence
            case "sidsam":
                return spectrum_utils.sidsam
            case "spectral_angle":
                return spectrum_utils.spectral_angle_objective_fn
            case "spectral_angle_w1":
                return spectrum_utils.spectral_angle_objective_fn_w1
            case _:
                raise ValueError(f"Objective function {self.cfg.objective_fn} not recognised")
        
    def fit_spectra(self):
        # create wrapper for function to allow parallel processing
        of = self.return_objective_fn()
        partial_wrapper = partial(
            spectrum_utils._wrapper, 
            of=of, 
            prism_spectra=self.spectra, 
            AOP_args=self.aop_args,
            endmember_array=self.endmembers.values,
            Rb_init=0.0001,
            bb_bounds=self.cfg.bb_bounds,
            Kd_bounds=self.cfg.Kd_bounds,
            H_bounds=self.cfg.H_bounds,
            method=self.cfg.solver,
            tol=self.cfg.tol
            )
        
        with mp.Pool() as pool:
            fitted_params = list(tqdm(pool.imap(partial_wrapper, self.spectra.index), total=len(self.spectra.index)))
        self.fitted_params = pd.DataFrame(fitted_params, index=self.spectra.index, columns=['bb', 'K', 'H'] + list(list(self.endmembers.index)))
   
    def calculate_error_metrics(self):
        """Reconstructing spectra from the fitted parameters"""
        self.metrics = spectrum_utils.calculate_metrics(self.spectra, self.fitted_params)
        
    def generate_results_df(self):
        """Generate a dataframe with a multiindex: run parameters, true spectra, fit results, error metrics"""
        self.results = spectrum_utils.generate_results_df(self.cfg, self.spectra, self.fitted_params, self.metrics)
        
    def generate_spectra_from_fits(self):
        self.fitted_spectra = spectrum_utils.generate_spectra_from_fits(self.fitted_params, self.spectra.columns, self.endmembers.values, self.aop_args)
    
    def calculate_error_metrics(self):
        self.error_metrics = spectrum_utils.calculate_metrics(self.spectra, self.fitted_spectra)
        
    def generate_fit_results(self):
        # combine fitted_params with fitted_spectra and metrics
        self.fit_results = spectrum_utils.generate_fit_results(self.fitted_params, self.fitted_spectra, self.error_metrics)
        # generate filename from summary_results index
        fits_dir_fp = file_ops.get_dir(file_ops.RESULTS_DIR_FP / "fits")
        # find maximum index in results_summary and create fp
        self.get_run_id()
        fits_fp = file_ops.get_f(fits_dir_fp / f"fit_results_{self.run_id}.csv")
        if self.save_fits:
            # save to csv if specified
            self.fit_results.to_csv(fits_fp, index=False)
            
    def generate_results_summary(self):
        # TODO: arrangement of method?
        # calculate results summary
        metrics_summary = spectrum_utils.generate_results_summary(self.error_metrics)
        
        # generate metadata
        self.generate_run_metadata()
        self.generate_glob_summary()
        self.generate_cfg_summary()
        if hasattr(self, 'e'):
            self.generate_error_summary()
        else:
            self.error_df = None
        # concatenate run summaries        
        self.results_summary = pd.concat([self.metadata_df, self.glob_summary, self.cfg_df, metrics_summary, self.error_df], axis=1)
        self.save_results_summary()
    
    def generate_cfg_summary(self):
        cfg_df = pd.DataFrame([self.cfg.__dict__])
        # create dataframe with run parameters and summary metrics
        multiindex_columns = pd.MultiIndex.from_product([['configuration'], cfg_df.columns])
        cfg_df.columns = multiindex_columns
        self.cfg_df = cfg_df
        
    def generate_run_metadata(self):
        # return date and time as multiindex df with metadata on first, header on second
        datetime = pd.Timestamp.now('UTC')
        # overkill for now, but may add further metadata in future
        multiindex = pd.MultiIndex.from_product([['metadata'], ['datetime (UTC)']])
        self.metadata_df = pd.DataFrame([[datetime]], columns=multiindex)
        
    def generate_glob_summary(self):
        # select keys of glob_cfg to include in summary
        glob_summary = {k: v for k, v in self.gcfg.__dict__.items() if k not in ["endmember_map", "endmember_schema"]}
        # create dataframe
        glob_summary_df = pd.DataFrame([glob_summary])
        # create multiindex columns
        multiindex_columns = pd.MultiIndex.from_product([['global_configuration'], glob_summary_df.columns])
        glob_summary_df.columns = multiindex_columns
        self.glob_summary = glob_summary_df
        
    def get_run_id(self):
        # get run id (maximum index in results_summary.csv)
        try:
            summary_csv = pd.read_csv(file_ops.get_f(file_ops.RESULTS_DIR_FP / "results_summary.csv"))
            self.run_id = summary_csv.index.max()
        except Exception:
            self.run_id = 1

    def save_results_summary(self):
        # save resulting fits to file
        results_dir_fp = file_ops.get_dir(file_ops.RESULTS_DIR_FP)
        # create results csv if doesn't already exist
        results_fp = results_dir_fp / "results_summary.csv"
        # write header if new file
        if not results_fp.exists():
            self.results_summary.to_csv(results_fp, index=False)
        else:
            # append results_summary to results_csv file
            self.results_summary.to_csv(results_fp, mode='a', header=False, index=False)
    
    def generate_error_summary(self):
        error_message = f"Error in {self.e_step}: {str(self.e)}\n{traceback.format_exc()}"
        logging.error(error_message)
        self.get_run_id()
        # Append the error message to the results summary
        error_df = pd.DataFrame([[self.run_id, self.e_step, error_message]], columns=['run_id', 'step', 'error_message'])
        error_df.columns = pd.MultiIndex.from_product([['error'], error_df.columns])
        self.error_df = error_df
        

    def run(self):
        """
        Runs the optimisation pipeline
        """
        pipeline_steps = [
            ('load_spectra', self.load_spectra),
            ('preprocess_spectra', self.preprocess_spectra),
            ('load_aop_model', self.load_aop_model),
            ('load_spectral_library', self.load_spectral_library),
            ('preprocess_spectral_library', self.preprocess_spectral_library),
            ('characterise_endmembers', self.characterise_endmembers),
            ('fit_spectra', self.fit_spectra),
            ('generate_spectra_from_fits', self.generate_spectra_from_fits),
            ('calculate_error_metrics', self.calculate_error_metrics)
        ]
        # # execute pipeline and catch errors
        # for step_name, step_method in pipeline_steps:
        
        #     step_method()
        #     print('step_name', step_name) # useful for debugging since logging otherwise hides until end

        # # generate results (these steps not guarded by error catcher intentionally)
        # self.generate_results_summary()
        # self.generate_fit_results()

        # execute pipeline and catch errors
        for step_name, step_method in pipeline_steps:
            try:
                step_method()
            except Exception as e:
                print('e', e) # useful for debugging since logging otherwise hides until end
                print('step_name', step_name) # useful for debugging since logging otherwise hides until end
                self.e = e
                self.e_step = step_name
        try:
            # generate results (these steps not guarded by error catcher intentionally)
            self.generate_results_summary()
            self.generate_fit_results()
            print(self.cfg)

        except Exception as e:
            print('e', e)
            print(self.cfg)
        

def run_pipeline(glob_cfg, run_cfgs):
    """
    Run the optimisation pipeline for a range of parameterisation schema
    """
    results = []
    glob_cfg = GlobalOptPipeConfig(glob_cfg)
    for cfg in tqdm(run_cfgs):
        run_cfg = RunOptPipeConfig(cfg)
        opt_pipe = OptPipe(glob_cfg, run_cfg)
        opt_pipe.run()


# optionally run as script
if __name__ == "__main__":
    # load config_dict
    config_dict = file_ops.read_yaml(file_ops.CONFIG_FP)
    run_pipeline(config_dict)
    
    
    

#### DEPRECATED ####
 
    # def generate_spectra_from_endmembers(self):
    #     for i, row in tqdm(self.fitted_params.iterrows(), total=len(self.fitted_params)):
    #         bb_m, bb_c, Kd_m, Kd_c = self.aop_args
    #         bb, K, H = row.values[:3]
    #         pred = spectrum_utils.sub_surface_reflectance_Rb(prism_spectra.columns, endmember_array, bb, K, H, AOP_args, *row.values[3:-2])

## been replaced by case switching
        # if self.cfg.endmember_type == "mean":
        #     self.endmembers = spectrum_utils.mean_endmembers(self.spectral_library)
        # elif self.cfg.endmember_type == "median":
        #     self.endmembers = spectrum_utils.median_endmembers(self.spectral_library)
        # elif self.cfg.endmember_type in ["pca", "nmf", "ica", "svd"]:
        #     self.endmembers = spectrum_utils.calculate_endmembers(self.spectral_library, self.cfg.endmember_type)
        # else:
        #     raise ValueError(f"Endmember type {self.cfg.endmember_type} not recognised")

## been replaced by error-protected pipeline
    # def run(self):
    #     """
    #     Runs the optimisation pipeline
    #     """
    #     # load spectra
    #     self.load_spectra()
    #     # preprocess observed spectra, normalising if specified
    #     self.preprocess_spectra()
    #     # load AOP model parameters
    #     self.load_aop_model()
    #     # load endmembers
    #     self.load_spectral_library()
    #     # preprocess endmembers
    #     self.preprocess_spectral_library()
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
        