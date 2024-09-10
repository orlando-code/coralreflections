# general
from dataclasses import dataclass
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

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
    endmember_map: dict
    # results_dir_fp: str
    
    def __init__(self, conf: dict):
        self.spectra_fp = conf["spectra_fp"]
        self.spectral_library_fp = conf["spectral_library_fp"]
        self.validation_data_fp = conf["validation_data_fp"]
        self.endmember_map = conf["endmember_map"]
        # self.results_dir_fp = conf["results_dir_fp"]
        

@dataclass
class RunOptPipeConfig:
    aop_group_num: int
    nir_wavelengths: tuple
    sensor_range: tuple
    endmember_type: str
    endmember_normalisation: str | bool
    spectra_normalisation: str | bool
    objective_fn: str
    
    
    def __init__(self, conf: dict):
        self.aop_group_num = conf["aop_group_num"]
        self.nir_wavelengths = conf["nir_wavelengths"]
        self.sensor_range = conf["sensor_range"]
        self.endmember_type = conf["endmember_type"]
        self.endmember_normalisation = conf["endmember_normalisation"]
        self.spectra_normalisation = conf["spectra_normalisation"]
        self.objective_fn = conf["objective_fn"]
    
    
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
        glint_inds = (self.raw_spectra.columns > min(self.cfg.nir_wavelengths)) & (self.raw_spectra.columns < max(self.cfg.nir_wavelengths))
        spectra_deglinted = self.raw_spectra.subtract(self.raw_spectra.loc[:, glint_inds].mean(axis=1), axis=0)
        # subsurface reflectance
        spectra_subsurface = spectra_deglinted / (0.518 + 1.562 * spectra_deglinted)
        # crop to sensor range
        crop_inds = (spectra_subsurface.columns > min(self.cfg.sensor_range)) & (spectra_subsurface.columns < max(self.cfg.sensor_range))
        # return corrected and cropped spectra for investigation
        self.spectra = spectra_subsurface.loc[:10, crop_inds]   # TEMP RESTRICTION FOR TESTING
        
        if self.cfg.spectra_normalisation:
            self.spectra = spectrum_utils.normalise_spectra(self.spectra, self.cfg.spectra_normalisation)

    def preprocess_spectral_library(self):
        """Go from raw spectral library to analysis-ready endmembers i.e. crop to relevant wavelengths"""
        crop_inds = (self.spectral_library.columns > min(self.cfg.sensor_range)) & (self.spectral_library.columns < max(self.cfg.sensor_range))
        self.spectral_library = self.spectral_library.loc[:, crop_inds]
    
    def convert_classes(self):
        """Map validation data classes to endmember classes"""
        endmember_map = self.gcfg.endmember_map
        mapped_df = pd.DataFrame(index=target_df.index, columns=endmember_map)

        for end_member_type, validation_fields in endmember_map.items():
            # fill in validation data with sum of all fields in the category
            mapped_df.loc[:, end_member_type] = target_df.loc[:, validation_fields].sum(axis=1) 
                
        self.validation_df = mapped_df
    
    def characterise_endmembers(self):
        """Characterise endmembers using specified method"""
        if self.cfg.endmember_type == "mean":
            self.endmembers = spectrum_utils.mean_endmembers(self.spectral_library)
        elif self.cfg.endmember_type == "pca":
            self.endmembers = spectrum_utils.do_pca(self.spectral_library)
        elif self.cfg.endmember_type == "truncated_svd":
            self.endmembers = spectrum_utils.do_trun_svd(self.spectral_library)
        else:
            raise ValueError(f"Endmember characterisation schema {self.cfg.endmember_type} not recognised")
        
        # if specified, normalise endmembers
        if self.cfg.endmember_normalisation:
            self.endmembers = self.normalise_spectra(self.spectra)
            
        self.endmember_array = np.array([spec.values for spec in self.endmembers.values()])

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
        if self.cfg.objective_fn == "r2":
            return spectrum_utils.r2_objective_fn
        elif self.cfg.objective_fn == "spectral_angle":
            return spectrum_utils.spectral_angle_objective_fn
        elif self.cfg.objective_fn == "spectral_angle_w1":
            return spectrum_utils.spectral_angle_objective_fn_w1
        else:
            raise ValueError(f"Objective function {self.cfg.objective_fn} not recognised")
        
    def fit_spectra(self):
        # create wrapper for function to allow parallel processing
        of = self.return_objective_fn()
        partial_wrapper = partial(
            spectrum_utils._wrapper, 
            of=of, 
            prism_spectra=self.spectra, 
            AOP_args=self.aop_args,
            endmember_array=self.endmember_array, 
            Rb_init=0.0001)
        
        with mp.Pool() as pool:
            fitted_params = list(tqdm(pool.imap(partial_wrapper, self.spectra.index), total=len(self.spectra.index)))
        self.fitted_params = pd.DataFrame(fitted_params, index=self.spectra.index, columns=['bb', 'K', 'H'] + list(list(self.endmembers.keys())))
   
    def calculate_error_metrics(self):
        """Reconstructing spectra from the fitted parameters"""
        self.metrics = spectrum_utils.calculate_metrics(self.spectra, self.fitted_params)
        
    def generate_results_df(self):
        """Generate a dataframe with a multiindex: run parameters, true spectra, fit results, error metrics"""
        self.results = spectrum_utils.generate_results_df(self.cfg, self.spectra, self.fitted_params, self.metrics)
        
    def generate_spectra_from_fits(self):
        self.fitted_spectra = spectrum_utils.generate_spectra_from_fits(self.fitted_params, self.spectra.columns, self.endmember_array, self.aop_args)
    
    def calculate_error_metrics(self):
        self.error_metrics = spectrum_utils.calculate_metrics(self.spectra, self.fitted_spectra)
        # save error metrics to file
        
    def generate_fit_results(self):
        # combine fitted_params with fitted_spectra and metrics
        self.fit_results = spectrum_utils.generate_fit_results(self.fitted_params, self.fitted_spectra, self.error_metrics)
        # generate filename from summary_results index
        fits_dir_fp = file_ops.get_dir(file_ops.RESULTS_DIR_FP / "fits")
        # find maximum index in results_summary and create fp
        self.get_run_id()
        fits_fp = file_ops.get_f(fits_dir_fp / f"fit_results_{self.run_id}.csv")
        # save csv
        self.fit_results.to_csv(fits_fp)
            
    def generate_results_summary(self):
        # calculate results summary
        metrics_summary = spectrum_utils.generate_results_summary(self.error_metrics)
        
        cfg_df = pd.DataFrame([self.cfg.__dict__])
        # create dataframe with run parameters and summary metrics
        multiindex_columns = pd.MultiIndex.from_product([['configuration'], cfg_df.columns])
        cfg_df.columns = multiindex_columns
        
        # concatenate configuration and metrics summary        
        self.results_summary = pd.concat([cfg_df, metrics_summary], axis=1)
        
    def get_run_id(self):
        # get run id (maximum index in results_summary.csv)
        summary_csv = pd.read_csv(file_ops.get_f(file_ops.RESULTS_DIR_FP / "results_summary.csv"))
        self.run_id = summary_csv.index.max()
        
    def save_results_summary(self):
        # save resulting fits to file
        results_dir_fp = file_ops.get_dir(file_ops.RESULTS_DIR_FP)
        # create results csv if doesn't already exist
        results_fp = file_ops.get_f(results_dir_fp / "results_summary.csv")
        # write header if new file
        if not results_fp.exists():
            self.results_summary.to_csv(results_fp, index=False)
        # append results_summary to results_csv
        self.results_summary.to_csv(results_fp, mode='a', header=False)
        
    def save_fits(self):
        # save results summary to next row in csv
        pass
        
    def run(self):
        """
        Runs the optimisation pipeline
        """
        # load spectra
        self.load_spectra()
        # preprocess observed spectra, normalising if specified
        self.preprocess_spectra()
        # load AOP model parameters
        self.load_aop_model()
        # load endmembers
        self.load_spectral_library()
        # preprocess endmembers
        self.preprocess_spectral_library()
        # generate specified endmember parameterisation, normalising if specified
        self.characterise_endmembers()
        # fit normalised spectra using specified objective function
        self.fit_spectra()
        # calculate resulting spectra
        self.generate_spectra_from_fits()
        # calculate error metrics and append to results
        self.calculate_error_metrics()
        # calculate summary of results
        self.generate_results_summary()
        # save resulting fit summary to file
        self.save_results_summary()
        # save resulting spectra and metrics to file
        self.generate_fit_results()
        # self.save_results()
        

def run_pipeline(configurations):
    """
    Run the optimisation pipeline for a range of parameterisation schema
    """
    results = []
    glob_cfg = GlobalOptPipeConfig(configurations["glob_cfg"])
    for conf in tqdm(configurations["cfgs"]):
        run_cfg = RunOptPipeConfig(conf)
        opt_pipe = OptPipe(glob_cfg, run_cfg)
        print("ended")
    #     opt_pipe.run()
        
    #     # append results to list
    #     results.append(opt_pipe.results)
        
    # results_df = pd.DataFrame(results)
    # results_df.to_csv("results.csv")
    

if __name__ == "__main__":
    # load configurations
    configurations = file_ops.read_yaml("configurations.yaml")
    run_pipeline(configurations)
    
    
    

#### DEPRECATED ####
 
    # def generate_spectra_from_endmembers(self):
    #     for i, row in tqdm(self.fitted_params.iterrows(), total=len(self.fitted_params)):
    #         bb_m, bb_c, Kd_m, Kd_c = self.aop_args
    #         bb, K, H = row.values[:3]
    #         pred = spectrum_utils.sub_surface_reflectance_Rb(prism_spectra.columns, endmember_array, bb, K, H, AOP_args, *row.values[3:-2])
