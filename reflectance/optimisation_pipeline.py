from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from dataclasses import dataclass
from tqdm.auto import tqdm


# custom
from reflectance import spectrum_utils, file_ops


@dataclass
class OptPipeConfig:
    # TODO: is this just for typing purposes?
    spectra_fp: str
    spectral_library_fp: str
    aop_group_num: int
    nir_wavelengths: tuple
    sensor_range: tuple
    scaler_type: str
    endmember_type: str
    endmember_map: dict
    
    def __init__(self, conf: dict):
        self.spectra_fp = conf["global_configurations"]["spectra_fp"]
        self.spectral_library_fp = conf["global_configurations"]["spectral_library_fp"]
        self.aop_group_num = conf["configurations"]["aop_group_num"]
        self.nir_wavelengths = conf["configurations"]["nir_wavelengths"]
        self.sensor_range = conf["configurations"]["sensor_range"]
        self.scaler_type = conf["configurations"]["scaler_type"]
        self.endmember_type = conf["configurations"]["endmember_type"]
        self.endmember_map = conf["endmember_map"]
    
    
class OptPipe():
    """
    Class to run the optimisation pipeline for a range of parameterisation schema
    """

    def __init__(self, config: OptPipeConfig):
        """Initialises class attributes"""
        self.gcfg = config["global_configurations"]
        self.cfg = config["configurations"]
        self.endmember_map = config["endmember_map"]

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
        self.spectra = spectra_subsurface.loc[:, crop_inds]
        
    def preprocess_spectral_library(self):
        """Go from raw spectral library to analysis-ready endmembers i.e. crop to relevant wavelengths"""
        crop_inds = (self.spectral_library.columns > min(self.cfg.sensor_range)) & (self.spectral_library.columns < max(self.cfg.sensor_range))
        self.spectral_library = self.spectral_library.loc[:, crop_inds]
    
    def convert_classes(self):
        """Map validation data classes to endmember classes"""
        pass
    
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
    
    def normalise_spectra(self):
        """Normalises the spectrum using specified scaler"""
        scaler = self.instantiate_scaler()
        self.norm_spectra = scaler.fit_transform(self.spectra)

    def instantiate_scaler(self):
        """Instantiate scaler"""
        if self.cfg.scaler_type == "zscore":
            self.scaler = StandardScaler()
        elif self.cfg.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif self.cfg.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif self.cfg.scaler_type == "maxabs":
            self.scaler = MaxAbsScaler()
        else:
            raise ValueError("Scaler not recognised")
            
        return self.scaler
    
    def load_spectra(self):
        """Load spectra"""
        self.raw_spectra = spectrum_utils.load_spectra(self.gcfg.spectra_fp)
    
    def load_aop_model(self):
        """Load the AOP model dependent on the specified group"""
        self.aop_model = spectrum_utils.load_aop_model(self.cfg.aop_group_num)
        
    def load_spectral_library(self):
        """Load spectral library"""
        self.spectral_library = spectrum_utils.load_spectral_library(self.gcfg.spectral_library_fp)
            
    def run(self):
        """
        Runs the optimisation pipeline
        """
        # create info directory
        # load spectra
        self.load_spectra()
        # preprocess observed spectra to fit
        self.preprocess_spectra()
        # normalise spectra using specified scaler
        self.normalise_spectra()
        # load AOP model parameters
        self.load_aop_model()
        # load endmembers and preprocess
        self.load_spectral_library()
        self.preprocess_spectral_library()
        # generate specified endmember parameterisation
        self.characterise_endmembers()
        # fit normalised spectra using specified objective function
        
        # calculate error metrics and append to results
        # save results
        # calculate resulting spectra
        # save resulting spectra
    


    
def run_pipeline(configurations):
    """
    Run the optimisation pipeline for a range of parameterisation schema
    """
    results = []
    for conf in tqdm(configurations["configurations"]):
        config = OptPipeConfig(conf)
        opt_pipe = OptPipe(config)
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