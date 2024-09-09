from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from dataclasses import dataclass


# custom
from reflectance import spectrum_utils, file_ops

def OptPipe():
    """
    Class to run the optimisation pipeline for a range of parameterisation schema
    """

    def __init__(self, config: OptPipeConfig):
        """Initialises class attributes"""
        self.raw_spectra = None

    def preprocess_spectrum(self):
        """Correct for glint, calculate subsurface reflectance, crop to sensor range"""
        # fetch indices within NIR wavelength range
        glint_inds = (self.raw_spectra.columns > min(self.nir_wavelengths)) & (self.raw_spectra.columns < max(self.nir_wavelengths))
        spectra_deglinted = self.raw_spectra.subtract(self.raw_spectra.loc[:, glint_inds].mean(axis=1), axis=0)
        # subsurface reflectance
        spectra_subsurface = spectra_deglinted / (0.518 + 1.562 * spectra_deglinted)
        # crop to sensor range
        crop_inds = (spectra_corrected.columns > min(self.sensor_range)) & (spectra_corrected.columns < max(self.sensor_range))
        # return corrected and cropped spectra for investigation
        self.spectra = spectra_subsurface.loc[:, crop_inds]
        
    def normalise_spectrum(self):
        """Normalises the spectrum using specified scaler"""
        scaler = self.instantiate_scaler()
        self.norm_spectra = scaler.fit_transform(self.spectra)
        # Normalise the spectrum
        pass
    
    def instantiate_scaler(self):
        """Instantiate scaler"""
        if self.scaler_type == "zscore":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif self.scaler_type == "maxabs":
            self.scaler = MaxAbsScaler()
        else:
            raise ValueError("Scaler not recognised")
            
        return self.scaler
    
    def load_aop_model(self):
        """Load the AOP model dependent on the specified group"""
        self.aop_model = spectrum_utils.read_aop_model(self.aop_group_num)

    def do_pca(self):
        """Return the specified number of principal components"""
        pass
    
    def generate_endmembers(self):
        """Generate endmembers using specified method"""
        pass
            
    def run(self):
        """
        Runs the optimisation pipeline
        """
        # create info directory
        # preprocess observed spectra to fit
        # normalise spectra using specified scaler
        # load AOP model parameters
        # load endmembers and generate specified endmember parameterisation
        # fit normalised spectra using specified objective function
        
        # calculate error metrics and append to results
        # save results
        # calculate resulting spectra
        # save resulting spectra
        
        pass
    

@dataclass
class OptPipeConfig:
    aop_group_num: int
    scaler_type: str
    nir_wavelengths: tuple
    sensor_range: tuple
    
    def __init__(self, conf: dict):
        self.aop_group_num = conf["aop_group_num"]
        self.scaler_type = conf["scaler_type"]
        self.nir_wavelengths = conf["nir_wavelengths"]
        self.sensor_range = conf["sensor_range"]
    
    