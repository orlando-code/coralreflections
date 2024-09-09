from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

def Class():
    """
    Class to run the optimisation pipeline
    """

    def __init__(self, **kwargs):
        """Initialises class attributes"""
        self.spectrum = kwargs


    def preprocess_spectrum(self):
        """Correct for glint, calculate subsurface reflectance, crop to sensor range"""
        continue
        
    def normalise_spectrum(self):
        """Normalises the spectrum using specified scaler"""
        scaler = self.instantiate_scaler()
        self.spectrum = scaler.fit_transform
        # Normalise the spectrum
        continue
    
    def instantiate_scaler(self):
        """Instantiate scaler"""
        if self.scaler is "zscore":
            self.scaler = StandardScaler()
        elif self.scaler is "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaler is "robust":
            self.scaler = RobustScaler()
        elif self.scaler is "maxabs":
            self.scaler = MaxAbsScaler()
        else:
            raise ValueError("Scaler not recognised")
            
        return self.scaler
    
    def load_aop_model(self):
        """Load the AOP model dependent on the specified group"""
        continue

    def do_pca(self):
        """Return the specified number of principal components"""
        continue
    
    def generate_endmembers(self):
        """Generate endmembers using specified method"""
        continue
            
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
        
        continue