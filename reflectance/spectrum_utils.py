# general
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import asdict

# stats
from sklearn.decomposition import PCA, TruncatedSVD, NMF, FastICA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import r2_score

# fitting
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import UnivariateSpline


### LOADING
# import resource data
try:
    from importlib import resources
    resource_dir = resources.files('reflectance') / 'resources'
except:
    from pathlib import Path
    resource_dir = Path(__file__).resolve().parent / 'resources'


def load_spectral_library(fp: Path="reflectance/resources/spectral_library_clean_v3_PRISM_wavebands.csv") -> pd.DataFrame:
    df = pd.read_csv(fp, skiprows=1)
    # correct column naming
    df.rename(columns={'wavelength': 'class'}, inplace=True)
    df.set_index('class', inplace=True)
    df.columns = df.columns.astype(float)
    return df.astype(float)


def load_spectra(fp: Path="data/CORAL_validation_spectra.csv") -> pd.DataFrame:
    """Load spectra from file"""
    spectra = pd.read_csv(fp)
    spectra.columns = spectra.columns.astype(float)
    return spectra


def load_aop_model(aop_group_num: int = 1) -> pd.DataFrame:
    """Load AOP model for specified group number
    N.B. Group 3 file was modified to remove a duplicate row of column headers
    """
    f_AOP_model = resource_dir / f'AOP_models_Group_{aop_group_num}.txt'
    with open(f_AOP_model, 'r') as f:
        start_found = False
        skiprows = 0
        while not start_found:
            line = f.readline()
            if line.startswith('wl,'):
                start_found = True
            else:
                skiprows += 1

    # read in wavelengths as df
    AOP_model = pd.read_csv(f_AOP_model, skiprows=skiprows - 1).set_index('wl')
    AOP_model.columns = ['Kd_m', 'Kd_c', 'bb_m', 'bb_c']
    return AOP_model


### PREPROCESSING
def deglint_spectra(spectra, nir_wavelengths: list[float]=None) -> pd.DataFrame:
    glint_inds = (spectra.columns > min(nir_wavelengths)) & (spectra.columns < max(nir_wavelengths))
    return spectra.subtract(spectra.loc[:, glint_inds].mean(axis=1), axis=0)


def retrieve_subsurface_reflectance(spectra: pd.DataFrame, constant: float=0.518, coeff: float=1.562) -> pd.DataFrame:
    """Retrieve subsurface reflectance (formula from Lee et al. 1998)"""
    return spectra_deglinted / (constant + coeff * spectra_deglinted)


def crop_spectra_to_range(spectra: pd.DataFrame, wv_range: tuple) -> pd.DataFrame:
    """Crop spectra to specified wavelength range"""
    return spectra.loc[:, (spectra.columns > min(wv_range)) & (spectra.columns < max(wv_range))]


### PHYISCAL CALCULATIONS
def sub_surface_reflectance(wv, bb, K, H, Rb, bb_m, bb_c, Kd_m, Kd_c):
    """Radiative transfer model for sub-surface reflectance.
    bb_lambda and K_lambda are calculated as a function of wavelength using the AOP model.
    Characterised by (fixed) coefficient and intercept from AOP model, with a scaling factor 
    set during optimisation.
    Types: arrays/pd.Series
    """
    bb_lambda = bb * bb_m + bb_c
    K_lambda = 2 * K * Kd_m + Kd_c
    return bb_lambda / K_lambda + (Rb - bb_lambda / K_lambda) * np.exp(-K_lambda * H)


def Rb_endmember(endmember_array, *X):
    """Return linear combination of spectrum, weighted by X vector"""
    return endmember_array.T.dot(X)


def sub_surface_reflectance_Rb(wv, endmember_array, bb, K, H, AOP_args, *Rb_args):
    bb_m, bb_c, Kd_m, Kd_c = AOP_args
    Rb = Rb_endmember(endmember_array, *Rb_args)
    return sub_surface_reflectance(wv, bb, K, H, Rb, bb_m, bb_c, Kd_m, Kd_c)


### FITTING
def _wrapper(
    i, of, prism_spectra, AOP_args, endmember_array,  Rb_init: float=0.0001, 
    bb_bounds: tuple = (0, 0.41123), Kd_bounds: tuple = (0.01688, 3.17231), H_bounds: tuple = (0, 50),
    end_member_bounds: tuple = (0, np.inf)):
    """
    Wrapper function for minimisation of objective function.
    
    Parameters:
    - i (int): Index of spectrum to fit.
    - of (function): Objective function to minimise.
    - prism_spectra (pd.DataFrame): DataFrame of observed spectra.
    - AOP_args (tuple): Tuple of backscatter and attenuation coefficients.
    - endmember_array (np.ndarray): Array of end member spectra.
    - Rb_init (float): Initial value for Rb: can't be 0 since spectral angle is undefined.
    - bb_bounds (tuple): Bounds for bb values.
    - Kd_bounds (tuple): Bounds for Kd values.
    - H_bounds (tuple): Bounds for H values.
    - end_member_bounds (tuple): Bounds for end member values.
    
    Returns:
    - np.ndarray: Fitted parameters.
    """
    fit = minimize(of,
            # initial parameter values
            x0=[0.1, 0.1, 0] + [Rb_init] * len(endmember_array),
            # extra arguments passsed to the object function (and its derivatives)
            args=(prism_spectra.loc[i], # spectrum to fit (obs)
                  *AOP_args,    # backscatter and attenuation coefficients (bb_m, bb_c, Kd_m, Kd_c)
                  endmember_array  # typical end-member spectra
                  ),
            # constrain values
            bounds=[bb_bounds, Kd_bounds, H_bounds] + [end_member_bounds] * len(endmember_array)) # may not always want to constrain this (e.g. for PCs)
    return fit.x


##### OBJECTIVE FUNCTIONS
def spectral_angle_objective_fn(x, obs, bb_m, bb_c, Kd_m, Kd_c, endmember_array):
    bb, K, H = x[:3]
    Rb_values = x[3:]
    Rb = Rb_endmember(endmember_array, *Rb_values)
    pred = sub_surface_reflectance(1, bb, K, H, Rb, bb_m, bb_c, Kd_m, Kd_c)
    return spectral_angle(pred, obs)


def r2_objective_fn(x, obs, bb_m, bb_c, Kd_m, Kd_c, endmember_array):
    bb, K, H = x[:3]
    Rb_values = x[3:]
    Rb = Rb_endmember(endmember_array, *Rb_values)
    pred = sub_surface_reflectance(1, bb, K, H, Rb, bb_m, bb_c, Kd_m, Kd_c)
    ssq = np.sum((obs - pred)**2)
    penalty = np.sum(np.array(Rb_values)**2)
    penalty_scale = ssq / max(penalty.max(), 1)  # doesn't this just remove the Rb penalty?
    return ssq + penalty_scale * penalty


def spectral_angle_objective_fn_w1(x, obs, bb_m, bb_c, Kd_m, Kd_c, endmember_array):
    bb, K, H = x[:3]
    bb, K, H, *Rb_values = x[:3]
    Rb_values = x[3:]
    Rb = Rb_endmember(endmember_array, *Rb_values)
    pred = sub_surface_reflectance(1, bb, K, H, Rb, bb_m, bb_c, Kd_m, Kd_c)
    # calculate rolling spectral angle between predicted and observed spectra
    spectral_angle_corrs = spectral_angle_correlation(Rb)
    # weight the regions of the spectra by the spectral angle correlation
    return -spectral_angle_corrs * spectral_angle(pred, obs)
    
    
# Define the helper function for generating spectra
def generate_spectrum(fitted_params, wvs: pd.Series, endmember_array: np.ndarray, AOP_args: tuple) -> pd.Series:
    bb_m, bb_c, Kd_m, Kd_c = AOP_args
    bb, K, H = fitted_params.values[:3]
    return sub_surface_reflectance_Rb(wvs, endmember_array, bb, K, H, AOP_args, *fitted_params.values[3:])


def generate_spectra_from_fits(fits: pd.DataFrame, wvs: pd.Series, endmember_array: np.ndarray, AOP_args: tuple) -> pd.DataFrame:
    spectra = fits.apply(
        generate_spectrum, 
        axis=1,
        args=(wvs, endmember_array, AOP_args)
    )
    spectra_df = pd.DataFrame(spectra.tolist(), index=fits.index, columns=wvs)
    return spectra_df


def calculate_metrics(observed_spectra: pd.DataFrame, fitted_spectra: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the r2 and spectral angle between observed and fitted spectra and return as a DataFrame
    """
    def calculate_row_metrics(row, observed_spectra, fitted_spectra):
        observed = observed_spectra.loc[row.name]
        fitted = fitted_spectra.loc[row.name]
        r2 = r2_score(observed, fitted)
        sa = spectral_angle(observed.values, fitted.values)
        return pd.Series({'r2': r2, 'spectral_angle': sa})
    
    metrics = observed_spectra.apply(
        calculate_row_metrics, 
        axis=1, 
        args=(observed_spectra, fitted_spectra)
    )
    return metrics


def generate_fit_results(fitted_params: pd.DataFrame, fitted_spectra: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Combine fitted parameters, spectra, and metrics into a single multiindex DataFrame"""
    multiindex_tuples = [('fitted_params', param) for param in fitted_params.columns] \
        + [('fitted_spectra', wl) for wl in fitted_spectra.columns] \
        + [('metrics', metric) for metric in metrics.columns]
    multiindex = pd.MultiIndex.from_tuples(multiindex_tuples)
    df = pd.DataFrame(columns=multiindex)
    
    df.loc[:, ('fitted_params', slice(None))] = fitted_params.values
    df.loc[:, ('fitted_spectra', slice(None))] = fitted_spectra.values
    df.loc[:, ('metrics', slice(None))] = metrics.values
    
    return df
    

def generate_results_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics from metrics DataFrame with a MultiIndex"""
    
    # Define first and second levels of the MultiIndex
    metrics_list = metrics.columns  # First level
    stats_list = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', '95%', 'max']   # Second level
    desc = metrics.describe().T
    
    multiindex = pd.MultiIndex.from_product([metrics_list, stats_list])
    summary = pd.DataFrame(index=[0], columns=multiindex)

    # calculate summary statistics for each metric
    for metric in metrics.columns:
        for stat in desc.columns:
            summary[(metric, stat)] = desc.loc[metric, stat]
        # additional custom metrics
        summary[(metric, '95%')] = 1.96 * metrics[metric].std()
    
    return summary

    
def generate_results_df(configuration: dict, observed_spectra: pd.DataFrame, fitted_spectra: pd.DataFrame, error_metrics: pd.DataFrame):
    """
    Generate a DataFrame of results from the configuration, observed spectra, fitted spectra, and error metrics.
    """
    multiindex_tuples = [('true_spectra', wl) for wl in observed_spectra.columns] \
        + [('fitted_spectra', wl) for wl in fitted_spectra.columns] \
        + [('error_metrics', metric) for metric in error_metrics.columns]
        # [('run_parameters', param) for param in asdict(configuration).keys()] \
    multiindex = pd.MultiIndex.from_tuples(multiindex_tuples)
    df = pd.DataFrame(columns=multiindex)
    
    # df.loc[:, ('run_parameters', slice(None))] = configuration
    df.loc[:, ('true_spectra', slice(None))] = observed_spectra.values
    df.loc[:, ('fitted_spectra', slice(None))] = fitted_spectra.values
    df.loc[:, ('error_metrics', slice(None))] = error_metrics.values
    
    return df  


### SPECTRAL ANGLE
def spectral_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Compute spectral angle between two spectra."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_theta = dot_product / (norm_a * norm_b)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip values to avoid numerical issues


def spectral_angle_correlation(spectra: np.ndarray) -> float:
    """Return a summary statistic for the similarity between a number of spectra"""
    matrix = spectral_angle_correlation_matrix(spectra)
    # calculate mean of upper triangle of matrix
    return np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)]), np.std(matrix[np.triu_indices(matrix.shape[0], k=1)])
    
    
def spectral_angle_correlation_matrix(spectra: np.ndarray) -> np.ndarray:
    """Compute the correlation matrix using spectral angle for an array of spectra."""
    dot_product_matrix = np.dot(spectra, spectra.T)
    norms = np.linalg.norm(spectra, axis=1)
    norm_matrix = np.outer(norms, norms)
    cos_theta_matrix = dot_product_matrix / norm_matrix
    
    # Clip values to the valid range of arccos to handle numerical issues
    cos_theta_matrix = np.clip(cos_theta_matrix, -1.0, 1.0)
    return np.arccos(cos_theta_matrix)


def calc_rolling_spectral_angle(wvs, spectra, wv_kernel_width, wv_kernel_displacement):
    """
    Calculate the rolling spectral angle between a spectrum and a set of end members.

    This function calculates the rolling spectral angle between a given spectrum and a set of end members
    using a specified kernel width and displacement. It returns the wavelength pairs and mean angles used
    in the calculation.

    Parameters:
    - wvs (np.ndarray): Array of wavelengths for the spectrum.
    - spectra (np.ndarray): Array of spectra for the end members.
    - wv_kernel_width (float | int): The width of the kernel used for calculating the rolling correlation.
    - wv_kernel_displacement (float | int): The displacement of the kernel for each step in the rolling correlation calculation.
    
    Returns:
    - wv_pairs (list of tuples): List of wavelength pairs used for each kernel.
    - mean_corrs (list of float): List of mean spectral angles for each kernel.
    """
    wv_pairs = [(wv, wv+wv_kernel_width) for wv in np.arange(wvs.min(), wvs.max()-wv_kernel_width, wv_kernel_displacement)]

    # calculate rolling spectral angles
    mean_corrs = []
    for wv_pair in wv_pairs:
        ids = (wvs > min(wv_pair)) & (wvs < max(wv_pair))
        mean_angle, _ = spectral_angle_correlation(spectra[:, ~ids])
        mean_corrs.append(mean_angle)

    return wv_pairs, mean_corrs


def instantiate_scaler(scaler_type: str):
    """Instantiate scaler"""
    match scaler_type:
        case "zscore":
            return StandardScaler()
        case "minmax":
            return MinMaxScaler()
        case "robust":
            return RobustScaler()
        case "maxabs":
            return MaxAbsScaler()
        case _:
            raise ValueError(f"Scaler_type {scaler_type} not recognised")
 
    
def normalise_spectra(spectra: pd.DataFrame, scaler_type: str) -> pd.DataFrame:
    scaler = instantiate_scaler(scaler_type)
    return pd.DataFrame(scaler.fit_transform(spectra.T).T, index=spectra.index, columns=spectra.columns)


### END MEMBER CHARACTERISATION
def mean_endmembers(spectral_library_df: pd.DataFrame, classes: list[str]=None) -> pd.DataFrame:
    """Calculate mean endmembers from spectral library."""
    return spectral_library_df.groupby('class').mean()

def median_endmembers(spectral_library_df: pd.DataFrame, classes: list[str]=None) -> pd.DataFrame:
    """Calculate median endmembers from spectral library."""
    return spectral_library_df.groupby('class').median()


def instantiate_decomposer(method: str, n_components: int):
    """Instantiate decomposition method"""
    
    match method:
        case 'pca':
            return PCA(n_components=n_components)
        case 'svd':
            return TruncatedSVD(n_components=n_components)
        case 'nmf':
            return NMF(n_components=n_components, init='random', random_state=0)
        case 'ica':
            return FastICA(n_components=n_components, random_state=0)
        case 'kpca':    # doesn't have 'components', but could be useful for focusing on different areas
            return KernelPCA(n_components=n_components, kernel='linear')
        case 'lda': # not doing a classification
            return LDA(n_components=n_components)
        case _:
            raise ValueError(f"Method {method} not recognised. Use 'pca', 'svd', 'nmf', 'ica', or 'kpca'.")


def group_classes(spectra: pd.DataFrame, map_dict: dict) -> pd.DataFrame:
    grouped_spectra = spectra.copy()
    category_to_group = {category: group for group, categories in map_dict.items() for category in categories}
    grouped_spectra.index = grouped_spectra.index.map(category_to_group)
    return grouped_spectra


def calculate_endmembers(spectral_library: pd.DataFrame, method: str='pca', n_components: int=3) -> pd.DataFrame:
    """Calculate endmembers from spectral library using specified decomposition method."""
    decomposer = instantiate_decomposer(method, n_components)
    components = {}

    for class_name, spectra in spectral_library.groupby('class'):
        decomposer.fit(spectra)
        for i in range(n_components):    
            components[(class_name, f'{method.upper()}_{i+1}')] = decomposer.components_[i]

    # Convert the dictionary to a DataFrame
    components_df = pd.DataFrame(components)
    components_df.columns = pd.MultiIndex.from_tuples(components_df.columns)
    components_df.index = spectral_library.columns
    components_df.index.name = 'wavelength'
    return components_df.T

### DEPRECATED ###
# # been surpassed by function for minimisation
# def sub_surface_reflectance(wv, bb, K, H, Rb):
#     sub = AOP_model.loc[wv]
#     bb_lambda = bb * sub.loc[wv, 'bb_m'] + sub.loc[wv, 'bb_c']
#     K_lambda = 2 * K * sub.loc[wv, 'Kd_m'] + sub.loc[wv, 'Kd_c']
#     return bb_lambda / K_lambda + (Rb - bb_lambda / K_lambda) * np.exp(-K_lambda * H)


# No longer using splines because...
# load splines
# Kd_splines = {}
# bb_splines = {}
# for f in resource_dir.glob('*.pkl'):
#     name = f.stem.split('_')[1]
#     with open(f, 'rb') as file:
#         if 'Kd' in f.stem:
#             Kd_splines[name] = pickle.load(file)
#         else:
#             bb_splines[name] = pickle.load(file)


# def sub_surface_reflectance(wv, bb, K, H, Rb, option='Group1'):
#     bb_lambda = bb * bb_splines[option](wv)
#     K_lambda = K * Kd_splines[option](wv)
    
#     return bb_lambda / K_lambda + (Rb - bb_lambda / K_lambda) * np.exp(-K_lambda * H)


# def r2_objective_fn(x, obs, bb_m, bb_c, Kd_m, Kd_c, endmember_array):
    # bb, K, H, Rb0, Rb1, Rb2, Rb3, Rb4, Rb5, Rb6, Rb7, Rb8, Rb9, Rb10 = x
    # Rb = Rb_endmember(endmember_array, Rb0, Rb1, Rb2, Rb3, Rb4, Rb5, Rb6, Rb7, Rb8, Rb9, Rb10)
    # pred = sub_surface_reflectance(1, bb, K, H, Rb, bb_m, bb_c, Kd_m, Kd_c)
    
    # ssq = np.sum((obs - pred)**2)
    # penalty = np.sum(np.array([Rb0, Rb1, Rb2, Rb3, Rb4, Rb5, Rb6, Rb7, Rb8, Rb9, Rb10])**2)
    # penalty_scale = ssq / max(penalty.max(), 1)  # doesn't this just remove the Rb penalty?
    # return ssq + penalty_scale * penalty
    

# def r2_objective_fn_4(x, obs, bb_m, bb_c, Kd_m, Kd_c, endmember_array):
#     bb, K, H, Rb0, Rb1, Rb2, Rb3 = x
#     Rb = Rb_endmember(endmember_array, Rb0, Rb1, Rb2, Rb3)
#     pred = sub_surface_reflectance(1, bb, K, H, Rb, bb_m, bb_c, Kd_m, Kd_c)
    
#     ssq = np.sum((obs - pred)**2)
#     penalty = np.sum(np.array([Rb0, Rb1, Rb2, Rb3])**2)
#     penalty_scale = ssq / max(penalty.max(), 1)  # doesn't this just remove the Rb penalty?
#     return ssq + penalty_scale * penalty


# # read in first AOP model (arbitrary choice), the functions looked less crazy than G2. Didn't look at G3.
# f_AOP_model = resource_dir / 'AOP_models_Group_1.txt'
# with open(f_AOP_model, 'r') as f:
#     start_found = False
#     skiprows = 0
#     while not start_found:
#         line = f.readline()
#         if line.startswith('wl,'):
#             start_found = True
#         else:
#             skiprows += 1

# # read in wavelengths as df
# AOP_model = pd.read_csv(f_AOP_model, skiprows=skiprows - 1).set_index('wl')
# AOP_model.columns = ['Kd_m', 'Kd_c', 'bb_m', 'bb_c']


# def wrapper_with_args(i):
#     return _wrapper(i, prism_spectra, AOP_args)


### originally mean_endmembers
    # if classes is None:
#     classes = spectral_library_df.index.unique()
    
# endmembers = {}
# for cat in classes:
#     ind = spectral_library_df.index == cat
#     endmembers[cat] = spectral_library_df.loc[ind].mean(axis=0)

# return endmembers
    
    
# def pca_endmembers(spectral_library: pd.DataFrame, classes: list[str]=None, n_components: int=3) -> dict:
#     """Calculate PCA endmembers from spectral library."""
#     if classes is None:
#         classes = spectral_library.index.unique()
        
#     pca = PCA(n_components=n_components)
#     pca.fit(spectral_library.T)
#     endmembers = {}
#     for i in range(n_components):
#         endmembers[f'PCA_{i}'] = pca.components_[i]
    
#     return endmembers


    
# # old attempts at weighting
# w = endmember_array.std(axis=0)
# w[0] = 1

# w = 0.5 * np.exp(0.01 * (wv - 450))
# w = 1 + 4 *  stats.norm.cdf(wv, loc=580, scale=20)


