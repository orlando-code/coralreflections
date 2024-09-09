# general
import numpy as np
import pandas as pd

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


def load_aop_model(aop_group_num: int = 1) -> pd.DataFrame:
    """Load AOP model for specified group number"""
    f_AOP_model = resource_dir / f'AOP_models_Group_{group_num}.txt'
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


### PREPROCESSING
def retrieve_subsurface_reflectance(spectra: pd.DataFrame) -> pd.DataFrame:
    """Retrieve subsurface reflectance (formula from Lee et al. 1998)"""
    return spectra_deglinted / (0.518 + 1.562 * spectra_deglinted)


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


def Rb_endmember(end_member_array, *X):
    """Return linear combination of spectrum, weighted by X vector"""
    return end_member_array.T.dot(X)


def sub_surface_reflectance_Rb(wv, end_member_array, bb, K, H, AOD_args, *Rb_args):
    bb_m, bb_c, Kd_m, Kd_c = AOD_args
    Rb = Rb_endmember(end_member_array, *Rb_args)
    return sub_surface_reflectance(wv, bb, K, H, Rb, bb_m, bb_c, Kd_m, Kd_c)


### FITTING
def _wrapper(i, of, prism_spectra, AOD_args, end_member_array,  Rb_init: float=0.1, end_member_bounds: tuple = (0, np.inf)):
    """
    Wrapper function for minimisation of objective function.
    
    Parameters:
    - i (int): Index of spectrum to fit.
    - of (function): Objective function to minimise.
    - prism_spectra (pd.DataFrame): DataFrame of observed spectra.
    - AOD_args (tuple): Tuple of backscatter and attenuation coefficients.
    - end_member_array (np.ndarray): Array of end member spectra.
    - Rb_init (float): Initial value for Rb.
    - end_member_bounds (tuple): Bounds for end member values.
    
    Returns:
    - np.ndarray: Fitted parameters.
    """
    fit = minimize(of,
            # initial parameter values
            x0=[0.1, 0.1, 0] + [Rb_init] * len(end_member_array),
            # extra arguments passsed to the object function (and its derivatives)
            args=(prism_spectra.loc[i], # spectrum to fit (obs)
                  *AOD_args,    # backscatter and attenuation coefficients (bb_m, bb_c, Kd_m, Kd_c)
                  end_member_array  # typical end-member spectra
                  ),
            # constrain values
            bounds=[(0, 0.41123), (0.01688, 3.17231), (0, 50)] + [end_member_bounds] * len(end_member_array)) # may not always want to constrain this (e.g. for PCs)
    return fit.x


##### OBJECTIVE FUNCTIONS
def spectral_angle_objective_fn(x, obs, bb_m, bb_c, Kd_m, Kd_c, end_member_array):
    bb, K, H = x[:3]
    Rb_values = x[3:]
    Rb = Rb_endmember(end_member_array, *Rb_values)
    pred = sub_surface_reflectance(1, bb, K, H, Rb, bb_m, bb_c, Kd_m, Kd_c)
    return spectral_angle(pred, obs)


def r2_objective_fn(x, obs, bb_m, bb_c, Kd_m, Kd_c, end_member_array):
    bb, K, H = x[:3]
    Rb_values = x[3:]
    Rb = Rb_endmember(end_member_array, *Rb_values)
    pred = sub_surface_reflectance(1, bb, K, H, Rb, bb_m, bb_c, Kd_m, Kd_c)
    ssq = np.sum((obs - pred)**2)
    penalty = np.sum(np.array(Rb_values)**2)
    penalty_scale = ssq / max(penalty.max(), 1)  # doesn't this just remove the Rb penalty?
    return ssq + penalty_scale * penalty


def weighted_spectral_angle_objective_fn(x, obs, bb_m, bb_c, Kd_m, Kd_c, end_member_array):
    bb, K, H = x[:3]
    Rb_values = x[3:]
    Rb = Rb_endmember(end_member_array, *Rb_values)
    pred = sub_surface_reflectance(1, bb, K, H, Rb, bb_m, bb_c, Kd_m, Kd_c)
    # calculate rolling spectral angle between predicted and observed spectra
    spectral_angle_corrs = spectral_angle_correlation(Rb)
    # weight the regions of the spectra by the spectral angle correlation
    return -spectral_angle_corrs * spectral_angle(pred, obs)
    

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


# def r2_objective_fn(x, obs, bb_m, bb_c, Kd_m, Kd_c, end_member_array):
    # bb, K, H, Rb0, Rb1, Rb2, Rb3, Rb4, Rb5, Rb6, Rb7, Rb8, Rb9, Rb10 = x
    # Rb = Rb_endmember(end_member_array, Rb0, Rb1, Rb2, Rb3, Rb4, Rb5, Rb6, Rb7, Rb8, Rb9, Rb10)
    # pred = sub_surface_reflectance(1, bb, K, H, Rb, bb_m, bb_c, Kd_m, Kd_c)
    
    # ssq = np.sum((obs - pred)**2)
    # penalty = np.sum(np.array([Rb0, Rb1, Rb2, Rb3, Rb4, Rb5, Rb6, Rb7, Rb8, Rb9, Rb10])**2)
    # penalty_scale = ssq / max(penalty.max(), 1)  # doesn't this just remove the Rb penalty?
    # return ssq + penalty_scale * penalty
    

# def r2_objective_fn_4(x, obs, bb_m, bb_c, Kd_m, Kd_c, end_member_array):
#     bb, K, H, Rb0, Rb1, Rb2, Rb3 = x
#     Rb = Rb_endmember(end_member_array, Rb0, Rb1, Rb2, Rb3)
#     pred = sub_surface_reflectance(1, bb, K, H, Rb, bb_m, bb_c, Kd_m, Kd_c)
    
#     ssq = np.sum((obs - pred)**2)
#     penalty = np.sum(np.array([Rb0, Rb1, Rb2, Rb3])**2)
#     penalty_scale = ssq / max(penalty.max(), 1)  # doesn't this just remove the Rb penalty?
#     return ssq + penalty_scale * penalty


# def wrapper_with_args(i):
#     return _wrapper(i, prism_spectra, AOD_args)


# # old attempts at weighting
# w = end_member_array.std(axis=0)
# w[0] = 1

# w = 0.5 * np.exp(0.01 * (wv - 450))
# w = 1 + 4 *  stats.norm.cdf(wv, loc=580, scale=20)
