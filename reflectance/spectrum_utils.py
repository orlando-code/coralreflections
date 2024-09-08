# general
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker

# fitting
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import UnivariateSpline


# import resource data
try:
    from importlib import resources
    resource_dir = resources.files('reflectance') / 'resources'
except:
    from pathlib import Path
    resource_dir = Path(__file__).resolve().parent / 'resources'


# read in first AOP model (arbitrary choice), the functions looked less crazy than G2. Didn't look at G3.
f_AOP_model = resource_dir / 'AOP_models_Group_1.txt'
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


# # old attempts at weighting
# w = end_member_array.std(axis=0)
# w[0] = 1

# w = 0.5 * np.exp(0.01 * (wv - 450))
# w = 1 + 4 *  stats.norm.cdf(wv, loc=580, scale=20)

def wrapper_with_args(i):
    return _wrapper(i, prism_spectra, AOD_args)


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


def _wrapper(i, of, prism_spectra, AOD_args, end_member_array,  Rb_init: float=0.1, end_member_bounds: tuple = (0, np.inf)):
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


def plot_spline_fits(smoothing_factors: list[float], spectrum: pd.Series, zoom_wvs: tuple[float, float]):
    """
    Plot spline fits for a given spectrum with various smoothing factors.

    Parameters:
    - smoothing_factors (list[float]): List of smoothing factors to be used for spline fitting.
    - spectrum (pd.Series): The spectrum data to be fitted, with the index representing wavelengths and values representing intensities.
    - zoom_wvs (tuple[float, float]): Tuple specifying the wavelength range to zoom in on for the zoomed plot.

    Returns:
    - None
    """
    fig = plt.figure(figsize=(14, len(smoothing_factors)*3))
    # one more plot than smoothing to also plot the original spectrum
    gs = GridSpec(len(smoothing_factors)+1, 5, figure=fig)
    fitted_ax = fig.add_subplot(gs[0, 0:4])
    fitted_ax.plot(spectrum.index, spectrum.values, label="spectrum", c='grey', zorder=-2)
    fitted_ax.grid(axis="x")

    zoom_fitted_ax = fig.add_subplot(gs[0, 4], sharey=fitted_ax)
    zoom_fitted_ax.plot(spectrum.index, spectrum.values, c='grey', zorder=-2)
    # formatting
    zoom_fitted_ax.set_xlim(*zoom_wvs)
    plt.setp(zoom_fitted_ax.get_yticklabels(), visible=False)
    zoom_fitted_ax.text(0.1, 0.1, f"zoomed spectrum", 
                        transform=zoom_fitted_ax.transAxes, 
                        fontsize=12, 
                        verticalalignment='bottom', 
                        horizontalalignment='left')

    for j, sf in enumerate(smoothing_factors):
        spline = UnivariateSpline(spectrum.index, spectrum.values, s=sf)
        # plot spline fit
        fitted_ax.plot(spectrum.index, spline(spectrum.index), label=f"spline fit, s={sf}", alpha=1, linestyle='--')
        # plot zoomed spline fit
        zoom_fitted_ax.plot(spectrum.index, spline(spectrum.index), label=f"spline fit, s={sf}", alpha=1, linestyle='--')
        
        # plot spectral residuals
        spectrum_ax = fig.add_subplot(gs[j+1, 0:4], sharex=fitted_ax)
        residuals = spectrum.values - spline(spectrum.index)
        spectrum_ax.scatter(spectrum.index, residuals, label=f"s={sf} residuals", s=3)
        spectrum_ax.hlines(0, spectrum.index.min(), spectrum.index.max(), color='r', linestyle='--', zorder=-2)
        # formatting
        spectrum_ax.set_xlim(spectrum.index.min(), spectrum.index.max())
        spectrum_ax.grid(axis="x")
        spectrum_ax.legend(loc="upper right")
        spectrum_ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        spectrum_ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        
        # plot histograms of residuals
        hist_ax = fig.add_subplot(gs[j+1, 4])
        counts, bins, _ = hist_ax.hist(residuals, bins=20, orientation='horizontal')
        hist_ax.hlines(0, 0, max(counts*1.1), color='r', linestyle='--')
        #formatting
        hist_ax.set_xlim(min(1.1*counts), max(1.1*counts))
        hist_ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        hist_ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # tidying up
    for k, ax in enumerate(fig.get_axes()):
        if ax != spectrum_ax and ax != zoom_fitted_ax and k%2 == 0:
            plt.setp(ax.get_xticklabels(), visible=False)
    spectrum_ax.set_xlabel("wavelength (nm)")
            
    fitted_ax.legend(loc="upper right");
    plt.tight_layout()



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


def visualise_rolling_spectral_correlation(endmembers, wv_kernel_width, wv_kernel_displacement):
    """
    Visualize the rolling spectral correlation for given end members.

    This function plots the spectra of the end members and their rolling spectral correlation
    using a specified kernel width and displacement. It also returns the wavelength pairs and
    mean correlations used in the calculation.

    Parameters:
    - endmembers (dict): Dictionary of end member spectra, where keys are category names and values are pandas Series with wavelengths as index and reflectance values as data.
    - wv_kernel_width (int): The width of the kernel used for calculating the rolling correlation.
    - wv_kernel_displacement (int): The displacement of the kernel for each step in the rolling correlation calculation.

    Returns:
    - wv_pairs (list of tuples): List of wavelength pairs used for each kernel.
    - mean_corrs (list of float): List of mean spectral angle correlations for each kernel.
    """
    f, ax_spectra = plt.subplots(1, figsize=(12, 6))
    ax_correlation = ax_spectra.twinx()

    # extract wavelengths from index of endmember dictionary's first entry
    wvs = next(iter(endmembers.values())).index
    end_member_spectra = np.array([spectrum.values for spectrum in endmembers.values()])
    wv_pairs, mean_corrs = calc_rolling_spectral_angle(wvs, end_member_spectra, wv_kernel_width, wv_kernel_displacement)
    x_coords = [np.mean(wv_pair) for wv_pair in wv_pairs]

    # plot endmember spectra
    for cat, spectrum in endmembers.items():
        ax_spectra.plot(wvs, endmembers[cat], label=cat, alpha=0.4)

    # plot horizontal error bars, width kenrel_width
    ax_correlation.errorbar(x_coords, mean_corrs, xerr=wv_kernel_width/2, fmt='x', color='k', alpha=0.5, label="horizontal bars = kernel span")
    ax_correlation.legend()
    
    # formatting
    ax_spectra.legend(bbox_to_anchor=(1.1, 0.5), title="End members")
    ax_spectra.grid('major', axis='x')
    ax_spectra.set_ylabel("Reflectance")
    ax_spectra.set_xlabel('Wavelength (nm)')
    ax_spectra.set_xlim(wvs.min(), wvs.max())

    ax_correlation.set_ylabel("Mean spectral angle correlation:\nLow is more correlated")
    ax_correlation.grid('major', axis='y');


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