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




def r2_objective_fn(x, obs, bb_m, bb_c, Kd_m, Kd_c, end_member_array):
    bb, K, H = x[:3]
    Rb_values = x[3:]
    Rb = Rb_endmember(end_member_array, *Rb_values)
    pred = sub_surface_reflectance(1, bb, K, H, Rb, bb_m, bb_c, Kd_m, Kd_c)
    ssq = np.sum((obs - pred)**2)
    penalty = np.sum(np.array(Rb_values)**2)
    penalty_scale = ssq / max(penalty.max(), 1)  # doesn't this just remove the Rb penalty?
    return ssq + penalty_scale * penalty


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


def _wrapper(i, prism_spectra, AOD_args, end_member_array, end_member_bounds: tuple = (0, np.inf)):
    fit = minimize(r2_objective_fn,
            # initial parameter values
            x0=[0.1, 0.1, 0] + [0] * len(end_member_array),
            # extra arguments passsed to the object function (and its derivatives)
            args=(prism_spectra.loc[i], # spectrum to fit (obs)
                  *AOD_args,    # backscatter and attenuation coefficients (bb_m, bb_c, Kd_m, Kd_c)
                  end_member_array  # typical end-member spectra
                  ),
            # constrain values
            bounds=[(0, 0.41123), (0.01688, 3.17231), (0, 50)] + [end_member_bounds] * len(end_member_array)) # may not always want to constrain this (e.g. for PCs)
    return fit.x


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


def visualise_rolling_spectral_correlation(end_members, kernel_width, kernel_displacement):
    f, ax_spectra = plt.subplots(1, figsize=(12, 6))
    ax_correlation = ax_spectra.twinx()

    choice_array = np.array([spectrum.values for spectrum in end_members.values()])

    # plot endmember spectra
    for cat, spectrum in end_members.items():
        ax_spectra.plot(spectrum.index, end_members[cat], label=cat, alpha=0.4)

    wv_pairs = [(wv, wv+2*kernel_width) for wv in np.arange(spectrum.index.min(), spectrum.index.max(), kernel_displacement)]
    x_coords = [np.mean(wv_pair) for wv_pair in wv_pairs]
    print(x_coords)
    # plot kernel correlations
    mean_corrs = []
    for wv_pair in wv_pairs:
        print(wv_pair)
        ids = (spectrum.index > min(wv_pair)) & (spectrum.index < max(wv_pair))
        mean, _ = spectral_angle_correlation(choice_array[:, ids])
        mean_corrs.append(mean)

    # ax_correlation.scatter(x_coords, mean_corrs, color='k', s=10, marker='x')
    # plot horizontal error bars, width kenrel_width
    ax_correlation.errorbar(x_coords, mean_corrs, xerr=kernel_width/2, fmt='x', color='k', alpha=0.5, label="horizontal bars = kernel span")
    ax_correlation.legend()
    
    # formatting
    ax_spectra.legend(bbox_to_anchor=(1.1, 0.7), title="End members")
    ax_spectra.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax_spectra.grid('major', axis='x')
    ax_spectra.set_ylabel("Reflectance")
    ax_spectra.set_xlabel('Wavelength (nm)')
    ax_spectra.set_xlim(spectrum.index.min(), spectrum.index.max())

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