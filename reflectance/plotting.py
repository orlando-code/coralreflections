# general
import numpy as np
import pandas as pd

# fitting
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
import plotly.graph_objs as go
import matplotlib.colors as mcolors

# metrics
from sklearn.metrics import r2_score

# custom
from reflectance import spectrum_utils


def format_axis_for_ppt(ax):
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    [ax.spines[pos].set_color("white") for pos in ax.spines.keys()]
    # ax.grid(True, color='white', ls='--', linewidth=0.5, alpha=0.5, which='both')
    return ax


class SpectralColour:
    """Define colour of spectra"""

    def __init__(
        self,
        blue_peak: float = 492.4,
        green_peak: float = 559.8,
        red_peak: float = 664.6,
        blue_width: float = 66,
        green_width: float = 36,
        red_width: float = 31,
    ):
        self.blue_peak = blue_peak
        self.green_peak = green_peak
        self.red_peak = red_peak
        self.blue_width = blue_width
        self.green_width = green_width
        self.red_width = red_width

    def generate_wv_lims(self):
        self.blue_wvs = spectrum_utils.range_from_centre_and_width(
            self.blue_peak, self.blue_width
        )
        self.green_wvs = spectrum_utils.range_from_centre_and_width(
            self.green_peak, self.green_width
        )
        self.red_wvs = spectrum_utils.range_from_centre_and_width(
            self.red_peak, self.red_width
        )
        return self.blue_wvs, self.green_wvs, self.red_wvs


def generate_colors_from_spectra(
    wvs: np.ndarray, spectra: np.ndarray, percentiles: tuple[float, float] = (1, 99)
):
    """Spectra in format N_samples, N_wavelengths"""
    blue_wvs, green_wvs, red_wvs = SpectralColour().generate_wv_lims()

    # generate spectra
    rgb_values = np.array(
        [
            spectrum_utils.rgb_from_hyperspectral(wvs, s, red_wvs, green_wvs, blue_wvs)
            for s in spectra
        ]
    )
    reds, greens, blues = rgb_values[:, 0], rgb_values[:, 1], rgb_values[:, 2]
    # normalise (assumption to scene)
    percentiles = [
        np.percentile(reds, [min(percentiles), max(percentiles)]),
        np.percentile(greens, [min(percentiles), max(percentiles)]),
        np.percentile(blues, [min(percentiles), max(percentiles)]),
    ]
    norm_reds, norm_greens, norm_blues = [
        (channel - p[0]) / (p[1] - p[0])
        for channel, p in zip([reds, greens, blues], percentiles)
    ]

    # min-max normalization
    maxes = np.max([norm_reds, norm_greens, norm_blues], axis=1)
    mins = np.min([norm_reds, norm_greens, norm_blues], axis=1)
    return (np.array([norm_reds, norm_greens, norm_blues]).T - mins) / (maxes - mins)


def plot_spline_fits(
    smoothing_factors: list[float], spectrum: pd.Series, zoom_wvs: tuple[float, float]
):
    """
    Plot spline fits for a given spectrum with various smoothing factors.

    Parameters:
    - smoothing_factors (list[float]): List of smoothing factors to be used for spline fitting.
    - spectrum (pd.Series): The spectrum data to be fitted, with the index representing wavelengths and values
        representing intensities.
    - zoom_wvs (tuple[float, float]): Tuple specifying the wavelength range to zoom in on for the zoomed plot.

    Returns:
    - None
    """
    fig = plt.figure(figsize=(14, len(smoothing_factors) * 3))
    # one more plot than smoothing to also plot the original spectrum
    gs = GridSpec(len(smoothing_factors) + 1, 5, figure=fig)
    fitted_ax = fig.add_subplot(gs[0, 0:4])
    fitted_ax.plot(
        spectrum.index, spectrum.values, label="spectrum", c="grey", zorder=-2
    )
    fitted_ax.grid(axis="x")

    zoom_fitted_ax = fig.add_subplot(gs[0, 4], sharey=fitted_ax)
    zoom_fitted_ax.plot(spectrum.index, spectrum.values, c="grey", zorder=-2)
    # formatting
    zoom_fitted_ax.set_xlim(*zoom_wvs)
    plt.setp(zoom_fitted_ax.get_yticklabels(), visible=False)
    zoom_fitted_ax.text(
        0.1,
        0.1,
        "zoomed spectrum",
        transform=zoom_fitted_ax.transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="left",
    )

    for j, sf in enumerate(smoothing_factors):
        spline = UnivariateSpline(spectrum.index, spectrum.values, s=sf)
        # plot spline fit
        fitted_ax.plot(
            spectrum.index,
            spline(spectrum.index),
            label=f"spline fit, s={sf}",
            alpha=1,
            linestyle="--",
        )
        # plot zoomed spline fit
        zoom_fitted_ax.plot(
            spectrum.index,
            spline(spectrum.index),
            label=f"spline fit, s={sf}",
            alpha=1,
            linestyle="--",
        )

        # plot spectral residuals
        spectrum_ax = fig.add_subplot(gs[j + 1, 0:4], sharex=fitted_ax)
        residuals = spectrum.values - spline(spectrum.index)
        spectrum_ax.scatter(spectrum.index, residuals, label=f"s={sf} residuals", s=3)
        spectrum_ax.hlines(
            0,
            spectrum.index.min(),
            spectrum.index.max(),
            color="r",
            linestyle="--",
            zorder=-2,
        )
        # formatting
        spectrum_ax.set_xlim(spectrum.index.min(), spectrum.index.max())
        spectrum_ax.grid(axis="x")
        spectrum_ax.legend(loc="upper right")
        spectrum_ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        spectrum_ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        # plot histograms of residuals
        hist_ax = fig.add_subplot(gs[j + 1, 4])
        counts, bins, _ = hist_ax.hist(residuals, bins=20, orientation="horizontal")
        hist_ax.hlines(0, 0, max(counts * 1.1), color="r", linestyle="--")
        # formatting
        hist_ax.set_xlim(min(1.1 * counts), max(1.1 * counts))
        hist_ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        hist_ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # tidying up
    for k, ax in enumerate(fig.get_axes()):
        if ax != spectrum_ax and ax != zoom_fitted_ax and k % 2 == 0:
            plt.setp(ax.get_xticklabels(), visible=False)
    spectrum_ax.set_xlabel("wavelength (nm)")

    fitted_ax.legend(loc="upper right")
    plt.tight_layout()


def plot_rolling_spectral_similarity(wv_pairs, mean_corrs, wvs, comp_spectra):
    """
    Visualize the rolling spectral correlation for given end members.
    """
    f, ax_spectra = plt.subplots(1, figsize=(12, 6))
    ax_correlation = ax_spectra.twinx()

    x_coords = [np.mean(wv_pair) for wv_pair in wv_pairs]

    # plot endmember spectra
    for spectrum in comp_spectra:
        ax_spectra.plot(wvs, spectrum, label="spectrum", alpha=1)

    # plot horizontal error bars, width kenrel_width
    ax_correlation.errorbar(
        x_coords,
        mean_corrs,
        xerr=min(np.diff(wv_pairs)) / 2,
        fmt="x",
        color="k",
        alpha=0.3,
        label="horizontal bars = kernel span",
    )
    ax_correlation.scatter(x_coords, mean_corrs, color="k", alpha=1, marker="x")
    ax_correlation.legend()

    # formatting
    ax_spectra.legend(bbox_to_anchor=(1.25, 0.5), title="End members")
    ax_spectra.grid("major", axis="x")
    ax_spectra.set_ylabel("Reflectance")
    ax_spectra.set_xlabel("Wavelength (nm)")
    ax_spectra.set_xlim(wvs.min(), wvs.max())

    ax_correlation.set_ylabel(
        "Mean spectral angle correlation:\nLow is more correlated"
    )
    ax_correlation.grid("major", axis="y")


def get_color_hex(cmap, value):
    rgba = cmap(value)  # Get RGBA values
    return mcolors.to_hex(rgba)  # Convert RGBA to hex


def plot_interactive_coral_algae_spectrum(coral, algae, n_samples, coralgal_cmap):
    fig = go.Figure()
    wavelengths = coral.columns

    for coral_prop in np.arange(0, 1.01, 0.05):
        algae_prop = 1 - coral_prop
        mean_spectrum_list = [
            np.mean(
                coral.sample(n=100).values * coral_prop
                + algae.sample(n=100).values * algae_prop,
                axis=0,
            )
            for _ in range(n_samples)
        ]
        for mean_spectrum in mean_spectrum_list:
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    x=wavelengths,
                    y=mean_spectrum,
                    mode="lines",
                    line=dict(color=get_color_hex(coralgal_cmap, coral_prop), width=1),
                    opacity=0.3,
                )
            )

    for i in range(n_samples):
        fig.data[i].visible = True

    steps = []
    for j, coral_prop in enumerate(np.arange(0, 1.01, 0.05)):
        step = dict(
            method="update",
            args=[{"visible": [i // n_samples == j for i in range(len(fig.data))]}],
            label=f"{coral_prop:.2f}",
        )
        for i in range(n_samples):
            fig.data[j * n_samples + i].showlegend = False
        steps.append(step)

    fig.update_layout(
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Coral Proportion: "},
                pad={"t": 50},
                steps=steps,
            )
        ],
        xaxis_title="Wavelength (nm)",
        yaxis_title="Reflectance",
        xaxis=dict(range=[400, 700]),
        yaxis=dict(range=[0, 0.2]),
        title="Interactive Coral-Algae Spectrum Blending",
    )

    fig.show()


def plot_single_fit(
    fitted_params, true_spectrum, AOP_args, endmember_array, endmember_cats
):
    """"""
    fig, axs = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(8, 6))

    wvs = true_spectrum.index
    fitted_spectrum = spectrum_utils.generate_spectrum(
        fitted_params, wvs, endmember_array, AOP_args
    )
    axs[0].plot(wvs, true_spectrum, label="spectrum")
    axs[0].plot(wvs, fitted_spectrum, color="red", alpha=0.7, label="fit")
    axs[0].legend(fontsize=8)

    axs[1].plot(
        wvs,
        spectrum_utils.Rb_endmember(
            endmember_array, *fitted_params[3 : 3 + len(endmember_array)]
        ),
        color="k",
        alpha=0.7,
        label="Extracted Rb",
    )

    endmember_contribution = endmember_array * fitted_params[
        3 : 3 + len(endmember_array)
    ].values.reshape(-1, 1)
    # generate colour as a sum of the components
    color_dict = {c: plt.cm.tab20(i) for i, c in enumerate(endmember_cats)}
    y = np.zeros(endmember_contribution.shape[1])
    for label, endmember in zip(endmember_cats, endmember_contribution):
        ynew = y + endmember
        axs[1].fill_between(
            wvs, y, ynew, label=label, lw=0, color=color_dict[label], alpha=0.5
        )
        y = ynew

    # formatting
    axs[1].set_xlim(wvs.min(), wvs.max())
    max_rb_ax = np.max(
        spectrum_utils.Rb_endmember(
            endmember_array, *fitted_params[3 : 3 + len(endmember_array)]
        )
    )
    max_r_ax = axs[0].get_ylim()[1]
    axs[1].set_ylim(
        bottom=np.min(endmember_contribution),
        top=max_r_ax if max_r_ax > max_rb_ax else max_rb_ax * 1.1,
    )  # surely endmember contribution is always less than total reflectance?
    axs[1].legend(bbox_to_anchor=(1, 1), fontsize=8)
    axs[1].set_xlabel("Wavelength (nm)")

    r2 = r2_score(true_spectrum, fitted_spectrum)
    plt.suptitle(
        f"r$^2$: {r2:.4f} | sa: {spectrum_utils.spectral_angle(true_spectrum, fitted_spectrum):.4f}"
    )


def plot_proportions(data: dict, true_ratio: list[float]):
    """Data dict should contain independent variable as keys and endmember contributions as values"""
    # calculate contributions
    means = []
    stds = []
    # std dev of endmember contributions
    endmember_contributions = {}
    for i in data.keys():
        endmember_contributions[i] = dfs[i].values
        means.append(endmember_contributions[i].mean(axis=0))
        stds.append(endmember_contributions[i].std(axis=0))
    means = np.array(means)
    stds = np.array(stds)

    # plot mean and std dev of endmember contributions
    fig, ax = plt.subplots(figsize=(12, 6))
    cs = ["g", "coral", "c"]
    lines = []
    for i, cat in enumerate(list(data.values())[0].columns):
        (line,) = ax.plot(noise_levels, means[:, i], label=cat, color=cs[i])
        lines.append(line)
        ax.fill_between(
            noise_levels,
            means[:, i] - stds[:, i],
            means[:, i] + stds[:, i],
            alpha=0.3,
            zorder=-1,
            color=cs[i],
        )
        ax.hlines(
            true_ratio[i],
            min(dfs.keys()),
            max(dfs.keys()),
            linestyle="--",
            color=cs[i],
            alpha=0.5,
        )

    std_patch = mpatches.Patch(
        color="grey", alpha=0.3, label="±1 Standard Deviation", lw=0
    )

    # Add the legend
    ax.legend(handles=lines + [std_patch])

    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Endmember Contribution")


# DEPRECATED
# def plot_rolling_spectral_similarity(
#     endmembers, wv_kernel_width, wv_kernel_displacement, similarity_fn
# ):
#     """
#     Visualize the rolling spectral correlation for given end members.

#     This function plots the spectra of the end members and their rolling spectral correlation
#     using a specified kernel width and displacement. It also returns the wavelength pairs and
#     mean correlations used in the calculation.

#     Parameters:
#     - endmembers (dict): Dictionary of end member spectra, where keys are category names and values are pandas Series
#         with wavelengths as index and reflectance values as data.
#     - wv_kernel_width (int): The width of the kernel used for calculating the rolling correlation.
#     - wv_kernel_displacement (int): The displacement of the kernel for each step in the rolling correlation calculation.

#     Returns:
#     - wv_pairs (list of tuples): List of wavelength pairs used for each kernel.
#     - mean_corrs (list of float): List of mean spectral angle correlations for each kernel.
#     """
#     f, ax_spectra = plt.subplots(1, figsize=(12, 6))
#     ax_correlation = ax_spectra.twinx()

#     # extract wavelengths from index of endmember dictionary's first entry
#     wvs = next(iter(endmembers.values())).index
#     end_member_spectra = np.array([spectrum.values for spectrum in endmembers.values()])
#     # TODO: should this calculation be within the function?
#     wv_pairs, mean_corrs = spectrum_utils.calc_rolling_similarity(
#         wvs, end_member_spectra, wv_kernel_width, wv_kernel_displacement, similarity_fn
#     )
#     x_coords = [np.mean(wv_pair) for wv_pair in wv_pairs]

#     # plot endmember spectra
#     for cat, spectrum in endmembers.items():
#         ax_spectra.plot(wvs, endmembers[cat], label=cat, alpha=0.4)

#     # plot horizontal error bars, width kenrel_width
#     ax_correlation.errorbar(
#         x_coords,
#         mean_corrs,
#         xerr=wv_kernel_width / 2,
#         fmt="x",
#         color="k",
#         alpha=0.5,
#         label="horizontal bars = kernel span",
#     )
#     ax_correlation.legend()

#     # formatting
#     ax_spectra.legend(bbox_to_anchor=(1.1, 0.5), title="End members")
#     ax_spectra.grid("major", axis="x")
#     ax_spectra.set_ylabel("Reflectance")
#     ax_spectra.set_xlabel("Wavelength (nm)")
#     ax_spectra.set_xlim(wvs.min(), wvs.max())

#     ax_correlation.set_ylabel(
#         "Mean spectral angle correlation:\nLow is more correlated"
#     )
#     ax_correlation.grid("major", axis="y")
