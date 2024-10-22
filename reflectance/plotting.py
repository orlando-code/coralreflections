# general
import numpy as np
import pandas as pd
import warnings

# fitting
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
import plotly.graph_objs as go
import matplotlib.colors as mcolors

# spatial
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xa
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
        nir_peak: float = 865,  # TODO: add nir width
    ):
        self.blue_peak = blue_peak
        self.green_peak = green_peak
        self.red_peak = red_peak
        self.blue_width = blue_width
        self.green_width = green_width
        self.red_width = red_width
        self.nir_peak = nir_peak

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


def generate_spectra_color(
    spectra_df: pd.DataFrame, vis_percentiles: tuple[float] = (1, 99), wvs=None
) -> np.ndarray:
    """
    Generate RGB visualisation of spectra from hyperspectral data.

    Parameters:
    spectra_df: (np.array) - dataframe containing spectra data in format N_samples, N_wavelengths
    vis_percentiles: (tuple) - percentiles for normalisation

    Returns:
    (np.ndarray) - array of RGB values for each spectrum
    """
    blue_wvs, green_wvs, red_wvs = SpectralColour().generate_wv_lims()

    wvs = np.array(spectra_df.columns) if wvs is None else wvs
    spectra = np.array(spectra_df)

    red_mask = (wvs > red_wvs[0]) & (wvs < red_wvs[1])
    green_mask = (wvs > green_wvs[0]) & (wvs < green_wvs[1])
    blue_mask = (wvs > blue_wvs[0]) & (wvs < blue_wvs[1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        red_vals = np.nanmean(spectra[:, red_mask], axis=1)
        green_vals = np.nanmean(spectra[:, green_mask], axis=1)
        blue_vals = np.nanmean(spectra[:, blue_mask], axis=1)
    return np.vstack((red_vals, green_vals, blue_vals)).T


def generate_and_visualise_spectral_colours(
    spectra_df, vis_percentiles=(1, 99), wvs=None
):
    rgb_values = generate_spectra_color(spectra_df, vis_percentiles, wvs)
    return visualise_spectral_colours(rgb_values, vis_percentiles)


def visualise_spectral_colours(
    rgb_values: np.array, vis_percentiles: tuple[float] = (1, 99)
) -> np.ndarray:
    # Calculate percentiles for normalization
    percentiles = np.nanpercentile(rgb_values, vis_percentiles, axis=0)

    # Clip the values to the calculated percentiles
    clipped_rgb_values = np.clip(rgb_values, percentiles[0], percentiles[1])

    # Normalize the clipped values to the range [0, 1]
    norm_rgb_values = (clipped_rgb_values - percentiles[0]) / (
        percentiles[1] - percentiles[0]
    )

    return norm_rgb_values


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


def plot_rolling_spectral_similarity(
    wv_pairs: list[tuple],
    mean_corrs: list[float],
    wvs: np.ndarray,
    comp_spectra: np.ndarray,
    spectra_names: list[str] = None,
):
    """
    Visualize the rolling spectral correlation for given end members.

    Parameters:
    - wv_pairs (list of tuples): List of wavelength pairs used for each kernel.
    - mean_corrs (list of float): List of mean spectral angle correlations for each kernel.
    - wvs (np.ndarray): Array of wavelengths.
    - comp_spectra (list of np.ndarray): List of component spectra to plot.
    """
    f, ax_spectra = plt.subplots(1, figsize=(12, 6))
    ax_correlation = ax_spectra.twinx()

    x_coords = [np.mean(wv_pair) for wv_pair in wv_pairs]

    # plot endmember spectra
    for i, spectrum in enumerate(comp_spectra):
        ax_spectra.plot(
            wvs, spectrum, label=spectra_names[i] if spectra_names else None, alpha=1
        )

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
    (
        ax_spectra.legend(bbox_to_anchor=(1.25, 0.5), title="End members")
        if spectra_names
        else None
    )
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
    fitted_params: np.ndarray | pd.Series,
    true_spectrum: pd.Series,
    AOP_args: tuple[np.ndarray],
    endmember_array: np.ndarray,
    endmember_cats: list[str],
):
    """"""
    fig, axs = plt.subplots(
        2, 1, sharex=True, sharey=True, constrained_layout=True, figsize=(8, 6)
    )

    # generate spectrum from fitted parameters
    wvs = true_spectrum.index
    fitted_spectrum = spectrum_utils.generate_spectrum(
        fitted_params, wvs, endmember_array, AOP_args
    )
    # plot real and fitted spectra
    axs[0].plot(wvs, true_spectrum, label="spectrum")
    axs[0].plot(wvs, fitted_spectrum, color="red", alpha=0.7, label="fit")
    axs[0].legend(bbox_to_anchor=(1.165, 1), fontsize=8)

    # plot endmember contributions
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
    for i, endmember in enumerate(endmember_contribution):
        ynew = np.array(y + np.array(endmember, dtype=np.float32))
        axs[1].fill_between(
            wvs,
            y,
            ynew,
            label=endmember_cats[i],
            lw=0,
            color=color_dict[endmember_cats[i]],
            alpha=0.5,
        )
        y = ynew

    # formatting
    axs[1].set_xlim(wvs.min(), wvs.max())
    axs[1].legend(bbox_to_anchor=(1, 1), fontsize=8)
    axs[1].set_xlabel("Wavelength (nm)")
    [ax.set_ylabel("Reflectance") for ax in axs]
    r2 = r2_score(true_spectrum, fitted_spectrum)
    plt.suptitle(
        f"r$^2$: {r2:.4f} | spectral angle: {spectrum_utils.spectral_angle(true_spectrum, fitted_spectrum):.4f}"
    )


def plot_good_bad_fits(
    ax: plt.Axes,
    spectra: pd.DataFrame,
    fits: pd.DataFrame,
    metadata: pd.DataFrame,
    metric: str = "r2",
    bad_fit_range: tuple[float, float] = [0, 0.1],
    metric_name: str = "r$^2$",
):
    """
    Plot the good and bad fits based on a given metric.

    Parameters:
    - ax (plt.Axes): The axes on which to plot.
    - metric (str): The metric to use for filtering.
    - bad_fit_range (tuple): The range of bad fits.
    - metric_name (str): The name of the metric.
    """
    bad_inds = (metadata[metric] < max(bad_fit_range)) & (
        metadata[metric] > min(bad_fit_range)
    )
    bad_fits = metadata[bad_inds]
    good_fits = metadata[~bad_inds]

    ax.plot(
        spectra.columns,
        spectra.loc[bad_fits.index].values.T,
        color="red",
        lw=0.6,
    )
    ax.plot(
        spectra.columns,
        spectra.loc[good_fits.index].values.T,
        color="k",
        alpha=0.1,
        lw=0.6,
        zorder=-2,
    )
    ax.plot(
        [],
        [],
        color="red",
        lw=0.6,
        label=f"bad fits: {min(bad_fit_range)} < {metric_name} < {max(bad_fit_range)}",
    )
    ax.set_title(
        f"Fitted simulated spectra (Rb + water column effects)\nNumber of bad fits: {bad_fits.shape[0]} (of {len(fits)})"
    )
    ax.set_xlim(spectra.columns.min(), spectra.columns.max())
    ax.legend()
    return ax


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


def initialise_square_plot_grid(
    n_plots, n_cols=3, figsize=(15, 15), constrained_layout: bool = True, dpi: int = 300
):
    if n_plots == 1:
        fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        return fig, axs
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6, n_rows * 3),
        constrained_layout=constrained_layout,
        dpi=dpi,
        sharex=True,
        sharey=True,
    )
    # remove empty axes
    for i in range(n_plots, n_rows * n_cols):
        fig.delaxes(axs.flatten()[i])
    return fig, axs


def plot_regression_axis(
    fig,
    ax,
    test_data: pd.DataFrame,
    pred_data: np.array,
    labels: pd.DataFrame,
    metadata: pd.DataFrame = None,
    color_by: str = "Depth",
):

    if metadata is not None:
        color_map = plt.cm.get_cmap("viridis_r" if color_by == "Depth" else "tab20")
        if color_by == "Locale":
            unique_locales = metadata["Locale"].unique()
            locale_to_color = {
                locale: color_map(i / len(unique_locales))
                for i, locale in enumerate(unique_locales)
            }
            colors = metadata["Locale"].map(locale_to_color)
            scatter = ax.scatter(test_data, pred_data, s=5, alpha=0.3, c=colors)
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=locale_to_color[locale],
                    markersize=5,
                    label=locale,
                )
                for locale in unique_locales
            ]
        elif color_by == "Depth":
            scatter = ax.scatter(
                test_data,
                pred_data,
                s=5,
                alpha=0.3,
                c=metadata["Depth"].values,
                cmap=color_map,
            )
            # if i == 0:
            cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02])
            cbar = fig.colorbar(scatter, cax=cbar_ax, orientation="horizontal")
            cbar.set_label("Depth")
            cbar.ax.tick_params(labelsize=6)

        if "*_std_dev" in metadata.columns:  # if std provided
            # plot error bars
            ax.errorbar(
                test_data,
                pred_data,
                # yerr=metadata["std_dev"].values,
                yerr=metadata[
                    metadata.columns[
                        [labels.columns[0] in col for col in metadata.columns]
                    ]
                ].values.squeeze(),
                fmt="none",
                alpha=0.01,
                color="k",
                zorder=-10,
            )
    else:
        scatter = ax.scatter(test_data, pred_data, s=5, alpha=0.3)

    ax.text(
        0.02,
        0.98,
        labels.columns[0],
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=6,
    )
    ax.axis("square")
    ax.set_xticks(np.arange(0, 1.1, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.5))
    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.plot([0, 1], [0, 1], color="white", ls="--", alpha=0.5)

    if np.sum(pred_data > 0.001):
        xs = np.linspace(0, 1, 100)
        try:
            p = np.polyfit(test_data.squeeze(), pred_data, 1)
            y_est = np.polyval(p, xs)
            ax.plot(xs, y_est, color="r", ls=":", alpha=0.8)
            # if test_data.shape
            y_err = test_data.squeeze().std() * np.sqrt(
                1 / len(xs) + (xs - xs.mean()) ** 2 / np.sum((xs - xs.mean()) ** 2)
            )
            ax.fill_between(xs, y_est - y_err, y_est + y_err, alpha=0.2)
            r2 = r2_score(test_data, pred_data)
            ax.set_title(
                f"$r^2$ = {r2:.2f}\nN = {len(np.nonzero(test_data)[0])}", fontsize=6
            )
            ax.text(
                0.98,
                0.02,
                f"m = {p[0]:.2f}\nc = {p[1]:.2f}",
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                fontsize=6,
            )
        except np.linalg.LinAlgError:
            pass

    if color_by == "Locale" and metadata is not None:
        fig.legend(handles=handles, loc="lower center", fontsize=6, ncol=len(handles))
    return fig, ax


# ML
def plot_regression_results(
    test_data: pd.DataFrame,
    pred_data: np.array,
    labels: pd.DataFrame,
    metadata: pd.DataFrame = None,
    color_by: str = "Depth",
):
    num_plots = test_data.shape[1]
    fig, axs = initialise_square_plot_grid(num_plots)

    xs = np.linspace(0, np.max(test_data), 100)
    color_map = plt.cm.get_cmap("viridis_r" if color_by == "Depth" else "tab20")
    max_val = max(np.max(test_data), np.max(pred_data))

    for i, (endmember, ax) in enumerate(
        zip(labels.columns, axs.flat if num_plots > 1 else axs)
    ):
        if len(labels.columns) > 1:  # multiclass
            pred = pred_data[:, i]
            true = test_data.iloc[:, i]
        else:
            pred = pred_data
            true = test_data.values[:, 0]

        fig, ax = plot_regression_axis(
            fig,
            ax,
            true,
            pred,
            labels.iloc[:, i : i + 1],
            metadata,
            color_by,
        )
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)

    return fig, axs


# PORT FROM SHIFTPY/REEFTRUTH
def get_n_colors_from_hexes(
    num: int,
    hex_list: list[str] = ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#d83c04"],
) -> list[str]:
    """
    from Wes Anderson: https://github.com/karthik/wesanderson/blob/master/R/colors.R
    Get a list of n colors from a list of hex codes.

    Args:
        num (int): The number of colors to return.
        hex_list (list[str]): The list of hex codes from which to create spectrum for sampling.

    Returns:
        list[str]: A list of n hex codes.
    """
    cmap = get_continuous_cmap(hex_list)
    colors = [cmap(i / num) for i in range(num)]
    hex_codes = [mcolors.to_hex(color) for color in colors]
    return hex_codes


class ColourMapGenerator:
    """
    Get a colormap for colorbar based on the specified type.

    Parameters
    ----------
    cbar_type (str, optional): The type of colormap to retrieve. Options are 'seq' for sequential colormap and 'div'
        for diverging colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap: The colormap object.
    """

    def __init__(self):
        self.sequential_hexes = ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#d83c04"]
        self.diverging_hexes = ["#3B9AB2", "#78B7C5", "#FFFFFF", "#E1AF00", "#d83c04"]
        self.cyclical_hexes = [
            "#3B9AB2",
            "#78B7C5",
            "#EBCC2A",
            "#E1AF00",
            "#d83c04",
            "#E1AF00",
            "#EBCC2A",
            "#78B7C5",
            "#3B9AB2",
        ]
        self.conf_mat_hexes = ["#EEEEEE", "#3B9AB2", "#cae7ed", "#d83c04", "#E1AF00"]
        self.residual_hexes = ["#3B9AB2", "#78B7C5", "#fafbfc", "#E1AF00", "#d83c04"]
        self.lim_red_hexes = ["#EBCC2A", "#E1AF00", "#d83c04"]
        self.lim_blue_hexes = ["#3B9AB2", "#78B7C5", "#FFFFFF"]

    def get_cmap(self, cbar_type, vmin=None, vmax=None):
        if cbar_type == "seq":
            return get_continuous_cmap(self.sequential_hexes)
        if cbar_type == "inc":
            return get_continuous_cmap(self.sequential_hexes[2:])
        elif cbar_type == "div":
            if not (vmin and vmax):
                raise ValueError(
                    "Minimum and maximum values needed for divergent colorbar"
                )
            cmap = get_continuous_cmap(self.diverging_hexes)
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            return cmap, norm
            # return get_continuous_cmap(self.diverging_hexes)
        elif cbar_type == "res":
            if not (vmin and vmax):
                raise ValueError(
                    "Minimum and maximum values needed for divergent colorbar"
                )
            cmap = get_continuous_cmap(self.residual_hexes)
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            return cmap, norm
        elif cbar_type == "cyc":
            return get_continuous_cmap(self.cyclical_hexes)
        elif cbar_type == "lim_blue":
            return get_continuous_cmap(self.lim_blue_hexes)
        elif cbar_type == "lim_red":
            return get_continuous_cmap(self.lim_red_hexes)
        elif cbar_type == "spatial_conf_matrix":
            return mcolors.ListedColormap(self.conf_mat_hexes)
        else:
            raise ValueError(f"{cbar_type} not recognised.")


def hex_to_rgb(value):
    """
    Convert a hexadecimal color code to RGB values.

    Parameters
    ----------
    value (str): The hexadecimal color code as a string of 6 characters.

    Returns
    -------
    tuple: A tuple of three RGB values.
    """
    value = value.strip("#")  # removes hash symbol if present
    hex_el = len(value)
    return tuple(
        int(value[i : i + hex_el // 3], 16)  # noqa
        for i in range(0, hex_el, hex_el // 3)
    )


def get_continuous_cmap(hex_list, float_list=None):
    """
    Create and return a color map that can be used in heat map figures.

    Parameters
    ----------
    hex_list (list of str): List of hex code strings representing colors.
    float_list (list of float, optional): List of floats between 0 and 1, same length as hex_list. Must start with 0
        and end with 1.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap: The created color map.
    """
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap("my_cmp", segmentdata=cdict, N=256)
    return cmp


def rgb_to_dec(value):
    """
    Convert RGB color values to decimal values (each value divided by 256).

    Parameters
    ----------
    value (list): A list of three RGB values.

    Returns
    -------
    list: A list of three decimal values.
    """
    return [v / 256 for v in value]


def generate_geo_axis(
    figsize: tuple[float, float] = (10, 10), map_proj=ccrs.PlateCarree(), dpi=300
):
    return plt.figure(figsize=figsize, dpi=dpi), plt.axes(projection=map_proj)


def plot_spatial(
    xa_da: xa.DataArray,
    fax: Axes = None,
    title: str = "default",
    figsize: tuple[float, float] = (10, 10),
    val_lims: tuple[float, float] = None,
    presentation_format: bool = False,
    labels: list[str] = ["l", "b"],
    cbar_dict: dict = None,
    cartopy_dict: dict = None,
    label_style_dict: dict = None,
    map_proj=ccrs.PlateCarree(),
    alpha: float = 1,
    extent: list[float] = None,
) -> tuple[Figure, Axes]:
    """
    Plot a spatial plot with colorbar, coastlines, landmasses, and gridlines.

    Parameters
    ----------
    xa_da (xa.DataArray): The input xarray DataArray representing the spatial data.
    title (str, optional): The title of the plot.
    cbar_name (str, optional): The name of the DataArray.
    val_lims (tuple[float, float], optional): The limits of the colorbar range.
    cmap_type (str, optional): The type of colormap to use.
    symmetric (bool, optional): Whether to make the colorbar symmetric around zero.
    edgecolor (str, optional): The edge color of the landmasses.
    orientation (str, optional): The orientation of the colorbar ('vertical' or 'horizontal').
    labels (list[str], optional): Which gridlines to include, as strings e.g. ["t","r","b","l"]
    map_proj (str, optional): The projection of the map.
    extent (list[float], optional): The extent of the plot as [min_lon, max_lon, min_lat, max_lat].

    Returns
    -------
    tuple: The figure and axes objects.
    TODO: saving option and tidy up presentation formatting
    """
    # may need to change this
    # for some reason fig not including axis ticks. Universal for other plotting
    if not fax:
        if extent == "global":
            fig, ax = generate_geo_axis(
                figsize=figsize,
                map_proj=ccrs.Robinson(central_longitude=180),  # TODO: less hard
            )
            ax.set_global()
        else:
            fig, ax = generate_geo_axis(figsize=figsize, map_proj=map_proj)

    else:
        fig, ax = fax[0], fax[1]

    if isinstance(extent, list):
        ax.set_extent(extent, crs=map_proj)

    default_cbar_dict = {
        "cbar_name": None,
        "cbar": True,
        "orientation": "vertical",
        "cbar_pad": 0.1,
        "cbar_frac": 0.025,
        "cmap_type": "seq",
        "fontsize": 14,
    }

    if cbar_dict:
        for k, v in cbar_dict.items():
            default_cbar_dict[k] = v
        if val_lims:
            default_cbar_dict["extend"] = "both"

    # if not cbarn_name specified, make name of variable
    cbar_name = default_cbar_dict["cbar_name"]
    if isinstance(xa_da, xa.DataArray) and not cbar_name:
        cbar_name = xa_da.name

    # # if title not specified, make title of variable at resolution
    # if title:
    #     if title == "default":
    #         resolution_d = np.mean(utils.calculate_spatial_resolution(xa_da))
    #         resolution_m = np.mean(utils.degrees_to_distances(resolution_d))
    #         title = (
    #             f"{cbar_name} at {resolution_d:.4f}° (~{resolution_m:.0f} m) resolution"
    #         )

    # if colorbar limits not specified, set to be maximum of array
    if not val_lims:  # TODO: allow dynamic specification of only one of min/max
        vmin, vmax = np.nanmin(xa_da.values), np.nanmax(xa_da.values)
    else:
        vmin, vmax = min(val_lims), max(val_lims)

    if (
        default_cbar_dict["cmap_type"] == "div"
        or default_cbar_dict["cmap_type"] == "res"
    ):
        if vmax < 0:
            vmax = 0.01
        cmap, norm = ColourMapGenerator().get_cmap(
            default_cbar_dict["cmap_type"], vmin, vmax
        )
    else:
        try:
            cmap = ColourMapGenerator().get_cmap(default_cbar_dict["cmap_type"])
        except:
            cmap = default_cbar_dict["cmap_type"]

    im = xa_da.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,  # for further formatting later
        transform=ccrs.PlateCarree(),
        alpha=alpha,
        norm=(
            norm
            if (
                default_cbar_dict["cmap_type"] == "div"
                or default_cbar_dict["cmap_type"] == "res"
            )
            else None
        ),
    )
    if presentation_format:
        fig, ax = customize_plot_colors(fig, ax)
        # ax.tick_params(axis="both", which="both", length=0)
    ax.tick_params(labelsize=2, axis="both")

    # nicely format spatial plot
    format_spatial_plot(
        image=im,
        fig=fig,
        ax=ax,
        title=title,
        # cbar_name=cbar_name,
        # cbar=default_cbar_dict["cbar"],
        # orientation=default_cbar_dict["orientation"],
        # cbar_pad=default_cbar_dict["cbar_pad"],
        # cbar_frac=default_cbar_dict["cbar_frac"],
        cartopy_dict=cartopy_dict,
        presentation_format=presentation_format,
        labels=labels,
        cbar_dict=default_cbar_dict,
        label_style_dict=label_style_dict,
    )

    # Add white line around the map
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(0.5)

    return fig, ax, im


def format_cbar(image, fig, ax, cbar_dict, labels: list[str] = ["l", "b"]):

    if cbar_dict["orientation"] == "vertical":
        cbar_rect = [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
        labels = [el if el != "b" else "t" for el in labels or []]
    else:
        cbar_rect = [
            ax.get_position().x0,
            ax.get_position().y0 - 0.04,
            ax.get_position().width,
            0.02,
        ]
        labels = [el if el != "b" else "t" for el in labels or []]
    cax = fig.add_axes(cbar_rect)

    cb = plt.colorbar(
        image,
        orientation=cbar_dict["orientation"],
        label=cbar_dict["cbar_name"],
        cax=cax,
        fraction=cbar_dict["cbar_frac"],
        extend=cbar_dict["extend"] if "extend" in cbar_dict else "neither",
        pad=cbar_dict["cbar_pad"],
    )
    if cbar_dict["orientation"] == "horizontal":
        cbar_ticks = cb.ax.get_xticklabels()
    else:
        cbar_ticks = cb.ax.get_yticklabels()
    cb.set_label(label=cbar_dict["cbar_name"], size=cbar_dict["fontsize"])

    return cb, cbar_ticks, labels


def format_cartopy_display(ax, cartopy_dict: dict = None):

    default_cartopy_dict = {
        "category": "physical",
        "name": "land",
        "scale": "10m",
        "edgecolor": "black",
        # "facecolor": (0, 0, 0, 0),  # "none"
        "facecolor": "white",
        "linewidth": 0.5,
        "alpha": 0.7,
        "labels": True,
    }

    if cartopy_dict:
        for k, v in cartopy_dict.items():
            default_cartopy_dict[k] = v

    ax.add_feature(
        cfeature.NaturalEarthFeature(
            default_cartopy_dict["category"],
            default_cartopy_dict["name"],
            default_cartopy_dict["scale"],
            edgecolor=default_cartopy_dict["edgecolor"],
            facecolor=default_cartopy_dict["facecolor"],
            linewidth=default_cartopy_dict["linewidth"],
            alpha=default_cartopy_dict["alpha"],
        )
    )

    return ax


def format_spatial_plot(
    image: xa.DataArray,
    fig: Figure,
    ax: Axes,
    title: str = None,
    # cbar: bool = True,
    # cmap_type: str = "seq",
    presentation_format: bool = False,
    labels: list[str] = ["l", "b"],
    cbar_dict: dict = None,
    cartopy_dict: dict = None,
    label_style_dict: dict = None,
) -> tuple[Figure, Axes]:
    """Format a spatial plot with a colorbar, title, coastlines and landmasses, and gridlines.

    Parameters
    ----------
        image (xa.DataArray): image data to plot.
        fig (Figure): figure object to plot onto.
        ax (Axes): axes object to plot onto.
        title (str): title of the plot.
        cbar_name (str): label of colorbar.
        cbar (bool): whether to include a colorbar.
        orientation (str): orientation of colorbar.
        cbar_pad (float): padding of colorbar.
        edgecolor (str): color of landmass edges.
        presentation_format (bool): whether to format for presentation.
        labels (list[str]): which gridlines to include, as strings e.g. ["t","r","b","l"]
        label_style_dict (dict): dictionary of label styles.

    Returns
    -------
        Figure, Axes
    """
    if cbar_dict and cbar_dict["cbar"]:
        cb, cbar_ticks, labels = format_cbar(image, fig, ax, cbar_dict, labels)

    ax = format_cartopy_display(ax, cartopy_dict)
    ax.set_title(title)

    # format ticks, gridlines, and colours
    ax.tick_params(axis="both", which="major")
    default_label_style_dict = {"fontsize": 8, "color": "black", "rotation": 30}

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        # x_inline=False, y_inline=False
    )
    gl.xlines = False
    gl.ylines = False

    if label_style_dict:
        for k, v in label_style_dict.items():
            default_label_style_dict[k] = v
    if presentation_format:
        default_label_style_dict["color"] = "white"
        if cbar_dict and cbar_dict["cbar"]:
            plt.setp(cbar_ticks, color="white")
            cb.set_label(cbar_dict["cbar_name"], color="white")

    gl.xlabel_style = default_label_style_dict
    gl.ylabel_style = default_label_style_dict

    if (
        not labels
    ):  # if no labels specified, set up something to iterate through returning nothing
        labels = [" "]
    if labels:
        # convert labels to relevant boolean: ["t","r","b","l"]
        gl.top_labels = "t" in labels
        gl.bottom_labels = "b" in labels
        gl.left_labels = "l" in labels
        gl.right_labels = "r" in labels

    return fig, ax


def get_n_colors_from_hexes(
    num: int,
    hex_list: list[str] = ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#d83c04"],
) -> list[str]:
    """
    from Wes Anderson: https://github.com/karthik/wesanderson/blob/master/R/colors.R
    Get a list of n colors from a list of hex codes.

    Args:
        num (int): The number of colors to return.
        hex_list (list[str]): The list of hex codes from which to create spectrum for sampling.

    Returns:
        list[str]: A list of n hex codes.
    """
    cmap = get_continuous_cmap(hex_list)
    colors = [cmap(i / num) for i in range(num)]
    hex_codes = [mcolors.to_hex(color) for color in colors]
    return hex_codes


class ColourMapGenerator:
    """
    Get a colormap for colorbar based on the specified type.

    Parameters
    ----------
    cbar_type (str, optional): The type of colormap to retrieve. Options are 'seq' for sequential colormap and 'div'
        for diverging colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap: The colormap object.
    """

    def __init__(self):
        self.sequential_hexes = ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#d83c04"]
        self.diverging_hexes = ["#3B9AB2", "#78B7C5", "#FFFFFF", "#E1AF00", "#d83c04"]
        self.cyclical_hexes = [
            "#3B9AB2",
            "#78B7C5",
            "#EBCC2A",
            "#E1AF00",
            "#d83c04",
            "#E1AF00",
            "#EBCC2A",
            "#78B7C5",
            "#3B9AB2",
        ]
        self.conf_mat_hexes = ["#EEEEEE", "#3B9AB2", "#cae7ed", "#d83c04", "#E1AF00"]
        self.residual_hexes = ["#3B9AB2", "#78B7C5", "#fafbfc", "#E1AF00", "#d83c04"]
        self.lim_red_hexes = ["#EBCC2A", "#E1AF00", "#d83c04"]
        self.lim_blue_hexes = ["#3B9AB2", "#78B7C5", "#FFFFFF"]

    def get_cmap(self, cbar_type, vmin=None, vmax=None):
        if cbar_type == "seq":
            return get_continuous_cmap(self.sequential_hexes)
        if cbar_type == "inc":
            return get_continuous_cmap(self.sequential_hexes[2:])
        elif cbar_type == "div":
            if not (vmin and vmax):
                raise ValueError(
                    "Minimum and maximum values needed for divergent colorbar"
                )
            cmap = get_continuous_cmap(self.diverging_hexes)
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            return cmap, norm
            # return get_continuous_cmap(self.diverging_hexes)
        elif cbar_type == "res":
            if not (vmin and vmax):
                raise ValueError(
                    "Minimum and maximum values needed for divergent colorbar"
                )
            cmap = get_continuous_cmap(self.residual_hexes)
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            return cmap, norm
        elif cbar_type == "cyc":
            return get_continuous_cmap(self.cyclical_hexes)
        elif cbar_type == "lim_blue":
            return get_continuous_cmap(self.lim_blue_hexes)
        elif cbar_type == "lim_red":
            return get_continuous_cmap(self.lim_red_hexes)
        elif cbar_type == "spatial_conf_matrix":
            return mcolors.ListedColormap(self.conf_mat_hexes)
        else:
            raise ValueError(f"{cbar_type} not recognised.")


def hex_to_rgb(value):
    """
    Convert a hexadecimal color code to RGB values.

    Parameters
    ----------
    value (str): The hexadecimal color code as a string of 6 characters.

    Returns
    -------
    tuple: A tuple of three RGB values.
    """
    value = value.strip("#")  # removes hash symbol if present
    hex_el = len(value)
    return tuple(
        int(value[i : i + hex_el // 3], 16)  # noqa
        for i in range(0, hex_el, hex_el // 3)
    )


def rgb_to_dec(value):
    """
    Convert RGB color values to decimal values (each value divided by 256).

    Parameters
    ----------
    value (list): A list of three RGB values.

    Returns
    -------
    list: A list of three decimal values.
    """
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    """
    Create and return a color map that can be used in heat map figures.

    Parameters
    ----------
    hex_list (list of str): List of hex code strings representing colors.
    float_list (list of float, optional): List of floats between 0 and 1, same length as hex_list. Must start with 0
        and end with 1.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap: The created color map.
    """
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap("my_cmp", segmentdata=cdict, N=256)
    return cmp


def customize_plot_colors(fig, ax, background_color="#212121", text_color="white"):
    # Set figure background color
    fig.patch.set_facecolor(background_color)

    # Set axis background color (if needed)
    ax.set_facecolor(background_color)

    # Set text color for all elements in the plot
    for text in fig.texts:
        text.set_color(text_color)
    for text in ax.texts:
        text.set_color(text_color)
    for text in ax.xaxis.get_ticklabels():
        text.set_color(text_color)
    for text in ax.yaxis.get_ticklabels():
        text.set_color(text_color)
    ax.title.set_color(text_color)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)

    # Set legend text color
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color(text_color)
    # # set cbar labels
    # cbar = ax.collections[0].colorbar
    # cbar.set_label(color=text_color)
    # cbar.ax.yaxis.label.set_color(text_color)
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    return fig, ax


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
