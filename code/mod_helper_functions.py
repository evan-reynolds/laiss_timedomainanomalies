import light_curve as lc
import numpy as np
import math
import pandas as pd
from pathlib import Path
import time
import antares_client
from astropy.table import MaskedColumn
from itertools import chain
from astropy.coordinates import SkyCoord
from PIL import Image
from astropy.io import fits
import astropy.units as u
import os
import tempfile
import matplotlib.pyplot as plt
from sfdmap2 import sfdmap
from dust_extinction.parameter_averages import G23
import constants
from scipy.stats import gamma, halfnorm, uniform
from astropy.table import Table
from astropy.table import MaskedColumn
from lightcurve_engineer import *
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from matplotlib.backends.backend_pdf import PdfPages
from astropy.visualization import AsinhStretch, PercentileInterval


from astro_prost.helpers import SnRateAbsmag
from astro_prost.associate import associate_sample

# GHOST getTransientHosts function with timeout
from timeout_decorator import timeout, TimeoutError

import sys
import warnings
from contextlib import contextmanager
import io
import logging
import requests


# @contextmanager
# def re_suppress_output():
#     with open(os.devnull, "w") as devnull:
#         old_stdout = sys.stdout
#         old_stderr = sys.stderr
#         sys.stdout = devnull
#         sys.stderr = devnull
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             try:
#                 yield
#             finally:
#                 sys.stdout = old_stdout
#                 sys.stderr = old_stderr


@contextmanager
def re_suppress_output():
    """Temporarily silence stdout, stderr, warnings *and* all logging messages < CRITICAL."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull

        logging.disable(logging.CRITICAL)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield
        finally:
            logging.disable(logging.NOTSET)
            sys.stdout, sys.stderr = old_stdout, old_stderr


def mod_extract_lc_and_host_features(
    ztf_id_ref,
    use_lc_for_ann_only_bool,
    show_lc=False,
    show_host=True,
    host_features=[],
    store_csv=False,
):
    start_time = time.time()
    df_path = "../timeseries"

    try:
        ref_info = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztf_id_ref)
        df_ref = ref_info.timeseries.to_pandas()
    except:
        print("antares_client can't find this object. Skip! Continue...")
        return

    df_ref_g = df_ref[(df_ref.ant_passband == "g") & (~df_ref.ant_mag.isna())]
    df_ref_r = df_ref[(df_ref.ant_passband == "R") & (~df_ref.ant_mag.isna())]

    try:
        mjd_idx_at_min_mag_r_ref = df_ref_r[["ant_mag"]].reset_index().idxmin().ant_mag
        mjd_idx_at_min_mag_g_ref = df_ref_g[["ant_mag"]].reset_index().idxmin().ant_mag
    except:
        print(f"No obs for {ztf_id_ref}. pass!\n")
        return

    if show_lc:
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.gca().invert_yaxis()

        ax.errorbar(
            x=df_ref_r.ant_mjd,
            y=df_ref_r.ant_mag,
            yerr=df_ref_r.ant_magerr,
            fmt="o",
            c="r",
            label=f"REF: {ztf_id_ref}",
        )
        ax.errorbar(
            x=df_ref_g.ant_mjd,
            y=df_ref_g.ant_mag,
            yerr=df_ref_g.ant_magerr,
            fmt="o",
            c="g",
        )
        plt.show()

    min_obs_count = 4

    lightcurve = ref_info.lightcurve
    feature_names, property_names, features_count = create_base_features_class(
        MAGN_EXTRACTOR, FLUX_EXTRACTOR
    )

    g_obs = list(get_detections(lightcurve, "g").ant_mjd.values)
    r_obs = list(get_detections(lightcurve, "R").ant_mjd.values)
    mjd_l = sorted(g_obs + r_obs)

    lc_properties_d_l = []
    len_det_counter_r, len_det_counter_g = 0, 0

    band_lc = lightcurve[(~lightcurve["ant_mag"].isna())]
    idx = ~MaskedColumn(band_lc["ant_mag"]).mask
    all_detections = remove_simultaneous_alerts(band_lc[idx])
    for ob, mjd in enumerate(mjd_l):  # requires 4 obs
        # do time evolution of detections - in chunks

        detections_pb = all_detections[all_detections["ant_mjd"].values <= mjd]
        lc_properties_d = {}
        for band, names in property_names.items():
            detections = detections_pb[detections_pb["ant_passband"] == band]

            # Ensure locus has >3 obs for calculation
            if len(detections) < min_obs_count:
                continue

            t = detections["ant_mjd"].values
            m = detections["ant_mag"].values
            merr = detections["ant_magerr"].values
            flux = np.power(10.0, -0.4 * m)
            fluxerr = (
                0.5 * flux * (np.power(10.0, 0.4 * merr) - np.power(10.0, -0.4 * merr))
            )

            magn_features = MAGN_EXTRACTOR(
                t,
                m,
                merr,
                fill_value=None,
            )
            flux_features = FLUX_EXTRACTOR(
                t,
                flux,
                fluxerr,
                fill_value=None,
            )

            # After successfully calculating features, set locus properties and tag
            lc_properties_d["obs_num"] = int(ob)
            lc_properties_d["mjd_cutoff"] = mjd
            lc_properties_d["ztf_object_id"] = ztf_id_ref
            # print(band, m)
            for name, value in zip(names, chain(magn_features, flux_features)):
                lc_properties_d[name] = value

        lc_properties_d_l.append(lc_properties_d)

    lc_properties_d_l = [d for d in lc_properties_d_l if d]
    lc_properties_df = pd.DataFrame(lc_properties_d_l)
    if len(lc_properties_df) == 0:
        print(f"Not enough obs for {ztf_id_ref}. pass!\n")
        return

    end_time = time.time()
    print(
        f"Extracted LC features for for {ztf_id_ref} in {(end_time - start_time):.2f}s!"
    )

    if not use_lc_for_ann_only_bool:

        # Get GHOST features
        ra, dec = np.mean(df_ref.ant_ra), np.mean(df_ref.ant_dec)
        snName = [ztf_id_ref, ztf_id_ref]
        snCoord = [
            SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"),
            SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            try:
                hosts = getTransientHosts_with_timeout(
                    transientName=snName,
                    transientCoord=snCoord,
                    GLADE=True,
                    verbose=0,
                    starcut="gentle",
                    ascentMatch=False,
                    savepath=tmp,
                    redo_search=False,
                )
            except:
                print(f"GHOST error for {ztf_id_ref}. Retry without GLADE. \n")
                hosts = getTransientHosts_with_timeout(
                    transientName=snName,
                    transientCoord=snCoord,
                    GLADE=False,
                    verbose=0,
                    starcut="gentle",
                    ascentMatch=False,
                    savepath=tmp,
                    redo_search=False,
                )

        if len(hosts) > 1:
            hosts = pd.DataFrame(hosts.loc[0]).T

        hosts_df = hosts[host_features]
        hosts_df = hosts_df[~hosts_df.isnull().any(axis=1)]

        if len(hosts_df) < 1:
            # if any features are nan, we can't use as input
            print(f"Some features are NaN for {ztf_id_ref}. Skip!\n")
            return

        if show_host:
            print(
                f"http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={hosts.raMean.values[0]}+{hosts.decMean.values[0]}&filter=color"
            )

        hosts_df = hosts[host_features]
        hosts_df = pd.concat([hosts_df] * len(lc_properties_df), ignore_index=True)

        lc_and_hosts_df = pd.concat([lc_properties_df, hosts_df], axis=1)
        lc_and_hosts_df = lc_and_hosts_df.set_index("ztf_object_id")
        lc_and_hosts_df["raMean"] = hosts.raMean.values[0]
        lc_and_hosts_df["decMean"] = hosts.decMean.values[0]
        if not os.path.exists(df_path):
            print(f"Creating path {df_path}.")
            os.makedirs(df_path)

        if store_csv:
            lc_and_hosts_df.to_csv(
                f"{df_path}/{lc_and_hosts_df.index[0]}_timeseries.csv"
            )
            print(f"Saved results for {ztf_id_ref}!\n")

    else:
        ra, dec = np.mean(df_ref.ant_ra), np.mean(df_ref.ant_dec)
        lc_properties_df = lc_properties_df.set_index("ztf_object_id")

        if store_csv:
            lc_properties_df.to_csv(
                f"{df_path}/{lc_properties_df.index[0]}_timeseries.csv"
            )
            print(f"Saved results for {ztf_id_ref}!\n")

    lc_and_hosts_df["ra"] = ra
    lc_and_hosts_df["dec"] = dec
    return lc_and_hosts_df


def get_base_name(path):
    p = Path(path)
    while True:
        stem = p.stem
        if stem == p.name:  # No more extensions to strip
            break
        p = Path(stem)  # Prepare for next iteration, if needed
    return stem


# GHOST getTransientHosts function with timeout
from timeout_decorator import timeout, TimeoutError


@timeout(
    120
)  # Set a timeout of 60 seconds to query GHOST throughput APIs for host galaxy data
def getTransientHosts_with_timeout(**args):
    return astro_ghost.ghostHelperFunctions.getTransientHosts(**args)


# Functions to extract light-curve features
def replace_magn_with_flux(s):
    if "magnitude" in s:
        return s.replace("magnitudes", "fluxes").replace("magnitude", "flux")
    return f"{s} for flux light curve"


def create_base_features_class(
    magn_extractor,
    flux_extractor,
    bands=(
        "R",
        "g",
    ),
):
    feature_names = [f"{name}_magn" for name in magn_extractor.names] + [
        f"{name}_flux" for name in flux_extractor.names
    ]

    property_names = {
        band: [f"feature_{name}_{band}".lower() for name in feature_names]
        for band in bands
    }

    features_count = len(feature_names)

    return feature_names, property_names, features_count


###### calculate relevant light curve features ########
MAGN_EXTRACTOR = lc.Extractor(
    lc.Amplitude(),
    lc.AndersonDarlingNormal(),
    lc.BeyondNStd(1.0),
    lc.BeyondNStd(2.0),
    lc.Cusum(),
    lc.EtaE(),
    lc.InterPercentileRange(0.02),
    lc.InterPercentileRange(0.1),
    lc.InterPercentileRange(0.25),
    lc.Kurtosis(),
    lc.LinearFit(),
    lc.LinearTrend(),
    lc.MagnitudePercentageRatio(0.4, 0.05),
    lc.MagnitudePercentageRatio(0.2, 0.05),
    lc.MaximumSlope(),
    lc.Mean(),
    lc.MedianAbsoluteDeviation(),
    lc.PercentAmplitude(),
    lc.PercentDifferenceMagnitudePercentile(0.05),
    lc.PercentDifferenceMagnitudePercentile(0.1),
    lc.MedianBufferRangePercentage(0.1),
    lc.MedianBufferRangePercentage(0.2),
    lc.Periodogram(
        peaks=5,
        resolution=10.0,
        max_freq_factor=2.0,
        nyquist="average",
        fast=True,
        features=(
            lc.Amplitude(),
            lc.BeyondNStd(2.0),
            lc.BeyondNStd(3.0),
            lc.StandardDeviation(),
        ),
    ),
    lc.ReducedChi2(),
    lc.Skew(),
    lc.StandardDeviation(),
    lc.StetsonK(),
    lc.WeightedMean(),
)

FLUX_EXTRACTOR = lc.Extractor(
    lc.AndersonDarlingNormal(),
    lc.Cusum(),
    lc.EtaE(),
    lc.ExcessVariance(),
    lc.Kurtosis(),
    lc.MeanVariance(),
    lc.ReducedChi2(),
    lc.Skew(),
    lc.StetsonK(),
)


def remove_simultaneous_alerts(table):
    """Remove alert duplicates"""
    dt = np.diff(table["ant_mjd"], append=np.inf)
    return table[dt != 0]


def get_detections(photometry, band):
    """Extract clean light curve in given band from locus photometry"""
    band_lc = photometry[
        (photometry["ant_passband"] == band) & (~photometry["ant_mag"].isna())
    ]
    idx = ~MaskedColumn(band_lc["ant_mag"]).mask
    detections = remove_simultaneous_alerts(band_lc[idx])
    return detections


def mod_plot_RFC_prob_vs_lc_ztfid(
    clf,
    anom_ztfid,
    host_ztf_id,
    anom_spec_cls,
    anom_spec_z,
    anom_thresh,
    lc_and_hosts_df,
    lc_and_hosts_df_120d,
    ref_info,
    savefig,
    figure_path,
):
    anom_thresh = anom_thresh
    anom_obj_df = lc_and_hosts_df_120d

    # try:
    pred_prob_anom = 100 * clf.predict_proba(anom_obj_df)
    pred_prob_anom[:, 0] = [round(a, 1) for a in pred_prob_anom[:, 0]]
    pred_prob_anom[:, 1] = [round(b, 1) for b in pred_prob_anom[:, 1]]
    num_anom_epochs = len(np.where(pred_prob_anom[:, 1] >= anom_thresh)[0])
    #    except:
    #    print(f"{anom_ztfid} has some NaN host galaxy values from PS1 catalog. Skip!")
    #    return

    try:
        anom_idx = lc_and_hosts_df.iloc[
            np.where(pred_prob_anom[:, 1] >= anom_thresh)[0][0]
        ].obs_num
        anom_idx_is = True
        print("Anomalous during timeseries!")

    except:
        print(
            f"Prediction doesn't exceed anom_threshold of {anom_thresh}% for {anom_ztfid}."
        )
        anom_idx_is = False

    max_anom_score = max(pred_prob_anom[:, 1])
    print("max_anom_score", round(max_anom_score, 1))
    print("num_anom_epochs", num_anom_epochs)

    ztf_id_ref = anom_ztfid

    ref_info = ref_info

    df_ref = ref_info.timeseries.to_pandas()

    df_ref_g = df_ref[(df_ref.ant_passband == "g") & (~df_ref.ant_mag.isna())]
    df_ref_r = df_ref[(df_ref.ant_passband == "R") & (~df_ref.ant_mag.isna())]

    mjd_idx_at_min_mag_r_ref = df_ref_r[["ant_mag"]].reset_index().idxmin().ant_mag
    mjd_idx_at_min_mag_g_ref = df_ref_g[["ant_mag"]].reset_index().idxmin().ant_mag

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 10))
    ax1.invert_yaxis()
    ax1.errorbar(
        x=df_ref_r.ant_mjd,
        y=df_ref_r.ant_mag,
        yerr=df_ref_r.ant_magerr,
        fmt="o",
        c="r",
        label=r"ZTF-$r$",
    )
    ax1.errorbar(
        x=df_ref_g.ant_mjd,
        y=df_ref_g.ant_mag,
        yerr=df_ref_g.ant_magerr,
        fmt="o",
        c="g",
        label=r"ZTF-$g$",
    )
    if anom_idx_is == True:
        ax1.axvline(
            x=lc_and_hosts_df[lc_and_hosts_df.obs_num == anom_idx].mjd_cutoff.values[0],
            label="Tag anomalous",
            color="dodgerblue",
            ls="--",
        )
        # ax1.axvline(x=59323, label="Orig. spectrum", color='darkviolet', ls='-.')
        mjd_cross_thresh = round(
            lc_and_hosts_df[lc_and_hosts_df.obs_num == anom_idx].mjd_cutoff.values[0], 3
        )

        left, right = ax1.get_xlim()
        mjd_anom_per = (mjd_cross_thresh - left) / (right - left)
        # mjd_anom_per2 = (59323 - left)/(right - left)
        plt.text(
            mjd_anom_per + 0.073,
            -0.075,
            f"t$_a$ = {int(mjd_cross_thresh)}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax1.transAxes,
            fontsize=16,
            color="dodgerblue",
        )
        # plt.text(mjd_anom_per2+0.12, 0.035, f"t$_s$ = {int(59323)}", horizontalalignment='center',
        # verticalalignment='center', transform=ax1.transAxes, fontsize=16, color='darkviolet')
        print("MJD crossed thresh:", mjd_cross_thresh)

    print(f"https://alerce.online/object/{anom_ztfid}")
    ax2.plot(
        lc_and_hosts_df.mjd_cutoff,
        pred_prob_anom[:, 0],
        drawstyle="steps",
        label=r"$p(Normal)$",
    )
    ax2.plot(
        lc_and_hosts_df.mjd_cutoff,
        pred_prob_anom[:, 1],
        drawstyle="steps",
        label=r"$p(Anomaly)$",
    )

    if anom_spec_z is None:
        anom_spec_z = "None"
    elif isinstance(anom_spec_z, float):
        anom_spec_z = round(anom_spec_z, 3)
    else:
        anom_spec_z = anom_spec_z
    ax1.set_title(
        rf"{anom_ztfid} ({anom_spec_cls}, $z$={anom_spec_z}) with host from {host_ztf_id}",
        pad=25,
    )
    plt.xlabel("MJD")
    ax1.set_ylabel("Magnitude")
    ax2.set_ylabel("Probability (%)")

    if anom_idx_is == True:
        ax1.legend(
            loc="upper right",
            ncol=3,
            bbox_to_anchor=(1.0, 1.12),
            frameon=False,
            fontsize=14,
        )
    # if anom_idx_is == True: ax1.legend(loc='upper right', ncol=4, bbox_to_anchor=(1.05,1.12), columnspacing=0.65, frameon=False, fontsize=14)
    else:
        ax1.legend(
            loc="upper right",
            ncol=2,
            bbox_to_anchor=(0.75, 1.12),
            frameon=False,
            fontsize=14,
        )
    ax2.legend(
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.87, 1.12),
        frameon=False,
        fontsize=14,
    )

    ax1.grid(True)
    ax2.grid(True)

    if savefig:
        plt.savefig(
            f"{figure_path}/{anom_ztfid}_AD_run_timeseries.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def plot_RFC_prob_vs_lc_yse_IAUid(
    clf,
    IAU_name,
    anom_ztfid,
    anom_spec_cls,
    anom_spec_z,
    anom_thresh,
    lc_and_hosts_df,
    lc_and_hosts_df_120d,
    yse_lightcurve,
    savefig,
    figure_path,
):
    anom_thresh = anom_thresh
    anom_obj_df = lc_and_hosts_df_120d

    try:
        pred_prob_anom = 100 * clf.predict_proba(anom_obj_df)
        pred_prob_anom[:, 0] = [round(a, 1) for a in pred_prob_anom[:, 0]]
        pred_prob_anom[:, 1] = [round(b, 1) for b in pred_prob_anom[:, 1]]
        num_anom_epochs = len(np.where(pred_prob_anom[:, 1] >= anom_thresh)[0])
    except:
        print(f"{anom_ztfid} has some NaN host galaxy values from PS1 catalog. Skip!")
        return

    try:
        anom_idx = lc_and_hosts_df.iloc[
            np.where(pred_prob_anom[:, 1] >= anom_thresh)[0][0]
        ].obs_num
        anom_idx_is = True
        print("Anomalous during timeseries!")

    except:
        print(
            f"Prediction doesn't exceed anom_threshold of {anom_thresh}% for {anom_ztfid}."
        )
        anom_idx_is = False

    max_anom_score = max(pred_prob_anom[:, 1])
    print("max_anom_score", round(max_anom_score, 1))
    print("num_anom_epochs", num_anom_epochs)

    ztf_id_ref = anom_ztfid

    df_ref_g = yse_lightcurve[(yse_lightcurve.FLT == "g")]
    df_ref_r = yse_lightcurve[(yse_lightcurve.FLT == "R")]

    mjd_idx_at_min_mag_r_ref = df_ref_r[["MAG"]].reset_index().idxmin().MAG
    mjd_idx_at_min_mag_g_ref = df_ref_g[["MAG"]].reset_index().idxmin().MAG

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 10))
    ax1.invert_yaxis()

    df_ref_g_ztf = df_ref_g[df_ref_g.TELESCOPE == "P48"]
    df_ref_g_ps1 = df_ref_g[df_ref_g.TELESCOPE == "Pan-STARRS1"]
    df_ref_r_ztf = df_ref_r[df_ref_r.TELESCOPE == "P48"]
    df_ref_r_ps1 = df_ref_r[df_ref_r.TELESCOPE == "Pan-STARRS1"]

    ax1.errorbar(
        x=df_ref_r_ztf.MJD,
        y=df_ref_r_ztf.MAG,
        yerr=df_ref_r_ztf.MAGERR,
        fmt="o",
        c="r",
        label=r"ZTF-$r$",
    )
    ax1.errorbar(
        x=df_ref_g_ztf.MJD,
        y=df_ref_g_ztf.MAG,
        yerr=df_ref_g_ztf.MAGERR,
        fmt="o",
        c="g",
        label=r"ZTF-$g$",
    )
    ax1.errorbar(
        x=df_ref_r_ps1.MJD,
        y=df_ref_r_ps1.MAG,
        yerr=df_ref_r_ps1.MAGERR,
        fmt="s",
        c="r",
        label=r"PS1-$r$",
    )
    ax1.errorbar(
        x=df_ref_g_ps1.MJD,
        y=df_ref_g_ps1.MAG,
        yerr=df_ref_g_ps1.MAGERR,
        fmt="s",
        c="g",
        label=r"PS1-$g$",
    )

    if anom_idx_is == True:
        ax1.axvline(
            x=lc_and_hosts_df[lc_and_hosts_df.obs_num == anom_idx].mjd_cutoff.values[0],
            label="Tagged anomalous",
            color="darkblue",
            ls="--",
        )
        mjd_cross_thresh = round(
            lc_and_hosts_df[lc_and_hosts_df.obs_num == anom_idx].mjd_cutoff.values[0], 3
        )

        left, right = ax1.get_xlim()
        mjd_anom_per = (mjd_cross_thresh - left) / (right - left)
        plt.text(
            mjd_anom_per + 0.073,
            -0.075,
            f"t = {int(mjd_cross_thresh)}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax1.transAxes,
            fontsize=16,
        )
        print("MJD crossed thresh:", mjd_cross_thresh)

    print(f"https://ziggy.ucolick.org/yse/transient_detail/{IAU_name}/")
    ax2.plot(lc_and_hosts_df.mjd_cutoff, pred_prob_anom[:, 0], label=r"$p(Normal)$")
    ax2.plot(lc_and_hosts_df.mjd_cutoff, pred_prob_anom[:, 1], label=r"$p(Anomaly)$")

    if anom_spec_z is None:
        anom_spec_z = "None"
    elif isinstance(anom_spec_z, float):
        anom_spec_z = round(anom_spec_z, 3)
    else:
        anom_spec_z = anom_spec_z
    ax1.set_title(rf"{anom_ztfid} ({anom_spec_cls}, $z$={anom_spec_z})", pad=25)
    plt.xlabel("MJD")
    ax1.set_ylabel("Magnitude")
    ax2.set_ylabel("Probability (%)")

    if anom_idx_is == True:
        ax1.legend(
            loc="upper right",
            ncol=5,
            bbox_to_anchor=(1.1, 1.12),
            columnspacing=0.45,
            frameon=False,
            fontsize=14,
        )
    else:
        ax1.legend(
            loc="upper right",
            ncol=4,
            bbox_to_anchor=(1.03, 1.12),
            frameon=False,
            fontsize=14,
        )
    ax2.legend(
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.87, 1.12),
        frameon=False,
        fontsize=14,
    )

    ax1.grid(True)
    ax2.grid(True)

    if savefig:
        plt.savefig(
            f"{figure_path}/{anom_ztfid}_AD_run_timeseries.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def extract_lc_and_host_features_YSE_snana_format(
    IAU_name, ztf_id_ref, yse_lightcurve, ra, dec, show_lc=False, show_host=False
):
    IAU_name = IAU_name
    df_path = "../timeseries"

    min_obs_count = 4

    lightcurve = yse_lightcurve
    feature_names, property_names, features_count = create_base_features_class(
        MAGN_EXTRACTOR, FLUX_EXTRACTOR
    )

    g_obs = list(yse_lightcurve[yse_lightcurve.FLT == "g"].MJD)
    r_obs = list(yse_lightcurve[yse_lightcurve.FLT == "R"].MJD)
    mjd_l = sorted(g_obs + r_obs)

    lc_properties_d_l = []
    len_det_counter_r, len_det_counter_g = 0, 0

    all_detections = yse_lightcurve
    for ob, mjd in enumerate(mjd_l):  # requires 4 obs
        # do time evolution of detections - in chunks
        detections_pb = all_detections[all_detections["MJD"].values <= mjd]
        # print(detections)
        lc_properties_d = {}
        for band, names in property_names.items():
            detections = detections_pb[detections_pb["FLT"] == band]

            # Ensure locus has >3 obs for calculation
            if len(detections) < min_obs_count:
                continue
            # print(detections)

            t = detections["MJD"].values
            m = detections["MAG"].values
            merr = detections["MAGERR"].values
            flux = detections["FLUXCAL"].values
            fluxerr = detections["FLUXCALERR"].values

            try:
                magn_features = MAGN_EXTRACTOR(
                    t,
                    m,
                    merr,
                    fill_value=None,
                )
            except:
                print(f"{IAU_name} is maybe not sorted?")
                return

            flux_features = FLUX_EXTRACTOR(
                t,
                flux,
                fluxerr,
                fill_value=None,
            )

            # After successfully calculating features, set locus properties and tag
            lc_properties_d["obs_num"] = int(ob)
            lc_properties_d["mjd_cutoff"] = mjd
            lc_properties_d["ztf_object_id"] = ztf_id_ref
            # print(band, m)
            for name, value in zip(names, chain(magn_features, flux_features)):
                lc_properties_d[name] = value
                # if name == "feature_amplitude_magn_g": print(m, value, band)
            # print("%%%%%%%%")
        lc_properties_d_l.append(lc_properties_d)

    lc_properties_d_l = [d for d in lc_properties_d_l if d]
    lc_properties_df = pd.DataFrame(lc_properties_d_l)
    if len(lc_properties_df) == 0:
        print(f"Not enough obs for {IAU_name}. pass!\n")
        return
    print(f"Extracted LC features for {IAU_name}/{ztf_id_ref}!")

    # Get GHOST features
    ra, dec = float(ra), float(dec)
    snName = [IAU_name, IAU_name]
    snCoord = [
        SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"),
        SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        try:
            hosts = getTransientHosts(
                transientName=snName,
                snCoord=snCoord,
                GLADE=True,
                verbose=0,
                starcut="gentle",
                ascentMatch=False,
                savepath=tmp,
                redo_search=False,
            )
        except:
            print(f"GHOST error for {IAU_name}. Retry without GLADE. \n")
            hosts = getTransientHosts(
                transientName=snName,
                snCoord=snCoord,
                GLADE=False,
                verbose=0,
                starcut="gentle",
                ascentMatch=False,
                savepath=tmp,
                redo_search=False,
            )

    if len(hosts) > 1:
        hosts = pd.DataFrame(hosts.loc[0]).T

    hosts_df = hosts[host_features]
    hosts_df = hosts_df[~hosts_df.isnull().any(axis=1)]

    if len(hosts_df) < 1:
        # if any features are nan, we can't use as input
        print(f"Some features are NaN for {IAU_name}. Skip!\n")
        return

    if show_host:
        print(
            f"http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={hosts.raMean.values[0]}+{hosts.decMean.values[0]}&filter=color"
        )

    hosts_df = hosts[host_features]
    hosts_df = pd.concat([hosts_df] * len(lc_properties_df), ignore_index=True)

    lc_and_hosts_df = pd.concat([lc_properties_df, hosts_df], axis=1)
    lc_and_hosts_df = lc_and_hosts_df.set_index("ztf_object_id")
    lc_and_hosts_df["raMean"] = hosts.raMean.values[0]
    lc_and_hosts_df["decMean"] = hosts.decMean.values[0]
    lc_and_hosts_df.to_csv(f"{df_path}/{lc_and_hosts_df.index[0]}_timeseries.csv")

    print(f"Saved results for {IAU_name}/{ztf_id_ref}!\n")


def panstarrs_image_filename(position, image_size=None, filter=None):
    """Query panstarrs service to get a list of image names
    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :size : int: cutout image size in pixels.
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :filename: str: file name of the cutout
    """

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = (
        f"{service}?ra={position.ra.degree}&dec={position.dec.degree}"
        f"&size={image_size}&format=fits&filters={filter}"
    )

    filename_table = pd.read_csv(url, delim_whitespace=True)["filename"]
    return filename_table[0] if len(filename_table) > 0 else None


def panstarrs_cutout(position, filename, image_size=None, filter=None):
    """
    Download Panstarrs cutout from their own service
    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size: int: size of cutout image in pixels
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """

    if filename:
        service = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
        fits_url = (
            f"{service}ra={position.ra.degree}&dec={position.dec.degree}"
            f"&size={image_size}&format=fits&red={filename}"
        )
        fits_image = fits.open(fits_url)
    else:
        fits_image = None

    return fits_image


def host_pdfs(
    ztfid_ref,
    df,
    figure_path,
    ann_num,
    save_pdf=True,
    imsizepix=100,
    change_contrast=False,
):
    ref_name = df.ZTFID[0]
    data = df

    if save_pdf:
        pdf_path = f"{figure_path}/{ztfid_ref}_host_thumbnails_ann={ann_num}.pdf"
        pdf_pages = PdfPages(pdf_path)

    total_plots = len(df)
    rows = 3  # Number of rows in the subplot grid
    cols = 3  # Number of columns in the subplot grid
    num_subplots = rows * cols  # Total number of subplots in each figure
    num_pages = math.ceil(total_plots / num_subplots)

    for page in range(num_pages):
        fig, axs = plt.subplots(rows, cols, figsize=(6, 6))

        for i in range(num_subplots):
            index = page * num_subplots + i

            if index >= total_plots:
                break

            d = df.iloc[index]
            ax = axs[i // cols, i % cols]
            ax.set_xticks([])
            ax.set_yticks([])

            try:  # Has host assoc
                sc = SkyCoord(d["HOST_RA"], d["HOST_DEC"], unit=u.deg)

                outfilename = f"../ps1_cutouts/{d['ZTFID']}_pscutout.fits"

                if os.path.isfile(outfilename):
                    print(
                        f"Remove previously saved cutout {d['ZTFID']}_pscutout.fits to download a new one"
                    )
                    os.remove(outfilename)

                if not os.path.exists(outfilename):
                    filename = panstarrs_image_filename(
                        sc, image_size=imsizepix, filter="r"
                    )
                    fits_image = panstarrs_cutout(
                        sc, filename, image_size=imsizepix, filter="r"
                    )
                    fits_image.writeto(outfilename)

                wcs = WCS(f"../ps1_cutouts/{d['ZTFID']}_pscutout.fits")

                imdata = fits.getdata(f"../ps1_cutouts/{d['ZTFID']}_pscutout.fits")

                if change_contrast:
                    transform = AsinhStretch() + PercentileInterval(93)
                else:
                    transform = AsinhStretch() + PercentileInterval(99.5)

                bfim = transform(imdata)
                ax.imshow(bfim, cmap="gray", origin="lower")
                ax.set_title(f"{d['ZTFID']}", pad=0.1, fontsize=18)

            except:
                # Use a red square image when there is no data
                imdata = Image.new("RGB", (100, 100), color=(255, 0, 0))  # red
                ax.imshow(imdata, origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f" ", pad=0.1, fontsize=18)

        # Remove axes labels
        for ax in axs.flat:
            ax.label_outer()
            ax.set_xticks([])
            ax.set_yticks([])

        # Reduce padding between subplots for a tighter layout
        plt.tight_layout(pad=0.1)

        plt.ion()
        plt.show()

        if save_pdf:
            pdf_pages.savefig(fig, bbox_inches="tight", pad_inches=0.1)
        else:
            plt.show()

        plt.close(fig)

    if save_pdf:
        pdf_pages.close()
        print(f"Host grid PDF saved at: {pdf_path}")


def re_getTnsData(ztf_id):
    locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztf_id)
    try:
        tns = locus.catalog_objects["tns_public_objects"][0]
        tns_name, tns_cls, tns_z = tns["name"], tns["type"], tns["redshift"]
    except:
        tns_name, tns_cls, tns_z = "No TNS", "---", -99
    if tns_cls == "":
        tns_cls, tns_ann_z = "---", -99
    return tns_name, tns_cls, tns_z


def re_getExtinctionCorrectedMag(
    transient_row,
    band,
    av_in_raw_df_bank,
    path_to_sfd_folder=None,
):
    central_wv_filters = {"g": 4849.11, "r": 6201.20, "i": 7534.96, "z": 8674.20}
    MW_RV = 3.1
    ext = G23(Rv=MW_RV)

    if av_in_raw_df_bank:
        MW_AV = transient_row["A_V"]
    else:
        m = sfdmap.SFDMap(path_to_sfd_folder)
        MW_EBV = m.ebv(float(transient_row["ra"]), float(transient_row["dec"]))
        MW_AV = MW_RV * MW_EBV

    wv_filter = central_wv_filters[band]
    A_filter = -2.5 * np.log10(ext.extinguish(wv_filter * u.AA, Av=MW_AV))

    return transient_row[band + "KronMag"] - A_filter


def re_build_dataset_bank(
    raw_df_bank,
    av_in_raw_df_bank,
    path_to_sfd_folder=None,
    theorized=False,
    path_to_dataset_bank=None,
    building_entire_df_bank=False,
    building_for_AD=False,
):

    raw_lc_features = constants.lc_features_const.copy()
    raw_host_features = constants.raw_host_features_const.copy()

    if av_in_raw_df_bank:
        if "A_V" not in raw_host_features:
            raw_host_features.append("A_V")
    else:
        for col in ["ra", "dec"]:
            if col not in raw_host_features:
                raw_host_features.insert(0, col)

    # if "ztf_object_id" is the index, move it to the first column
    if raw_df_bank.index.name == "ztf_object_id":
        raw_df_bank = raw_df_bank.reset_index()

    if theorized:
        raw_features = raw_lc_features
        raw_feats_no_ztf = raw_lc_features
    else:
        raw_features = ["ztf_object_id"] + raw_lc_features + raw_host_features
        raw_feats_no_ztf = raw_lc_features + raw_host_features

    # Check to make sure all required features are in the raw data
    missing_cols = [col for col in raw_features if col not in raw_df_bank.columns]
    if missing_cols:
        print(
            f"KeyError: The following columns for this transient are not in the raw data: {missing_cols}. Abort!"
        )
        return

    # Impute missing features
    test_dataset_bank = raw_df_bank.replace([np.inf, -np.inf, -999], np.nan).dropna(
        subset=raw_features
    )

    nan_cols = [
        col
        for col in raw_features
        if raw_df_bank[col].replace([np.inf, -np.inf, -999], np.nan).isna().all()
    ]

    if not building_for_AD:
        print(
            f"There are {len(raw_df_bank) - len(test_dataset_bank)} of {len(raw_df_bank)} rows in the timeseries dataframe with 1 or more NA features."
        )
        if len(nan_cols) != 0:
            print(
                f"The following {len(nan_cols)} feature(s) are NaN for all measurements: {nan_cols}."
            )
        print("Imputing features (if necessary)...")

    wip_dataset_bank = raw_df_bank

    if building_entire_df_bank:
        X = raw_df_bank[raw_feats_no_ztf]

        feat_imputer = KNNImputer(weights="distance").fit(X)
        imputed_filt_arr = feat_imputer.transform(X)
    else:
        true_raw_df_bank = pd.read_csv(path_to_dataset_bank)
        X = true_raw_df_bank[raw_feats_no_ztf]

        if building_for_AD:
            # Use mean imputation
            feat_imputer = SimpleImputer(strategy="mean").fit(X)
        else:
            # Use KNN imputation
            feat_imputer = KNNImputer(weights="distance").fit(X)

        imputed_filt_arr = feat_imputer.transform(wip_dataset_bank[raw_feats_no_ztf])

    imputed_filt_df = pd.DataFrame(imputed_filt_arr, columns=raw_feats_no_ztf)
    imputed_filt_df.index = raw_df_bank.index

    wip_dataset_bank[raw_feats_no_ztf] = imputed_filt_df

    wip_dataset_bank = wip_dataset_bank.replace([np.inf, -np.inf, -999], np.nan).dropna(
        subset=raw_features
    )

    if not building_for_AD:
        if not wip_dataset_bank.empty:
            print("Successfully imputed features.")
        else:
            print("Failed to impute features.")

    # Engineer the remaining features
    if not theorized:
        if not building_for_AD:
            print(f"Engineering remaining features...")
        # Correct host magnitude features for dust
        for band in ["g", "r", "i", "z"]:
            wip_dataset_bank[band + "KronMagCorrected"] = wip_dataset_bank.apply(
                lambda row: re_getExtinctionCorrectedMag(
                    transient_row=row,
                    band=band,
                    av_in_raw_df_bank=av_in_raw_df_bank,
                    path_to_sfd_folder=path_to_sfd_folder,
                ),
                axis=1,
            )

        # Create color features
        wip_dataset_bank["gminusrKronMag"] = (
            wip_dataset_bank["gKronMag"] - wip_dataset_bank["rKronMag"]
        )
        wip_dataset_bank["rminusiKronMag"] = (
            wip_dataset_bank["rKronMag"] - wip_dataset_bank["iKronMag"]
        )
        wip_dataset_bank["iminuszKronMag"] = (
            wip_dataset_bank["iKronMag"] - wip_dataset_bank["zKronMag"]
        )

        # Calculate color uncertainties
        wip_dataset_bank["gminusrKronMagErr"] = np.sqrt(
            wip_dataset_bank["gKronMagErr"] ** 2 + wip_dataset_bank["rKronMagErr"] ** 2
        )
        wip_dataset_bank["rminusiKronMagErr"] = np.sqrt(
            wip_dataset_bank["rKronMagErr"] ** 2 + wip_dataset_bank["iKronMagErr"] ** 2
        )
        wip_dataset_bank["iminuszKronMagErr"] = np.sqrt(
            wip_dataset_bank["iKronMagErr"] ** 2 + wip_dataset_bank["zKronMagErr"] ** 2
        )

    final_df_bank = wip_dataset_bank

    return final_df_bank


def re_extract_lc_and_host_features(
    ztf_id,
    path_to_timeseries_folder,
    path_to_sfd_data_folder,
    path_to_dataset_bank=None,
    theorized_lightcurve_df=None,
    show_lc=False,
    show_host=True,
    store_csv=False,
    building_for_AD=False,
    swapped_host=False,
):
    start_time = time.time()
    df_path = path_to_timeseries_folder

    # Look up transient
    if theorized_lightcurve_df is not None:
        df_ref = theorized_lightcurve_df
        # Ensure correct capitalization of passbands ('g' and 'R')
        df_ref["ant_passband"] = df_ref["ant_passband"].replace({"G": "g", "r": "R"})
    else:
        try:
            ref_info = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztf_id)
            df_ref = ref_info.timeseries.to_pandas()
        except:
            print("antares_client can't find this object. Abort!")
            raise ValueError(f"antares_client can't find object {ztf_id}.")

    # Check for observations
    df_ref_g = df_ref[(df_ref.ant_passband == "g") & (~df_ref.ant_mag.isna())]
    df_ref_r = df_ref[(df_ref.ant_passband == "R") & (~df_ref.ant_mag.isna())]
    try:
        mjd_idx_at_min_mag_r_ref = df_ref_r[["ant_mag"]].reset_index().idxmin().ant_mag
        mjd_idx_at_min_mag_g_ref = df_ref_g[["ant_mag"]].reset_index().idxmin().ant_mag
    except:
        raise ValueError(
            f"No observations for {ztf_id if theorized_lightcurve_df is None else 'theorized lightcurve'}. Abort!\n"
        )

    # Plot lightcurve
    if show_lc:
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.gca().invert_yaxis()

        ax.errorbar(
            x=df_ref_r.ant_mjd,
            y=df_ref_r.ant_mag,
            yerr=df_ref_r.ant_magerr,
            fmt="o",
            c="r",
            label=f"REF: {ztf_id}",
        )
        ax.errorbar(
            x=df_ref_g.ant_mjd,
            y=df_ref_g.ant_mag,
            yerr=df_ref_g.ant_magerr,
            fmt="o",
            c="g",
        )
        plt.show()

    # Pull required lightcurve features:
    if theorized_lightcurve_df is None:
        lightcurve = df_ref[["ant_passband", "ant_mjd", "ant_mag", "ant_magerr"]]
    else:
        lightcurve = theorized_lightcurve_df

    lightcurve = lightcurve.sort_values(by="ant_mjd").reset_index(drop=True).dropna()
    min_obs_count = 5
    if len(lightcurve) < min_obs_count:
        raise ValueError(
            f"Not enough observations for {ztf_id if theorized_lightcurve_df is None else 'theorized lightcurve'}. Abort!\n"
        )

    # Engineer features in time
    lc_col_names = constants.lc_features_const.copy()
    lc_timeseries_feat_df = pd.DataFrame(
        columns=["ztf_object_id"] + ["obs_num"] + ["mjd_cutoff"] + lc_col_names
    )
    for i in range(min_obs_count, len(lightcurve) + 1):

        lightcurve_subset = lightcurve.iloc[:i]
        time_mjd = lightcurve_subset["ant_mjd"].iloc[-1]

        # Engineer lightcurve features
        df_g = lightcurve_subset[lightcurve_subset["ant_passband"] == "g"]
        time_g = df_g["ant_mjd"].tolist()
        mag_g = df_g["ant_mag"].tolist()
        err_g = df_g["ant_magerr"].tolist()

        df_r = lightcurve_subset[lightcurve_subset["ant_passband"] == "R"]
        time_r = df_r["ant_mjd"].tolist()
        mag_r = df_r["ant_mag"].tolist()
        err_r = df_r["ant_magerr"].tolist()

        try:
            extractor = SupernovaFeatureExtractor(
                time_g=time_g,
                mag_g=mag_g,
                err_g=err_g,
                time_r=time_r,
                mag_r=mag_r,
                err_r=err_r,
                ZTFID=ztf_id,
            )

            engineered_lc_properties_df = extractor.extract_features(
                return_uncertainty=True
            )
        except:
            continue

        if engineered_lc_properties_df is not None:

            engineered_lc_properties_df.insert(0, "mjd_cutoff", time_mjd)
            engineered_lc_properties_df.insert(0, "obs_num", int(i))
            engineered_lc_properties_df.insert(
                0,
                "ztf_object_id",
                ztf_id if theorized_lightcurve_df is None else "theorized_lightcurve",
            )

            if lc_timeseries_feat_df.empty:
                lc_timeseries_feat_df = engineered_lc_properties_df
            else:
                lc_timeseries_feat_df = pd.concat(
                    [lc_timeseries_feat_df, engineered_lc_properties_df],
                    ignore_index=True,
                )

    end_time = time.time()

    if lc_timeseries_feat_df.empty and not swapped_host:
        raise ValueError(
            f"Failed to extract features for {ztf_id if theorized_lightcurve_df is None else 'theorized lightcurve'}"
        )

    print(
        f"Extracted lightcurve features for {ztf_id if theorized_lightcurve_df is None else 'theorized lightcurve'} in {(end_time - start_time):.2f}s!"
    )

    # Get PROST features
    if theorized_lightcurve_df is None:
        print("Searching for host galaxy...")
        ra, dec = np.mean(df_ref.ant_ra), np.mean(df_ref.ant_dec)
        snName = [ztf_id, ztf_id]
        snCoord = [
            SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"),
            SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            # define priors for properties
            priorfunc_offset = uniform(loc=0, scale=5)

            likefunc_offset = gamma(a=0.75)

            priors = {"offset": priorfunc_offset}
            likes = {"offset": likefunc_offset}

            transient_catalog = pd.DataFrame(
                {"IAUID": [snName], "RA": [ra], "Dec": [dec]}
            )

            catalogs = ["panstarrs"]
            transient_coord_cols = ("RA", "Dec")
            transient_name_col = "IAUID"
            verbose = 0
            parallel = False
            save = False
            progress_bar = False
            cat_cols = True
            with re_suppress_output():
                hosts = associate_sample(
                    transient_catalog,
                    coord_cols=transient_coord_cols,
                    priors=priors,
                    likes=likes,
                    catalogs=catalogs,
                    parallel=parallel,
                    save=save,
                    progress_bar=progress_bar,
                    cat_cols=cat_cols,
                    calc_host_props=False,
                )
            hosts.rename(
                columns={"host_ra": "raMean", "host_dec": "decMean"}, inplace=True
            )

            if len(hosts) >= 1:
                hosts_df = pd.DataFrame(hosts.loc[0]).T
            else:
                print(f"Cannot identify host galaxy for {ztf_id}. Abort!\n")
                return

            # Check if required host features are missing
            try:
                raw_host_feature_check = constants.raw_host_features_const.copy()
                hosts_df = hosts[raw_host_feature_check]
            except KeyError:
                print(
                    f"KeyError: The following columns are not in the identified host feature set. Try engineering: {[col for col in raw_host_feature_check if col not in hosts_df.columns]}.\nAbort!"
                )
                return
            hosts_df = hosts_df[~hosts_df.isnull().any(axis=1)]
            if len(hosts_df) < 1:
                # if any features are nan, we can't use as input
                print(f"Some features are NaN for {ztf_id}. Abort!\n")
                return

            if show_host:
                if not building_for_AD:
                    print(
                        f"Host galaxy identified for {ztf_id}: http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={hosts.raMean.values[0]}+{hosts.decMean.values[0]}&filter=color"
                    )
                else:
                    print("Host identified.")

        if not lc_timeseries_feat_df.empty:
            hosts_df = pd.concat(
                [hosts_df] * len(lc_timeseries_feat_df), ignore_index=True
            )
            lc_and_hosts_df = pd.concat([lc_timeseries_feat_df, hosts_df], axis=1)
        else:
            lc_timeseries_feat_df.loc[0, "ztf_object_id"] = (
                ztf_id if theorized_lightcurve_df is None else "theorized_lightcurve"
            )
            lc_and_hosts_df = pd.concat([lc_timeseries_feat_df, hosts_df], axis=1)

        lc_and_hosts_df = lc_and_hosts_df.set_index("ztf_object_id")

        lc_and_hosts_df["raMean"] = hosts.raMean.values[0]
        lc_and_hosts_df["decMean"] = hosts.decMean.values[0]

        if not os.path.exists(df_path):
            print(f"Creating path {df_path}.")
            os.makedirs(df_path)

        # Lightcurve ra and dec may be needed in feature engineering
        lc_and_hosts_df["ra"] = ra
        lc_and_hosts_df["dec"] = dec

    # Engineer additonal features in build_dataset_bank function
    if building_for_AD:
        print("Engineering features...")
    lc_and_hosts_df_hydrated = re_build_dataset_bank(
        raw_df_bank=(
            lc_and_hosts_df
            if theorized_lightcurve_df is None
            else lc_timeseries_feat_df
        ),
        av_in_raw_df_bank=False,
        path_to_sfd_folder=path_to_sfd_data_folder,
        theorized=True if theorized_lightcurve_df is not None else False,
        path_to_dataset_bank=path_to_dataset_bank,
        building_for_AD=building_for_AD,
    )
    if building_for_AD:
        print("Finished engineering features.\n")

    if store_csv and not lc_and_hosts_df_hydrated.empty:
        os.makedirs(df_path, exist_ok=True)
        if theorized_lightcurve_df is None:
            lc_and_hosts_df_hydrated.to_csv(f"{df_path}/{ztf_id}_timeseries.csv")
            print(f"Saved timeseries features for {ztf_id}!\n")
        else:
            lc_and_hosts_df_hydrated.to_csv(f"{df_path}/theorized_timeseries.csv")
            print(f"Saved timeseries features for theorized lightcurve!\n")

    return lc_and_hosts_df_hydrated


def _ps1_list_filenames(ra_deg, dec_deg, flt):
    """
    Return the first stack FITS filename for (ra,dec) and *flt* or None.
    """
    url = (
        "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
        f"?ra={ra_deg}&dec={dec_deg}&filters={flt}"
    )
    for line in requests.get(url, timeout=20).text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for tok in line.split():
            if tok.endswith(".fits"):
                return tok
    return None


def fetch_ps1_cutout(ra_deg, dec_deg, *, size_pix=100, flt="r"):
    """
    Grayscale cut-out (2-D float) in a single PS1 filter.
    """
    fits_name = _ps1_list_filenames(ra_deg, dec_deg, flt)
    if fits_name is None:
        raise RuntimeError(f"No {flt}-band stack at this position")

    url = (
        "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
        f"?ra={ra_deg}&dec={dec_deg}&size={size_pix}"
        f"&format=fits&filters={flt}&red={fits_name}"
    )
    r = requests.get(url, timeout=40)
    if r.status_code == 400:
        raise RuntimeError("Outside PS1 footprint or no data in this filter")
    r.raise_for_status()

    with fits.open(io.BytesIO(r.content)) as hdul:
        data = hdul[0].data.astype(float)

    if data is None or data.size == 0 or (data != data).all():
        raise RuntimeError("Empty FITS array returned")

    data[data != data] = 0.0
    return data


def fetch_ps1_rgb_jpeg(ra_deg, dec_deg, *, size_pix=100):
    """
    Colour JPEG (H,W,3  uint8) using PS1 g/r/i stacks.
    Falls back by *raising* RuntimeError when the server lacks colour data.
    """
    url = (
        "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
        f"?ra={ra_deg}&dec={dec_deg}&size={size_pix}"
        f"&format=jpeg&filters=grizy&red=i&green=r&blue=g&autoscale=99.5"
    )
    r = requests.get(url, timeout=40)
    if r.status_code == 400:
        raise RuntimeError("Outside PS1 footprint or no colour data here")
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return np.array(img)


def re_plot_hosts(
    ztfid_ref,
    plot_label,
    df,
    figure_path,
    ann_num,
    save_pdf=True,
    imsizepix=100,
    change_contrast=False,
    prefer_color=True,
):
    """
    Build 3×3 grids of PS1 thumbnails for each row in *df* and write a PDF.

    Set *prefer_color=False* for r-band grayscale only.  With *prefer_color=True*
    (default) the code *tries* colour first and quietly falls back to grayscale
    when colour isn’t available.
    """
    os.makedirs(figure_path, exist_ok=True)
    host_grid_path = figure_path + "/host_grids"
    Path(host_grid_path).mkdir(parents=True, exist_ok=True)
    pdf_path = Path(host_grid_path) / f"{plot_label}_host_thumbnails_ann={ann_num}.pdf"
    pdf_pages = PdfPages(pdf_path) if save_pdf else None

    logging.basicConfig(level=logging.INFO, format="%(levelname)7s : %(message)s")
    rows = cols = 3
    per_page = rows * cols
    pages = math.ceil(len(df) / per_page)

    for pg in range(pages):
        fig, axs = plt.subplots(rows, cols, figsize=(6, 6))
        axs = axs.ravel()

        for k in range(per_page):
            idx = pg * per_page + k
            ax = axs[k]
            ax.set_xticks([])
            ax.set_yticks([])

            if idx >= len(df):
                ax.axis("off")
                continue

            row = df.iloc[idx]
            ztfid, ra, dec = (
                str(row["ZTFID"]),
                float(row["HOST_RA"]),
                float(row["HOST_DEC"]),
            )

            try:
                # validate coordinates
                if np.isnan(ra) or np.isnan(dec):
                    raise ValueError("NaN coordinate")
                SkyCoord(ra * u.deg, dec * u.deg)

                # Attempt colour first (if requested), then grayscale fallback
                if prefer_color:
                    try:
                        im = fetch_ps1_rgb_jpeg(ra, dec, size_pix=imsizepix)
                        ax.imshow(im, origin="lower")
                    except Exception as col_err:
                        im = fetch_ps1_cutout(ra, dec, size_pix=imsizepix, flt="r")
                        stretch = AsinhStretch() + PercentileInterval(
                            93 if change_contrast else 99.5
                        )
                        ax.imshow(stretch(im), cmap="gray", origin="lower")
                else:
                    im = fetch_ps1_cutout(ra, dec, size_pix=imsizepix, flt="r")
                    stretch = AsinhStretch() + PercentileInterval(
                        93 if change_contrast else 99.5
                    )
                    ax.imshow(stretch(im), cmap="gray", origin="lower")

                ax.set_title(ztfid, fontsize=8, pad=1.5)

            except Exception as e:
                logging.warning(f"{ztfid}: {e}")
                ax.imshow(np.full((imsizepix, imsizepix, 3), [1.0, 0, 0]))
                ax.set_title("", fontsize=8, pad=1.5)

        plt.tight_layout(pad=0.2)
        if pdf_pages:
            pdf_pages.savefig(fig, bbox_inches="tight", pad_inches=0.05)
        plt.show(block=False)
        plt.close(fig)

    if pdf_pages:
        pdf_pages.close()
        print(f"PDF written to {pdf_path}\n")


def re_check_anom_and_plot(
    clf,
    input_ztf_id,
    swapped_host_ztf_id,
    input_spec_cls,
    input_spec_z,
    anom_thresh,
    timeseries_df_full,
    timeseries_df_features_only,
    ref_info,
    savefig,
    figure_path,
):
    anom_obj_df = timeseries_df_features_only

    pred_prob_anom = 100 * clf.predict_proba(anom_obj_df)
    pred_prob_anom[:, 0] = [round(a, 1) for a in pred_prob_anom[:, 0]]
    pred_prob_anom[:, 1] = [round(b, 1) for b in pred_prob_anom[:, 1]]
    num_anom_epochs = len(np.where(pred_prob_anom[:, 1] >= anom_thresh)[0])

    try:
        anom_idx = timeseries_df_full.iloc[
            np.where(pred_prob_anom[:, 1] >= anom_thresh)[0][0]
        ].obs_num
        anom_idx_is = True
        print("Anomalous during timeseries!")

    except:
        print(
            f"Prediction doesn't exceed anom_threshold of {anom_thresh}% for {input_ztf_id}."
            + (f" with host from {swapped_host_ztf_id}" if swapped_host_ztf_id else "")
        )
        anom_idx_is = False

    max_anom_score = max(pred_prob_anom[:, 1])
    print("max_anom_score", round(max_anom_score, 1))
    print("num_anom_epochs", num_anom_epochs, "\n")

    df_ref = ref_info.timeseries.to_pandas()

    df_ref_g = df_ref[(df_ref.ant_passband == "g") & (~df_ref.ant_mag.isna())]
    df_ref_r = df_ref[(df_ref.ant_passband == "R") & (~df_ref.ant_mag.isna())]

    mjd_idx_at_min_mag_r_ref = df_ref_r[["ant_mag"]].reset_index().idxmin().ant_mag
    mjd_idx_at_min_mag_g_ref = df_ref_g[["ant_mag"]].reset_index().idxmin().ant_mag

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 10))
    ax1.invert_yaxis()
    ax1.errorbar(
        x=df_ref_r.ant_mjd,
        y=df_ref_r.ant_mag,
        yerr=df_ref_r.ant_magerr,
        fmt="o",
        c="r",
        label=r"ZTF-$r$",
    )
    ax1.errorbar(
        x=df_ref_g.ant_mjd,
        y=df_ref_g.ant_mag,
        yerr=df_ref_g.ant_magerr,
        fmt="o",
        c="g",
        label=r"ZTF-$g$",
    )
    if anom_idx_is == True:
        ax1.axvline(
            x=timeseries_df_full[
                timeseries_df_full.obs_num == anom_idx
            ].mjd_cutoff.values[0],
            label="Tag anomalous",
            color="dodgerblue",
            ls="--",
        )
        mjd_cross_thresh = round(
            timeseries_df_full[
                timeseries_df_full.obs_num == anom_idx
            ].mjd_cutoff.values[0],
            3,
        )

        left, right = ax1.get_xlim()
        mjd_anom_per = (mjd_cross_thresh - left) / (right - left)
        plt.text(
            mjd_anom_per + 0.073,
            -0.075,
            f"t$_a$ = {int(mjd_cross_thresh)}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax1.transAxes,
            fontsize=16,
            color="dodgerblue",
        )
        print("MJD crossed thresh:", mjd_cross_thresh)

    ax2.plot(
        timeseries_df_full.mjd_cutoff,
        pred_prob_anom[:, 0],
        drawstyle="steps",
        label=r"$p(Normal)$",
    )
    ax2.plot(
        timeseries_df_full.mjd_cutoff,
        pred_prob_anom[:, 1],
        drawstyle="steps",
        label=r"$p(Anomaly)$",
    )

    if input_spec_z is None:
        input_spec_z = "None"
    elif isinstance(input_spec_z, float):
        input_spec_z = round(input_spec_z, 3)
    else:
        input_spec_z = input_spec_z
    ax1.set_title(
        rf"{input_ztf_id} ({input_spec_cls}, $z$={input_spec_z})"
        + (f" with host from {swapped_host_ztf_id}" if swapped_host_ztf_id else ""),
        pad=25,
    )
    plt.xlabel("MJD")
    ax1.set_ylabel("Magnitude")
    ax2.set_ylabel("Probability (%)")

    if anom_idx_is == True:
        ax1.legend(
            loc="upper right",
            ncol=3,
            bbox_to_anchor=(1.0, 1.12),
            frameon=False,
            fontsize=14,
        )
    else:
        ax1.legend(
            loc="upper right",
            ncol=2,
            bbox_to_anchor=(0.75, 1.12),
            frameon=False,
            fontsize=14,
        )
    ax2.legend(
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.87, 1.12),
        frameon=False,
        fontsize=14,
    )

    ax1.grid(True)
    ax2.grid(True)

    if savefig:
        os.makedirs(figure_path, exist_ok=True)
        os.makedirs(figure_path + "/AD", exist_ok=True)
        plt.savefig(
            (
                f"{figure_path}/AD/{input_ztf_id}"
                + (f"_w_host_{swapped_host_ztf_id}" if swapped_host_ztf_id else "")
                + "_AD.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        print(
            "Saved anomaly detection chart to:"
            + f"{figure_path}/AD/{input_ztf_id}"
            + (f"_w_host_{swapped_host_ztf_id}" if swapped_host_ztf_id else "")
            + "_AD.pdf"
        )
    plt.show()
