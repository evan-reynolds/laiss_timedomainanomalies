import astro_ghost
from astro_ghost.ghostHelperFunctions import getGHOST

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

# GHOST getTransientHosts function with timeout
from timeout_decorator import timeout, TimeoutError


def mod_extract_lc_and_host_features(
    ztf_id_ref,
    use_lc_for_ann_only_bool,
    show_lc=False,
    show_host=True,
    host_features=[],
):
    start_time = time.time()
    ztf_id_ref = ztf_id_ref  #'ZTF20aalxlis' #'ZTF21abmspzt'
    df_path = "../timeseries"

    # try:
    ref_info = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztf_id_ref)
    df_ref = ref_info.timeseries.to_pandas()
    # except:
    #    print("antares_client can't find this object. Skip! Continue...")
    #    return

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
    # print("lightcurve", lightcurve)
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
        # print(detections)
        lc_properties_d = {}
        for band, names in property_names.items():
            detections = detections_pb[detections_pb["ant_passband"] == band]

            # Ensure locus has >3 obs for calculation
            if len(detections) < min_obs_count:
                continue
            # print(detections)

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
                # if name == "feature_amplitude_magn_g": print(m, value, band)
            # print("%%%%%%%%")
        lc_properties_d_l.append(lc_properties_d)

    lc_properties_d_l = [d for d in lc_properties_d_l if d]
    lc_properties_df = pd.DataFrame(lc_properties_d_l)
    if len(lc_properties_df) == 0:
        print(f"Not enough obs for {ztf_id_ref}. pass!\n")
        return
    print(f"Extracted LC features for {ztf_id_ref}!")

    end_time = time.time()
    print(f"Extracted LC features in {(end_time - start_time):.2f}s!")

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
        lc_and_hosts_df.to_csv(f"{df_path}/{lc_and_hosts_df.index[0]}_timeseries.csv")

    else:
        print("Saving for lc timeseries only")
        lc_properties_df = lc_properties_df.set_index("ztf_object_id")
        lc_properties_df.to_csv(f"{df_path}/{lc_properties_df.index[0]}_timeseries.csv")

    print(f"Saved results for {ztf_id_ref}!\n")
