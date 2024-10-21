import os

# a silly hack to switch the working directory to the one where this file is located
working_dir = os.path.abspath("")

import time
import datetime
import math
import numpy as np
import pandas as pd
import sys
import astro_ghost
import os
import subprocess

os.chdir(working_dir)
sys.path.append("code/")

from helper_functions import *
from laiss_functions import *
from mod_helper_functions import *
from mod_laiss_functions import *

import requests
from requests.auth import HTTPBasicAuth

import warnings

warnings.filterwarnings("ignore")

with open("data/host_features.txt") as host_f:
    host_features = [line.strip() for line in host_f.readlines()]

with open("../data/lc_features.txt") as lc_f:
    lc_features = [line.strip() for line in lc_f.readlines()]

lc_and_host_features = host_features + lc_features

transient_df = pd.read_csv("data/ZTFBTS.txt")

storage_df = pd.DataFrame()
for idx, ztf_id in enumerate(transient_df["ZTFID"]):
    print(
        f"Processing {ztf_id}, transient number {idx+1} of {len(transient_df['ZTFID'])}"
    )

    feat_df = mod_extract_lc_and_host_features(
        ztf_id_ref=ztf_id,
        use_lc_for_ann_only_bool=False,
        show_lc=False,
        show_host=False,
        host_features=host_features,
        store_csv=False,
    )

    if feat_df is not None:
        feat_df = feat_df.dropna()
        try:
            lc_and_hosts_df_120d = feat_df[lc_and_host_features]
        except:
            print(f"{ztf_id} has some NaN LC features. Skip!")
            continue

        anom_obj_df = pd.DataFrame(lc_and_hosts_df_120d.iloc[-1]).T
        anom_obj_df["ztf_id"] = ztf_id
        cols = ["ztf_id"] + [col for col in anom_obj_df.columns if col != "ztf_id"]
        anom_obj_df = anom_obj_df[cols]
        anom_obj_df.reset_index(drop=True, inplace=True)

        storage_df = pd.concat([storage_df, anom_obj_df])

    if idx >= 3:
        break

storage_df.to_csv("data/transient_feature_lib.csv", index=False)
