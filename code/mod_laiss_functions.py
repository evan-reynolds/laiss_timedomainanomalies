from mod_helper_functions import *
from helper_functions import *
import pandas as pd
import numpy as np
import os
import sys
import annoy
from annoy import AnnoyIndex
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from kneed import KneeLocator
import pickle


def mod_build_indexed_sample(
    fn="", lc_features=[], host_features=[], pca=True, save=True
):
    host = True
    fn_stem = get_base_name(fn)

    if len(lc_features) < 1:
        print("Error! Must select at least one LC feature.")
        sys.exit()
    if len(host_features) < 1:
        host = False
    lc_and_host_features = lc_features + host_features
    data = pd.read_csv(fn, compression="gzip")
    data = data.set_index("ztf_object_id")
    if host:
        data = data[lc_and_host_features]
    else:
        data = data[lc_features]

    data = data.dropna()

    # LC + host features annoy index, w/ PCA
    feat_arr = np.array(data)
    idx_arr = np.array(data.index)

    if pca:
        scaler = preprocessing.StandardScaler()

        # Set a random seed for PCA
        random_seed = 42  # Choose your desired random seed

        # Scale the features
        feat_arr_scaled = scaler.fit_transform(feat_arr)

        # Initialize PCA with 60 principal components
        n_components = 60
        pcaModel = PCA(n_components=n_components, random_state=random_seed)

        # Apply PCA
        feat_arr_scaled_pca = pcaModel.fit_transform(feat_arr_scaled)

    # Create or load the ANNOY index
    index_nm = f"{fn_stem}_pca{pca}_host{host}_annoy_index"
    if save:
        # Save the index array to a binary file
        np.save(f"../data/{index_nm}_idx_arr.npy", idx_arr)
        np.save(f"../data/{index_nm}_feat_arr.npy", feat_arr)
        if pca:
            np.save(f"../data/{index_nm}_feat_arr_scaled.npy", feat_arr_scaled)
            np.save(f"../data/{index_nm}_feat_arr_scaled_pca.npy", feat_arr_scaled_pca)

    # Create or load the ANNOY index
    index_file = f"../data/{index_nm}.ann"  # Choose a filename
    if pca:
        index_dim = feat_arr_scaled_pca.shape[1]
    else:
        index_dim = feat_arr.shape[1]  # Dimension of the index

    # Check if the index file exists
    if not os.path.exists(index_file):
        print("Saving new ANNOY index")
        # If the index file doesn't exist, create and build the index
        index = annoy.AnnoyIndex(index_dim, metric="manhattan")

        # Add items to the index
        for i in range(len(idx_arr)):
            if pca:
                index.add_item(i, feat_arr_scaled_pca[i])
            else:
                index.add_item(i, feat_arr[i])
        # Build the index
        index.build(1000)  # 1000 trees

        if save:
            # Save the index to a file
            index.save(index_file)
    else:
        print("Loading previously saved ANNOY index")
        # If the index file exists, load it
        index = annoy.AnnoyIndex(index_dim, metric="manhattan")
        index.load(index_file)
        idx_arr = np.load(f"../data/{index_nm}_idx_arr.npy", allow_pickle=True)

    # binary_index = f"../data/{index_nm}_idx_arr.npy"
    return index_nm


def mod_LAISS(
    LC_l_or_ztfid_ref,
    HOST_l_or_ztfid_ref,
    lc_features,
    host_features=[],
    n=8,
    use_lc_for_ann_only_bool=False,
    use_ysepz_phot_snana_file=False,
    show_lightcurves_grid=False,
    show_hosts_grid=False,
    run_AD_model=False,
    savetables=False,
    savefigs=False,
    ad_params={},
):
    print("Running LAISS...")
    lc_and_host_features = lc_features + host_features
    start_time = time.time()
    ann_num = n
    l_or_ztfid_refs = [LC_l_or_ztfid_ref, HOST_l_or_ztfid_ref]

    if use_ysepz_phot_snana_file:
        IAU_name = input("Input the IAU (TNS) name here, like: 2023abc\t")
        print("IAU_name:", IAU_name)
        ysepz_snana_fp = f"../ysepz_snana_phot_files/{IAU_name}_data.snana.txt"
        print(f"Looking for file {ysepz_snana_fp}...")

        # Initialize variables to store the values
        ra = None
        dec = None

        # Open the file for reading
        with open(ysepz_snana_fp, "r") as file:
            # Read lines one by one
            for line in file:
                # Check if the line starts with '#'
                if line.startswith("#"):
                    # Split the line into key and value
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        # Check if the key is 'RA' or 'DEC'
                        if key == "# RA":
                            ra = value
                        elif key == "# DEC":
                            dec = value

        SN_df = pd.read_csv(ysepz_snana_fp, comment="#", delimiter="\s+")
        SN_df = SN_df[
            (SN_df.FLT == "r-ZTF")
            | (SN_df.FLT == "g-ZTF")
            | (SN_df.FLT == "g")
            | (SN_df.FLT == "r")
        ].reset_index(drop=True)
        SN_df["FLT"] = SN_df["FLT"].map(
            {"g-ZTF": "g", "g": "g", "r-ZTF": "R", "r": "R"}
        )
        SN_df = SN_df.sort_values("MJD")
        SN_df = SN_df.dropna()
        SN_df = SN_df.drop_duplicates(keep="first")
        SN_df = SN_df.drop_duplicates(subset=["MJD"], keep="first")
        print("Using S/N cut of 3...")
        SN_df = SN_df[SN_df.FLUXCAL >= 3 * SN_df.FLUXCALERR]  # SNR >= 3

    ############# LOOP HERE #############
    host = False
    n_flag = False
    for i, l_or_ztfid_ref in enumerate(l_or_ztfid_refs):
        if i == 1:
            host = True

        figure_path = f"../LAISS_run/{l_or_ztfid_ref}/figures"
        if savefigs:
            if not os.path.exists(figure_path):
                print(f"Making figures directory {figure_path}")
                os.makedirs(figure_path)

        table_path = f"../LAISS_run/{l_or_ztfid_ref}/tables"
        if savetables:
            if not os.path.exists(table_path):
                print(f"Making tables directory {table_path}")
                os.makedirs(table_path)

        needs_reextraction_for_AD = False
        l_or_ztfid_ref_in_dataset_bank = False

        host_df_ztf_id_l, host_df_ra_l, host_df_dec_l = [], [], []

        if l_or_ztfid_ref.startswith("ANT"):
            # Get locus data using antares_client
            try:
                locus = antares_client.search.get_by_id(l_or_ztfid_ref)
            except:
                print(
                    f"Can't get locus. Check that {l_or_ztfid_ref} is a legimiate loci! Exiting..."
                )
                return
            ztfid_ref = locus.properties["ztf_object_id"]
            needs_reextraction_for_AD = True

            if "tns_public_objects" not in locus.catalogs:
                tns_name, tns_cls, tns_z = "No TNS", "---", -99
            else:
                tns = locus.catalog_objects["tns_public_objects"][0]
                tns_name, tns_cls, tns_z = tns["name"], tns["type"], tns["redshift"]
            if tns_cls == "":
                tns_cls, tns_ann_z = "---", -99

            # Extract the relevant features
            try:
                locus_feat_arr_lc = [locus.properties[f] for f in lc_features]
                locus_feat_arr_host = [locus.properties[f] for f in host_features]
                print(locus.properties["raMean"], locus.properties["decMean"])
                if host:
                    HOST_locus_feat_arr = locus_feat_arr_lc + locus_feat_arr_host
                    print(
                        f"HOST : http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={locus.properties['raMean']}+{locus.properties['decMean']}&filter=color\n"
                    )
                    host_df_ztf_id_l.append(ztfid_ref), host_df_ra_l.append(
                        locus.properties["raMean"]
                    ), host_df_dec_l.append(locus.properties["decMean"])
                else:
                    LC_locus_feat_arr = locus_feat_arr_lc + locus_feat_arr_host

            except:
                print(
                    f"{l_or_ztfid_ref} is not in dataset_bank. Checking if made before..."
                )
                if os.path.exists(f"../timeseries/{ztfid_ref}_timeseries.csv"):
                    print(f"{ztfid_ref} is already made. Continue!\n")
                else:
                    print("Re-extracting features")
                    if use_ysepz_phot_snana_file:
                        print("Using YSE-PZ SNANA Photometry file...")
                        extract_lc_and_host_features_YSE_snana_format(
                            IAU_name=IAU_name,
                            ztf_id_ref=l_or_ztfid_ref,
                            yse_lightcurve=SN_df,
                            ra=ra,
                            dec=dec,
                            show_lc=False,
                            show_host=True,
                        )
                    else:
                        extract_lc_and_host_features(
                            ztf_id_ref=ztfid_ref,
                            use_lc_for_ann_only_bool=use_lc_for_ann_only_bool,
                            show_lc=False,
                            show_host=True,
                        )

                try:
                    lc_and_hosts_df = pd.read_csv(
                        f"../timeseries/{ztfid_ref}_timeseries.csv"
                    )
                except:
                    print(
                        f"couldn't feature space as func of time for {ztfid_ref}. pass."
                    )
                    return

                # try:
                lc_and_hosts_df_120d = lc_and_hosts_df[lc_and_host_features]
                # except:
                #    print(f"{ztfid_ref} has some NaN LC features. Skip!")
                #    return

                anom_obj_df = pd.DataFrame(
                    lc_and_hosts_df_120d.iloc[-1]
                ).T  # last row of df to test "full LC only"

                if host:
                    HOST_locus_feat_arr = anom_obj_df.values[0]
                    lc_and_hosts_df = lc_and_hosts_df.dropna()
                    print(
                        f"HOST : http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={lc_and_hosts_df['raMean']}+{lc_and_hosts_df['decMean']}&filter=color\n"
                    )
                    host_df_ztf_id_l.append(ztfid_ref), host_df_ra_l.append(
                        locus.properties["raMean"]
                    ), host_df_dec_l.append(locus.properties["decMean"])
                else:
                    LC_locus_feat_arr = anom_obj_df.values[0]

        elif l_or_ztfid_ref.startswith("ZTF"):
            # Assuming you have a list of feature values
            # n = n+1 # because object in dataset chooses itself as ANN=0
            ztfid_ref = l_or_ztfid_ref

            try:
                dataset_bank_orig = pd.read_csv(
                    "../data/dataset_bank_orig_5472objs.csv.gz",
                    compression="gzip",
                    index_col=0,
                )
                locus_feat_arr = dataset_bank_orig.loc[ztfid_ref]
                if host:
                    HOST_locus_feat_arr = locus_feat_arr[lc_and_host_features].values
                else:
                    LC_locus_feat_arr = locus_feat_arr[lc_and_host_features].values
                needs_reextraction_for_AD = True
                l_or_ztfid_ref_in_dataset_bank = True
                print(f"{l_or_ztfid_ref} is in dataset_bank")
                if not n_flag:
                    n = n + 1
                    n_flag = True

            except:
                print(
                    f"{l_or_ztfid_ref} is not in dataset_bank. Checking if made before..."
                )
                if os.path.exists(f"../timeseries/{l_or_ztfid_ref}_timeseries.csv"):
                    print(f"{l_or_ztfid_ref} is already made. Continue!\n")

                else:
                    print("Re-extracting LC+Host features")
                    # try:
                    if use_ysepz_phot_snana_file:
                        print("Using YSE-PZ SNANA Photometry file...")
                        extract_lc_and_host_features_YSE_snana_format(
                            IAU_name=IAU_name,
                            ztf_id_ref=l_or_ztfid_ref,
                            yse_lightcurve=SN_df,
                            ra=ra,
                            dec=dec,
                            show_lc=False,
                            show_host=True,
                            host_features=host_features,
                        )
                    else:
                        extract_lc_and_host_features(
                            ztf_id_ref=ztfid_ref,
                            use_lc_for_ann_only_bool=use_lc_for_ann_only_bool,
                            show_lc=False,
                            show_host=True,
                            host_features=host_features,
                        )

                try:
                    lc_and_hosts_df = pd.read_csv(
                        f"../timeseries/{l_or_ztfid_ref}_timeseries.csv"
                    )
                except:
                    print(
                        f"couldn't feature space as func of time for {l_or_ztfid_ref}. pass."
                    )
                    return

                if not use_lc_for_ann_only_bool:
                    if host:
                        print(
                            f"HOST : http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={lc_and_hosts_df.iloc[0]['raMean']}+{lc_and_hosts_df.iloc[0]['decMean']}&filter=color\n"
                        )
                        host_df_ztf_id_l.append(ztfid_ref), host_df_ra_l.append(
                            lc_and_hosts_df.iloc[0]["raMean"]
                        ), host_df_dec_l.append(lc_and_hosts_df.iloc[0]["decMean"])

                    lc_and_hosts_df = (
                        lc_and_hosts_df.dropna()
                    )  # if this drops all rows, that means something is nan from a 0 or nan entry (check data file)

                    try:
                        # print(lc_and_hosts_df.columns.values)
                        lc_and_hosts_df_120d = lc_and_hosts_df[lc_and_host_features]
                    except:
                        print(f"{ztfid_ref} has some NaN LC features. Skip!")
                        return

                    anom_obj_df = pd.DataFrame(
                        lc_and_hosts_df_120d.iloc[-1]
                    ).T  # last row of df to test "full LC only"
                    if host:
                        HOST_locus_feat_arr = anom_obj_df.values[0]
                    else:
                        LC_locus_feat_arr = anom_obj_df.values[0]

                if use_lc_for_ann_only_bool:
                    try:
                        lc_only_df = lc_and_hosts_df.copy()
                        lc_only_df = lc_only_df.dropna()
                        lc_only_df = lc_only_df[lc_features]
                        lc_and_hosts_df_120d = lc_only_df.copy()

                        anom_obj_df = pd.DataFrame(
                            lc_and_hosts_df_120d.iloc[-1]
                        ).T  # last row of df to test "full LC only"
                        if host:
                            HOST_locus_feat_arr = anom_obj_df.values[0]
                        else:
                            LC_locus_feat_arr = anom_obj_df.values[0]
                    except:
                        print(f"{ztfid_ref} doesn't have enough g or r obs. Skip!")
                        return

            locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztfid_ref)
            try:
                tns = locus.catalog_objects["tns_public_objects"][0]
                tns_name, tns_cls, tns_z = tns["name"], tns["type"], tns["redshift"]
            except:
                tns_name, tns_cls, tns_z = "No TNS", "---", -99
            if tns_cls == "":
                tns_cls, tns_ann_z = "---", -99

        else:
            raise ValueError(
                "Input must be a string (l or ztfid_ref) or a list of feature values"
            )
        if host:
            HOST_ztfid_ref = ztfid_ref
            HOST_tns_name, HOST_tns_cls, HOST_tns_z = tns_name, tns_cls, tns_z
        else:
            LC_ztfid_ref = ztfid_ref
            LC_tns_name, LC_tns_cls, LC_tns_z = tns_name, tns_cls, tns_z

    ###### END OF LOOP ######
    # Create new feature away with mixed lc and host features
    subset_lc_features = LC_locus_feat_arr[:62]
    subset_temp_host_features = HOST_locus_feat_arr[-58:]
    locus_feat_arr = np.concatenate((subset_lc_features, subset_temp_host_features))

    if not use_lc_for_ann_only_bool:
        # 1. Scale locus_feat_arr using the same scaler (Standard Scaler)
        scaler = preprocessing.StandardScaler()
        trained_PCA_feat_arr = np.load(
            f"../data/dataset_bank_orig_5472objs_pcaTrue_hostTrue_annoy_index_feat_arr.npy",
            allow_pickle=True,
        )

        trained_PCA_feat_arr_scaled = scaler.fit_transform(
            trained_PCA_feat_arr
        )  # scaler needs to be fit first to the same data as trained

        locus_feat_arr_scaled = scaler.transform(
            [locus_feat_arr]
        )  # scaler transform new data

        # 2. Transform the scaled locus_feat_arr using the same PCA model (60 PCs, RS=42)
        n_components = 60
        random_seed = 42
        pca = PCA(n_components=n_components, random_state=random_seed)
        trained_PCA_feat_arr_scaled_pca = pca.fit_transform(
            trained_PCA_feat_arr_scaled
        )  # pca needs to be fit first to the same data as trained
        locus_feat_arr_pca = pca.transform(
            locus_feat_arr_scaled
        )  # pca transform  new data

        # Create or load the ANNOY index
        # index_nm = "../dataset_bank_60pca_annoy_index" #5k, 1000 trees
        # index_file = "../dataset_bank_60pca_annoy_index.ann" #5k, 1000 trees
        index_nm = "../data/dataset_bank_orig_5472objs_pcaTrue_hostTrue_annoy_index"
        index_file = index_nm + ".ann"
        index_dim = n_components  # Dimension of the PCA index

        # 3. Use the ANNOY index to find nearest neighbors
        print("Loading previously saved ANNOY LC+HOST PCA=60 index")
        print(index_file)

        index = annoy.AnnoyIndex(index_dim, metric="manhattan")
        index.load(index_file)
        idx_arr = np.load(f"{index_nm}_idx_arr.npy", allow_pickle=True)

        ann_start_time = time.time()
        ann_indexes, ann_dists = index.get_nns_by_vector(
            locus_feat_arr_pca[0], n=n, include_distances=True
        )
        ann_alerce_links = [
            f"https://alerce.online/object/{idx_arr[i]}" for i in ann_indexes
        ]
        ann_end_time = time.time()

    else:

        if l_or_ztfid_ref_in_dataset_bank:
            locus_feat_arr = locus_feat_arr[0:62]

        # 1, 2, 3. Don't use PCA at all. Just use LC features only + ANNOY index to find nearest neighbors
        # Create or load the ANNOY index

        # TODO: make argument for which file
        # index_nm = "dataset_bank_LCfeats_only_annoy_index" #5k, 1000 trees
        # index_file = "../dataset_bank_LCfeats_only_annoy_index.ann" #5k, 1000 trees

        # index_nm = "../bigbank_90k_LCfeats_only_annoy_index_100trees" #90k, 100 trees
        # index_file = "../bigbank_90k_LCfeats_only_annoy_index_100trees.ann" #90k, 100 trees

        # index_nm = "../" #90k, 1000 trees
        # index_file = "../bigbank_90k_LCfeats_only_annoy_index_1000trees.ann" #90k, 1000 trees
        index_file = "../data/loci_df_271688objects_cut_stars_and_gal_plane_pcaFalse_hostFalse_annoy_index.ann"
        index_nm = get_base_name(index_file)
        index_dim = 62  # Dimension of the index

        print("Loading previously saved ANNOY LC-only index")
        print(index_file)
        index = annoy.AnnoyIndex(index_dim, metric="manhattan")
        index.load(index_file)
        idx_arr = np.load(f"../data/{index_nm}_idx_arr.npy", allow_pickle=True)

        ann_start_time = time.time()
        ann_indexes, ann_dists = index.get_nns_by_vector(
            locus_feat_arr, n=n, include_distances=True
        )
        ann_alerce_links = [
            f"https://alerce.online/object/{idx_arr[i]}" for i in ann_indexes
        ]
        ann_end_time = time.time()

    # 4. Get TNS, spec. class of ANNs
    tns_ann_names, tns_ann_classes, tns_ann_zs = [], [], []
    ann_locus_l = []
    for i in ann_indexes:
        ann_locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=idx_arr[i])
        ann_locus_l.append(ann_locus)
        try:
            ann_tns = ann_locus.catalog_objects["tns_public_objects"][0]
            tns_ann_name, tns_ann_cls, tns_ann_z = (
                ann_tns["name"],
                ann_tns["type"],
                ann_tns["redshift"],
            )
        except:
            tns_ann_name, tns_ann_cls, tns_ann_z = "No TNS", "---", -99
        if tns_ann_cls == "":
            tns_ann_cls, tns_ann_z = "---", -99
        tns_ann_names.append(tns_ann_name), tns_ann_classes.append(
            tns_ann_cls
        ), tns_ann_zs.append(tns_ann_z)
        host_df_ztf_id_l.append(idx_arr[i])

    # Print the nearest neighbors
    print("\t\t\t\t\t   ZTFID IAU_NAME SPEC Z")
    print(
        f"LC REF: https://alerce.online/object/{LC_ztfid_ref} {LC_tns_name} {LC_tns_cls} {LC_tns_z}"
    )
    print(
        f"HOST REF: https://alerce.online/object/{HOST_ztfid_ref} {HOST_tns_name} {HOST_tns_cls} {HOST_tns_z}"
    )

    ann_num_l = []
    for i, (al, iau_name, spec_cls, z) in enumerate(
        zip(ann_alerce_links, tns_ann_names, tns_ann_classes, tns_ann_zs)
    ):
        if l_or_ztfid_ref.startswith("ZTF"):
            if i == 0:
                # continue
                pass
            print(f"ANN={i}: {al} {iau_name} {spec_cls}, {z}")
            ann_num_l.append(i)
        else:
            print(f"ANN={i+1}: {al} {iau_name} {spec_cls} {z}")
            ann_num_l.append(i + 1)

    end_time = time.time()
    ann_elapsed_time = ann_end_time - ann_start_time
    elapsed_time = end_time - start_time
    print(f"\nANN elapsed_time = {round(ann_elapsed_time, 3)} s")
    print(f"\ntotal elapsed_time = {round(elapsed_time, 3)} s\n")

    if savetables:
        print("Saving reference+ANN table...")
        if l_or_ztfid_ref_in_dataset_bank:
            ref_and_ann_df = pd.DataFrame(
                zip(
                    host_df_ztf_id_l,
                    list(range(0, n + 1)),
                    tns_ann_names,
                    tns_ann_classes,
                    tns_ann_zs,
                ),
                columns=["ZTFID", "ANN_NUM", "IAU_NAME", "SPEC_CLS", "Z"],
            )
        else:
            ref_and_ann_df = pd.DataFrame(
                zip(
                    host_df_ztf_id_l,
                    list(range(0, n + 1)),
                    [HOST_tns_name] + tns_ann_names,
                    [HOST_tns_cls] + tns_ann_classes,
                    [HOST_tns_z] + tns_ann_zs,
                ),
                columns=["ZTFID", "ANN_NUM", "IAU_NAME", "SPEC_CLS", "Z"],
            )
        ref_and_ann_df.to_csv(
            f"{table_path}/{ztfid_ref}_ann={ann_num}.csv", index=False
        )
        print(f"CSV saved at: {table_path}/{ztfid_ref}_ann={ann_num}.csv")

    if show_lightcurves_grid:
        print("Making a plot of stacked lightcurves...")

        if LC_tns_z is None:
            LC_tns_z = "None"
        elif isinstance(LC_tns_z, float):
            LC_tns_z = round(LC_tns_z, 3)
        else:
            LC_tns_z = LC_tns_z

        if use_ysepz_phot_snana_file:
            try:
                df_ref = SN_df
            except:
                print("No timeseries data...pass!")
                pass

            fig, ax = plt.subplots(figsize=(9.5, 6))

            df_ref_g = df_ref[(df_ref.FLT == "g") & (~df_ref.MAG.isna())]
            df_ref_r = df_ref[(df_ref.FLT == "R") & (~df_ref.MAG.isna())]

            mjd_idx_at_min_mag_r_ref = df_ref_r[["MAG"]].reset_index().idxmin().MAG
            mjd_idx_at_min_mag_g_ref = df_ref_g[["MAG"]].reset_index().idxmin().MAG

            ax.errorbar(
                x=df_ref_r.MJD - df_ref_r.MJD.iloc[mjd_idx_at_min_mag_r_ref],
                y=df_ref_r.MAG.min() - df_ref_r.MAG,
                yerr=df_ref_r.MAGERR,
                fmt="o",
                c="r",
                label=f"LC REF: {LC_ztfid_ref}, HOST REF: {HOST_ztfid_ref}, For LC: d=0\n{LC_tns_name},\t{LC_tns_cls},\tz={LC_tns_z}",
            )
            ax.errorbar(
                x=df_ref_g.MJD - df_ref_g.MJD.iloc[mjd_idx_at_min_mag_g_ref],
                y=df_ref_g.MAG.min() - df_ref_g.MAG,
                yerr=df_ref_g.MAGERR,
                fmt="o",
                c="g",
            )

        else:
            ref_info = antares_client.search.get_by_ztf_object_id(
                ztf_object_id=LC_ztfid_ref
            )
            try:
                df_ref = ref_info.timeseries.to_pandas()
            except:
                print("No timeseries data...pass!")
                pass

            fig, ax = plt.subplots(figsize=(9.5, 6))

            df_ref_g = df_ref[(df_ref.ant_passband == "g") & (~df_ref.ant_mag.isna())]
            df_ref_r = df_ref[(df_ref.ant_passband == "R") & (~df_ref.ant_mag.isna())]

            mjd_idx_at_min_mag_r_ref = (
                df_ref_r[["ant_mag"]].reset_index().idxmin().ant_mag
            )
            mjd_idx_at_min_mag_g_ref = (
                df_ref_g[["ant_mag"]].reset_index().idxmin().ant_mag
            )

            ax.errorbar(
                x=df_ref_r.ant_mjd - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref],
                y=df_ref_r.ant_mag.min() - df_ref_r.ant_mag,
                yerr=df_ref_r.ant_magerr,
                fmt="o",
                c="r",
                label=f"LC REF: {LC_ztfid_ref}, HOST REF: {HOST_ztfid_ref}, For LC: d=0\n{LC_tns_name},\t{LC_tns_cls},\tz={LC_tns_z}",
            )
            ax.errorbar(
                x=df_ref_g.ant_mjd - df_ref_g.ant_mjd.iloc[mjd_idx_at_min_mag_g_ref],
                y=df_ref_g.ant_mag.min() - df_ref_g.ant_mag,
                yerr=df_ref_g.ant_magerr,
                fmt="o",
                c="g",
            )

        markers = ["s", "*", "x", "P", "^", "v", "D", "<", ">", "8", "p", "x"]
        consts = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]

        if l_or_ztfid_ref_in_dataset_bank:
            ann_locus_l = ann_locus_l[1:]
            host_df_ztf_id_l = host_df_ztf_id_l
            ann_dists = ann_dists[1:]
            tns_ann_names = tns_ann_names[1:]
            tns_ann_classes = tns_ann_classes[1:]
            tns_ann_zs = tns_ann_zs[1:]

        for num, (l_info, ztfname, dist, iau_name, spec_cls, z) in enumerate(
            zip(
                ann_locus_l,
                host_df_ztf_id_l[1:],
                ann_dists,
                tns_ann_names,
                tns_ann_classes,
                tns_ann_zs,
            )
        ):
            try:
                alpha = 0.25
                c1 = "darkred"
                c2 = "darkgreen"

                if ztfname == "ZTF21achjwus" or ztfname == "ZTF20acnznol":
                    alpha = 0.75

                df_knn = l_info.timeseries.to_pandas()

                df_g = df_knn[(df_knn.ant_passband == "g") & (~df_knn.ant_mag.isna())]
                df_r = df_knn[(df_knn.ant_passband == "R") & (~df_knn.ant_mag.isna())]

                mjd_idx_at_min_mag_r = df_r[["ant_mag"]].reset_index().idxmin().ant_mag
                mjd_idx_at_min_mag_g = df_g[["ant_mag"]].reset_index().idxmin().ant_mag

                ax.errorbar(
                    x=df_r.ant_mjd - df_r.ant_mjd.iloc[mjd_idx_at_min_mag_r],
                    y=df_r.ant_mag.min() - df_r.ant_mag,
                    yerr=df_r.ant_magerr,
                    fmt=markers[num],
                    c=c1,
                    alpha=alpha,
                    label=f"ANN={num}: {ztfname}, d={int(dist)}\n{iau_name},\t{spec_cls},\tz={round(z, 3)}",
                )
                ax.errorbar(
                    x=df_g.ant_mjd - df_g.ant_mjd.iloc[mjd_idx_at_min_mag_g],
                    y=df_g.ant_mag.min() - df_g.ant_mag,
                    yerr=df_g.ant_magerr,
                    fmt=markers[num],
                    c=c2,
                    alpha=alpha,
                )
                # ax.text(df_ref_r.ant_mjd.iloc[-1]-df_ref_r.ant_mjd.iloc[0]+15, df_r.ant_mag[-1]-df_r.ant_mag.min(), s=f'ANN={num+1}: {has_tns_knn}   {tns_cls_knn}')

                plt.ylabel("Apparent Mag. + Constant")
                # plt.xlabel('Days of event') # make iloc[0]
                plt.xlabel(
                    "Days since peak ($r$, $g$ indep.)"
                )  # (need r, g to be same)

                if use_ysepz_phot_snana_file:
                    if (
                        df_ref_r.MJD.iloc[0]
                        - df_ref_r.MJD.iloc[mjd_idx_at_min_mag_r_ref]
                        <= 10
                    ):
                        plt.xlim(
                            (
                                df_ref_r.MJD.iloc[0]
                                - df_ref_r.MJD.iloc[mjd_idx_at_min_mag_r_ref]
                            )
                            - 20,
                            df_ref_r.MJD.iloc[-1] - df_ref_r.MJD.iloc[0] + 15,
                        )
                    else:
                        plt.xlim(
                            2
                            * (
                                df_ref_r.MJD.iloc[0]
                                - df_ref_r.MJD.iloc[mjd_idx_at_min_mag_r_ref]
                            ),
                            df_ref_r.MJD.iloc[-1] - df_ref_r.MJD.iloc[0] + 15,
                        )

                else:
                    if (
                        df_ref_r.ant_mjd.iloc[0]
                        - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]
                        <= 10
                    ):
                        plt.xlim(
                            (
                                df_ref_r.ant_mjd.iloc[0]
                                - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]
                            )
                            - 20,
                            df_ref_r.ant_mjd.iloc[-1] - df_ref_r.ant_mjd.iloc[0] + 15,
                        )
                    else:
                        plt.xlim(
                            2
                            * (
                                df_ref_r.ant_mjd.iloc[0]
                                - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]
                            ),
                            df_ref_r.ant_mjd.iloc[-1] - df_ref_r.ant_mjd.iloc[0] + 15,
                        )

                plt.legend(
                    frameon=False,
                    loc="upper right",
                    bbox_to_anchor=(0.52, 0.85, 0.5, 0.5),
                    ncol=3,
                    columnspacing=0.75,
                    prop={"size": 12},
                )

                plt.grid(True)

                plt.xlim(-24, 107)

            except Exception as e:
                print(
                    f"Something went wrong with plotting {ztfname}! Error is {e}. Continue..."
                )

        if savefigs:
            print("Saving stacked lightcurve...")
            plt.savefig(
                f"{figure_path}/{LC_ztfid_ref}_stacked_lightcurve_ann={ann_num}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            print(
                f"PDF saved at: {figure_path}/{LC_ztfid_ref}_stacked_lightcurve_ann={ann_num}.pdf"
            )

        plt.show()

    if show_hosts_grid:  # TODO: Get images to show (currently all red squares)
        print("\nGenerating hosts grid plot...")

        dataset_bank_orig_w_hosts_ra_dec = pd.read_csv(
            "../data/dataset_bank_orig_w_hosts_ra_dec_5472objs.csv.gz",
            compression="gzip",
            index_col=0,
        )
        for j, ztfid in enumerate(
            host_df_ztf_id_l
        ):  # first entry is reference, which we already calculated
            if j == 0:
                try:
                    print(
                        f"REF.  ({ztfid}): http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={host_df_ra_l[0]}+{host_df_dec_l[0]}&filter=color"
                    )
                    continue
                except:
                    print(
                        f"REF.  ({ztfid}): http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={dataset_bank_orig_w_hosts_ra_dec.loc[ztfid].raMean}+{dataset_bank_orig_w_hosts_ra_dec.loc[ztfid].decMean}&filter=color"
                    )
                    pass
            h_ra, h_dec = (
                dataset_bank_orig_w_hosts_ra_dec.loc[ztfid].raMean,
                dataset_bank_orig_w_hosts_ra_dec.loc[ztfid].decMean,
            )
            host_df_ra_l.append(h_ra), host_df_dec_l.append(h_dec)
            if j == 0:
                continue
            print(
                f"ANN={j} ({ztfid}): http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={h_ra}+{h_dec}&filter=color"
            )
        host_5ann_df = pd.DataFrame(
            zip(host_df_ztf_id_l, host_df_ra_l, host_df_dec_l),
            columns=["ZTFID", "HOST_RA", "HOST_DEC"],
        )
        if savefigs:
            print("Saving host thumbnails pdf...")
            host_pdfs(
                ztfid_ref=HOST_ztfid_ref,
                df=host_5ann_df,
                figure_path=figure_path,
                ann_num=ann_num,
                save_pdf=True,
            )
        else:
            host_pdfs(
                ztfid_ref=HOST_ztfid_ref,
                df=host_5ann_df,
                figure_path=figure_path,
                ann_num=ann_num,
                save_pdf=False,
            )

        if savetables:
            print("Saving host thumbnails table...")
            host_5ann_df.to_csv(
                f"{table_path}/{HOST_ztfid_ref}_host_thumbnails_ann={ann_num}.csv",
                index=False,
            )
            print(
                f"CSV saved at: {table_path}/{HOST_ztfid_ref}_host_thumbnails_ann={ann_num}.csv"
            )

    if run_AD_model:
        n_estimators = ad_params["n_estimators"]
        max_depth = ad_params["max_depth"]
        random_state = ad_params["random_state"]
        max_features = ad_params["max_features"]

        figure_path = f"../models/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced/figures"
        model_path = f"../models/SMOTE_train_test_70-30_min14_kneighbors8/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced/model"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        with open(
            f"{model_path}/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced.pkl",
            "rb",
        ) as f:
            clf = pickle.load(f)

        print("\nRunning AD Model...")
        if needs_reextraction_for_AD:
            print("Needs re-extraction for full timeseries.")
            print("Checking if made before...")
            if os.path.exists(f".../timeseries/{LC_ztfid_ref}_timeseries.csv"):
                print(f"{LC_ztfid_ref} is already made. Continue!\n")
            else:
                print("Re-extracting LC+HOST features")
                if use_ysepz_phot_snana_file:
                    print("Using YSE-PZ SNANA Photometry file...")
                    extract_lc_and_host_features_YSE_snana_format(
                        IAU_name=IAU_name,
                        ztf_id_ref=LC_l_or_ztfid_ref,
                        yse_lightcurve=SN_df,
                        ra=ra,
                        dec=dec,
                        show_lc=False,
                        show_host=True,
                        host_features=host_features,
                    )
                else:
                    extract_lc_and_host_features(
                        ztf_id_ref=LC_ztfid_ref,
                        use_lc_for_ann_only_bool=use_lc_for_ann_only_bool,
                        show_lc=False,
                        show_host=True,
                        host_features=host_features,
                    )

            try:
                lc_and_hosts_df = pd.read_csv(
                    f"../timeseries/{LC_ztfid_ref}_timeseries.csv"
                )
            except:
                print(
                    f"couldn't feature space as func of time for {LC_ztfid_ref}. pass."
                )
                return

            try:
                print(
                    f"HOST : http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={lc_and_hosts_df.iloc[0]['raMean']}+{lc_and_hosts_df.iloc[0]['decMean']}&filter=color\n"
                )
            except:
                pass

            lc_and_hosts_df = lc_and_hosts_df.dropna()
            try:
                lc_and_hosts_df_120d = lc_and_hosts_df[lc_and_host_features]
            except:
                print(f"{LC_ztfid_ref} has some NaN LC features. Skip!")

        if use_ysepz_phot_snana_file:
            plot_RFC_prob_vs_lc_yse_IAUid(
                clf=clf,
                IAU_name=IAU_name,
                anom_ztfid=LC_l_or_ztfid_ref,
                anom_spec_cls=LC_tns_cls,
                anom_spec_z=LC_tns_z,
                anom_thresh=50,
                lc_and_hosts_df=lc_and_hosts_df,
                lc_and_hosts_df_120d=lc_and_hosts_df_120d,
                yse_lightcurve=SN_df,
                savefig=savefigs,
                figure_path=figure_path,
            )
        else:
            plot_RFC_prob_vs_lc_ztfid(
                clf=clf,
                anom_ztfid=LC_ztfid_ref,
                anom_spec_cls=LC_tns_cls,
                anom_spec_z=LC_tns_z,
                anom_thresh=50,
                lc_and_hosts_df=lc_and_hosts_df,
                lc_and_hosts_df_120d=lc_and_hosts_df_120d,
                ref_info=locus,
                savefig=savefigs,
                figure_path=figure_path,
            )


def simple_LAISS(
    LC_l_or_ztfid_ref,
    HOST_l_or_ztfid_ref,
    lc_features,
    host_features=[],
    use_pca_for_nn=True,
    search_k=1000,
    n=8,
    show_lightcurves_grid=False,
    run_AD_model=False,
    ad_params={},
):
    print("Running LAISS...")
    lc_and_host_features = lc_features + host_features
    start_time = time.time()
    ann_num = n
    l_or_ztfid_refs = [LC_l_or_ztfid_ref, HOST_l_or_ztfid_ref]

    ############# LOOP HERE #############
    host = False
    n_flag = False
    for i, l_or_ztfid_ref in enumerate(l_or_ztfid_refs):
        if i == 1:
            host = True

        figure_path = f"../LAISS_run/{l_or_ztfid_ref}/figures"
        table_path = f"../LAISS_run/{l_or_ztfid_ref}/tables"

        l_or_ztfid_ref_in_dataset_bank = False
        host_df_ztf_id_l, host_df_ra_l, host_df_dec_l = [], [], []

        if l_or_ztfid_ref.startswith("ZTF"):
            ztfid_ref = l_or_ztfid_ref

            try:
                dataset_bank_orig = pd.read_csv(
                    "../data/dataset_bank_orig_5472objs.csv.gz",
                    compression="gzip",
                    index_col=0,
                )
                locus_feat_arr = dataset_bank_orig.loc[ztfid_ref]
                if host:
                    HOST_locus_feat_arr = locus_feat_arr[lc_and_host_features].values
                else:
                    LC_locus_feat_arr = locus_feat_arr[lc_and_host_features].values

                l_or_ztfid_ref_in_dataset_bank = True
                print(f"{l_or_ztfid_ref} is in dataset_bank")

                if not n_flag:
                    n = n + 1
                    n_flag = True

            except:
                print(
                    f"{l_or_ztfid_ref} is not in dataset_bank. Checking if made before..."
                )
                if os.path.exists(f"../timeseries/{l_or_ztfid_ref}_timeseries.csv"):
                    print(f"{l_or_ztfid_ref} is already made. Continue!\n")

                else:
                    print("Re-extracting LC+Host features")
                    extract_lc_and_host_features(
                        ztf_id_ref=ztfid_ref,
                        use_lc_for_ann_only_bool=False,
                        show_lc=False,
                        show_host=True,
                        host_features=host_features,
                    )

                try:
                    lc_and_hosts_df = pd.read_csv(
                        f"../timeseries/{l_or_ztfid_ref}_timeseries.csv"
                    )
                except:
                    print(
                        f"couldn't feature space as function of time for {l_or_ztfid_ref}. pass."
                    )
                    return

                if host:
                    print(
                        f"HOST : http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={lc_and_hosts_df.iloc[0]['raMean']}+{lc_and_hosts_df.iloc[0]['decMean']}&filter=color\n"
                    )
                    host_df_ztf_id_l.append(ztfid_ref), host_df_ra_l.append(
                        lc_and_hosts_df.iloc[0]["raMean"]
                    ), host_df_dec_l.append(lc_and_hosts_df.iloc[0]["decMean"])

                lc_and_hosts_df = (
                    lc_and_hosts_df.dropna()
                )  # if this drops all rows, that means something is nan from a 0 or nan entry (check data file)

                try:
                    lc_and_hosts_df_120d = lc_and_hosts_df[lc_and_host_features]
                except:
                    print(f"{ztfid_ref} has some NaN LC features. Skip!")
                    return

                anom_obj_df = pd.DataFrame(
                    lc_and_hosts_df_120d.iloc[-1]
                ).T  # last row of df to test "full LC only"
                if host:
                    HOST_locus_feat_arr = anom_obj_df.values[0]
                else:
                    LC_locus_feat_arr = anom_obj_df.values[0]

            locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztfid_ref)
            try:
                tns = locus.catalog_objects["tns_public_objects"][0]
                tns_name, tns_cls, tns_z = tns["name"], tns["type"], tns["redshift"]
            except:
                tns_name, tns_cls, tns_z = "No TNS", "---", -99
            if tns_cls == "":
                tns_cls, tns_ann_z = "---", -99

        else:
            raise ValueError("Input must be a string representing a ztfid_ref)")

        if host:
            HOST_ztfid_ref = ztfid_ref
            HOST_tns_name, HOST_tns_cls, HOST_tns_z = tns_name, tns_cls, tns_z
        else:
            LC_ztfid_ref = ztfid_ref
            LC_tns_name, LC_tns_cls, LC_tns_z = tns_name, tns_cls, tns_z

    ###### END OF LOOP ######
    # Create new feature array with mixed lc and host features

    subset_lc_features = LC_locus_feat_arr[:62]
    subset_temp_host_features = HOST_locus_feat_arr[-58:]
    locus_feat_arr = np.concatenate((subset_lc_features, subset_temp_host_features))

    if use_pca_for_nn:
        # 1. Scale locus_feat_arr using the same scaler (Standard Scaler)
        scaler = preprocessing.StandardScaler()
        trained_PCA_feat_arr = np.load(
            f"../data/dataset_bank_orig_5472objs_pcaTrue_hostTrue_annoy_index_feat_arr.npy",
            allow_pickle=True,
        )

        trained_PCA_feat_arr_scaled = scaler.fit_transform(
            trained_PCA_feat_arr
        )  # scaler needs to be fit first to the same data as trained

        locus_feat_arr_scaled = scaler.transform(
            [locus_feat_arr]
        )  # scaler transform new data

        # 2. Transform the scaled locus_feat_arr using the same PCA model (60 PCs, RS=42)
        n_components = 60
        random_seed = 42
        pca = PCA(n_components=n_components, random_state=random_seed)
        trained_PCA_feat_arr_scaled_pca = pca.fit_transform(
            trained_PCA_feat_arr_scaled
        )  # pca needs to be fit first to the same data as trained
        locus_feat_arr_pca = pca.transform(
            locus_feat_arr_scaled
        )  # pca transform  new data

        # Create or load the ANNOY index
        # index_nm = "../dataset_bank_60pca_annoy_index" #5k, 1000 trees
        # index_file = "../dataset_bank_60pca_annoy_index.ann" #5k, 1000 trees
        index_nm = "../data/dataset_bank_orig_5472objs_pcaTrue_hostTrue_annoy_index"
        index_file = index_nm + ".ann"
        index_dim = n_components  # Dimension of the PCA index

        # 3. Use the ANNOY index to find nearest neighbors
        print("Loading previously saved ANNOY LC+HOST PCA=60 index")
        print(index_file)

        index = annoy.AnnoyIndex(index_dim, metric="manhattan")
        index.load(index_file)
        idx_arr = np.load(f"{index_nm}_idx_arr.npy", allow_pickle=True)

        ann_start_time = time.time()
        ann_indexes, ann_dists = index.get_nns_by_vector(
            locus_feat_arr_pca[0], n=n, search_k=search_k, include_distances=True
        )
        ann_alerce_links = [
            f"https://alerce.online/object/{idx_arr[i]}" for i in ann_indexes
        ]
        ann_end_time = time.time()
    else:
        # Create or load the ANNOY index
        index_nm = "../data/loci_df_271688objects_cut_stars_and_gal_plane_pcaFalse_hostFalse_annoy_index"
        index_file = index_nm + ".ann"
        index_dim = 62

        # 3. Use the ANNOY index to find nearest neighbors
        print("Loading previously saved ANNOY LC+HOST index without PCA:")
        print(index_file)

        index = annoy.AnnoyIndex(index_dim, metric="manhattan")
        index.load(index_file)
        idx_arr = np.load(f"{index_nm}_idx_arr.npy", allow_pickle=True)

        ann_start_time = time.time()
        ann_indexes, ann_dists = index.get_nns_by_vector(
            locus_feat_arr[:62], n=n, search_k=search_k, include_distances=True
        )
        print(ann_indexes)
        ann_alerce_links = [
            f"https://alerce.online/object/{idx_arr[i]}" for i in ann_indexes
        ]
        ann_end_time = time.time()

    # 4. Get TNS, spec. class of ANNs
    tns_ann_names, tns_ann_classes, tns_ann_zs = [], [], []
    ann_locus_l = []
    for i in ann_indexes:
        ann_locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=idx_arr[i])
        ann_locus_l.append(ann_locus)
        try:
            ann_tns = ann_locus.catalog_objects["tns_public_objects"][0]
            tns_ann_name, tns_ann_cls, tns_ann_z = (
                ann_tns["name"],
                ann_tns["type"],
                ann_tns["redshift"],
            )
        except:
            tns_ann_name, tns_ann_cls, tns_ann_z = "No TNS", "---", -99
        if tns_ann_cls == "":
            tns_ann_cls, tns_ann_z = "---", -99
        tns_ann_names.append(tns_ann_name), tns_ann_classes.append(
            tns_ann_cls
        ), tns_ann_zs.append(tns_ann_z)
        host_df_ztf_id_l.append(idx_arr[i])

    # Print the nearest neighbors
    print("\t\t\t\t\t   ZTFID IAU_NAME SPEC Z")
    print(
        f"LC REF: https://alerce.online/object/{LC_ztfid_ref} {LC_tns_name} {LC_tns_cls} {LC_tns_z}"
    )
    print(
        f"HOST REF: https://alerce.online/object/{HOST_ztfid_ref} {HOST_tns_name} {HOST_tns_cls} {HOST_tns_z}"
    )

    ann_num_l = []
    for i, (al, iau_name, spec_cls, z) in enumerate(
        zip(ann_alerce_links, tns_ann_names, tns_ann_classes, tns_ann_zs)
    ):
        if l_or_ztfid_ref.startswith("ZTF"):
            if i == 0:
                # continue
                pass
            print(f"ANN={i}: {al} {iau_name} {spec_cls}, {z}")
            ann_num_l.append(i)
        else:
            print(f"ANN={i+1}: {al} {iau_name} {spec_cls} {z}")
            ann_num_l.append(i + 1)

    end_time = time.time()
    ann_elapsed_time = ann_end_time - ann_start_time
    elapsed_time = end_time - start_time
    print(f"\nANN elapsed_time = {round(ann_elapsed_time, 3)} s")
    print(f"\ntotal elapsed_time = {round(elapsed_time, 3)} s\n")

    if show_lightcurves_grid:
        print("Making a plot of stacked lightcurves...")

        if LC_tns_z is None:
            LC_tns_z = "None"
        elif isinstance(LC_tns_z, float):
            LC_tns_z = round(LC_tns_z, 3)
        else:
            LC_tns_z = LC_tns_z

        ref_info = antares_client.search.get_by_ztf_object_id(
            ztf_object_id=LC_ztfid_ref
        )
        try:
            df_ref = ref_info.timeseries.to_pandas()
        except:
            print("No timeseries data...pass!")
            pass

        fig, ax = plt.subplots(figsize=(9.5, 6))

        df_ref_g = df_ref[(df_ref.ant_passband == "g") & (~df_ref.ant_mag.isna())]
        df_ref_r = df_ref[(df_ref.ant_passband == "R") & (~df_ref.ant_mag.isna())]

        mjd_idx_at_min_mag_r_ref = df_ref_r[["ant_mag"]].reset_index().idxmin().ant_mag
        mjd_idx_at_min_mag_g_ref = df_ref_g[["ant_mag"]].reset_index().idxmin().ant_mag

        ax.errorbar(
            x=df_ref_r.ant_mjd - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref],
            y=df_ref_r.ant_mag.min() - df_ref_r.ant_mag,
            yerr=df_ref_r.ant_magerr,
            fmt="o",
            c="r",
            label=f"LC REF: {LC_ztfid_ref}, HOST REF: {HOST_ztfid_ref}, For LC: d=0\n{LC_tns_name},\t{LC_tns_cls},\tz={LC_tns_z}",
        )
        ax.errorbar(
            x=df_ref_g.ant_mjd - df_ref_g.ant_mjd.iloc[mjd_idx_at_min_mag_g_ref],
            y=df_ref_g.ant_mag.min() - df_ref_g.ant_mag,
            yerr=df_ref_g.ant_magerr,
            fmt="o",
            c="g",
        )

        markers = ["s", "*", "x", "P", "^", "v", "D", "<", ">", "8", "p", "x"]
        consts = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]

        if l_or_ztfid_ref_in_dataset_bank:
            ann_locus_l = ann_locus_l[1:]
            host_df_ztf_id_l = host_df_ztf_id_l
            ann_dists = ann_dists[1:]
            tns_ann_names = tns_ann_names[1:]
            tns_ann_classes = tns_ann_classes[1:]
            tns_ann_zs = tns_ann_zs[1:]

        for num, (l_info, ztfname, dist, iau_name, spec_cls, z) in enumerate(
            zip(
                ann_locus_l,
                host_df_ztf_id_l[1:],
                ann_dists,
                tns_ann_names,
                tns_ann_classes,
                tns_ann_zs,
            )
        ):
            try:
                alpha = 0.25
                c1 = "darkred"
                c2 = "darkgreen"

                if ztfname == "ZTF21achjwus" or ztfname == "ZTF20acnznol":
                    alpha = 0.75

                df_knn = l_info.timeseries.to_pandas()

                df_g = df_knn[(df_knn.ant_passband == "g") & (~df_knn.ant_mag.isna())]
                df_r = df_knn[(df_knn.ant_passband == "R") & (~df_knn.ant_mag.isna())]

                mjd_idx_at_min_mag_r = df_r[["ant_mag"]].reset_index().idxmin().ant_mag
                mjd_idx_at_min_mag_g = df_g[["ant_mag"]].reset_index().idxmin().ant_mag

                ax.errorbar(
                    x=df_r.ant_mjd - df_r.ant_mjd.iloc[mjd_idx_at_min_mag_r],
                    y=df_r.ant_mag.min() - df_r.ant_mag,
                    yerr=df_r.ant_magerr,
                    fmt=markers[num],
                    c=c1,
                    alpha=alpha,
                    label=f"ANN={num}: {ztfname}, d={round(dist, 2)}\n{iau_name},\t{spec_cls},\tz={round(z, 3)}",
                )
                ax.errorbar(
                    x=df_g.ant_mjd - df_g.ant_mjd.iloc[mjd_idx_at_min_mag_g],
                    y=df_g.ant_mag.min() - df_g.ant_mag,
                    yerr=df_g.ant_magerr,
                    fmt=markers[num],
                    c=c2,
                    alpha=alpha,
                )
                # ax.text(df_ref_r.ant_mjd.iloc[-1]-df_ref_r.ant_mjd.iloc[0]+15, df_r.ant_mag[-1]-df_r.ant_mag.min(), s=f'ANN={num+1}: {has_tns_knn}   {tns_cls_knn}')

                plt.ylabel("Apparent Mag. + Constant")
                # plt.xlabel('Days of event') # make iloc[0]
                plt.xlabel(
                    "Days since peak ($r$, $g$ indep.)"
                )  # (need r, g to be same)

                if (
                    df_ref_r.ant_mjd.iloc[0]
                    - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]
                    <= 10
                ):
                    plt.xlim(
                        (
                            df_ref_r.ant_mjd.iloc[0]
                            - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]
                        )
                        - 20,
                        df_ref_r.ant_mjd.iloc[-1] - df_ref_r.ant_mjd.iloc[0] + 15,
                    )
                else:
                    plt.xlim(
                        2
                        * (
                            df_ref_r.ant_mjd.iloc[0]
                            - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]
                        ),
                        df_ref_r.ant_mjd.iloc[-1] - df_ref_r.ant_mjd.iloc[0] + 15,
                    )

                plt.legend(
                    frameon=False,
                    loc="upper right",
                    bbox_to_anchor=(0.52, 0.85, 0.5, 0.5),
                    ncol=3,
                    columnspacing=0.75,
                    prop={"size": 12},
                )

                plt.grid(True)

                plt.xlim(-24, 107)

            except Exception as e:
                print(
                    f"Something went wrong with plotting {ztfname}! Error is {e}. Continue..."
                )
        plt.show()

    if run_AD_model:
        n_estimators = ad_params["n_estimators"]
        max_depth = ad_params["max_depth"]
        random_state = ad_params["random_state"]
        max_features = ad_params["max_features"]

        figure_path = f"../models/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced/figures"
        model_path = f"../models/SMOTE_train_test_70-30_min14_kneighbors8/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced/model"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        with open(
            f"{model_path}/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced.pkl",
            "rb",
        ) as f:
            clf = pickle.load(f)

        print("\nRunning AD Model...")

        host = False
        for idx, ztf_id_temp in enumerate([LC_ztfid_ref, HOST_ztfid_ref]):
            if idx == 1:
                host = True

            print(f"Checking if {ztf_id_temp} made before...")

            timeseries_dir = os.path.abspath("../timeseries")
            file_path = os.path.join(timeseries_dir, f"{ztf_id_temp}_timeseries.csv")

            if os.path.exists(file_path):
                print(f"{ztf_id_temp} is already made. Continue!\n")
            else:
                print(f"Re-extracting LC+HOST features for {ztf_id_temp}")
                extract_lc_and_host_features(
                    ztf_id_ref=ztf_id_temp,
                    use_lc_for_ann_only_bool=False,
                    show_lc=False,
                    show_host=False,
                    host_features=host_features,
                )
                print(f"Completed re-extraction for {ztf_id_temp}")

            try:
                lc_and_hosts_df = pd.read_csv(
                    f"../timeseries/{ztf_id_temp}_timeseries.csv"
                )
            except:
                print(
                    f"couldn't feature space as func of time for {ztf_id_temp}. pass."
                )
                return

            # try:
            #     print(
            #         f"HOST : http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={lc_and_hosts_df.iloc[0]['raMean']}+{lc_and_hosts_df.iloc[0]['decMean']}&filter=color\n"
            #     )
            # except:
            #     pass

            lc_and_hosts_df = lc_and_hosts_df.dropna()
            try:
                lc_and_hosts_df_120d = lc_and_hosts_df[lc_and_host_features]
            except:
                print(f"{ztf_id_temp} has some NaN LC features. Skip!")

            if not host:
                LC_lc_and_hosts_df = lc_and_hosts_df
                LC_lc_and_hosts_df_120d = lc_and_hosts_df_120d
                LC_locus = antares_client.search.get_by_ztf_object_id(
                    ztf_object_id=ztf_id_temp
                )
            elif host:
                HOST_lc_and_hosts_df = lc_and_hosts_df
                HOST_lc_and_hosts_df_120d = lc_and_hosts_df_120d

        # Create combined dataframe for anomaly detection
        LC_HOST_COMBINED_lc_and_hosts_df = LC_lc_and_hosts_df
        same_value_columns = HOST_lc_and_hosts_df[host_features].apply(
            lambda x: x.nunique() == 1, axis=0
        )
        for column in host_features:
            if same_value_columns[
                column
            ]:  # Check if all rows in the column are the same
                LC_HOST_COMBINED_lc_and_hosts_df[column] = HOST_lc_and_hosts_df[
                    column
                ].iloc[
                    0
                ]  # Replace with the single host feat value
            else:
                print(f"ERROR: INCONSISTENT HOST FEATURE: {column}")

        LC_HOST_COMBINED_lc_and_hosts_df_120d = LC_HOST_COMBINED_lc_and_hosts_df[
            lc_and_host_features
        ]

        mod_plot_RFC_prob_vs_lc_ztfid(
            clf=clf,
            anom_ztfid=LC_ztfid_ref,
            host_ztf_id=HOST_ztfid_ref,
            anom_spec_cls=LC_tns_cls,
            anom_spec_z=LC_tns_z,
            anom_thresh=50,
            lc_and_hosts_df=LC_HOST_COMBINED_lc_and_hosts_df,
            lc_and_hosts_df_120d=LC_HOST_COMBINED_lc_and_hosts_df_120d,
            ref_info=LC_locus,
            savefig=False,
            figure_path=figure_path,
        )


def LAISS_primer(
    LC_l_or_ztfid_ref,
    HOST_l_or_ztfid_ref,
    lc_features,
    host_features=[],
):
    lc_and_host_features = lc_features + host_features
    l_or_ztfid_refs = [LC_l_or_ztfid_ref, HOST_l_or_ztfid_ref]

    host = False
    n_flag = False
    # Loop through lightcurve object and host object
    for i, l_or_ztfid_ref in enumerate(l_or_ztfid_refs):
        if i == 1:
            host = True

        l_or_ztfid_ref_in_dataset_bank = False
        host_df_ztf_id_l, host_df_ra_l, host_df_dec_l = [], [], []

        if l_or_ztfid_ref.startswith("ZTF"):
            ztfid_ref = l_or_ztfid_ref

            try:
                dataset_bank_orig = pd.read_csv(
                    "../data/dataset_bank_orig_5472objs.csv.gz",
                    compression="gzip",
                    index_col=0,
                )
                locus_feat_arr = dataset_bank_orig.loc[ztfid_ref]
                if host:
                    HOST_locus_feat_arr = locus_feat_arr[lc_and_host_features].values
                else:
                    LC_locus_feat_arr = locus_feat_arr[lc_and_host_features].values

                l_or_ztfid_ref_in_dataset_bank = True
                print(f"{l_or_ztfid_ref} is in dataset_bank")

                if not n_flag:
                    n = n + 1
                    n_flag = True

            except:
                print(
                    f"{l_or_ztfid_ref} is not in dataset_bank. Checking if made before..."
                )
                if os.path.exists(f"../timeseries/{l_or_ztfid_ref}_timeseries.csv"):
                    print(f"{l_or_ztfid_ref} is already made. Continue!\n")

                else:
                    print("Re-extracting LC+Host features")
                    extract_lc_and_host_features(
                        ztf_id_ref=ztfid_ref,
                        use_lc_for_ann_only_bool=False,
                        show_lc=False,
                        show_host=True,
                        host_features=host_features,
                    )

                try:
                    lc_and_hosts_df = pd.read_csv(
                        f"../timeseries/{l_or_ztfid_ref}_timeseries.csv"
                    )
                except:
                    print(
                        f"couldn't feature space as function of time for {l_or_ztfid_ref}. pass."
                    )
                    return

                if host:
                    print(
                        f"HOST : http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={lc_and_hosts_df.iloc[0]['raMean']}+{lc_and_hosts_df.iloc[0]['decMean']}&filter=color\n"
                    )
                    host_df_ztf_id_l.append(ztfid_ref), host_df_ra_l.append(
                        lc_and_hosts_df.iloc[0]["raMean"]
                    ), host_df_dec_l.append(lc_and_hosts_df.iloc[0]["decMean"])

                lc_and_hosts_df = (
                    lc_and_hosts_df.dropna()
                )  # if this drops all rows, that means something is nan from a 0 or nan entry (check data file)

                try:
                    lc_and_hosts_df_120d = lc_and_hosts_df[lc_and_host_features]
                except:
                    print(f"{ztfid_ref} has some NaN LC features. Skip!")
                    return

                anom_obj_df = pd.DataFrame(
                    lc_and_hosts_df_120d.iloc[-1]
                ).T  # last row of df to test "full LC only"
                if host:
                    HOST_locus_feat_arr = anom_obj_df.values[0]
                else:
                    LC_locus_feat_arr = anom_obj_df.values[0]

            locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztfid_ref)
            try:
                tns = locus.catalog_objects["tns_public_objects"][0]
                tns_name, tns_cls, tns_z = tns["name"], tns["type"], tns["redshift"]
            except:
                tns_name, tns_cls, tns_z = "No TNS", "---", -99
            if tns_cls == "":
                tns_cls, tns_ann_z = "---", -99

        else:
            raise ValueError("Input must be a string representing a ztfid_ref)")

        if host:
            HOST_ztfid_ref = ztfid_ref
            HOST_tns_name, HOST_tns_cls, HOST_tns_z = tns_name, tns_cls, tns_z
        else:
            LC_ztfid_ref = ztfid_ref
            LC_tns_name, LC_tns_cls, LC_tns_z = tns_name, tns_cls, tns_z

    # Create new feature array with mixed lc and host features
    subset_lc_features = LC_locus_feat_arr[:62]
    subset_temp_host_features = HOST_locus_feat_arr[-58:]
    locus_feat_arr = np.concatenate((subset_lc_features, subset_temp_host_features))

    output_dict = {
        "HOST_ztfid_ref": HOST_ztfid_ref,
        "HOST_tns_name": HOST_tns_name,
        "HOST_tns_cls": HOST_tns_cls,
        "HOST_tns_z": HOST_tns_z,
        "host_df_ztf_id_l": host_df_ztf_id_l,
        "LC_ztfid_ref": LC_ztfid_ref,
        "LC_tns_name": LC_tns_name,
        "LC_tns_cls": LC_tns_cls,
        "LC_tns_z": LC_tns_z,
        "locus_feat_arr": locus_feat_arr,
        "l_or_ztfid_ref_in_dataset_bank": l_or_ztfid_ref_in_dataset_bank,
    }

    return output_dict


def LAISS_nearest_neighbors(
    laiss_dict,
    use_pca_for_nn=True,
    annoy_index_file_path="",
    n=8,
    search_k=1000,
    show_lightcurves_grid=False,
    store_results=False,
):
    start_time = time.time()
    if use_pca_for_nn:
        # 1. Scale locus_feat_arr using the same scaler (Standard Scaler)
        scaler = preprocessing.StandardScaler()
        trained_PCA_feat_arr = np.load(
            annoy_index_file_path + "_feat_arr.npy",
            allow_pickle=True,
        )

        trained_PCA_feat_arr_scaled = scaler.fit_transform(
            trained_PCA_feat_arr
        )  # scaler needs to be fit first to the same data as trained

        locus_feat_arr_scaled = scaler.transform(
            [laiss_dict["locus_feat_arr"]]
        )  # scaler transform new data

        # 2. Transform the scaled locus_feat_arr using the same PCA model (60 PCs, RS=42)
        n_components = 60
        random_seed = 42
        pca = PCA(n_components=n_components, random_state=random_seed)
        trained_PCA_feat_arr_scaled_pca = pca.fit_transform(
            trained_PCA_feat_arr_scaled
        )  # pca needs to be fit first to the same data as trained
        locus_feat_arr_pca = pca.transform(
            locus_feat_arr_scaled
        )  # pca transform  new data

        index_nm = annoy_index_file_path
        index_file = index_nm + ".ann"
        index_dim = n_components  # Dimension of the PCA index

        # 3. Use the ANNOY index to find nearest neighbors
        print("Loading previously saved ANNOY LC+HOST PCA=60 index")
        print(index_file)

        index = annoy.AnnoyIndex(index_dim, metric="manhattan")
        index.load(index_file)
        idx_arr = np.load(f"{index_nm}_idx_arr.npy", allow_pickle=True)

        ann_start_time = time.time()
        ann_indexes, ann_dists = index.get_nns_by_vector(
            locus_feat_arr_pca[0], n=n, search_k=search_k, include_distances=True
        )
        ann_alerce_links = [
            f"https://alerce.online/object/{idx_arr[i]}" for i in ann_indexes
        ]
        ann_end_time = time.time()
    else:
        # Create or load the ANNOY index
        index_nm = annoy_index_file_path
        index_file = index_nm + ".ann"
        index_dim = 62

        # 3. Use the ANNOY index to find nearest neighbors
        print("Loading previously saved ANNOY LC+HOST index without PCA:")
        print(index_file)

        index = annoy.AnnoyIndex(index_dim, metric="manhattan")
        index.load(index_file)
        idx_arr = np.load(f"{index_nm}_idx_arr.npy", allow_pickle=True)

        ann_start_time = time.time()
        ann_indexes, ann_dists = index.get_nns_by_vector(
            laiss_dict["locus_feat_arr"][:62],
            n=n,
            search_k=search_k,
            include_distances=True,
        )
        print(ann_indexes)
        ann_alerce_links = [
            f"https://alerce.online/object/{idx_arr[i]}" for i in ann_indexes
        ]
        ann_end_time = time.time()

    # 4. Get TNS, spec. class of ANNs
    tns_ann_names, tns_ann_classes, tns_ann_zs = [], [], []
    ann_locus_l = []
    for i in ann_indexes:
        ann_locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=idx_arr[i])
        ann_locus_l.append(ann_locus)
        try:
            ann_tns = ann_locus.catalog_objects["tns_public_objects"][0]
            tns_ann_name, tns_ann_cls, tns_ann_z = (
                ann_tns["name"],
                ann_tns["type"],
                ann_tns["redshift"],
            )
        except:
            tns_ann_name, tns_ann_cls, tns_ann_z = "No TNS", "---", -99
        if tns_ann_cls == "":
            tns_ann_cls, tns_ann_z = "---", -99
        tns_ann_names.append(tns_ann_name), tns_ann_classes.append(
            tns_ann_cls
        ), tns_ann_zs.append(tns_ann_z)
        laiss_dict["host_df_ztf_id_l"].append(idx_arr[i])

    # Print the nearest neighbors
    print("\t\t\t\t\t   ZTFID IAU_NAME SPEC Z")
    print(
        f"LC REF: https://alerce.online/object/{laiss_dict['LC_ztfid_ref']} {laiss_dict['LC_tns_name']} {laiss_dict['LC_tns_cls']} {laiss_dict['LC_tns_z']}"
    )
    print(
        f"HOST REF: https://alerce.online/object/{laiss_dict['HOST_ztfid_ref']} {laiss_dict['HOST_tns_name']} {laiss_dict['HOST_tns_cls']} {laiss_dict['HOST_tns_z']}"
    )

    ann_num_l = []
    if store_results:
        storage = []
    for i, (al, iau_name, spec_cls, z, dist) in enumerate(
        zip(ann_alerce_links, tns_ann_names, tns_ann_classes, tns_ann_zs, ann_dists)
    ):
        if i == 0:
            # continue
            pass
        print(f"ANN={i}: {al} {iau_name} {spec_cls}, {z}")
        ann_num_l.append(i)
        if store_results:
            neighbor_dict = {
                "lightcurve_ztf": laiss_dict["LC_ztfid_ref"],
                "host_ztf": laiss_dict["HOST_ztfid_ref"],
                "neighbor_num": i,
                "ztf_link": al,
                "dist": dist,
                "iau_name": iau_name,
                "spec_cls": spec_cls,
                "z": z,
            }
            storage.append(neighbor_dict)

    end_time = time.time()
    ann_elapsed_time = ann_end_time - ann_start_time
    elapsed_time = end_time - start_time
    print(f"\nANN elapsed_time = {round(ann_elapsed_time, 3)} s")
    print(f"\ntotal elapsed_time = {round(elapsed_time, 3)} s\n")

    if show_lightcurves_grid:
        print("Making a plot of stacked lightcurves...")

        if laiss_dict["LC_tns_z"] is None:
            laiss_dict["LC_tns_z"] = "None"
        elif isinstance(laiss_dict["LC_tns_z"], float):
            laiss_dict["LC_tns_z"] = round(laiss_dict["LC_tns_z"], 3)
        else:
            laiss_dict["LC_tns_z"] = laiss_dict["LC_tns_z"]

        ref_info = antares_client.search.get_by_ztf_object_id(
            ztf_object_id=laiss_dict["LC_ztfid_ref"]
        )
        try:
            df_ref = ref_info.timeseries.to_pandas()
        except:
            print("No timeseries data...pass!")
            pass

        fig, ax = plt.subplots(figsize=(9.5, 6))

        df_ref_g = df_ref[(df_ref.ant_passband == "g") & (~df_ref.ant_mag.isna())]
        df_ref_r = df_ref[(df_ref.ant_passband == "R") & (~df_ref.ant_mag.isna())]

        mjd_idx_at_min_mag_r_ref = df_ref_r[["ant_mag"]].reset_index().idxmin().ant_mag
        mjd_idx_at_min_mag_g_ref = df_ref_g[["ant_mag"]].reset_index().idxmin().ant_mag

        ax.errorbar(
            x=df_ref_r.ant_mjd - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref],
            y=df_ref_r.ant_mag.min() - df_ref_r.ant_mag,
            yerr=df_ref_r.ant_magerr,
            fmt="o",
            c="r",
            label=f"LC REF: {laiss_dict['LC_ztfid_ref']}, HOST REF: {laiss_dict['HOST_ztfid_ref']}, For LC: d=0\n{laiss_dict['LC_tns_name']},\t{laiss_dict['LC_tns_cls']},\tz={laiss_dict['LC_tns_z']}",
        )
        ax.errorbar(
            x=df_ref_g.ant_mjd - df_ref_g.ant_mjd.iloc[mjd_idx_at_min_mag_g_ref],
            y=df_ref_g.ant_mag.min() - df_ref_g.ant_mag,
            yerr=df_ref_g.ant_magerr,
            fmt="o",
            c="g",
        )

        markers = ["s", "*", "x", "P", "^", "v", "D", "<", ">", "8", "p", "x"]
        consts = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]

        if laiss_dict["l_or_ztfid_ref_in_dataset_bank"]:
            ann_locus_l = ann_locus_l[1:]
            host_df_ztf_id_l = laiss_dict["host_df_ztf_id_l"]
            ann_dists = ann_dists[1:]
            tns_ann_names = tns_ann_names[1:]
            tns_ann_classes = tns_ann_classes[1:]
            tns_ann_zs = tns_ann_zs[1:]

        for num, (l_info, ztfname, dist, iau_name, spec_cls, z) in enumerate(
            zip(
                ann_locus_l,
                laiss_dict["host_df_ztf_id_l"][1:],
                ann_dists,
                tns_ann_names,
                tns_ann_classes,
                tns_ann_zs,
            )
        ):
            try:
                alpha = 0.25
                c1 = "darkred"
                c2 = "darkgreen"

                if ztfname == "ZTF21achjwus" or ztfname == "ZTF20acnznol":
                    alpha = 0.75

                df_knn = l_info.timeseries.to_pandas()

                df_g = df_knn[(df_knn.ant_passband == "g") & (~df_knn.ant_mag.isna())]
                df_r = df_knn[(df_knn.ant_passband == "R") & (~df_knn.ant_mag.isna())]

                mjd_idx_at_min_mag_r = df_r[["ant_mag"]].reset_index().idxmin().ant_mag
                mjd_idx_at_min_mag_g = df_g[["ant_mag"]].reset_index().idxmin().ant_mag

                ax.errorbar(
                    x=df_r.ant_mjd - df_r.ant_mjd.iloc[mjd_idx_at_min_mag_r],
                    y=df_r.ant_mag.min() - df_r.ant_mag,
                    yerr=df_r.ant_magerr,
                    fmt=markers[num],
                    c=c1,
                    alpha=alpha,
                    label=f"ANN={num}: {ztfname}, d={round(dist, 2)}\n{iau_name},\t{spec_cls},\tz={round(z, 3)}",
                )
                ax.errorbar(
                    x=df_g.ant_mjd - df_g.ant_mjd.iloc[mjd_idx_at_min_mag_g],
                    y=df_g.ant_mag.min() - df_g.ant_mag,
                    yerr=df_g.ant_magerr,
                    fmt=markers[num],
                    c=c2,
                    alpha=alpha,
                )

                plt.ylabel("Apparent Mag. + Constant")
                # plt.xlabel('Days of event') # make iloc[0]
                plt.xlabel(
                    "Days since peak ($r$, $g$ indep.)"
                )  # (need r, g to be same)

                if (
                    df_ref_r.ant_mjd.iloc[0]
                    - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]
                    <= 10
                ):
                    plt.xlim(
                        (
                            df_ref_r.ant_mjd.iloc[0]
                            - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]
                        )
                        - 20,
                        df_ref_r.ant_mjd.iloc[-1] - df_ref_r.ant_mjd.iloc[0] + 15,
                    )
                else:
                    plt.xlim(
                        2
                        * (
                            df_ref_r.ant_mjd.iloc[0]
                            - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]
                        ),
                        df_ref_r.ant_mjd.iloc[-1] - df_ref_r.ant_mjd.iloc[0] + 15,
                    )

                plt.legend(
                    frameon=False,
                    loc="upper right",
                    bbox_to_anchor=(0.52, 0.85, 0.5, 0.5),
                    ncol=3,
                    columnspacing=0.75,
                    prop={"size": 12},
                )

                plt.grid(True)

                plt.xlim(-24, 107)

            except Exception as e:
                print(
                    f"Something went wrong with plotting {ztfname}! Error is {e}. Continue..."
                )

        plt.show()

    if store_results:
        return pd.DataFrame(storage)


def LAISS_AD(
    laiss_dict,
    lc_features,
    host_features=[],
    ad_params={},
):
    lc_and_host_features = lc_features + host_features
    n_estimators = ad_params["n_estimators"]
    max_depth = ad_params["max_depth"]
    random_state = ad_params["random_state"]
    max_features = ad_params["max_features"]

    figure_path = f"../models/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced/figures"
    model_path = f"../models/SMOTE_train_test_70-30_min14_kneighbors8/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced/model"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with open(
        f"{model_path}/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced.pkl",
        "rb",
    ) as f:
        clf = pickle.load(f)

    print("\nRunning AD Model...")

    host = False
    for idx, ztf_id_temp in enumerate(
        [laiss_dict["LC_ztfid_ref"], laiss_dict["HOST_ztfid_ref"]]
    ):
        if idx == 1:
            host = True

        print(f"Checking if {ztf_id_temp} made before...")

        timeseries_dir = os.path.abspath("../timeseries")
        file_path = os.path.join(timeseries_dir, f"{ztf_id_temp}_timeseries.csv")

        if os.path.exists(file_path):
            print(f"{ztf_id_temp} is already made. Continue!\n")
        else:
            print(f"Re-extracting LC+HOST features for {ztf_id_temp}")
            extract_lc_and_host_features(
                ztf_id_ref=ztf_id_temp,
                use_lc_for_ann_only_bool=False,
                show_lc=False,
                show_host=False,
                host_features=host_features,
            )
            print(f"Completed re-extraction for {ztf_id_temp}")

        try:
            lc_and_hosts_df = pd.read_csv(f"../timeseries/{ztf_id_temp}_timeseries.csv")
        except:
            print(f"couldn't feature space as func of time for {ztf_id_temp}. pass.")
            return

        lc_and_hosts_df = lc_and_hosts_df.dropna()
        try:
            lc_and_hosts_df_120d = lc_and_hosts_df[lc_and_host_features]
        except:
            print(f"{ztf_id_temp} has some NaN LC features. Skip!")

        if not host:
            LC_lc_and_hosts_df = lc_and_hosts_df
            LC_lc_and_hosts_df_120d = lc_and_hosts_df_120d
            LC_locus = antares_client.search.get_by_ztf_object_id(
                ztf_object_id=ztf_id_temp
            )
        elif host:
            HOST_lc_and_hosts_df = lc_and_hosts_df
            HOST_lc_and_hosts_df_120d = lc_and_hosts_df_120d

    # Create combined dataframe for anomaly detection
    LC_HOST_COMBINED_lc_and_hosts_df = LC_lc_and_hosts_df
    same_value_columns = HOST_lc_and_hosts_df[host_features].apply(
        lambda x: x.nunique() == 1, axis=0
    )
    for column in host_features:
        if same_value_columns[column]:  # Check if all rows in the column are the same
            LC_HOST_COMBINED_lc_and_hosts_df[column] = HOST_lc_and_hosts_df[
                column
            ].iloc[
                0
            ]  # Replace with the single host feat value
        else:
            print(f"ERROR: INCONSISTENT HOST FEATURE: {column}")

    LC_HOST_COMBINED_lc_and_hosts_df_120d = LC_HOST_COMBINED_lc_and_hosts_df[
        lc_and_host_features
    ]

    mod_plot_RFC_prob_vs_lc_ztfid(
        clf=clf,
        anom_ztfid=laiss_dict["LC_ztfid_ref"],
        host_ztf_id=laiss_dict["HOST_ztfid_ref"],
        anom_spec_cls=laiss_dict["LC_tns_cls"],
        anom_spec_z=laiss_dict["LC_tns_z"],
        anom_thresh=50,
        lc_and_hosts_df=LC_HOST_COMBINED_lc_and_hosts_df,
        lc_and_hosts_df_120d=LC_HOST_COMBINED_lc_and_hosts_df_120d,
        ref_info=LC_locus,
        savefig=False,
        figure_path=figure_path,
    )


def host_only_build_indexed_sample(
    fn="",
    host_features=[],
    pca=True,
    n_components=None,
    save=True,
    force_recreation_of_index=False,
):
    data = pd.read_csv(fn)
    data = data.set_index("ztf_object_id")
    data = data[host_features]
    data = data.dropna()

    # Host features annoy index, w/ PCA
    feat_arr = np.array(data)
    idx_arr = np.array(data.index)

    if pca:
        scaler = preprocessing.StandardScaler()

        # Set a random seed for PCA
        random_seed = 88

        # Scale the features
        feat_arr_scaled = scaler.fit_transform(feat_arr)

        # Initialize PCA
        pcaModel = PCA(n_components=n_components, random_state=random_seed)

        # Apply PCA
        feat_arr_scaled_pca = pcaModel.fit_transform(feat_arr_scaled)

    # Save the index array to a binary file
    index_nm = f"host_only_laiss_annoy_index_pca{pca}"
    if save:
        np.save(f"../data/{index_nm}_idx_arr.npy", idx_arr)
        np.save(f"../data/{index_nm}_feat_arr.npy", feat_arr)
        if pca:
            np.save(f"../data/{index_nm}_feat_arr_scaled.npy", feat_arr_scaled)
            np.save(f"../data/{index_nm}_feat_arr_scaled_pca.npy", feat_arr_scaled_pca)

    # Create or load the ANNOY index
    index_file = f"../data/{index_nm}.ann"  # Choose a filename
    if pca:
        index_dim = feat_arr_scaled_pca.shape[1]
    else:
        index_dim = feat_arr.shape[1]  # Dimension of the index

    if not os.path.exists(index_file) or force_recreation_of_index:
        print(f"Building new ANNOY index with {data.shape[0]} transients...")
        # If the index file doesn't exist, create and build the index
        index = annoy.AnnoyIndex(index_dim, metric="manhattan")

        # Add items to the index
        for i in range(len(idx_arr)):
            if pca:
                index.add_item(i, feat_arr_scaled_pca[i])
            else:
                index.add_item(i, feat_arr[i])
        # Build the index
        index.build(1000)  # 1000 trees

        if save:
            # Save the index to a file
            index.save(index_file)
    else:
        print("Loading previously saved ANNOY index...")
        # If the index file exists, load it
        index = annoy.AnnoyIndex(index_dim, metric="manhattan")
        index.load(index_file)
        idx_arr = np.load(f"../data/{index_nm}_idx_arr.npy", allow_pickle=True)

    print("Done!")

    return index_nm


def host_only_LAISS_primer(ztf_id, dataset_bank_path, host_features=[]):

    l_or_ztfid_ref_in_dataset_bank = False
    host_df_ztf_id_l, host_df_ra_l, host_df_dec_l = [], [], []

    if ztf_id.startswith("ZTF"):

        try:
            dataset_bank = pd.read_csv(dataset_bank_path, index_col=0)[
                host_features
            ].dropna()
            locus_feat_arr = dataset_bank.loc[ztf_id]
            locus_feat_arr = locus_feat_arr[host_features].values

            l_or_ztfid_ref_in_dataset_bank = True
            print(f"{ztf_id} is in dataset_bank. Continuing...")

        except:
            print(
                f"{ztf_id} has NA features or is not in dataset bank. Cannot calculate new feature space. Abort!"
            )
            sys.exit(1)
            return
            # print(f"{ztf_id} is not in dataset_bank. Checking if made before...")
            # if os.path.exists(f"../timeseries/{ztf_id}_timeseries.csv"):
            #     print(f"{ztf_id} is already made. Continue!\n")

            # else:
            #     print("Re-extracting LC+Host features")
            #     extract_lc_and_host_features(
            #         ztf_id_ref=ztf_id,
            #         use_lc_for_ann_only_bool=False,
            #         show_lc=False,
            #         show_host=True,
            #         host_features=host_features,
            #     )

            # try:
            #     lc_and_hosts_df = pd.read_csv(f"../timeseries/{ztf_id}_timeseries.csv")
            # except:
            #     print(f"couldn't feature space as function of time for {ztf_id}. pass.")
            #     return

            # print(
            #     f"HOST : http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={lc_and_hosts_df.iloc[0]['raMean']}+{lc_and_hosts_df.iloc[0]['decMean']}&filter=color\n"
            # )
            # host_df_ztf_id_l.append(ztf_id), host_df_ra_l.append(
            #     lc_and_hosts_df.iloc[0]["raMean"]
            # ), host_df_dec_l.append(lc_and_hosts_df.iloc[0]["decMean"])

            # try:
            #     host_feature_df = lc_and_hosts_df[host_features]
            # except:
            #     print(f"{ztf_id} has some NaN host features. Skip!")
            #     return

            # host_feature_df = (
            #     host_feature_df.dropna()
            # )  # if this drops all rows, that means something is nan from a 0 or nan entry (check data file)

            # anom_obj_df = pd.DataFrame(
            #     host_feature_df.iloc[-1]
            # ).T  # last row of df to test "full LC only"

            # locus_feat_arr = anom_obj_df.values[0]

        locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztf_id)
        try:
            tns = locus.catalog_objects["tns_public_objects"][0]
            tns_name, tns_cls, tns_z = tns["name"], tns["type"], tns["redshift"]
        except:
            tns_name, tns_cls, tns_z = "No TNS", "---", -99
        if tns_cls == "":
            tns_cls, tns_ann_z = "---", -99

    else:
        raise ValueError("Input must be a string representing a ztfid_ref)")

    HOST_ztfid_ref = ztf_id
    HOST_tns_name, HOST_tns_cls, HOST_tns_z = tns_name, tns_cls, tns_z

    output_dict = {
        "HOST_ztfid_ref": HOST_ztfid_ref,
        "HOST_tns_name": HOST_tns_name,
        "HOST_tns_cls": HOST_tns_cls,
        "HOST_tns_z": HOST_tns_z,
        "host_df_ztf_id_l": host_df_ztf_id_l,  # This is just HOST_ztfid_ref but in a length-1 list
        "locus_feat_arr": locus_feat_arr,
        "l_or_ztfid_ref_in_dataset_bank": l_or_ztfid_ref_in_dataset_bank,
    }

    print("Created output dictionary!")

    return output_dict


def host_only_LAISS_nearest_neighbors(
    laiss_dict,
    use_pca_for_nn=True,
    n_components=15,
    annoy_index_file_path="",
    n=8,
    max_neighbor_dist=np.inf,
    search_k=1000,
    return_results=False,
):
    start_time = time.time()

    locus_feat_arr = laiss_dict["locus_feat_arr"]

    if use_pca_for_nn:
        # 1. Scale locus_feat_arr using the same scaler (Standard Scaler)
        scaler = preprocessing.StandardScaler()
        trained_PCA_feat_arr = np.load(
            annoy_index_file_path + "_feat_arr.npy",
            allow_pickle=True,
        )

        trained_PCA_feat_arr_scaled = scaler.fit_transform(
            trained_PCA_feat_arr
        )  # scaler needs to be fit first to the same data as trained

        locus_feat_arr_scaled = scaler.transform(
            [laiss_dict["locus_feat_arr"]]
        )  # scaler transform new data

        # 2. Transform the scaled locus_feat_arr using the same PCA model
        n_components = n_components
        random_seed = 88
        pca = PCA(n_components=n_components, random_state=random_seed)
        trained_PCA_feat_arr_scaled_pca = pca.fit_transform(
            trained_PCA_feat_arr_scaled
        )  # pca needs to be fit first to the same data as trained
        locus_feat_arr_pca = pca.transform(
            locus_feat_arr_scaled
        )  # pca transform  new data

        index_nm = annoy_index_file_path
        index_file = index_nm + ".ann"
        index_dim = n_components  # Dimension of the PCA index

        # 3. Use the ANNOY index to find nearest neighbors
        print(f"Loading previously saved ANNOY PCA={n_components} index")
        print(index_file)

        index = annoy.AnnoyIndex(index_dim, metric="manhattan")
        index.load(index_file)
        idx_arr = np.load(f"{index_nm}_idx_arr.npy", allow_pickle=True)

        ann_start_time = time.time()
        ann_indexes, ann_dists = index.get_nns_by_vector(
            locus_feat_arr_pca[0], n=n, search_k=search_k, include_distances=True
        )
        ann_alerce_links = [
            f"https://alerce.online/object/{idx_arr[i]}" for i in ann_indexes
        ]
        ann_end_time = time.time()
    else:
        # Create or load the ANNOY index
        index_nm = annoy_index_file_path
        index_file = index_nm + ".ann"
        index_dim = len(laiss_dict["locus_feat_arr"])

        # 3. Use the ANNOY index to find nearest neighbors
        print("Loading previously saved ANNOY index without PCA:")
        print(index_file)

        index = annoy.AnnoyIndex(index_dim, metric="manhattan")
        index.load(index_file)
        idx_arr = np.load(f"{index_nm}_idx_arr.npy", allow_pickle=True)

        ann_start_time = time.time()
        ann_indexes, ann_dists = index.get_nns_by_vector(
            locus_feat_arr,
            n=n,
            search_k=search_k,
            include_distances=True,
        )
        ann_alerce_links = [
            f"https://alerce.online/object/{idx_arr[i]}" for i in ann_indexes
        ]
        ann_end_time = time.time()

    # Plot neighbor number vs distance and find optimal number of neighbors:
    neighbor_numbers_for_plot = list(range(1, len(ann_dists)))
    dists_for_plot = ann_dists[1:]

    knee = KneeLocator(
        neighbor_numbers_for_plot,
        dists_for_plot,
        curve="concave",
        direction="increasing",
    )
    optimal_k = knee.knee

    plt.figure(figsize=(10, 4))
    plt.plot(neighbor_numbers_for_plot, dists_for_plot, marker="o", label="Distances")
    if optimal_k:
        plt.axvline(
            optimal_k, color="red", linestyle="--", label=f"Elbow at {optimal_k}"
        )
    plt.xlabel("Neighbor Number")
    plt.ylabel("Distance")
    plt.title("Distance for Closest Neighbors")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(
        f"Suggested number of neighbors: {optimal_k}. Suggested Maximum Distance: {round(dists_for_plot[optimal_k-1], 2)}"
    )

    # Filter for distance:
    filtered_neighbors = [
        (idx, dist)
        for idx, dist in zip(ann_indexes, ann_dists)
        if dist <= abs(max_neighbor_dist)
    ]
    ann_indexes, ann_dists = (
        zip(*filtered_neighbors) if filtered_neighbors else ([], [])
    )
    ann_indexes = list(ann_indexes)
    ann_dists = list(ann_dists)
    # Minus 1 because host itself will always be a neighbor (dist = 0)
    number_of_neighbors_found = len(ann_indexes) - 1
    if number_of_neighbors_found == 0:
        print(
            f"No neighbors found for distance threshold of {abs(max_neighbor_dist)}. Please try a larger maximum distance."
        )
    else:
        print("Number of neighbors found:", number_of_neighbors_found)

    # 4. Get TNS, spec. class of ANNs
    tns_ann_names, tns_ann_classes, tns_ann_zs = [], [], []
    ann_locus_l = []
    for i in ann_indexes:
        ann_locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=idx_arr[i])
        ann_locus_l.append(ann_locus)
        try:
            ann_tns = ann_locus.catalog_objects["tns_public_objects"][0]
            tns_ann_name, tns_ann_cls, tns_ann_z = (
                ann_tns["name"],
                ann_tns["type"],
                ann_tns["redshift"],
            )
        except:
            tns_ann_name, tns_ann_cls, tns_ann_z = "No TNS", "---", -99
        if tns_ann_cls == "":
            tns_ann_cls, tns_ann_z = "---", -99
        tns_ann_names.append(tns_ann_name), tns_ann_classes.append(
            tns_ann_cls
        ), tns_ann_zs.append(tns_ann_z)
        laiss_dict["host_df_ztf_id_l"].append(idx_arr[i])

    # Print the nearest neighbors
    print("\t\t\t\t\t   ZTFID IAU_NAME SPEC Z")
    # print(
    #     f"LC REF: https://alerce.online/object/{laiss_dict['LC_ztfid_ref']} {laiss_dict['LC_tns_name']} {laiss_dict['LC_tns_cls']} {laiss_dict['LC_tns_z']}"
    # )
    print(
        f"HOST REF: https://alerce.online/object/{laiss_dict['HOST_ztfid_ref']} {laiss_dict['HOST_tns_name']} {laiss_dict['HOST_tns_cls']} {laiss_dict['HOST_tns_z']}"
    )

    ann_num_l = []
    if return_results:
        storage = []
    for i, (al, iau_name, spec_cls, z, dist) in enumerate(
        zip(ann_alerce_links, tns_ann_names, tns_ann_classes, tns_ann_zs, ann_dists)
    ):
        if i == 0:
            continue
        print(f"ANN={i}: {al} {iau_name} {spec_cls}, {z}")
        ann_num_l.append(i)
        if return_results:
            neighbor_dict = {
                "input_host_ztf_id": laiss_dict["HOST_ztfid_ref"],
                "neighbor_num": i,
                "ztf_link": al,
                "dist": dist,
                "iau_name": iau_name,
                "spec_cls": spec_cls,
                "z": z,
            }
            storage.append(neighbor_dict)

    end_time = time.time()
    ann_elapsed_time = ann_end_time - ann_start_time
    elapsed_time = end_time - start_time
    print(f"\nANN elapsed_time = {round(ann_elapsed_time, 3)} s")
    print(f"\ntotal elapsed_time = {round(elapsed_time, 3)} s\n")

    if return_results:
        return pd.DataFrame(storage)


def re_build_indexed_sample(
    dataset_bank_path,
    lc_features=[],
    host_features=[],
    use_pca=False,
    n_components=None,
    num_trees=1000,
    index_folder_relative_path="",
    save=True,
    force_recreation_of_index=False,
):
    df_bank = pd.read_csv(dataset_bank_path)

    # Confirm that the first column is the ZTF ID, and index by ZTF ID
    if df_bank.columns[0] != "ztf_object_id":
        print(
            f"Error: Expected first column in dataset bank to be '{ztf_object_id}', but got '{df_bank.columns[0]}' instead."
        )
        sys.exit(1)
    df_bank = df_bank.set_index("ztf_object_id")

    # Ensure proper user input of features
    num_lc_features = len(lc_features)
    num_host_features = len(host_features)
    if num_lc_features + num_host_features == 0:
        print("Error: must provide at least one lightcurve or host feature.")
        sys.exit(1)
    if num_lc_features == 0:
        print(
            f"No lightcurve features provided. Running host-only LAISS with {num_host_features} features."
        )
    if num_host_features == 0:
        print(
            f"No host features provided. Running lightcurve-only LAISS with {num_lc_features} features."
        )

    # Filtering dataset bank for provided features
    df_bank = df_bank[lc_features + host_features]
    df_bank = df_bank.dropna()
    feat_arr = np.array(df_bank)
    idx_arr = np.array(df_bank.index)

    # Apply PCA if necessary
    if use_pca:
        random_seed = 88
        scaler = preprocessing.StandardScaler()

        feat_arr_scaled = scaler.fit_transform(feat_arr)

        pcaModel = PCA(n_components=n_components, random_state=random_seed)
        feat_arr_scaled_pca = pcaModel.fit_transform(feat_arr_scaled)

    # Save PCA and non-PCA index arrays to binary files
    index_stem_name = (
        f"re_laiss_annoy_index_pca{use_pca}"
        + (f"_{n_components}comps" if use_pca else "")
        + f"_{num_lc_features}lc_{num_host_features}host"
    )
    index_stem_name_with_path = index_folder_relative_path + "/" + index_stem_name
    if save:
        np.save(f"{index_stem_name_with_path}_idx_arr.npy", idx_arr)
        np.save(f"{index_stem_name_with_path}_feat_arr.npy", feat_arr)
        if use_pca:
            np.save(
                f"{index_stem_name_with_path}_feat_arr_scaled.npy",
                feat_arr_scaled,
            )
            np.save(
                f"{index_stem_name_with_path}_feat_arr_scaled_pca.npy",
                feat_arr_scaled_pca,
            )

    # Create or load the ANNOY index:
    index_file = f"{index_stem_name_with_path}.ann"
    index_dim = feat_arr_scaled_pca.shape[1] if use_pca else feat_arr.shape[1]

    # If the ANNOY index already exists, use it
    if os.path.exists(index_file) and not force_recreation_of_index:
        print("Loading previously saved ANNOY index...")
        index = annoy.AnnoyIndex(index_dim, metric="manhattan")
        index.load(index_file)
        idx_arr = np.load(
            f"{index_stem_name_with_path}_idx_arr.npy",
            allow_pickle=True,
        )

    # Otherwise, create a new index
    else:
        print(f"Building new ANNOY index with {df_bank.shape[0]} transients...")

        index = annoy.AnnoyIndex(index_dim, metric="manhattan")
        for i in range(len(idx_arr)):
            index.add_item(i, feat_arr_scaled_pca[i] if use_pca else feat_arr[i])

        index.build(num_trees)

        if save:
            index.save(index_file)

    print("Done!")

    return index_stem_name_with_path


def re_get_timeseries_df(
    ztf_id,
    path_to_timeseries_folder,
    path_to_sfd_data_folder,
):
    # Check if timeseries already made
    print(f"Checking if timeseries dataframe for {ztf_id} already exists...")
    if os.path.exists(f"{path_to_timeseries_folder}/{ztf_id}_timeseries.csv"):
        timeseries_df = pd.read_csv(
            f"{path_to_timeseries_folder}/{ztf_id}_timeseries.csv"
        )
        print(f"Timeseries dataframe for {ztf_id} is already made. Continue!\n")

    # If timeseries is not already made, create it by extracting features
    else:
        print(
            f"Timeseries dataframe does not exist. Re-extracting lightcurve and host features for {ztf_id}."
        )
        timeseries_df = re_extract_lc_and_host_features(
            ztf_id=ztf_id,
            path_to_timeseries_folder=path_to_timeseries_folder,
            path_to_sfd_data_folder=path_to_sfd_data_folder,
            show_lc=False,
            show_host=True,
            store_csv=True,
        )
    return timeseries_df


def re_LAISS_primer(
    lc_ztf_id,
    dataset_bank_path,
    path_to_timeseries_folder,
    path_to_sfd_data_folder,
    host_ztf_id=None,
    lc_features=[],
    host_features=[],
):

    feature_names = lc_features + host_features

    # Loop through lightcurve object and host object to create feature array
    for ztf_id, host_loop in [(lc_ztf_id, False), (host_ztf_id, True)]:

        # Skip host loop if not swapping out host galaxy
        if ztf_id is None:
            continue

        ztf_id_in_dataset_bank = False

        # Check if ztf_id is in dataset bank
        try:
            df_bank = pd.read_csv(dataset_bank_path, index_col=0)
            # Check to make sure all features are in the dataset bank
            missing_cols = [col for col in feature_names if col not in df_bank.columns]
            if missing_cols:
                print(
                    f"KeyError: The following columns are not in the raw data provided: {missing_cols}. Abort!"
                )
                return
            locus_feat_arr = df_bank.loc[ztf_id]

            ztf_id_in_dataset_bank = True
            print(f"{ztf_id} is in dataset_bank.")

        # If ztf_if is not in dataset bank...
        except:
            # Extract timeseries dataframe
            print(f"{ztf_id} is not in dataset_bank.")
            timeseries_df = re_get_timeseries_df(
                ztf_id=ztf_id,
                path_to_timeseries_folder=path_to_timeseries_folder,
                path_to_sfd_data_folder=path_to_sfd_data_folder,
            )

            timeseries_df = timeseries_df[lc_features + host_features]
            timeseries_df = timeseries_df.dropna()

            if timeseries_df.empty:
                print(f"{ztf_id} has some NaN features. Abort!")
                sys.exit(1)

            # Extract feature array from timeseries dataframe
            locus_feat_arr_df = pd.DataFrame(timeseries_df.iloc[-1]).T
            locus_feat_arr = locus_feat_arr_df.iloc[0]

        # Pull TNS data for ztf_id
        locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztf_id)
        try:
            tns = locus.catalog_objects["tns_public_objects"][0]
            tns_name, tns_cls, tns_z = tns["name"], tns["type"], tns["redshift"]
        except:
            tns_name, tns_cls, tns_z = "No TNS", "---", -99
        if tns_cls == "":
            tns_cls, tns_ann_z = "---", -99

        if host_loop:
            host_tns_name, host_tns_cls, host_tns_z = tns_name, tns_cls, tns_z
            host_ztf_id_in_dataset_bank = ztf_id_in_dataset_bank
            host_locus_feat_arr = locus_feat_arr
        else:
            lc_tns_name, lc_tns_cls, lc_tns_z = tns_name, tns_cls, tns_z
            lc_ztf_id_in_dataset_bank = ztf_id_in_dataset_bank
            lc_locus_feat_arr = locus_feat_arr

    # Make final feature array
    if host_ztf_id is None:
        # Not swapping out host, use features from lightcurve ztf_id
        locus_feat_arr = lc_locus_feat_arr[feature_names].values
    else:
        # Create new feature array with mixed lc and host features
        subset_lc_features = lc_locus_feat_arr[lc_features]
        subset_host_features = host_locus_feat_arr[host_features]
        locus_feat_arr = np.concatenate((subset_lc_features, subset_host_features))

    output_dict = {
        # host data is optional, it's only if the user decides to swap in a new host
        "host_ztf_id": host_ztf_id if host_ztf_id is not None else None,
        "host_tns_name": host_tns_name if host_ztf_id is not None else None,
        "host_tns_cls": host_tns_cls if host_ztf_id is not None else None,
        "host_tns_z": host_tns_z if host_ztf_id is not None else None,
        "host_ztf_id_in_dataset_bank": (
            host_ztf_id_in_dataset_bank if host_ztf_id is not None else None
        ),
        "lc_ztf_id": lc_ztf_id,
        "lc_tns_name": lc_tns_name,
        "lc_tns_cls": lc_tns_cls,
        "lc_tns_z": lc_tns_z,
        "lc_ztf_id_in_dataset_bank": lc_ztf_id_in_dataset_bank,
        "locus_feat_arr": locus_feat_arr,
    }

    return output_dict


def re_plot_lightcurves(
    primer_dict,
    neighbor_ztfids,
    ann_locus_l,
    ann_dists,
    tns_ann_names,
    tns_ann_classes,
    tns_ann_zs,
):
    print("Making a plot of stacked lightcurves...")

    if primer_dict["lc_tns_z"] is None:
        primer_dict["lc_tns_z"] = "None"
    elif isinstance(primer_dict["lc_tns_z"], float):
        primer_dict["lc_tns_z"] = round(primer_dict["lc_tns_z"], 3)
    else:
        primer_dict["lc_tns_z"] = primer_dict["lc_tns_z"]

    ref_info = antares_client.search.get_by_ztf_object_id(
        ztf_object_id=primer_dict["lc_ztf_id"]
    )
    try:
        df_ref = ref_info.timeseries.to_pandas()
    except:
        print("No timeseries data...pass!")
        pass

    fig, ax = plt.subplots(figsize=(9.5, 6))

    df_ref_g = df_ref[(df_ref.ant_passband == "g") & (~df_ref.ant_mag.isna())]
    df_ref_r = df_ref[(df_ref.ant_passband == "R") & (~df_ref.ant_mag.isna())]

    mjd_idx_at_min_mag_r_ref = df_ref_r[["ant_mag"]].reset_index().idxmin().ant_mag
    mjd_idx_at_min_mag_g_ref = df_ref_g[["ant_mag"]].reset_index().idxmin().ant_mag

    ax.errorbar(
        x=df_ref_r.ant_mjd - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref],
        y=df_ref_r.ant_mag.min() - df_ref_r.ant_mag,
        yerr=df_ref_r.ant_magerr,
        fmt="o",
        c="r",
        label=(
            f"{primer_dict['lc_ztf_id']}"
            + (
                f" w/ host from {primer_dict['host_ztf_id']},"
                if primer_dict["host_ztf_id"] is not None
                else ","
            )
            + f"\nd=0, {primer_dict['lc_tns_name']}, {primer_dict['lc_tns_cls']}, z={primer_dict['lc_tns_z']}"
        ),
    )
    ax.errorbar(
        x=df_ref_g.ant_mjd - df_ref_g.ant_mjd.iloc[mjd_idx_at_min_mag_g_ref],
        y=df_ref_g.ant_mag.min() - df_ref_g.ant_mag,
        yerr=df_ref_g.ant_magerr,
        fmt="o",
        c="g",
    )

    markers = ["s", "*", "x", "P", "^", "v", "D", "<", ">", "8", "p", "x"]
    consts = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]

    for num, (l_info, ztfname, dist, iau_name, spec_cls, z) in enumerate(
        zip(
            ann_locus_l,
            neighbor_ztfids,
            ann_dists,
            tns_ann_names,
            tns_ann_classes,
            tns_ann_zs,
        )
    ):
        # Plots up to 8 neighbors
        if num + 1 > 8:
            print(
                "Lightcurve plotter only plots up to 8 neighbors. Stopping at neighbor 8."
            )
            break
        try:
            alpha = 0.25
            c1 = "darkred"
            c2 = "darkgreen"

            df_knn = l_info.timeseries.to_pandas()

            df_g = df_knn[(df_knn.ant_passband == "g") & (~df_knn.ant_mag.isna())]
            df_r = df_knn[(df_knn.ant_passband == "R") & (~df_knn.ant_mag.isna())]

            mjd_idx_at_min_mag_r = df_r[["ant_mag"]].reset_index().idxmin().ant_mag
            mjd_idx_at_min_mag_g = df_g[["ant_mag"]].reset_index().idxmin().ant_mag

            ax.errorbar(
                x=df_r.ant_mjd - df_r.ant_mjd.iloc[mjd_idx_at_min_mag_r],
                y=df_r.ant_mag.min() - df_r.ant_mag,
                yerr=df_r.ant_magerr,
                fmt=markers[num],
                c=c1,
                alpha=alpha,
                label=f"ANN={num+1}:{ztfname}, d={round(dist, 2)},\n{iau_name}, {spec_cls}, z={round(z, 3)}",
            )
            ax.errorbar(
                x=df_g.ant_mjd - df_g.ant_mjd.iloc[mjd_idx_at_min_mag_g],
                y=df_g.ant_mag.min() - df_g.ant_mag,
                yerr=df_g.ant_magerr,
                fmt=markers[num],
                c=c2,
                alpha=alpha,
            )

            plt.ylabel("Apparent Mag. + Constant")
            plt.xlabel("Days since peak ($r$, $g$ indep.)")  # (need r, g to be same)

            if (
                df_ref_r.ant_mjd.iloc[0]
                - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]
                <= 10
            ):
                plt.xlim(
                    (
                        df_ref_r.ant_mjd.iloc[0]
                        - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]
                    )
                    - 20,
                    df_ref_r.ant_mjd.iloc[-1] - df_ref_r.ant_mjd.iloc[0] + 15,
                )
            else:
                plt.xlim(
                    2
                    * (
                        df_ref_r.ant_mjd.iloc[0]
                        - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]
                    ),
                    df_ref_r.ant_mjd.iloc[-1] - df_ref_r.ant_mjd.iloc[0] + 15,
                )

            shift, scale = 1.4, 0.975
            if len(neighbor_ztfids) <= 2:
                shift = 1.175
                scale = 0.9
            elif len(neighbor_ztfids) <= 5:
                shift = 1.3
                scale = 0.925

            plt.legend(
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, shift),
                ncol=3,
                prop={"size": 10},
            )
            plt.grid(True)

            # Shrink axes to leave space above for the legend
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height * scale])

        except Exception as e:
            print(
                f"Something went wrong with plotting {ztfname}! Error is {e}. Continue..."
            )

    plt.show()


def re_LAISS_nearest_neighbors(
    primer_dict,
    annoy_index_file_stem,
    use_pca=False,
    num_pca_components=15,
    n=8,
    suggest_neighbor_num=False,
    max_neighbor_dist=None,
    search_k=1000,
    return_results=False,
):
    start_time = time.time()
    index_file = annoy_index_file_stem + ".ann"

    if n is None or n <= 0:
        print("Neighbor number must be a nonzero integer. Abort!")
        return

    if use_pca:
        # 1. Scale locus_feat_arr using the same scaler (Standard Scaler)
        scaler = preprocessing.StandardScaler()
        trained_PCA_feat_arr = np.load(
            annoy_index_file_stem + "_feat_arr.npy",
            allow_pickle=True,
        )

        # scaler needs to be fit first to the same data as trained
        trained_PCA_feat_arr_scaled = scaler.fit_transform(trained_PCA_feat_arr)

        # scaler transform new data
        locus_feat_arr_scaled = scaler.transform([primer_dict["locus_feat_arr"]])

        # 2. Transform the scaled locus_feat_arr using the same PCA model
        random_seed = 88
        pca = PCA(n_components=num_pca_components, random_state=random_seed)

        # pca needs to be fit first to the same data as trained
        trained_PCA_feat_arr_scaled_pca = pca.fit_transform(trained_PCA_feat_arr_scaled)

        # pca transform new data
        locus_feat_arr_pca = pca.transform(locus_feat_arr_scaled)

        print(
            f"Loading previously saved ANNOY PCA={num_pca_components} index:",
            index_file,
        )
        index_dim = num_pca_components
        query_vector = locus_feat_arr_pca[0]

    else:
        print("Loading previously saved ANNOY index without PCA:", index_file)
        index_dim = len(primer_dict["locus_feat_arr"])
        query_vector = primer_dict["locus_feat_arr"]

    # 3. Use the ANNOY index to find nearest neighbors (common to both branches)
    index = annoy.AnnoyIndex(index_dim, metric="manhattan")
    index.load(index_file)
    idx_arr = np.load(f"{annoy_index_file_stem}_idx_arr.npy", allow_pickle=True)

    ann_start_time = time.time()
    ann_indexes, ann_dists = index.get_nns_by_vector(
        query_vector, n=n, search_k=search_k, include_distances=True
    )

    if round(ann_dists[0]) == 0:
        print(
            "First neighbor is input transient, so it will be excluded. The final neighbor count will be one less than expected."
        )
        # drop first neighbor, which is input transient
        ann_dists = ann_dists[1:]
        ann_indexes = ann_indexes[1:]

    ann_alerce_links = [
        f"https://alerce.online/object/{idx_arr[i]}" for i in ann_indexes
    ]
    ann_end_time = time.time()

    # Find optimal number of neighbors
    if suggest_neighbor_num:
        number_of_neighbors_found = len(ann_dists)
        neighbor_numbers_for_plot = list(range(1, number_of_neighbors_found + 1))

        knee = KneeLocator(
            neighbor_numbers_for_plot,
            ann_dists,
            curve="concave",
            direction="increasing",
        )
        optimal_n = knee.knee

        if optimal_n is not None:
            print(
                f"Suggested number of neighbors is {optimal_n}, chosen by comparing {n} neighbors."
            )
        else:
            print(
                "Couldn't identify optimal number of neighbors. Try a larger neighbor pool."
            )

        plt.figure(figsize=(10, 4))
        plt.plot(
            neighbor_numbers_for_plot,
            ann_dists,
            marker="o",
            label="Distances",
        )
        if optimal_n:
            plt.axvline(
                optimal_n, color="red", linestyle="--", label=f"Elbow at {optimal_n}"
            )
        plt.xlabel("Neighbor Number")
        plt.ylabel("Distance")
        plt.title("Distance for Closest Neighbors")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Filter neighbors for maximum distance, if provided
    if max_neighbor_dist is not None:
        filtered_neighbors = [
            (idx, dist)
            for idx, dist in zip(ann_indexes, ann_dists)
            if dist <= abs(max_neighbor_dist)
        ]
        ann_indexes, ann_dists = (
            zip(*filtered_neighbors) if filtered_neighbors else ([], [])
        )
        ann_indexes = list(ann_indexes)
        ann_dists = list(ann_dists)

        if len(ann_dists) == 0:
            print(
                f"No neighbors found for distance threshold of {abs(max_neighbor_dist)}. Try a larger maximum distance."
            )
        else:
            print(
                f"Found {len(ann_dists)} neighbors for distance threshold of {abs(max_neighbor_dist)}."
            )

    # 4. Get TNS, spec. class of neighbors
    tns_ann_names, tns_ann_classes, tns_ann_zs, neighbor_ztfids = [], [], [], []
    ann_locus_l = []
    for i in ann_indexes:
        neighbor_ztfids.append(idx_arr[i])
        ann_locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=idx_arr[i])
        ann_locus_l.append(ann_locus)
        try:
            ann_tns = ann_locus.catalog_objects["tns_public_objects"][0]
            tns_ann_name, tns_ann_cls, tns_ann_z = (
                ann_tns["name"],
                ann_tns["type"],
                ann_tns["redshift"],
            )
        except:
            tns_ann_name, tns_ann_cls, tns_ann_z = "No TNS", "---", -99
        if tns_ann_cls == "":
            tns_ann_cls, tns_ann_z = "---", -99
        tns_ann_names.append(tns_ann_name), tns_ann_classes.append(
            tns_ann_cls
        ), tns_ann_zs.append(tns_ann_z)

    # Print the nearest neighbors and organize them for storage
    print("\t\t\t\t\t\t ZTFID     IAU_NAME SPEC  Z")
    print(
        f"Input transient: https://alerce.online/object/{primer_dict['lc_ztf_id']} {primer_dict['lc_tns_name']} {primer_dict['lc_tns_cls']} {primer_dict['lc_tns_z']}"
    )
    if primer_dict["host_ztf_id"] is not None:
        print(
            f"Transient with host swapped into input: https://alerce.online/object/{primer_dict['host_ztf_id']} {primer_dict['host_tns_name']} {primer_dict['host_tns_cls']} {primer_dict['host_tns_z']}"
        )

    # Print lightcurves
    re_plot_lightcurves(
        primer_dict=primer_dict,
        neighbor_ztfids=neighbor_ztfids,
        ann_locus_l=ann_locus_l,
        ann_dists=ann_dists,
        tns_ann_names=tns_ann_names,
        tns_ann_classes=tns_ann_classes,
        tns_ann_zs=tns_ann_zs,
    )

    storage = []
    neighbor_num = 1
    for al, iau_name, spec_cls, z, dist in zip(
        ann_alerce_links, tns_ann_names, tns_ann_classes, tns_ann_zs, ann_dists
    ):
        print(f"ANN={neighbor_num}: {al} {iau_name} {spec_cls}, {z}")
        if return_results:
            neighbor_dict = {
                "input_ztf_id": primer_dict["lc_ztf_id"],
                "input_swapped_host_ztf_id": primer_dict["host_ztf_id"],
                "neighbor_num": neighbor_num,
                "ztf_link": al,
                "dist": dist,
                "iau_name": iau_name,
                "spec_cls": spec_cls,
                "z": z,
            }
            storage.append(neighbor_dict)
        neighbor_num += 1

    end_time = time.time()
    ann_elapsed_time = ann_end_time - ann_start_time
    elapsed_time = end_time - start_time
    print(f"\nANN elapsed_time: {round(ann_elapsed_time, 3)} s")
    print(f"\ntotal elapsed_time: {round(elapsed_time, 3)} s\n")

    if return_results:
        return pd.DataFrame(storage)


def re_anomaly_detection(
    laiss_dict,
    lc_features,
    host_features,
    path_to_timeseries_folder,
    path_to_sfd_data_folder,
    n_estimators,
    max_depth,
    random_state,
    max_features,
):
    # Load the model
    figure_path = f"../models/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced/figures"
    model_path = f"../models/SMOTE_train_test_70-30_min14_kneighbors8/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced/model"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with open(
        f"{model_path}/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced.pkl",
        "rb",
    ) as f:
        clf = pickle.load(f)

    print("\nRunning AD Model...")

    # Load the timeseries dataframe
    timeseries_df = re_get_timeseries_df(
        ztf_id=laiss_dict["lc_ztf_id"],
        path_to_timeseries_folder=path_to_timeseries_folder,
        path_to_sfd_data_folder=path_to_sfd_data_folder,
    )

    if laiss_dict["host_ztf_id"] is not None:
        # Swap in the host galaxy
        swapped_host_timeseries_df = re_get_timeseries_df(
            ztf_id=laiss_dict["host_ztf_id"],
            path_to_timeseries_folder=path_to_timeseries_folder,
            path_to_sfd_data_folder=path_to_sfd_data_folder,
        )

        host_values = swapped_host_timeseries_df[host_features].iloc[0]
        for col in host_features:
            timeseries_df[col] = host_values[col]

    timeseries_df_filt_feats = timeseries_df[lc_features + host_features]
    input_lightcurve_locus = antares_client.search.get_by_ztf_object_id(
        ztf_object_id=laiss_dict["lc_ztf_id"]
    )

    re_AD_and_plots(
        clf=clf,
        input_ztf_id=laiss_dict["lc_ztf_id"],
        swapped_host_ztf_id=laiss_dict["host_ztf_id"],
        input_spec_cls=laiss_dict["lc_tns_cls"],
        input_spec_z=laiss_dict["lc_tns_z"],
        anom_thresh=50,
        timeseries_df_full=timeseries_df,
        timeseries_df_features_only=timeseries_df_filt_feats,
        ref_info=input_lightcurve_locus,
        savefig=False,
        figure_path=figure_path,
    )
    return


def re_LAISS(
    transient_ztf_id,  # transient on which to run laiss
    path_to_dataset_bank,
    path_to_timeseries_folder,
    host_ztf_id_to_swap_in=None,  # will swap the host galaxy of the input transient to this transient's host
    host_feature_names=[],  # Leave blank for lightcurve-only LAISS
    lc_feature_names=[],  # Leave blank for host-only LAISS
    path_to_sfd_data_folder="../data/sfddata-master",  # to correct extracted magnitudes for dust; not needed if transient_ztf_id in dataset bank
    use_pca=False,
    num_pca_components=20,  # Only matters if use_pca = True
    force_recreation_of_annoy_index=False,  # Rebuild indexed space for ANNOY even if it already exists
    index_folder_relative_path="../data/re_LAISS/index_files",  # folder to store ANNOY indices
    path_to_ghost_database="../data/host_info",
    neighbors=10,  # will return this number of neighbors unless filtered by max_neighbor_distance
    suggest_neighbor_num=False,  # plot distances of neighbors to help choose optimal neighbor number
    max_neighbor_distance=None,  # optional, will return all neighbors below this distance (but no more than the 'neighbors' argument)
    search_k=5000,  # for ANNOY search
    return_neighbor_results=True,  # returns a list of neighbor dictionaries
    run_AD=True,  # run anomaly detection
    n_estimators=100,  # anomaly detection parameter
    max_depth=35,  # anomaly detection parameter
    random_state=11,  # anomaly detection parameter
    max_features=35,  # anomaly detection parameter
):
    # build ANNOY indexed sample from dataset bank
    index_stem_name_with_path = re_build_indexed_sample(
        dataset_bank_path=path_to_dataset_bank,
        lc_features=lc_feature_names,
        host_features=host_feature_names,
        use_pca=use_pca,
        n_components=num_pca_components,
        num_trees=1000,
        index_folder_relative_path=index_folder_relative_path,
        save=True,
        force_recreation_of_index=force_recreation_of_annoy_index,
    )

    # set up GHOST file structure
    host_path = path_to_ghost_database
    if not os.path.exists(host_path):
        os.makedirs(host_path)
    os.environ["GHOST_PATH"] = host_path

    # run primer
    primer_dict = re_LAISS_primer(
        lc_ztf_id=transient_ztf_id,
        host_ztf_id=host_ztf_id_to_swap_in,
        dataset_bank_path=path_to_dataset_bank,
        path_to_timeseries_folder=path_to_timeseries_folder,
        path_to_sfd_data_folder=path_to_sfd_data_folder,
        lc_features=lc_feature_names,
        host_features=host_feature_names,
    )

    # nearest neighbors search
    nearest_neighbors_df = re_LAISS_nearest_neighbors(
        primer_dict=primer_dict,
        annoy_index_file_stem=index_stem_name_with_path,
        use_pca=use_pca,
        num_pca_components=num_pca_components,
        n=neighbors,
        suggest_neighbor_num=suggest_neighbor_num,
        max_neighbor_dist=max_neighbor_distance,
        search_k=search_k,
        return_results=return_neighbor_results,
    )

    # annomaly detection
    if run_AD:
        re_anomaly_detection(
            laiss_dict=primer_dict,
            lc_features=lc_feature_names,
            host_features=host_feature_names,
            path_to_timeseries_folder=path_to_timeseries_folder,
            path_to_sfd_data_folder=path_to_sfd_data_folder,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            max_features=max_features,
        )

    if return_neighbor_results:
        return nearest_neighbors_df
    return
