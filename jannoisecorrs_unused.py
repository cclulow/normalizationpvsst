import numpy as np
import os
import pandas as pd
from neural_data_object import NeuralData
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy.stats import ranksums, pearsonr, spearmanr, sem
from scipy.stats import ttest_ind, mannwhitneyu, ranksums, ks_2samp
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from statsmodels.formula.api import mixedlm
from scipy.stats import ranksums
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import power_divergence, norm
from scipy.stats import ranksums, mannwhitneyu, ks_2samp, pearsonr, spearmanr, linregress
from statsmodels.stats.power import TTestIndPower

save_dir = '/home/maclean/data_proc/noise_corr'
os.makedirs(save_dir, exist_ok=True)
OSI_THRESHOLD = 0.25  
std_f_before = 30
std_f_after = 60
micedates = {
    #'18S': ['071924', '072024'],
    '21N': ['022424', '022524'],
    '30G': ['082424'], #, '082524'
    '30N': ['062824', '062924', '070124', '070224'],
    '32L': ['042524'],
    '32L2': ['053024'],
    '37L': ['070424', '070524'],
    '37L2': ['070924', '071024'],
    '37R': ['070224', '070324'],
    '37R2': ['070924', '071024'],
    '42N': ['022524', '022624'],
    '44N': ['022524', '022624'],
    '42R2': ['073024', '073124'], 
    '42R': ['072224', '072324'],

    
    #'32L'; ['042424']
    #'42R': [, '072424', '072524']
    #'42R2': [072924]
}
def load_data(mouse, date, base_path='/media/maclean/3d4f021d-2b12-4203-943d-4f1052e2ccf0/1104_hal_data/'):
    proc_dir = os.path.join(base_path, mouse, date, 'proc')
    data = NeuralData(mouse, date, base_name=base_path, use_l='cascade')

    proc_dir = os.path.join(base_path, mouse, date, 'proc')
    # loading NeuralData object
    data = NeuralData(mouse, date, base_name=base_path, use_l='cascade')

    # loading visual responsiveness data
    vis_resp_cells = data.vis_responsive_cells() if hasattr(data, 'vis_responsive_cells') else None
    if vis_resp_cells is None or not vis_resp_cells.any():
        print(f"No visually responsive cells for Mouse {mouse}, Date {date}.")
        vis_resp_cells = None

    # loading red labels directly from red_labels.npy
    red_labels_path = os.path.join(proc_dir, 'red_labels.npy')
    if os.path.exists(red_labels_path):
        red_labels = np.load(red_labels_path)
    else:
        print(f"No red labels found in {red_labels_path}. Treating all spikes as excitatory.")
        red_labels = None
    
    trial_info = {}
    if hasattr(data, 'stim_info'):
        stim_info = data.stim_info  # trial stimulus information
        orientations = stim_info[0]  # orientations of stimuli
        is_plaid = stim_info[1] == 1  # plaid (1) or grating (0)
        unique_stim = np.unique(stim_info, axis=1)  # unique stimulus combinations
        orientation_trial_map = {}  # map orientations to trial indices
        for stim_idx in range(unique_stim.shape[1]):
            # trials matching the unique orientation and plaid status
            trial_indices = np.logical_and(
                stim_info[0] == unique_stim[0, stim_idx],
                stim_info[1] == unique_stim[1, stim_idx]
            )
            orientation_trial_map[(unique_stim[0, stim_idx], unique_stim[1, stim_idx])] = trial_indices

        trial_info = {
            'orientations': orientations,
            'is_plaid': is_plaid,
            'orientation_trial_map': orientation_trial_map,  
            'unique_stim': unique_stim.T  # unique stimuli combinations -- idk if i want trial_info since i already rewrote into stim_info
        }
    else:
        print(f"No trial information found for Mouse {mouse}, Date {date}.")
        trial_info = None

    orientations = stim_info[0]  # grating orientation per trial
    is_plaid = stim_info[1] == 1  # plaid (True) or grating (False)

    osis = data.get_osis() if hasattr(data, 'get_osis') else None

    return data, vis_resp_cells, stim_info, red_labels, orientations, is_plaid, osis
def plot_kde_pv_vs_sst(save_dir):
    results_path = os.path.join(save_dir, "noise_correlations_pairwise.csv")
    if not os.path.exists(results_path):
        print(f"Error: File {results_path} not found.")
        return
    
    results_df = pd.read_csv(results_path)

    pv_mice = ['18S', '21N', '30G', '37L', '37L2', '37R', '37R2', '42N', '44N']
    sst_mice = ['30N', '32L', '32L2', '42R', '42R2']

    results_df["Inhibitory Population"] = results_df["Mouse"].apply(
        lambda mouse: "PV" if mouse in pv_mice else "SST" if mouse in sst_mice else None
    )
    results_df["Inhibitory Population"].fillna("Other", inplace=True)

    metrics = ["Exc-Red Noise Correlation", "Red-Red Noise Correlation", "Exc-Exc Noise Correlation"]

    for metric in metrics:
        pv_data = results_df.loc[
            (results_df["Inhibitory Population"] == "PV") & results_df[metric].notna(), metric
        ]
        sst_data = results_df.loc[
            (results_df["Inhibitory Population"] == "SST") & results_df[metric].notna(), metric
        ]

        if not pv_data.empty:
            print(f"{metric} - PV Avg: {pv_data.mean():.3f}, Std: {pv_data.std():.3f}")
        else:
            print(f"Warning: No PV data available for {metric}")

        if not sst_data.empty:
            print(f"{metric} - SST Avg: {sst_data.mean():.3f}, Std: {sst_data.std():.3f}")
        else:
            print(f"Warning: No SST data available for {metric}")

        plt.figure(figsize=(10, 7))
        if not pv_data.empty:
            sns.kdeplot(pv_data, fill=True, label=f"PV ({metric})", bw_adjust=0.5)
        if not sst_data.empty:
            sns.kdeplot(sst_data, fill=True, label=f"SST ({metric})", bw_adjust=0.5)

        plt.title(f"KDE of {metric} (PV vs SST)")
        plt.xlabel(metric)
        plt.ylabel("Density")
        plt.legend(title="Population")
        plt.grid(True)

        plot_path = os.path.join(save_dir, f"KDE_{metric.replace(' ', '_')}_PV_vs_SST.png")
        plt.savefig(plot_path)
        plt.close()

    print("KDE plots generated and saved successfully.")
def analyze_neurons():
    all_results = []
    for mouse, dates in micedates.items():
        for date in dates:
            try:
                # Load data
                data, vis_resp_cells, stim_info, red_labels, orientations, is_plaid, osis = load_data(mouse, date)
                
                # Apply vis_resp_cells mask to keep only visually responsive neurons
                if vis_resp_cells is not None:
                    neuron_indices = np.where(vis_resp_cells)[0]  # ROI indices of vis-responsive neurons
                else:
                    neuron_indices = np.arange(data.spks.shape[0])  # Default to all neurons
                
                n_neurons = len(neuron_indices)  # Update n_neurons count

                if red_labels is None or np.sum(red_labels) == 0:
                    print(f"Skipping Mouse {mouse}, Date {date} - No red-labeled neurons.")
                    continue

                print(f"\nProcessing Neurons for Mouse: {mouse}, Date: {date}")

                # Identify PV and SST populations
                pv_mice = ['18S', '21N', '30G', '37L', '37L2', '37R', '37R2', '42N', '44N']
                sst_mice = ['30N', '32L', '32L2', '42R', '42R2']
                inhibitory_population = "PV" if mouse in pv_mice else "SST" if mouse in sst_mice else None

                # Compute noise correlations
                results = compute_noise_correlations(data, red_labels, vis_resp_cells)

                roi_to_sequential = {roi: i for i, roi in enumerate(neuron_indices)}
                red_neuron_to_sequential = {roi: i for i, roi in enumerate(neuron_indices) if red_labels[roi]}

                neuron_indices_fixed = np.array([roi_to_sequential[roi] for roi in neuron_indices if roi in roi_to_sequential])

                vis_resp_red_indices = np.array([roi for roi in neuron_indices if red_labels[roi]])
                vis_resp_exc_indices = np.array([roi for roi in neuron_indices if not red_labels[roi]])

                vis_resp_red_indices_fixed = np.array([roi_to_sequential[roi] for roi in vis_resp_red_indices if roi in roi_to_sequential])
                vis_resp_exc_indices_fixed = np.array([roi_to_sequential[roi] for roi in vis_resp_exc_indices if roi in roi_to_sequential])

                n_vis_resp = len(neuron_indices_fixed)  # Visually responsive count
                
                # Extract correlation matrices correctly (fixed on 2/6)
                raw_red_red_corrs = results["Red-Red Noise Correlation"].reshape(n_vis_resp, n_vis_resp)
                raw_exc_red_corrs = results["Exc-Red Noise Correlation"].reshape(n_vis_resp, n_vis_resp)
                raw_exc_exc_corrs = results["Exc-Exc Noise Correlation"].reshape(n_vis_resp, n_vis_resp)

                # Extract submatrices using properly mapped indices
                red_red_corrs = raw_red_red_corrs[np.ix_(vis_resp_red_indices_fixed, vis_resp_red_indices_fixed)]
                exc_red_corrs = raw_exc_red_corrs[np.ix_(vis_resp_exc_indices_fixed, vis_resp_red_indices_fixed)]
                exc_exc_corrs = raw_exc_exc_corrs[np.ix_(vis_resp_exc_indices_fixed, vis_resp_exc_indices_fixed)]

                #breaking this shit down by steps because i am so lost and i need this to run

                # Step 2: Ensure sequential indices exist
                neuron_indices_fixed = np.array([roi_to_sequential[roi] for roi in neuron_indices if roi in roi_to_sequential])

                # Step 3: Iterate through neuron pairs and store correlations
                for i, neuron1 in enumerate(neuron_indices_fixed):
                    for j, neuron2 in enumerate(neuron_indices_fixed):
                        if neuron1 == neuron2:
                            continue  # Skip self-correlations

                        pv_mice = ['18S', '21N', '30G', '37L', '37L2', '37R', '37R2', '42N', '44N']
                        sst_mice = ['30N', '32L', '32L2', '42R', '42R2']

                        pair_result = {
                            "Mouse": mouse,
                            "Date": date,
                            "Neuron1": neuron1,
                            "Neuron2": neuron2,
                            "Inhibitory Population": "PV" if mouse in pv_mice else "SST" if mouse in sst_mice else "Other"
                        }
                        

                        # ✅ Store Exc-Exc Correlations
                        if neuron1 in vis_resp_exc_indices_fixed and neuron2 in vis_resp_exc_indices_fixed:
                            i1 = np.where(vis_resp_exc_indices_fixed == neuron1)[0]
                            i2 = np.where(vis_resp_exc_indices_fixed == neuron2)[0]
                            if i1.size > 0 and i2.size > 0:
                                pair_result["Exc-Exc Noise Correlation"] = exc_exc_corrs[i1[0], i2[0]]
                            else:
                                pair_result["Exc-Exc Noise Correlation"] = np.nan
                        else:
                            pair_result["Exc-Exc Noise Correlation"] = np.nan

                        # ✅ Store Red-Red Correlations
                        if neuron1 in vis_resp_red_indices_fixed and neuron2 in vis_resp_red_indices_fixed:
                            i1 = np.where(vis_resp_red_indices_fixed == neuron1)[0]
                            i2 = np.where(vis_resp_red_indices_fixed == neuron2)[0]

                            if i1.size > 0 and i2.size > 0:
                                pair_result["Red-Red Noise Correlation"] = red_red_corrs[i1[0], i2[0]]
                                print(f"🔴 Storing Red-Red Correlation: ({neuron1}, {neuron2}) = {pair_result['Red-Red Noise Correlation']}")
                            else:
                                pair_result["Red-Red Noise Correlation"] = np.nan
                                print(f"❌ Red-Red Correlation Skipped: ({neuron1}, {neuron2}) - Index Issue")
                        else:
                            pair_result["Red-Red Noise Correlation"] = np.nan

                        if (neuron1 in vis_resp_exc_indices_fixed and neuron2 in vis_resp_red_indices_fixed) or \
                        (neuron1 in vis_resp_red_indices_fixed and neuron2 in vis_resp_exc_indices_fixed):
                            
                            i1 = np.where(vis_resp_exc_indices_fixed == (neuron1 if neuron1 in vis_resp_exc_indices_fixed else neuron2))[0]
                            i2 = np.where(vis_resp_red_indices_fixed == (neuron2 if neuron2 in vis_resp_red_indices_fixed else neuron1))[0]

                            if i1.size > 0 and i2.size > 0:
                                pair_result["Exc-Red Noise Correlation"] = exc_red_corrs[i1[0], i2[0]]
                            else:
                                pair_result["Exc-Red Noise Correlation"] = np.nan
                                print(f"❌ Exc-Red Correlation Skipped: ({neuron1}, {neuron2}) - Index Issue")
                        else:
                            pair_result["Exc-Red Noise Correlation"] = np.nan

                        # Append results
                        all_results.append(pair_result)

                # ✅ Print Summary Statistics
                num_exc_exc = sum(1 for pair in all_results if not np.isnan(pair.get("Exc-Exc Noise Correlation", np.nan)))
                num_red_red = sum(1 for pair in all_results if not np.isnan(pair.get("Red-Red Noise Correlation", np.nan)))
                num_exc_red = sum(1 for pair in all_results if not np.isnan(pair.get("Exc-Red Noise Correlation", np.nan)))

                print(f"\n🔵 Total Exc-Exc Correlations Stored: {num_exc_exc}/{len(all_results)}")
                print(f"🔴 Total Red-Red Correlations Stored: {num_red_red}/{len(all_results)}")
                print(f"🟢 Total Exc-Red Correlations Stored: {num_exc_red}/{len(all_results)}")

                print(f"✅ Successfully appended {len(all_results)} neuron pairs to all_results.")

            except Exception as e:
                print(f"Error processing Mouse {mouse}, Date {date}: {e}")
            
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(save_dir, "noise_correlations_pairwise.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Pairwise noise correlations saved to {results_path}.")
        
        # Plot KDE for PV vs SST populations
        plot_kde_pv_vs_sst(save_dir)
    else:
        print("No results to save.")

#trial-specific noise corrs, segmented by population
def plot_kde_pv_vs_sst_trial_segmented(results_df, save_dir):
    """
    Plot KDE for noise correlations segmented by trial type (Grating, Plaid) 
    and inhibitory population (PV, SST).
    """
    print("\nGenerating KDE plots segmented by trial type and population...")

    # PV and SST mice
    pv_mice = ['18S', '21N', '30G', '37L', '37L2', '37R', '37R2', '42N', '44N']
    sst_mice = ['30N', '32L', '32L2', '42R', '42R2']

    # add column for inhibitory population
    results_df["Inhibitory Population"] = results_df["Mouse"].apply(
        lambda mouse: "PV" if mouse in pv_mice else "SST" if mouse in sst_mice else None
    )
    results_df["Inhibitory Population"].fillna("Other", inplace=True)

    # metrics to analyze
    metrics = ["Exc-Red Noise Correlation", "Red-Red Noise Correlation", "Exc-Exc Noise Correlation"]

    for metric in metrics:
        # filter data for PV and SST, segmented by trial type (NEW)
        grating_pv_data = results_df[
            (results_df["Inhibitory Population"] == "PV") & 
            (results_df["Trial Type"] == "Grating")
        ][metric].dropna()
        grating_sst_data = results_df[
            (results_df["Inhibitory Population"] == "SST") & 
            (results_df["Trial Type"] == "Grating")
        ][metric].dropna()
        plaid_pv_data = results_df[
            (results_df["Inhibitory Population"] == "PV") & 
            (results_df["Trial Type"] == "Plaid")
        ][metric].dropna()
        plaid_sst_data = results_df[
            (results_df["Inhibitory Population"] == "SST") & 
            (results_df["Trial Type"] == "Plaid")
        ][metric].dropna()

        # display summary statistics
        print(f"{metric} - Grating PV Avg: {grating_pv_data.mean():.3f}, Std: {grating_pv_data.std():.3f}")
        print(f"{metric} - Grating SST Avg: {grating_sst_data.mean():.3f}, Std: {grating_sst_data.std():.3f}")
        print(f"{metric} - Plaid PV Avg: {plaid_pv_data.mean():.3f}, Std: {plaid_pv_data.std():.3f}")
        print(f"{metric} - Plaid SST Avg: {plaid_sst_data.mean():.3f}, Std: {plaid_sst_data.std():.3f}")

        # plot KDE
        plt.figure(figsize=(10, 7))
        sns.kdeplot(grating_pv_data, fill=True, label=f"Grating PV ({metric})", bw_adjust=0.5)
        sns.kdeplot(grating_sst_data, fill=True, label=f"Grating SST ({metric})", bw_adjust=0.5)
        sns.kdeplot(plaid_pv_data, fill=True, label=f"Plaid PV ({metric})", bw_adjust=0.5)
        sns.kdeplot(plaid_sst_data, fill=True, label=f"Plaid SST ({metric})", bw_adjust=0.5)
        plt.title(f"KDE of {metric} (Grating vs Plaid, PV vs SST)")
        plt.xlabel(metric)
        plt.ylabel("Density")
        plt.legend(title="Trial Type and Population")
        plt.grid(True)
        plot_path = os.path.join(save_dir, f"KDE_{metric.replace(' ', '_')}_Grating_vs_Plaid_PV_vs_SST.png")
        plt.savefig(plot_path)
        plt.close()
def analyze_trial_specific_noise_correlations():
    trial_results = []

    for mouse, dates in micedates.items():
        for date in dates:
            try:
                data, vis_resp_cells, stim_info, red_labels, orientations, is_plaid, osis = load_data(mouse, date)

                if vis_resp_cells is not None:
                    neuron_indices = np.where(vis_resp_cells)[0]
                else:
                    neuron_indices = np.arange(data.spks_tr.shape[0])

                if red_labels is None or np.sum(red_labels) == 0:
                    continue

                pv_mice = ['18S', '21N', '30G', '37L', '37L2', '37R', '37R2', '42N', '44N']
                sst_mice = ['30N', '32L', '32L2', '42R', '42R2']
                inhibitory_population = "PV" if mouse in pv_mice else "SST" if mouse in sst_mice else None

                roi_to_sequential = {roi: i for i, roi in enumerate(neuron_indices)}
                red_neuron_to_sequential = {roi: i for i, roi in enumerate(neuron_indices) if red_labels[roi]}

                neuron_indices_fixed = np.array([roi_to_sequential[roi] for roi in neuron_indices if roi in roi_to_sequential])

                vis_resp_red_indices = np.array([roi for roi in neuron_indices if red_labels[roi]])
                vis_resp_exc_indices = np.array([roi for roi in neuron_indices if not red_labels[roi]])

                vis_resp_red_indices_fixed = np.array([roi_to_sequential[roi] for roi in vis_resp_red_indices if roi in roi_to_sequential])
                vis_resp_exc_indices_fixed = np.array([roi_to_sequential[roi] for roi in vis_resp_exc_indices if roi in roi_to_sequential])

                for trial_type in ["Grating", "Plaid"]:
                    is_plaid_trial = trial_type == "Plaid"
                    trial_indices = stim_info[1] == int(is_plaid_trial)

                    if trial_indices.shape[0] != data.spks_tr.shape[1]:
                        raise ValueError("Mismatch between trial indices and spks_tr dimensions!")

                    filtered_spks_tr = data.spks_tr[:, trial_indices, :]

                    if filtered_spks_tr.shape[1] == 0:
                        continue

                    n_neurons = filtered_spks_tr.shape[0]

                    reshaped_spks = filtered_spks_tr[:, :, data.resp_start:data.resp_end].mean(axis=2)
                    noise_corrs = np.corrcoef(reshaped_spks.reshape(n_neurons, -1))
                    np.fill_diagonal(noise_corrs, np.nan)

                    raw_red_red_corrs = noise_corrs[np.ix_(vis_resp_red_indices_fixed, vis_resp_red_indices_fixed)]
                    raw_exc_red_corrs = noise_corrs[np.ix_(vis_resp_exc_indices_fixed, vis_resp_red_indices_fixed)]
                    raw_exc_exc_corrs = noise_corrs[np.ix_(vis_resp_exc_indices_fixed, vis_resp_exc_indices_fixed)]

                    for i, neuron1 in enumerate(neuron_indices_fixed):
                        for j, neuron2 in enumerate(neuron_indices_fixed):
                            if neuron1 == neuron2:
                                continue

                            pair_result = {
                                "Mouse": mouse,
                                "Date": date,
                                "Trial Type": trial_type,
                                "Neuron1": neuron1,
                                "Neuron2": neuron2,
                                "Inhibitory Population": inhibitory_population
                            }

                            if neuron1 in vis_resp_exc_indices_fixed and neuron2 in vis_resp_exc_indices_fixed:
                                i1 = np.where(vis_resp_exc_indices_fixed == neuron1)[0]
                                i2 = np.where(vis_resp_exc_indices_fixed == neuron2)[0]
                                if i1.size > 0 and i2.size > 0:
                                    pair_result["Exc-Exc Noise Correlation"] = raw_exc_exc_corrs[i1[0], i2[0]]
                                else:
                                    pair_result["Exc-Exc Noise Correlation"] = np.nan
                            else:
                                pair_result["Exc-Exc Noise Correlation"] = np.nan

                            if neuron1 in vis_resp_red_indices_fixed and neuron2 in vis_resp_red_indices_fixed:
                                i1 = np.where(vis_resp_red_indices_fixed == neuron1)[0]
                                i2 = np.where(vis_resp_red_indices_fixed == neuron2)[0]
                                if i1.size > 0 and i2.size > 0:
                                    pair_result["Red-Red Noise Correlation"] = raw_red_red_corrs[i1[0], i2[0]]
                                else:
                                    pair_result["Red-Red Noise Correlation"] = np.nan
                            else:
                                pair_result["Red-Red Noise Correlation"] = np.nan

                            if (neuron1 in vis_resp_exc_indices_fixed and neuron2 in vis_resp_red_indices_fixed) or \
                            (neuron1 in vis_resp_red_indices_fixed and neuron2 in vis_resp_exc_indices_fixed):
                                i1 = np.where(vis_resp_exc_indices_fixed == (neuron1 if neuron1 in vis_resp_exc_indices_fixed else neuron2))[0]
                                i2 = np.where(vis_resp_red_indices_fixed == (neuron2 if neuron2 in vis_resp_red_indices_fixed else neuron1))[0]
                                if i1.size > 0 and i2.size > 0:
                                    pair_result["Exc-Red Noise Correlation"] = raw_exc_red_corrs[i1[0], i2[0]]
                                else:
                                    pair_result["Exc-Red Noise Correlation"] = np.nan
                            else:
                                pair_result["Exc-Red Noise Correlation"] = np.nan

                            trial_results.append(pair_result)

            except Exception as e:
                print(f"Error processing Mouse {mouse}, Date {date}: {e}")

    trial_results_df = pd.DataFrame(trial_results)
    results_path = os.path.join(save_dir, "trial_specific_noise_correlations.csv")
    trial_results_df.to_csv(results_path, index=False)
    print(f"Trial-specific noise correlations saved to {results_path}.")
    results_df = pd.read_csv(os.path.join(save_dir, "trial_specific_noise_correlations.csv"))
    pv_df = results_df[results_df["Inhibitory Population"] == "PV"]
    sst_df = results_df[results_df["Inhibitory Population"] == "SST"]

    num_exc_exc_pv = pv_df["Exc-Exc Noise Correlation"].count()
    num_exc_red_pv = pv_df["Exc-Red Noise Correlation"].count()
    num_red_red_pv = pv_df["Red-Red Noise Correlation"].count()

    num_exc_exc_sst = sst_df["Exc-Exc Noise Correlation"].count()
    num_exc_red_sst = sst_df["Exc-Red Noise Correlation"].count()
    num_red_red_sst = sst_df["Red-Red Noise Correlation"].count()

    print(f"\nTotal Exc-Exc Correlations Stored (PV): {num_exc_exc_pv}")
    print(f"Total Exc-Red Correlations Stored (PV): {num_exc_red_pv}")
    print(f"Total Red-Red Correlations Stored (PV): {num_red_red_pv}")

    print(f"Total Exc-Exc Correlations Stored (SST): {num_exc_exc_sst}")
    print(f"Total Exc-Red Correlations Stored (SST): {num_exc_red_sst}")
    print(f"Total Red-Red Correlations Stored (SST): {num_red_red_sst}")

    plot_kde_pv_vs_sst_trial_segmented(results_df, save_dir)
#

def plot_orientation_segmented_results():
    results_path = os.path.join(save_dir, "orientation_segmented_noise_correlations.csv")
    if not os.path.exists(results_path):
        print(f"Error: File {results_path} not found.")
        return
    
    results_df = pd.read_csv(results_path)

    if "Angular Difference" not in results_df.columns:
        print("Error: 'Angular Difference' column missing in results file.")
        return
    
    # Convert angles to float for consistency
    results_df["Angular Difference"] = results_df["Angular Difference"].astype(float)

    # Group angles (handle angle symmetry)
    group_mapping = {
        "0": [0.0],
        "22.5": [22.5, 157.5],
        "45": [45.0, 135.0],
        "67.5": [67.5, 112.5],
        "90": [90.0]
    }

    plaid_grouped_data = {group: [] for group in group_mapping.keys()}
    grating_grouped_data = {group: [] for group in group_mapping.keys()}

    for group, angles in group_mapping.items():
        plaid_grouped_data[group] = results_df.loc[
            results_df["Angular Difference"].isin(angles), "Plaid Corr"
        ].dropna().tolist()

        grating_grouped_data[group] = results_df.loc[
            results_df["Angular Difference"].isin(angles), "Grating Corr"
        ].dropna().tolist()

    # Convert grouped data into DataFrames
    plaid_data = [(group, val) for group, data in plaid_grouped_data.items() for val in data]
    grating_data = [(group, val) for group, data in grating_grouped_data.items() for val in data]

    plaid_df = pd.DataFrame(plaid_data, columns=["Group", "Noise Correlation"])
    grating_df = pd.DataFrame(grating_data, columns=["Group", "Noise Correlation"])

    # Ensure there is data before plotting
    if not plaid_df.empty:
        plt.figure(figsize=(12, 8))
        for group, data in plaid_grouped_data.items():
            if data:  # Only plot if data exists
                sns.kdeplot(data, fill=True, label=f"Plaid {group}")
        plt.title("KDE of Noise Correlations (Plaid Groups)")
        plt.xlabel("Noise Correlation")
        plt.ylabel("Density")
        plt.legend(title="Groups")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "kde_plaid_combined_groups.png"))
        plt.close()
    else:
        print("No valid Plaid noise correlation data available for plotting.")

    if not grating_df.empty:
        plt.figure(figsize=(12, 8))
        for group, data in grating_grouped_data.items():
            if data:  
                sns.kdeplot(data, fill=True, label=f"Grating {group}")
        plt.title("KDE of Noise Correlations (Grating Groups)")
        plt.xlabel("Noise Correlation")
        plt.ylabel("Density")
        plt.legend(title="Groups")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "kde_grating_combined_groups.png"))
        plt.close()
    else:
        print("No valid Grating noise correlation data available for plotting.")
def plot_trend_analysis_orientation_segmented_with_stats():
    # Load the results from the CSV
    results_path = os.path.join(save_dir, "orientation_segmented_noise_correlations.csv")
    results_df = pd.read_csv(results_path)

    results_df["Noise Corr Diff (Plaid-Grating)"] = results_df["Plaid Corr"] - results_df["Grating Corr"]

    # compute normalization index and osi differences between neuron pairs
    results_df["NI Difference"] = abs(results_df["Normalization Index (NI) Neuron1"] - results_df["Normalization Index (NI) Neuron2"])
    results_df["OSI Difference"] = abs(results_df["OSI Neuron1"] - results_df["OSI Neuron2"])

    # Extract the plaid and grating columns
    plaid_cols = [col for col in results_df.columns if col.startswith("Plaid Corr")]
    grating_cols = [col for col in results_df.columns if col.startswith("Grating Corr")]

    results_df["Angular Difference"] = results_df["Angular Difference"].astype(float)

    group_mapping = {
        "1": [0.0],
        "2/8": [22.5, 157.5],
        "3/7": [45.0, 135.0],
        "4/6": [67.5, 112.5],
        "5": [90.0],
    }

    # Combine plaid groups
    plaid_grouped_data = {group: [] for group in group_mapping.keys()}
    grating_grouped_data = {group: [] for group in group_mapping.keys()}

    for group, angles in group_mapping.items():
        plaid_grouped_data[group] = results_df.loc[
            results_df["Angular Difference"].isin(angles), "Plaid Corr"
        ].dropna().values.tolist()

        grating_grouped_data[group] = results_df.loc[
            results_df["Angular Difference"].isin(angles), "Grating Corr"
        ].dropna().values.tolist()


    # Compute group statistics (mean, SEM)
    plaid_means = []
    plaid_errors = []
    plaid_angles = []

    grating_means = []
    grating_errors = []
    grating_angles = []

    group_angles = {
        "1": 0.0,
        "2/8": 22.5,
        "3/7": 45.0,
        "4/6": 67.5,
        "5": 90.0,
    }

    for group, data in plaid_grouped_data.items():
        if len(data) > 0:
            plaid_angles.append(group_angles[group])
            plaid_means.append(np.mean(data))
            plaid_errors.append(sem(data))  # Standard error of the mean (SEM)

    for group, data in grating_grouped_data.items():
        if len(data) > 0:
            grating_angles.append(group_angles[group])
            grating_means.append(np.mean(data))
            grating_errors.append(sem(data))

    # Convert angles to NumPy arrays for fitting
    plaid_angles = np.array(plaid_angles, dtype=float)
    grating_angles = np.array(grating_angles, dtype=float)

    for metric in ["NI Difference", "OSI Difference"]:
        pearson_corr, pearson_p = pearsonr(results_df["Noise Corr Diff (Plaid-Grating)"].dropna(), results_df[metric].dropna())
        spearman_corr, spearman_p = spearmanr(results_df["Noise Corr Diff (Plaid-Grating)"].dropna(), results_df[metric].dropna())

        print(f"\n--- Correlation between {metric} and Noise Corr Diff (Plaid-Grating) ---")
        print(f"Pearson correlation: r = {pearson_corr:.3f}, p-value = {pearson_p:.3e}")
        print(f"Spearman correlation: ρ = {spearman_corr:.3f}, p-value = {spearman_p:.3e}")

        if pearson_p < 0.05:
            print("Pearson correlation is significant (p < 0.05)")
        if spearman_p < 0.05:
            print("Spearman correlation is significant (p < 0.05)")

    # Fit regression models
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c

    def exponential(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Fit plaid to a quadratic model
    plaid_params, _ = curve_fit(quadratic, plaid_angles, plaid_means)
    plaid_fitted = quadratic(plaid_angles, *plaid_params)
    plaid_r2 = 1 - (np.sum((plaid_means - plaid_fitted) ** 2) / np.sum((plaid_means - np.mean(plaid_means)) ** 2))

    # Fit grating to an exponential model
    grating_params, _ = curve_fit(exponential, grating_angles, grating_means)
    grating_fitted = exponential(grating_angles, *grating_params)
    grating_r2 = 1 - (np.sum((grating_means - grating_fitted) ** 2) / np.sum((grating_means - np.mean(grating_means)) ** 2))
    # scatter plots of noise correlation difference vs ni/osi difference
    for metric in ["NI Difference", "OSI Difference"]:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=results_df[metric], y=results_df["Noise Corr Diff (Plaid-Grating)"], color="blue", alpha=0.7)
        plt.title(f"Scatter Plot: Noise Corr Diff vs {metric}")
        plt.xlabel(metric)
        plt.ylabel("Noise Corr Diff (Plaid-Grating)")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"scatter_noise_corr_diff_vs_{metric.replace(' ', '_').lower()}.png"))
        plt.close()
    # Scatter plot with error bars and regression fits
    plt.figure(figsize=(10, 6))

    # Plaid data
    plt.errorbar(plaid_angles, plaid_means, yerr=plaid_errors, fmt="o", color="blue", label="Plaid (Mean ± SEM)")
    plt.plot(plaid_angles, plaid_fitted, linestyle="--", color="blue", label=f"Quadratic Fit (R²={plaid_r2:.2f})")

    # Grating data
    plt.errorbar(grating_angles, grating_means, yerr=grating_errors, fmt="o", color="orange", label="Grating (Mean ± SEM)")
    plt.plot(grating_angles, grating_fitted, linestyle="--", color="orange", label=f"Exponential Fit (R²={grating_r2:.2f})")

    plt.title("Angular Difference vs Noise Correlation (with Best Fits)")
    plt.xlabel("Angular Difference (Degrees)")
    plt.ylabel("Mean Noise Correlation")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "scatter_with_best_fit_and_error_bars.png"))
    plt.close()

    # Violin plots for plaid and grating
    plaid_data = [(group_angles[group], val) for group, values in plaid_grouped_data.items() for val in values]
    plaid_df = pd.DataFrame(plaid_data, columns=["Angular Difference in Tuning", "Noise Correlation"])

    grating_data = [(group_angles[group], val) for group, values in grating_grouped_data.items() for val in values]
    grating_df = pd.DataFrame(grating_data, columns=["Angular Difference in Tuning", "Noise Correlation"])

    plt.figure(figsize=(12, 6))
    plaid_means = plaid_df.groupby("Angular Difference in Tuning")["Noise Correlation"].mean()
    plaid_sems = plaid_df.groupby("Angular Difference in Tuning")["Noise Correlation"].sem()

    plt.bar(plaid_means.index, plaid_means, yerr=plaid_sems, capsize=5, color="purple", alpha=0.7, width=10)
    plt.title("Bar Plot: Noise Correlations (Plaid) with SEM")
    plt.xlabel("Angular Difference in Tuning")
    plt.ylabel("Mean Noise Correlation")
    plt.xticks([0, 22.5, 45, 67.5, 90])  # Ensure correct tick marks
    plt.grid(True, axis="y")
    plt.savefig(os.path.join(save_dir, "bar_plaid_groups_sem.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    grating_means = grating_df.groupby("Angular Difference in Tuning")["Noise Correlation"].mean()
    grating_sems = grating_df.groupby("Angular Difference in Tuning")["Noise Correlation"].sem()

    plt.bar(grating_means.index, grating_means, yerr=grating_sems, capsize=5, color="pink", alpha=0.7, width=10)
    plt.title("Bar Plot: Noise Correlations (Grating) with SEM")
    plt.xlabel("Angular Difference in Tuning")
    plt.ylabel("Mean Noise Correlation")
    plt.xticks([0, 22.5, 45, 67.5, 90])  # Ensure correct tick marks
    plt.grid(True, axis="y")
    plt.savefig(os.path.join(save_dir, "bar_grating_groups_sem.png"))
    plt.close()

    # Statistical comparison for 90° (Group 5) plaid vs grating
    plaid_90 = plaid_grouped_data["5"]
    grating_90 = grating_grouped_data["5"]

    if len(plaid_90) > 0 and len(grating_90) > 0:
        print("\n--- Statistical Comparison (90° Plaid vs Grating) ---")
        
        # **1. Wilcoxon Rank-Sum Test (Mann-Whitney U Equivalent)**
        stat_ranksum, p_ranksum = ranksums(plaid_90, grating_90)
        print(f"Ranksum Test Statistic: {stat_ranksum:.3f}, p-value: {p_ranksum:.3e}")
        if p_ranksum < 0.05:
            print("Result: Significant difference detected (p < 0.05).")
        else:
            print("Result: No significant difference detected (p >= 0.05).")
        
        # **2. Independent Two-Tailed T-Test**
        stat_ttest, p_ttest = ttest_ind(plaid_90, grating_90, equal_var=False)  # Welch's T-test
        print(f"T-Test Statistic: {stat_ttest:.3f}, p-value: {p_ttest:.3e}")
        if p_ttest < 0.05:
            print("Result: Significant difference detected (p < 0.05).")
        else:
            print("Result: No significant difference detected (p >= 0.05).")
        
        # **3. Mann-Whitney U Test (Non-parametric)**
        stat_mann, p_mann = mannwhitneyu(plaid_90, grating_90, alternative='two-sided')
        print(f"Mann-Whitney U Statistic: {stat_mann:.3f}, p-value: {p_mann:.3e}")
        if p_mann < 0.05:
            print("Result: Significant difference detected (p < 0.05).")
        else:
            print("Result: No significant difference detected (p >= 0.05).")
    # Summary of fits
    print("\n--- Fit Summary ---")
    print(f"Plaid Quadratic Fit Parameters: a={plaid_params[0]:.3f}, b={plaid_params[1]:.3f}, c={plaid_params[2]:.3f}")
    print(f"Plaid R²: {plaid_r2:.3f}")
    print(f"Grating Exponential Fit Parameters: a={grating_params[0]:.3f}, b={grating_params[1]:.3f}, c={grating_params[2]:.3f}")
    print(f"Grating R²: {grating_r2:.3f}")

    print("\nPlots and statistical analysis completed.")
def analyze_orthogonal_noise_correlations():
    # load the orientation-segmented results
    results_path = os.path.join(save_dir, "orientation_segmented_noise_correlations.csv")
    if not os.path.exists(results_path):
        print(f"Error: File {results_path} not found. Run `orientation_segmentation` first.")
        return

    results_df = pd.read_csv(results_path)

    # ensure necessary columns exist
    required_cols = {"Mouse", "Date", "Neuron1", "Neuron2", "Angular Difference", "Plaid Corr", "Grating Corr"}
    if not required_cols.issubset(results_df.columns):
        print(f"Error: Required columns missing from {results_path}. Ensure `orientation_segmentation` ran correctly.")
        return

    # filter for neuron pairs with an angular difference of 90 degrees
    orthogonal_pairs = results_df[results_df["Angular Difference"] == 90].copy()

    # compute noise correlation difference (plaid - grating)
    orthogonal_pairs["Noise Corr Diff (Plaid-Grating)"] = orthogonal_pairs["Plaid Corr"] - orthogonal_pairs["Grating Corr"]

    orthogonal_pairs["NI Average"] = (orthogonal_pairs["Normalization Index (NI) Neuron1"] + orthogonal_pairs["Normalization Index (NI) Neuron2"]) / 2
    orthogonal_pairs["NI Difference"] = np.abs(orthogonal_pairs["Normalization Index (NI) Neuron1"] - orthogonal_pairs["Normalization Index (NI) Neuron2"])

    # compute osi average and osi difference
    orthogonal_pairs["OSI Average"] = (orthogonal_pairs["OSI Neuron1"] + orthogonal_pairs["OSI Neuron2"]) / 2
    orthogonal_pairs["OSI Difference"] = np.abs(orthogonal_pairs["OSI Neuron1"] - orthogonal_pairs["OSI Neuron2"])

    # drop rows with NaN values in the difference column
    orthogonal_pairs.dropna(subset=["Noise Corr Diff (Plaid-Grating)"], inplace=True)

    if orthogonal_pairs.empty:
        print("No valid orthogonal neuron pairs found for analysis.")
        return

    # correlation analysis
    correlation_metrics = ["NI Difference", "OSI Difference", "NI Average", "OSI Average"]
    for metric in correlation_metrics:
        x = orthogonal_pairs["Noise Corr Diff (Plaid-Grating)"].dropna()
        y = orthogonal_pairs[metric].dropna()

        # ensure x and y have the same length
        min_length = min(len(x), len(y))
        x, y = x.iloc[:min_length], y.iloc[:min_length]

        # pearson correlation
        pearson_corr, pearson_p = pearsonr(x, y)
        print(f"{metric} vs Noise Corr Diff (Pearson): r = {pearson_corr:.3f}, p = {pearson_p:.3e}")

        # spearman correlation
        spearman_corr, spearman_p = spearmanr(x, y)
        print(f"{metric} vs Noise Corr Diff (Spearman): r = {spearman_corr:.3f}, p = {spearman_p:.3e}")

        # scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x, y=y, color="blue")
        plt.xlabel("Noise Corr Diff (Plaid-Grating)")
        plt.ylabel(metric)
        plt.title(f"Noise Corr Diff vs {metric}")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"scatter_noise_corr_diff_vs_{metric.replace(' ', '_').lower()}.png"))
        plt.close()

    # save filtered results for reference
    filtered_results_path = os.path.join(save_dir, "orthogonal_noise_correlation_analysis.csv")
    orthogonal_pairs.to_csv(filtered_results_path, index=False)
    print(f"Filtered results saved to {filtered_results_path}.")

    # determine threshold for top 10% of pairs with the largest noise correlation change
    threshold = orthogonal_pairs["Noise Corr Diff (Plaid-Grating)"].quantile(0.90)

    # identify top 10% pairs
    top_10_percent_df = orthogonal_pairs[orthogonal_pairs["Noise Corr Diff (Plaid-Grating)"] >= threshold]
    rest_df = orthogonal_pairs[orthogonal_pairs["Noise Corr Diff (Plaid-Grating)"] < threshold]

    # metrics to analyze
    metrics = ["Plaid Corr", "Grating Corr"]

    # compute mean and standard deviation for each metric
    print("\n--- top 10% vs entire population comparison ---")
    for metric in metrics:
        top_mean = top_10_percent_df[metric].mean()
        top_std = top_10_percent_df[metric].std()
        rest_mean = rest_df[metric].mean()
        rest_std = rest_df[metric].std()

        print(f"{metric}:")
        print(f"  top 10% - mean: {top_mean:.3f}, std: {top_std:.3f}")
        print(f"  rest - mean: {rest_mean:.3f}, std: {rest_std:.3f}")

        # perform statistical tests
        t_stat, t_p_value = ttest_ind(top_10_percent_df[metric].dropna(), rest_df[metric].dropna(), equal_var=False)
        print(f"  t-test - statistic: {t_stat:.3f}, p-value: {t_p_value:.3e}")

        u_stat, u_p_value = mannwhitneyu(top_10_percent_df[metric].dropna(), rest_df[metric].dropna(), alternative='two-sided')
        print(f"  mann-whitney u test - statistic: {u_stat:.3f}, p-value: {u_p_value:.3e}")

    # scatter plot comparing noise corr diff with plaid and grating correlations for top 10%
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=orthogonal_pairs["Noise Corr Diff (Plaid-Grating)"],
            y=orthogonal_pairs[metric],
            color="gray",
            label="all orthogonal pairs"
        )
        sns.scatterplot(
            x=top_10_percent_df["Noise Corr Diff (Plaid-Grating)"],
            y=top_10_percent_df[metric],
            color="red",
            label="top 10% pairs"
        )
        plt.title(f"noise corr diff vs {metric} (top 10% highlighted)")
        plt.xlabel("noise corr diff (plaid-grating)")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"scatter_top10_vs_{metric.replace(' ', '_').lower()}.png"))
        plt.close()

    print("top 10% analysis and scatter plots completed.")
def sst_analysis(save_dir):
    """
    Performs a comprehensive analysis of SST neurons, examining:
    1. Plaid vs. Grating noise correlations
    2. Excitatory-SST (E-I) vs. Excitatory-Excitatory (E-E) interactions
    3. SST-SST (I-I) interactions
    4. Regression analysis of OSI, NI, and Fano Factor as predictors
    5. Angular difference effects

    Parameters:
    - save_dir (str): Directory containing the dataset.

    Returns:
    - None (prints analysis results).
    """

    # 🔹 Load Dataset
    file_path = os.path.join(save_dir, "orientation_segmented_noise_correlations.csv")
    if not os.path.exists(file_path):
        print(f"⚠ Missing file: {file_path}")
        return
    
    df = pd.read_csv(file_path)

    # 🔹 Filter for SST-specific neuron pairs
    df_sst = df[df["Inhibitory Population"] == "SST"]

    if df_sst.empty:
        print("⚠ No SST data found. Check if 'Inhibitory Population' column is formatted correctly.")
        return
    print(f"✔ SST-specific data extracted: {len(df_sst)} rows")

    ### 1️⃣ Plaid vs. Grating Comparisons ###
    print("\n📊 **Plaid vs. Grating Noise Correlations (SST Pairs)**")

    plaid = df_sst[df_sst["Trial Type"] == "Plaid"]
    grating = df_sst[df_sst["Trial Type"] == "Grating"]

    for noise_corr in ["Exc-Red Noise Correlation", "Red-Red Noise Correlation"]:
        if noise_corr in df_sst.columns:
            plaid_data = plaid[noise_corr].dropna()
            grating_data = grating[noise_corr].dropna()

            print(f"  🔍 {noise_corr} - Plaid: {len(plaid_data)}, Grating: {len(grating_data)}")

            if len(plaid_data) > 0 and len(grating_data) > 0:
                stat, p_value = mannwhitneyu(grating_data, plaid_data, alternative="two-sided")
                print(f"  🔹 {noise_corr}: Mann-Whitney U = {stat:.3f}, p = {p_value:.3e}")
            else:
                print(f"  ⚠ Not enough valid data for {noise_corr}")

    ### 2️⃣ Excitatory-SST (E-I) vs. Excitatory-Excitatory (E-E) ###
    print("\n📊 **Excitatory-SST vs. Excitatory-Excitatory Noise Correlation**")

    exc_sst = df_sst[(df_sst["Neuron1 Type"] == "Excitatory") | (df_sst["Neuron2 Type"] == "Excitatory")]
    exc_exc = df[(df["Neuron1 Type"] == "Excitatory") & (df["Neuron2 Type"] == "Excitatory")]

    if "Exc-Red Noise Correlation" in df_sst.columns and "Exc-Exc Noise Correlation" in df.columns:
        exc_sst_data = exc_sst["Exc-Red Noise Correlation"].dropna()
        exc_exc_data = exc_exc["Exc-Exc Noise Correlation"].dropna()

        if len(exc_sst_data) > 0 and len(exc_exc_data) > 0:
            stat, p_value = mannwhitneyu(exc_sst_data, exc_exc_data, alternative="two-sided")
            print(f"  🔹 Exc-SST vs. Exc-Exc: Mann-Whitney U = {stat:.3f}, p = {p_value:.3e}")
        else:
            print(f"  ⚠ Not enough valid data for Exc-SST vs. Exc-Exc")

    ### 3️⃣ SST-SST (I-I) Interactions ###
    print("\n📊 **SST-SST vs. Exc-Exc Noise Correlation**")

    sst_sst = df_sst[(df_sst["Neuron1 Type"] == "Inhibitory") & (df_sst["Neuron2 Type"] == "Inhibitory")]

    if not sst_sst.empty and "Red-Red Noise Correlation" in df_sst.columns:
        sst_sst_data = sst_sst["Red-Red Noise Correlation"].dropna()
        exc_exc_data = exc_exc["Exc-Exc Noise Correlation"].dropna()

        if len(sst_sst_data) > 0 and len(exc_exc_data) > 0:
            stat, p_value = mannwhitneyu(sst_sst_data, exc_exc_data, alternative="two-sided")
            print(f"  🔹 SST-SST vs. Exc-Exc: Mann-Whitney U = {stat:.3f}, p = {p_value:.3e}")
        else:
            print(f"  ⚠ Not enough valid data for SST-SST vs. Exc-Exc")

    ### 4️⃣ Regression Analysis for Predictors (OSI, NI, Fano Factor) ###
    print("\n📊 **Regression Analysis: SST Noise Correlation Predictors**")

    predictors = [
        "Normalization Index (NI) Neuron1", "Normalization Index (NI) Neuron2",
        "OSI Neuron1", "OSI Neuron2",
        "Fano Factor Neuron1", "Fano Factor Neuron2",
        "Angular Difference"
    ]

    for noise_corr in ["Exc-Red Noise Correlation", "Red-Red Noise Correlation"]:
        if noise_corr in df_sst.columns:
            df_sst_reg = df_sst.dropna(subset=[noise_corr] + predictors)

            if len(df_sst_reg) > 10:
                X = df_sst_reg[predictors]
                X = sm.add_constant(X)
                y = df_sst_reg[noise_corr]

                model = sm.OLS(y, X).fit()
                print(f"\n📊 **Regression: {noise_corr}**")
                print(model.summary())
            else:
                print(f"⚠ Not enough data for regression ({noise_corr})")

    ### 5️⃣ Angular Difference Effects ###
    print("\n📊 **Angular Difference Effects on SST Noise Correlation**")

    angle_groups = {
        "0°": [0],
        "22.5°-157.5°": [22.5, 157.5],
        "45°-135°": [45, 135],
        "67.5°-112.5°": [67.5, 112.5],
        "90°": [90]
    }

    for noise_corr in ["Exc-Red Noise Correlation", "Red-Red Noise Correlation"]:
        if noise_corr in df_sst.columns:
            for angle_label, angles in angle_groups.items():
                angle_df = df_sst[df_sst["Angular Difference"].isin(angles)][noise_corr].dropna()
                print(f"  🔍 {noise_corr} - {angle_label}: {len(angle_df)} values, Mean = {angle_df.mean():.3f}, SD = {angle_df.std():.3f}")

    print("\n✅ **SST Analysis Complete!**")

def pv_analysis(save_dir):
    """
    Performs a comprehensive analysis of PV neurons, examining:
    1. Plaid vs. Grating noise correlations
    2. Excitatory-SST (E-I) vs. Excitatory-Excitatory (E-E) interactions
    3. SST-SST (I-I) interactions
    4. Regression analysis of OSI, NI, and Fano Factor as predictors
    5. Angular difference effects

    Parameters:
    - save_dir (str): Directory containing the dataset.

    Returns:
    - None (prints analysis results).
    """

    # 🔹 Load Dataset
    file_path = os.path.join(save_dir, "orientation_segmented_noise_correlations.csv")
    if not os.path.exists(file_path):
        print(f"⚠ Missing file: {file_path}")
        return
    
    df = pd.read_csv(file_path)

    # 🔹 Filter for SST-specific neuron pairs
    df_pv = df[df["Inhibitory Population"] == "PV"]

    if df_pv.empty:
        print("⚠ No SST data found. Check if 'Inhibitory Population' column is formatted correctly.")
        return
    print(f"✔ SST-specific data extracted: {len(df_pv)} rows")

    ### 1️⃣ Plaid vs. Grating Comparisons ###
    print("\n📊 **Plaid vs. Grating Noise Correlations (SST Pairs)**")

    plaid = df_pv[df_pv["Trial Type"] == "Plaid"]
    grating = df_pv[df_pv["Trial Type"] == "Grating"]

    for noise_corr in ["Exc-Red Noise Correlation", "Red-Red Noise Correlation"]:
        if noise_corr in df_pv.columns:
            plaid_data = plaid[noise_corr].dropna()
            grating_data = grating[noise_corr].dropna()

            print(f"  🔍 {noise_corr} - Plaid: {len(plaid_data)}, Grating: {len(grating_data)}")

            if len(plaid_data) > 0 and len(grating_data) > 0:
                stat, p_value = mannwhitneyu(grating_data, plaid_data, alternative="two-sided")
                print(f"  🔹 {noise_corr}: Mann-Whitney U = {stat:.3f}, p = {p_value:.3e}")
            else:
                print(f"  ⚠ Not enough valid data for {noise_corr}")

    ### 2️⃣ Excitatory-SST (E-I) vs. Excitatory-Excitatory (E-E) ###
    print("\n📊 **Excitatory-SST vs. Excitatory-Excitatory Noise Correlation**")

    exc_pv = df_pv[(df_pv["Neuron1 Type"] == "Excitatory") | (df_pv["Neuron2 Type"] == "Excitatory")]
    exc_exc = df[(df["Neuron1 Type"] == "Excitatory") & (df["Neuron2 Type"] == "Excitatory")]

    if "Exc-Red Noise Correlation" in df_pv.columns and "Exc-Exc Noise Correlation" in df.columns:
        exc_pv_data = exc_pv["Exc-Red Noise Correlation"].dropna()
        exc_exc_data = exc_exc["Exc-Exc Noise Correlation"].dropna()

        if len(exc_pv_data) > 0 and len(exc_exc_data) > 0:
            stat, p_value = mannwhitneyu(exc_pv_data, exc_exc_data, alternative="two-sided")
            print(f"  🔹 Exc-SST vs. Exc-Exc: Mann-Whitney U = {stat:.3f}, p = {p_value:.3e}")
        else:
            print(f"  ⚠ Not enough valid data for Exc-SST vs. Exc-Exc")

    ### 3️⃣ SST-SST (I-I) Interactions ###
    print("\n📊 **SST-SST vs. Exc-Exc Noise Correlation**")

    pv_pv = df_pv[(df_pv["Neuron1 Type"] == "Inhibitory") & (df_pv["Neuron2 Type"] == "Inhibitory")]

    if not pv_pv.empty and "Red-Red Noise Correlation" in df_pv.columns:
        pv_pv_data = pv_pv["Red-Red Noise Correlation"].dropna()
        exc_exc_data = exc_exc["Exc-Exc Noise Correlation"].dropna()

        if len(pv_pv_data) > 0 and len(exc_exc_data) > 0:
            stat, p_value = mannwhitneyu(pv_pv_data, exc_exc_data, alternative="two-sided")
            print(f"  🔹 PV-PV vs. Exc-Exc: Mann-Whitney U = {stat:.3f}, p = {p_value:.3e}")
        else:
            print(f"  ⚠ Not enough valid data for PV-PV vs. Exc-Exc")

    ### 4️⃣ Regression Analysis for Predictors (OSI, NI, Fano Factor) ###
    print("\n📊 **Regression Analysis: PV Noise Correlation Predictors**")

    predictors = [
        "Normalization Index (NI) Neuron1", "Normalization Index (NI) Neuron2",
        "OSI Neuron1", "OSI Neuron2",
        "Fano Factor Neuron1", "Fano Factor Neuron2",
        "Angular Difference"
    ]

    for noise_corr in ["Exc-Red Noise Correlation", "Red-Red Noise Correlation"]:
        if noise_corr in df_pv.columns:
            df_pv_reg = df_pv.dropna(subset=[noise_corr] + predictors)

            if len(df_pv_reg) > 10:
                X = df_pv_reg[predictors]
                X = sm.add_constant(X)
                y = df_pv_reg[noise_corr]

                model = sm.OLS(y, X).fit()
                print(f"\n📊 **Regression: {noise_corr}**")
                print(model.summary())
            else:
                print(f"⚠ Not enough data for regression ({noise_corr})")

    ### 5️⃣ Angular Difference Effects ###
    print("\n📊 **Angular Difference Effects on SST Noise Correlation**")

    angle_groups = {
        "0°": [0],
        "22.5°-157.5°": [22.5, 157.5],
        "45°-135°": [45, 135],
        "67.5°-112.5°": [67.5, 112.5],
        "90°": [90]
    }

    for noise_corr in ["Exc-Red Noise Correlation", "Red-Red Noise Correlation"]:
        if noise_corr in df_pv.columns:
            for angle_label, angles in angle_groups.items():
                angle_df = df_pv[df_pv["Angular Difference"].isin(angles)][noise_corr].dropna()
                print(f"  🔍 {noise_corr} - {angle_label}: {len(angle_df)} values, Mean = {angle_df.mean():.3f}, SD = {angle_df.std():.3f}")

    print("\n✅ **PV Analysis Complete!**")

def plot_hypothesis(save_dir):
    # Load the dataset
    file_path = os.path.join(save_dir, "orientation_segmented_noise_correlations.csv")
    if not os.path.exists(file_path):
        print(f"⚠ Missing file: {file_path}")
        return
    
    df = pd.read_csv(file_path)

    # 🔹 Filter for SST pairs during Plaid stimuli
    df_sst_exc = df[
        (df["Inhibitory Population"] == "SST") &
        (df["Trial Type"] == "Plaid") &
        ((df["Neuron1 Type"] == "Excitatory") | (df["Neuron2 Type"] == "Excitatory"))
    ]

    df_sst_exc = df_sst_exc[df_sst_exc["Exc-Red Noise Correlation"] <= 0.8]

    if df_sst_exc.empty:
        print("⚠ No valid SST-Excitatory Plaid data found.")
        return
    
    print(f"✔ Extracted {len(df_sst_exc)} SST-Excitatory rows for analysis.")

    # Identify the NI of the excitatory neuron
    df_sst_exc["NI Excitatory"] = df_sst_exc.apply(
        lambda row: row["Normalization Index (NI) Neuron1"] if row["Neuron1 Type"] == "Excitatory"
        else row["Normalization Index (NI) Neuron2"], axis=1
    )

    # Extract noise correlation
    df_sst_exc["Exc-Red Noise Correlation"] = df_sst_exc["Exc-Red Noise Correlation"]

    plt.figure(figsize=(10, 6))
    sns.regplot(
        x="NI Excitatory",
        y="Exc-Red Noise Correlation",
        data=df_sst_exc,
        scatter_kws={"s": 10, "alpha": 0.7},
        line_kws={"color": "red"},
    )
    plt.title("Noise Correlation vs. Normalization Index of Excitatory Neurons (SST, Plaid)")
    plt.xlabel("Normalization Index (NI) of Excitatory Neuron")
    plt.ylabel("Exc-Red Noise Correlation")
    plt.grid(True)
    plt.savefig(f"{save_dir}/scatter_plot_regression.png")
    plt.show()

    # 📊 **KDE plot of noise correlation distribution by NI category**
    bins = [-np.inf, 0, 0.33, np.inf]
    labels = ["NI < 0", "0 ≤ NI ≤ 0.33", "NI > 0.33"]
    df_sst_exc["NI Category"] = pd.cut(df_sst_exc["NI Excitatory"], bins=bins, labels=labels)

    plt.figure(figsize=(10, 6))
    for category in labels:
        subset = df_sst_exc[df_sst_exc["NI Category"] == category]
        sns.kdeplot(subset["Exc-Red Noise Correlation"], label=category, fill=True, alpha=0.5)

    plt.title("KDE of Noise Correlation by Normalization Index Category (SST, Plaid)")
    plt.xlabel("Exc-Red Noise Correlation")
    plt.ylabel("Density")
    plt.legend(title="NI Category")
    plt.grid(True)
    plt.savefig(f"{save_dir}/kde_plot_NI_category.png")
    plt.show()


