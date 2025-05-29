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
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import statsmodels.api as sm
from scipy.stats import f_oneway, ttest_ind, levene, ks_2samp, bartlett
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors



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

color_palette = {
    "Exc-Exc Plaid": mcolors.to_rgba("#99d98c", alpha=0.7),  # Light green
    "Exc": mcolors.to_rgba("#99d98c", alpha=0.7),  # Light green
    "Exc-Exc Grating": mcolors.to_rgba("#386641", alpha=0.9),  # Dark green

    "PV-PV Plaid": mcolors.to_rgba("#ffadad", alpha=0.7),  # Light red
    "PV": mcolors.to_rgba("#ffadad", alpha=0.7),  # Light red
    "PV-PV Grating": mcolors.to_rgba("#b91c1c", alpha=0.9),  # Dark red

    "SST-SST Plaid": mcolors.to_rgba("#a0c4ff", alpha=0.7),  # Light blue
    "SST": mcolors.to_rgba("#a0c4ff", alpha=0.7),  # Light blue
    "SST-SST Grating": mcolors.to_rgba("#1d4ed8", alpha=0.9)  # Dark blue
}

def load_data(mouse, date, base_path='/media/maclean/3d4f021d-2b12-4203-943d-4f1052e2ccf0/1104_hal_data/'):
    proc_dir = os.path.join(base_path, mouse, date, 'proc')
    data = NeuralData(mouse, date, base_name=base_path, use_l='cascade')

    data.resp_start = std_f_before
    data.resp_end = std_f_after

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

#noise corrs, generally segmented by population, thank god thank jesus thank everything 20 hours of debugging later it is finally fixed to be pairwise
def compute_noise_correlations(data, red_labels, vis_resp_cells):
    """
    Compute noise correlations while ensuring only visually responsive neurons are considered.

    Parameters:
        data: NeuralData object for accessing noise correlations.
        red_labels (np.ndarray): Boolean array indicating red-labeled neurons.
        vis_resp_cells (np.ndarray): Boolean array of visually responsive cells.

    Returns:
        dict: Dictionary containing noise correlation results.
    """
    # Step 1: Get full noise correlation matrix
    all_noise_corrs = data.noise_correlations()  # Shape (stimuli, neurons, neurons)
    noise_corrs = np.nanmean(all_noise_corrs, axis=0)  # Shape (neurons, neurons)

    # Step 2: Apply vis_resp_cells filter
    if vis_resp_cells is not None:
        neuron_indices = np.where(vis_resp_cells)[0]  # Convert boolean mask to indices
        noise_corrs = noise_corrs[np.ix_(neuron_indices, neuron_indices)]
        red_labels = red_labels[neuron_indices] if red_labels is not None else None  # Apply filtering
    
    # Step 3: Remove self-correlations
    np.fill_diagonal(noise_corrs, np.nan)  

    # Step 4: Get number of neurons after filtering
    n_neurons = noise_corrs.shape[0]
    results = {}

    if red_labels is not None:
        if len(red_labels) != n_neurons:
            raise ValueError(f"Red labels size ({len(red_labels)}) does not match number of neurons ({n_neurons}).")

        # Step 5: Get red and excitatory neuron indices
        red_indices = np.where(red_labels)[0]  # Indices of red-labeled neurons
        exc_indices = np.where(~red_labels)[0]  # Indices of excitatory neurons

        # Step 6: Red-Red Correlations
        red_red_corrs = np.full((n_neurons, n_neurons), np.nan)  
        if len(red_indices) > 0:
            red_red_corrs[np.ix_(red_indices, red_indices)] = noise_corrs[np.ix_(red_indices, red_indices)]
        results['Red-Red Noise Correlation'] = red_red_corrs.flatten()

        # Step 7: Exc-Red Correlations
        exc_red_corrs = np.full((n_neurons, n_neurons), np.nan)  
        if len(red_indices) > 0:
            exc_red_corrs[np.ix_(exc_indices, red_indices)] = noise_corrs[np.ix_(exc_indices, red_indices)]
        results['Exc-Red Noise Correlation'] = exc_red_corrs.flatten()

        # Step 8: Exc-Exc Correlations
        exc_exc_corrs = np.full((n_neurons, n_neurons), np.nan)  
        if len(exc_indices) > 0:
            exc_exc_corrs[np.ix_(exc_indices, exc_indices)] = noise_corrs[np.ix_(exc_indices, exc_indices)]
        results['Exc-Exc Noise Correlation'] = exc_exc_corrs.flatten()

    else:
        print("No red-labeled neurons found. Treating all neurons as excitatory.")
        results['Red-Red Noise Correlation'] = None
        results['Exc-Red Noise Correlation'] = None
        results['Exc-Exc Noise Correlation'] = np.nanmean(noise_corrs, axis=1)

    return results

# ANGULAR DIFFERENCE noise correlations
def orientation_segmentation(save_dir):
    """Extracts noise correlations for orientation-segmented data, adds angular difference, and excludes rows with missing OSI."""

    trial_results = []

    for mouse, dates in micedates.items():
        for date in dates:
            try:
                data, vis_resp_cells, stim_info, red_labels, orientations, is_plaid, osis = load_data(mouse, date)

                if vis_resp_cells is not None:
                    neuron_indices = np.where(vis_resp_cells)[0]
                else:
                    neuron_indices = np.arange(data.spks_tr.shape[0])
                
                ni_values = data.normalization_indices() if hasattr(data, "normalization_indices") else np.full(len(neuron_indices), np.nan)
                osi_values = osis if osis is not None else np.full(len(neuron_indices), np.nan)

                

                if red_labels is None or np.sum(red_labels) == 0:
                    continue

                # **Classify Mouse as PV or SST**
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

                # **Compute Preferred Orientation for Each Neuron**
                pref_ori = np.full(len(neuron_indices), np.nan)  # Default to NaN
                for idx, neuron in enumerate(neuron_indices):
                    mean_responses = []
                    for ori in np.unique(orientations):
                        ori_trials = orientations == ori
                        mean_responses.append(data.spks_tr[neuron, ori_trials, data.resp_start:data.resp_end].mean())

                    if len(mean_responses) > 0 and not np.all(np.isnan(mean_responses)):
                        pref_ori[idx] = np.unique(orientations)[np.argmax(mean_responses)]

                # **Compute Noise Correlations for Each Trial Type (Plaid, Grating)**
                for trial_type in ["Grating", "Plaid"]:
                    is_plaid_trial = trial_type == "Plaid"
                    trial_indices = stim_info[1] == int(is_plaid_trial)

                    if trial_indices.sum() == 0:
                        print(f"⚠ WARNING: No trials selected for {trial_type} (Mouse {mouse}, Date {date})!")

                    if trial_indices.shape[0] != data.spks_tr.shape[1]:
                        raise ValueError("Mismatch between trial indices and spks_tr dimensions!")

                    filtered_spks_tr = data.spks_tr[:, trial_indices, :]
                    if filtered_spks_tr.shape[1] == 0:
                        continue  # Skip if no trials

                    # Step 1: Compute mean response over the response window
                    reshaped_spks = filtered_spks_tr[:, :, data.resp_start:data.resp_end].mean(axis=2)

                    # Step 2: Apply visual responsiveness filtering before correlation
                    reshaped_spks_filtered = reshaped_spks[neuron_indices, :]  # Fix the neuron mismatch!

                    fano_factors_per_neuron = {}  # Store computed Fano Factors for quick lookup

                    # Dictionary to store per-stimulus Fano Factors before averaging
                    fano_factors_grating = {}
                    fano_factors_plaid = {}

                    for neuron_idx in neuron_indices_fixed:  
                        neuron_spikes = data.spks_tr[neuron_idx, :, :]  # Extract spikes for the neuron
                        
                        # Compute Fano factor separately for each stimulus type
                        unique_stimuli = np.unique(stim_info.T, axis=0)  # Unique combinations of (orientation, plaid status)

                        fano_factors_per_stim = []

                        for stim in unique_stimuli:
                            trial_indices = np.logical_and(
                                stim_info[0] == stim[0],  # Matching orientation
                                stim_info[1] == stim[1]   # Matching plaid/grating status
                            )
                            
                            trial_summed_spikes = neuron_spikes[trial_indices, data.resp_start:data.resp_end].sum(axis=1)

                            mean_spikes = np.mean(trial_summed_spikes)
                            variance_spikes = np.var(trial_summed_spikes)
                            
                            fano_factor = variance_spikes / mean_spikes if mean_spikes > 0 else np.nan
                            fano_factors_per_stim.append(fano_factor)
                        
                        # Compute the overall Fano factor by averaging across stimuli
                        fano_factors_per_neuron[neuron_idx] = np.nanmean(fano_factors_per_stim)

                        # Store separately for grating and plaid conditions
                        fano_grating = [fano_factors_per_stim[i] for i, s in enumerate(unique_stimuli) if s[1] == 0]
                        fano_plaid = [fano_factors_per_stim[i] for i, s in enumerate(unique_stimuli) if s[1] == 1]

                        # Compute averages correctly
                        fano_factors_grating[neuron_idx] = np.nanmean(fano_grating) if len(fano_grating) > 0 else np.nan
                        fano_factors_plaid[neuron_idx] = np.nanmean(fano_plaid) if len(fano_plaid) > 0 else np.nan

                    # Compute overall averages
                    avg_fano_factor_grating = np.nanmean(list(fano_factors_grating.values()))
                    avg_fano_factor_plaid = np.nanmean(list(fano_factors_plaid.values()))

                    # Step 3: Compute noise correlations only for visually responsive neurons
                    noise_corrs = np.corrcoef(reshaped_spks_filtered.reshape(len(neuron_indices), -1))
                    np.fill_diagonal(noise_corrs, np.nan)

                    # **Extract Noise Correlations for Different Pairs**
                    raw_red_red_corrs = noise_corrs[np.ix_(vis_resp_red_indices_fixed, vis_resp_red_indices_fixed)]
                    raw_exc_red_corrs = noise_corrs[np.ix_(vis_resp_exc_indices_fixed, vis_resp_red_indices_fixed)]
                    raw_exc_exc_corrs = noise_corrs[np.ix_(vis_resp_exc_indices_fixed, vis_resp_exc_indices_fixed)]

                    # **Pairwise Comparisons for Neurons**
                    for i, neuron1 in enumerate(neuron_indices_fixed):
                        if np.isnan(pref_ori[i]):
                            continue  # Skip if preferred orientation is undefined

                        for j, neuron2 in enumerate(neuron_indices_fixed):
                            if neuron1 == neuron2 or np.isnan(pref_ori[j]):
                                continue  # Skip self-comparisons or undefined orientations

                            angular_difference = np.mod(abs(pref_ori[i] - pref_ori[j]), 180)

                            pair_result = {
                                "Mouse": mouse,
                                "Date": date,
                                "Trial Type": trial_type,
                                "Neuron1": neuron1,
                                "Neuron2": neuron2,
                                "Angular Difference": angular_difference,
                                "Inhibitory Population": inhibitory_population,
                                "Normalization Index (NI) Neuron1": ni_values[neuron1],
                                "Normalization Index (NI) Neuron2": ni_values[neuron2],
                                "OSI Neuron1": osi_values[neuron1],
                                "OSI Neuron2": osi_values[neuron2]
                            }

                            # **Assign the correct noise correlation column based on neuron pair type**
                            if neuron1 in vis_resp_exc_indices_fixed and neuron2 in vis_resp_exc_indices_fixed:
                                i1, i2 = np.where(vis_resp_exc_indices_fixed == neuron1)[0], np.where(vis_resp_exc_indices_fixed == neuron2)[0]
                                pair_result["Exc-Exc Noise Correlation"] = raw_exc_exc_corrs[i1[0], i2[0]] if i1.size > 0 and i2.size > 0 else np.nan

                            if neuron1 in vis_resp_red_indices_fixed and neuron2 in vis_resp_red_indices_fixed:
                                i1, i2 = np.where(vis_resp_red_indices_fixed == neuron1)[0], np.where(vis_resp_red_indices_fixed == neuron2)[0]
                                pair_result["Red-Red Noise Correlation"] = raw_red_red_corrs[i1[0], i2[0]] if i1.size > 0 and i2.size > 0 else np.nan

                            if (neuron1 in vis_resp_exc_indices_fixed and neuron2 in vis_resp_red_indices_fixed) or \
                               (neuron1 in vis_resp_red_indices_fixed and neuron2 in vis_resp_exc_indices_fixed):
                                i1, i2 = np.where(vis_resp_exc_indices_fixed == (neuron1 if neuron1 in vis_resp_exc_indices_fixed else neuron2))[0], \
                                         np.where(vis_resp_red_indices_fixed == (neuron2 if neuron2 in vis_resp_red_indices_fixed else neuron1))[0]
                                pair_result["Exc-Red Noise Correlation"] = raw_exc_red_corrs[i1[0], i2[0]] if i1.size > 0 and i2.size > 0 else np.nan

                            if trial_type == "Grating":
                                pair_result["Fano Factor Neuron1"] = fano_factors_grating.get(neuron1, np.nan)
                                pair_result["Fano Factor Neuron2"] = fano_factors_grating.get(neuron2, np.nan)
                            elif trial_type == "Plaid":
                                pair_result["Fano Factor Neuron1"] = fano_factors_plaid.get(neuron1, np.nan)
                                pair_result["Fano Factor Neuron2"] = fano_factors_plaid.get(neuron2, np.nan)

                            # **Classify Neuron1 and Neuron2 as Excitatory or Inhibitory**
                            def classify_neuron(neuron):
                                """Determine if a neuron is excitatory or inhibitory based on the types of correlations it appears in."""
                                neuron_corr_types = {
                                    "Exc-Exc": neuron in vis_resp_exc_indices_fixed,
                                    "Exc-Red": neuron in vis_resp_red_indices_fixed and neuron in vis_resp_exc_indices_fixed,
                                    "Red-Red": neuron in vis_resp_red_indices_fixed
                                }

                                if neuron_corr_types["Exc-Exc"] or neuron_corr_types["Exc-Red"] and not neuron_corr_types["Red-Red"]:
                                    return "Excitatory"
                                elif neuron_corr_types["Red-Red"] or neuron_corr_types["Exc-Red"] and not neuron_corr_types["Exc-Exc"]:
                                    return "Inhibitory"
                                else:
                                    return "Unknown"

                            # Assign neuron types
                            pair_result["Neuron1 Type"] = classify_neuron(neuron1)
                            pair_result["Neuron2 Type"] = classify_neuron(neuron2)
                            
                            trial_results.append(pair_result)

                print(f"Processed {len(trial_results)} neuron pairs for Mouse {mouse}, Date {date}.")

            except Exception as e:
                print(f"Error processing Mouse {mouse}, Date {date}: {e}")

    # **Save results to CSV**
    if trial_results:
        results_df = pd.DataFrame(trial_results)
        desired_order = [
            "Mouse", "Date", "Trial Type", "Neuron1", "Neuron2", "Angular Difference", 
            "Inhibitory Population", 
            "Normalization Index (NI) Neuron1", "Normalization Index (NI) Neuron2", 
            "OSI Neuron1", "OSI Neuron2", 
            "Fano Factor Neuron1", "Fano Factor Neuron2",
            "Exc-Exc Noise Correlation", "Exc-Red Noise Correlation", "Red-Red Noise Correlation", "Neuron1 Type", "Neuron2 Type"
        ]

        results_df = results_df[desired_order]  # Reorder columns
        results_path = os.path.join(save_dir, "orientation_segmented_noise_correlations.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Orientation-segmented noise correlations saved to {results_path}.")
    else:
        print("No results to save.")

# ANGULAR DIFFERENCE noise correlations
def orientation_segmentation2(save_dir):
    """Extracts noise correlations for orientation-segmented data, adds angular difference, and excludes rows with missing OSI."""

    trial_results = []

    for mouse, dates in micedates.items():
        for date in dates:
            try:
                data, vis_resp_cells, stim_info, red_labels, orientations, is_plaid, osis = load_data(mouse, date)

                if vis_resp_cells is not None:
                    neuron_indices = np.where(vis_resp_cells)[0]
                else:
                    neuron_indices = np.arange(data.spks_tr.shape[0])
                
                ni_values = data.normalization_indices() if hasattr(data, "normalization_indices") else np.full(len(neuron_indices), np.nan)
                osi_values = osis if osis is not None else np.full(len(neuron_indices), np.nan)

                

                if red_labels is None or np.sum(red_labels) == 0:
                    continue

                # **Classify Mouse as PV or SST**
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

                # **Compute Preferred Orientation for Each Neuron**
                pref_ori = np.full(len(neuron_indices), np.nan)  # Default to NaN
                for idx, neuron in enumerate(neuron_indices):
                    mean_responses = []
                    for ori in np.unique(orientations):
                        ori_trials = orientations == ori
                        mean_responses.append(data.spks_tr[neuron, ori_trials, data.resp_start:data.resp_end].mean())

                    if len(mean_responses) > 0 and not np.all(np.isnan(mean_responses)):
                        pref_ori[idx] = np.unique(orientations)[np.argmax(mean_responses)]

                # **Compute Noise Correlations for Each Trial Type (Plaid, Grating)**
                for trial_type in ["Grating", "Plaid"]:
                    is_plaid_trial = trial_type == "Plaid"
                    trial_indices = stim_info[1] == int(is_plaid_trial)

                    if trial_indices.sum() == 0:
                        print(f"⚠ WARNING: No trials selected for {trial_type} (Mouse {mouse}, Date {date})!")

                    if trial_indices.shape[0] != data.spks_tr.shape[1]:
                        raise ValueError("Mismatch between trial indices and spks_tr dimensions!")

                    filtered_spks_tr = data.spks_tr[:, trial_indices, :]
                    if filtered_spks_tr.shape[1] == 0:
                        continue  # Skip if no trials

                    # Step 1: Compute mean response over the response window
                    reshaped_spks = filtered_spks_tr[:, :, data.resp_start:data.resp_end].mean(axis=2)

                    # Step 2: Apply visual responsiveness filtering before correlation
                    reshaped_spks_filtered = reshaped_spks[neuron_indices, :]  # Fix the neuron mismatch!

                    fano_factors_per_neuron = {}  # Store computed Fano Factors for quick lookup

                    # Dictionary to store per-stimulus Fano Factors before averaging
                    fano_factors_grating = {}
                    fano_factors_plaid = {}

                    for neuron_idx in neuron_indices_fixed:  
                        neuron_spikes = data.spks_tr[neuron_idx, :, :]  # Extract spikes for the neuron
                        
                        # Compute Fano factor separately for each stimulus type
                        unique_stimuli = np.unique(stim_info.T, axis=0)  # Unique combinations of (orientation, plaid status)

                        fano_factors_per_stim = []

                        for stim in unique_stimuli:
                            trial_indices = np.logical_and(
                                stim_info[0] == stim[0],  # Matching orientation
                                stim_info[1] == stim[1]   # Matching plaid/grating status
                            )
                            
                            trial_summed_spikes = neuron_spikes[trial_indices, data.resp_start:data.resp_end].sum(axis=1)

                            mean_spikes = np.mean(trial_summed_spikes)
                            variance_spikes = np.var(trial_summed_spikes)
                            
                            fano_factor = variance_spikes / mean_spikes if mean_spikes > 0 else np.nan
                            fano_factors_per_stim.append(fano_factor)
                        
                        # Compute the overall Fano factor by averaging across stimuli
                        fano_factors_per_neuron[neuron_idx] = np.nanmean(fano_factors_per_stim)

                        # Store separately for grating and plaid conditions
                        fano_grating = [fano_factors_per_stim[i] for i, s in enumerate(unique_stimuli) if s[1] == 0]
                        fano_plaid = [fano_factors_per_stim[i] for i, s in enumerate(unique_stimuli) if s[1] == 1]

                        # Compute averages correctly
                        fano_factors_grating[neuron_idx] = np.nanmean(fano_grating) if len(fano_grating) > 0 else np.nan
                        fano_factors_plaid[neuron_idx] = np.nanmean(fano_plaid) if len(fano_plaid) > 0 else np.nan

                    # Compute overall averages
                    avg_fano_factor_grating = np.nanmean(list(fano_factors_grating.values()))
                    avg_fano_factor_plaid = np.nanmean(list(fano_factors_plaid.values()))

                    # Step 3: Compute noise correlations only for visually responsive neurons
                    noise_corrs = np.corrcoef(reshaped_spks_filtered.reshape(len(neuron_indices), -1))
                    np.fill_diagonal(noise_corrs, np.nan)

                    # **Extract Noise Correlations for Different Pairs**
                    raw_red_red_corrs = noise_corrs[np.ix_(vis_resp_red_indices_fixed, vis_resp_red_indices_fixed)]
                    raw_exc_red_corrs = noise_corrs[np.ix_(vis_resp_exc_indices_fixed, vis_resp_red_indices_fixed)]
                    raw_exc_exc_corrs = noise_corrs[np.ix_(vis_resp_exc_indices_fixed, vis_resp_exc_indices_fixed)]

                    # **Pairwise Comparisons for Neurons**
                    for i, neuron1 in enumerate(neuron_indices_fixed):
                        if np.isnan(pref_ori[i]):
                            continue  # Skip if preferred orientation is undefined

                        for j, neuron2 in enumerate(neuron_indices_fixed):
                            if neuron1 == neuron2 or np.isnan(pref_ori[j]):
                                continue  # Skip self-comparisons or undefined orientations

                            angular_difference = np.mod(abs(pref_ori[i] - pref_ori[j]), 180)

                            pair_result = {
                                "Mouse": mouse,
                                "Date": date,
                                "Trial Type": trial_type,
                                "Neuron1": neuron1,
                                "Neuron2": neuron2,
                                "Angular Difference": angular_difference,
                                "Inhibitory Population": inhibitory_population,
                                "Normalization Index (NI) Neuron1": ni_values[neuron1],
                                "Normalization Index (NI) Neuron2": ni_values[neuron2],
                                "OSI Neuron1": osi_values[neuron1],
                                "OSI Neuron2": osi_values[neuron2]
                            }

                            # **Assign the correct noise correlation column based on neuron pair type**
                            if neuron1 in vis_resp_exc_indices_fixed and neuron2 in vis_resp_exc_indices_fixed:
                                i1, i2 = np.where(vis_resp_exc_indices_fixed == neuron1)[0], np.where(vis_resp_exc_indices_fixed == neuron2)[0]
                                pair_result["Exc-Exc Noise Correlation"] = raw_exc_exc_corrs[i1[0], i2[0]] if i1.size > 0 and i2.size > 0 else np.nan

                            if neuron1 in vis_resp_red_indices_fixed and neuron2 in vis_resp_red_indices_fixed:
                                i1, i2 = np.where(vis_resp_red_indices_fixed == neuron1)[0], np.where(vis_resp_red_indices_fixed == neuron2)[0]
                                pair_result["Red-Red Noise Correlation"] = raw_red_red_corrs[i1[0], i2[0]] if i1.size > 0 and i2.size > 0 else np.nan

                            if (neuron1 in vis_resp_exc_indices_fixed and neuron2 in vis_resp_red_indices_fixed) or \
                               (neuron1 in vis_resp_red_indices_fixed and neuron2 in vis_resp_exc_indices_fixed):
                                i1, i2 = np.where(vis_resp_exc_indices_fixed == (neuron1 if neuron1 in vis_resp_exc_indices_fixed else neuron2))[0], \
                                         np.where(vis_resp_red_indices_fixed == (neuron2 if neuron2 in vis_resp_red_indices_fixed else neuron1))[0]
                                pair_result["Exc-Red Noise Correlation"] = raw_exc_red_corrs[i1[0], i2[0]] if i1.size > 0 and i2.size > 0 else np.nan

                            if trial_type == "Grating":
                                pair_result["Fano Factor Neuron1"] = fano_factors_grating.get(neuron1, np.nan)
                                pair_result["Fano Factor Neuron2"] = fano_factors_grating.get(neuron2, np.nan)
                            elif trial_type == "Plaid":
                                pair_result["Fano Factor Neuron1"] = fano_factors_plaid.get(neuron1, np.nan)
                                pair_result["Fano Factor Neuron2"] = fano_factors_plaid.get(neuron2, np.nan)

                            # **Classify Neuron1 and Neuron2 as Excitatory or Inhibitory**
                            def classify_neuron(neuron):
                                """Determine if a neuron is excitatory or inhibitory based on the types of correlations it appears in."""
                                neuron_corr_types = {
                                    "Exc-Exc": neuron in vis_resp_exc_indices_fixed,
                                    "Exc-Red": neuron in vis_resp_red_indices_fixed and neuron in vis_resp_exc_indices_fixed,
                                    "Red-Red": neuron in vis_resp_red_indices_fixed
                                }

                                if neuron_corr_types["Exc-Exc"] or neuron_corr_types["Exc-Red"] and not neuron_corr_types["Red-Red"]:
                                    return "Excitatory"
                                elif neuron_corr_types["Red-Red"] or neuron_corr_types["Exc-Red"] and not neuron_corr_types["Exc-Exc"]:
                                    return "Inhibitory"
                                else:
                                    return "Unknown"

                            # Assign neuron types
                            pair_result["Neuron1 Type"] = classify_neuron(neuron1)
                            pair_result["Neuron2 Type"] = classify_neuron(neuron2)
                            
                            trial_results.append(pair_result)

                print(f"Processed {len(trial_results)} neuron pairs for Mouse {mouse}, Date {date}.")

            except Exception as e:
                print(f"Error processing Mouse {mouse}, Date {date}: {e}")

    # **Save results to CSV**
    if trial_results:
        results_df = pd.DataFrame(trial_results)
        desired_order = [
            "Mouse", "Date", "Trial Type", "Neuron1", "Neuron2", "Angular Difference", 
            "Inhibitory Population", 
            "Normalization Index (NI) Neuron1", "Normalization Index (NI) Neuron2", 
            "OSI Neuron1", "OSI Neuron2", 
            "Fano Factor Neuron1", "Fano Factor Neuron2",
            "Exc-Exc Noise Correlation", "Exc-Red Noise Correlation", "Red-Red Noise Correlation", "Neuron1 Type", "Neuron2 Type"
        ]

        results_df = results_df[desired_order]  # Reorder columns
        results_path = os.path.join(save_dir, "orientation_segmented_noise_correlations2.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Orientation-segmented noise correlations saved to {results_path}.")
    else:
        print("No results to save.")





#stat test setup (pre-analysis)
def cohen_d(group1, group2):
    """Calculate Cohen's d for effect size."""
    mean_diff = np.abs(np.mean(group1) - np.mean(group2))
    pooled_std = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)
    return mean_diff / pooled_std if pooled_std > 0 else np.nan
def regression_analysis(x, y, description):
    """Perform Pearson/Spearman correlation and linear regression."""
    if len(x) > 2 and len(y) > 2:
        pearson_corr, p_pearson = pearsonr(x, y)
        spearman_corr, p_spearman = spearmanr(x, y)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        print(f"\n📈 Regression: {description}")
        print(f"  Pearson r = {pearson_corr:.3f}, p = {p_pearson:.3e}")
        print(f"  Spearman ρ = {spearman_corr:.3f}, p = {p_spearman:.3e}")
        print(f"  Linear Fit: y = {slope:.3f}x + {intercept:.3f}, R² = {r_value**2:.3f}, p = {p_value:.3e}")
    else:
        print(f"⚠ Not enough data for regression in {description}")
def run_stat_tests(group1, group2, description):
    """Run statistical comparisons, including effect size and power analysis."""
    if not group1.empty and not group2.empty:
        print(f"\n🔹 {description}")
        print(f"  Group 1 Mean: {group1.mean():.3f}, SD: {group1.std():.3f}, n={len(group1)}")
        print(f"  Group 2 Mean: {group2.mean():.3f}, SD: {group2.std():.3f}, n={len(group2)}")

        # Statistical Tests
        stat_ranksum, p_ranksum = ranksums(group1, group2)
        stat_mann, p_mann = mannwhitneyu(group1, group2, alternative="two-sided")
        stat_ks, p_ks = ks_2samp(group1, group2)

        print(f"  Ranksum Test: Stat={stat_ranksum:.3f}, p={p_ranksum:.3e}")
        print(f"  Mann-Whitney U: Stat={stat_mann:.3f}, p={p_mann:.3e}")
        print(f"  Kolmogorov-Smirnov (K-S) Test: Stat={stat_ks:.3f}, p={p_ks:.3e}")

        # Effect Size
        effect_size = cohen_d(group1, group2)
        print(f"  Effect Size (Cohen's d): {effect_size:.3f}")

        # Power Analysis
        power_analysis = TTestIndPower()
        power = power_analysis.solve_power(effect_size, nobs1=len(group1), ratio=len(group2)/len(group1), alpha=0.05)
        print(f"  Power Analysis: {power:.3f} (1-β)")

    else:
        print(f"  ⚠ Not enough data for {description}")


#stat tests/analysis summary
def summarize_all_results(save_dir):
    """Runs all statistical comparisons and plots side-by-side."""

    # Load CSVs
    files = {
        "Pairwise": "noise_correlations_pairwise.csv",
        "Trial-Specific": "trial_specific_noise_correlations.csv",
        "Orientation-Segmented": "orientation_segmented_noise_correlations.csv",
        "Orthogonal": "orthogonal_noise_correlation_analysis.csv",
        "Red Cell Orientation": "red_cell_orientation_segmented_noise_correlations.csv",
    }

    dataframes = {}
    for name, file in files.items():
        path = os.path.join(save_dir, file)
        if os.path.exists(path):
            dataframes[name] = pd.read_csv(path)
            print(f"✔ Loaded {name} - {len(dataframes[name])} rows")
        else:
            print(f"⚠ Missing file: {file}")

    print("\n--- 📊 Running Statistical Comparisons ---")

    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    def save_plot(fig, filename):
        """Save figure to disk."""
        filepath = os.path.join(plots_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_kde_multi(data_groups, labels, title, filename, colors):
        """Plot KDE for multiple groups with different color themes."""
        fig, ax = plt.subplots(figsize=(7,5))
        for data, label, color in zip(data_groups, labels, colors):
            sns.kdeplot(data, label=label, fill=True, alpha=0.6, color=color, bw_adjust=0.5)
        plt.title(title)
        plt.xlabel("Noise Correlation")
        plt.ylabel("Density")
        plt.legend()
        save_plot(fig, filename)

    def plot_violin(data, labels, title, filename, colors):
        """Plot and save violin plot with custom colors and outlier removal using IQR."""
        
        def remove_outliers(series):
            """Remove outliers using the IQR method."""
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return series[(series >= lower_bound) & (series <= upper_bound)]

        # Filter each dataset to remove outliers and exclude values equal to 1.0
        filtered_data = [remove_outliers(d[d != 1.0]) for d in data]  

        fig, ax = plt.subplots(figsize=(8,6))
        sns.violinplot(data=filtered_data, palette=colors)
        plt.xticks(ticks=range(len(labels)), labels=labels, rotation=30)
        plt.title(title)
        plt.ylabel("Noise Correlation")
        save_plot(fig, filename)
    
    def plot_bar_with_sem(data, labels, title, filename, colors):
        """Plot and save bar plot with means and SEM error bars, removing outliers using IQR."""
        
        def remove_outliers(series):
            """Remove outliers using the IQR method."""
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return series[(series >= lower_bound) & (series <= upper_bound)]

        # Filter each dataset to remove outliers and exclude values equal to 1.0
        filtered_data = [remove_outliers(d[d != 1.0]) for d in data]  

        # Compute means and SEMs
        means = [d.mean() for d in filtered_data]
        sems = [d.std() / np.sqrt(len(d)) for d in filtered_data]  # SEM = std / sqrt(n)

        # Create the bar plot
        fig, ax = plt.subplots(figsize=(8,6))
        ax.bar(labels, means, yerr=sems, capsize=5, color=colors, alpha=0.7, edgecolor='black')

        # Format plot
        plt.xticks(rotation=30)
        plt.title(title)
        plt.ylabel("Noise Correlation")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        
        # Save the figure
        save_plot(fig, filename)
    
    def plot_grouped_bar(data, column, group_labels, title, filename, colors):
        """
        Plots grouped bar plots for different conditions (e.g., PV vs SST, Plaid vs Grating),
        grouped by Angular Difference.

        Parameters:
            - data: DataFrame containing the values
            - column: Name of the column to plot
            - group_labels: List of tuples (filter1, filter2, label)
            - title: Title of the plot
            - filename: Output file name
            - colors: List of colors for the bars
        """
        means, sems = [], []

        for angles in angle_groups.values():
            angle_means, angle_sems = [], []
            
            for filter1, filter2, _ in group_labels:
                filtered_df = data[data["Angular Difference"].isin(angles)]
                if filter1:
                    filtered_df = filtered_df[filtered_df["Inhibitory Population"] == filter1]
                if filter2:
                    filtered_df = filtered_df[filtered_df["Trial Type"] == filter2]

                # Get mean & SEM for the selected group
                values = filtered_df[column].dropna()
                angle_means.append(values.mean())
                angle_sems.append(values.sem())

            means.append(angle_means)
            sems.append(angle_sems)

        # Convert to numpy for easy plotting
        means, sems = np.array(means), np.array(sems)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(angle_groups))  # X-axis positions
        width = 0.3  # Bar width

        for i, (label, color) in enumerate(zip([g[2] for g in group_labels], colors)):
            ax.bar(x + (i - 0.5) * width, means[:, i], width, yerr=sems[:, i], capsize=5, label=label, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(angle_groups.keys())
        ax.set_xlabel("Angular Difference")
        ax.set_ylabel(column)
        ax.set_title(title)
        ax.legend()
        save_plot(fig, filename)
    
    """
    # **1️⃣ Pairwise Noise Correlations (Main Stats + KDEs + Violin)**
    if "Pairwise" in dataframes:
        df = dataframes["Pairwise"]

        # **Extract Groups**
        groups = {
            "Exc-Exc PV": df.loc[df["Inhibitory Population"] == "PV", "Exc-Exc Noise Correlation"].dropna(),
            "Exc-Red PV": df.loc[df["Inhibitory Population"] == "PV", "Exc-Red Noise Correlation"].dropna(),
            "Red-Red PV": df.loc[df["Inhibitory Population"] == "PV", "Red-Red Noise Correlation"].dropna(),
            "Exc-Exc SST": df.loc[df["Inhibitory Population"] == "SST", "Exc-Exc Noise Correlation"].dropna(),
            "Exc-Red SST": df.loc[df["Inhibitory Population"] == "SST", "Exc-Red Noise Correlation"].dropna(),
            "Red-Red SST": df.loc[df["Inhibitory Population"] == "SST", "Red-Red Noise Correlation"].dropna(),
        }

        plot_kde_multi(
            [groups["Exc-Exc PV"], groups["Exc-Red PV"], groups["Red-Red PV"]],
            ["Exc-Exc PV", "Exc-Red PV", "Red-Red PV"],
            "KDE: PV Groups",
            "kde_pv_groups.png",
            ["#0D47A1", "#1976D2", "#64B5F6"]
        )

        plot_kde_multi(
            [groups["Exc-Exc SST"], groups["Exc-Red SST"], groups["Red-Red SST"]],
            ["Exc-Exc SST", "Exc-Red SST", "Red-Red SST"],
            "KDE: SST Groups",
            "kde_sst_groups.png",
            ["#B71C1C", "#E53935", "#EF9A9A"]
        )

        plot_kde_multi(
            [groups["Exc-Red PV"], groups["Exc-Red SST"], groups["Red-Red PV"], groups["Red-Red SST"]],
            ["Exc-Red PV", "Exc-Red SST", "Red-Red PV", "Red-Red SST"],
            "KDE: Exc-Red & Red-Red Groups",
            "kde_exc_red_red_red_groups.png",
            ["#1976D2", "#E53935", "#64B5F6", "#EF9A9A"]
        )

        # **Violin Plots for PV and SST**
        plot_violin(
            [groups["Exc-Exc PV"], groups["Exc-Red PV"], groups["Red-Red PV"]],
            ["Exc-Exc PV", "Exc-Red PV", "Red-Red PV"],
            "Violin: PV Groups",
            "violin_pv_groups.png",
            ["#66BB6A", "#42A5F5", "#EF5350"]
        )

        plot_violin(
            [groups["Exc-Exc SST"], groups["Exc-Red SST"], groups["Red-Red SST"]],
            ["Exc-Exc SST", "Exc-Red SST", "Red-Red SST"],
            "Violin: SST Groups",
            "violin_sst_groups.png",
            ["#66BB6A", "#42A5F5", "#EF5350"]
        )

        # **Print Means, SDs, and Sample Sizes**
        print("\n📊 **Pairwise Noise Correlation Summary:**")
        for group_name, data in groups.items():
            print(f"  {group_name} | Mean: {data.mean():.3f}, SD: {data.std():.3f}, n={len(data)}")

        # **Statistical Tests for Pairwise Comparisons**
        print("\n📊 **Statistical Tests for Pairwise Comparisons:**")
        for g1 in groups:
            for g2 in groups:
                if g1 != g2:
                    run_stat_tests(groups[g1], groups[g2], f"{g1} vs {g2}")

    # **2️⃣ Trial-Specific Noise Correlations (Plaid & Grating Comparisons)**
    if "Trial-Specific" in dataframes:
        df = dataframes["Trial-Specific"]

        # **Extract Plaid & Grating Groups**
        plaid_df = df[df["Trial Type"] == "Plaid"]
        grating_df = df[df["Trial Type"] == "Grating"]

        plaid_groups = {
            "Exc-Exc PV Plaid": plaid_df.loc[plaid_df["Inhibitory Population"] == "PV", "Exc-Exc Noise Correlation"].dropna(),
            "Exc-Red PV Plaid": plaid_df.loc[plaid_df["Inhibitory Population"] == "PV", "Exc-Red Noise Correlation"].dropna(),
            "Red-Red PV Plaid": plaid_df.loc[plaid_df["Inhibitory Population"] == "PV", "Red-Red Noise Correlation"].dropna(),
            "Exc-Exc SST Plaid": plaid_df.loc[plaid_df["Inhibitory Population"] == "SST", "Exc-Exc Noise Correlation"].dropna(),
            "Exc-Red SST Plaid": plaid_df.loc[plaid_df["Inhibitory Population"] == "SST", "Exc-Red Noise Correlation"].dropna(),
            "Red-Red SST Plaid": plaid_df.loc[plaid_df["Inhibitory Population"] == "SST", "Red-Red Noise Correlation"].dropna(),
        }

        grating_groups = {
            "Exc-Exc PV Grating": grating_df.loc[grating_df["Inhibitory Population"] == "PV", "Exc-Exc Noise Correlation"].dropna(),
            "Exc-Red PV Grating": grating_df.loc[grating_df["Inhibitory Population"] == "PV", "Exc-Red Noise Correlation"].dropna(),
            "Red-Red PV Grating": grating_df.loc[grating_df["Inhibitory Population"] == "PV", "Red-Red Noise Correlation"].dropna(),
            "Exc-Exc SST Grating": grating_df.loc[grating_df["Inhibitory Population"] == "SST", "Exc-Exc Noise Correlation"].dropna(),
            "Exc-Red SST Grating": grating_df.loc[grating_df["Inhibitory Population"] == "SST", "Exc-Red Noise Correlation"].dropna(),
            "Red-Red SST Grating": grating_df.loc[grating_df["Inhibitory Population"] == "SST", "Red-Red Noise Correlation"].dropna(),
        }

        # **Print Means, SDs, and Sample Sizes for Plaid & Grating**
        print("\n📊 **Plaid Noise Correlation Summary:**")
        for group_name, data in plaid_groups.items():
            print(f"  {group_name} | Mean: {data.mean():.3f}, SD: {data.std():.3f}, n={len(data)}")

        print("\n📊 **Grating Noise Correlation Summary:**")
        for group_name, data in grating_groups.items():
            print(f"  {group_name} | Mean: {data.mean():.3f}, SD: {data.std():.3f}, n={len(data)}")

        # **Statistical Tests (Within Plaid, Within Grating, Plaid vs Grating)**
        print("\n📊 **Statistical Tests for Plaid:**")
        for g1 in plaid_groups:
            for g2 in plaid_groups:
                if g1 != g2:
                    run_stat_tests(plaid_groups[g1], plaid_groups[g2], f"{g1} vs {g2}")

        print("\n📊 **Statistical Tests for Grating:**")
        for g1 in grating_groups:
            for g2 in grating_groups:
                if g1 != g2:
                    run_stat_tests(grating_groups[g1], grating_groups[g2], f"{g1} vs {g2}")

        print("\n📊 **Statistical Tests for Plaid vs Grating (Same Groups):**")
        for key in plaid_groups.keys():
            run_stat_tests(plaid_groups[key], grating_groups[key.replace("Plaid", "Grating")], f"{key} vs {key.replace('Plaid', 'Grating')}")

        # **Plot KDEs for Plaid and Grating separately**
        plot_kde_multi(
            [plaid_groups["Exc-Exc PV Plaid"], plaid_groups["Exc-Red PV Plaid"], plaid_groups["Red-Red PV Plaid"]],
            ["Exc-Exc PV", "Exc-Red PV", "Red-Red PV"],
            "KDE: PV Groups (Plaid)",
            "kde_pv_groups_plaid.png",
            ["#0D47A1", "#1976D2", "#64B5F6"]
        )

        plot_kde_multi(
            [grating_groups["Exc-Exc PV Grating"], grating_groups["Exc-Red PV Grating"], grating_groups["Red-Red PV Grating"]],
            ["Exc-Exc PV", "Exc-Red PV", "Red-Red PV"],
            "KDE: PV Groups (Grating)",
            "kde_pv_groups_grating.png",
            ["#0D47A1", "#1976D2", "#64B5F6"]
        )

        plot_kde_multi(
            [plaid_groups["Exc-Exc SST Plaid"], plaid_groups["Exc-Red SST Plaid"], plaid_groups["Red-Red SST Plaid"]],
            ["Exc-Exc SST", "Exc-Red SST", "Red-Red SST"],
            "KDE: SST Groups (Plaid)",
            "kde_sst_groups_plaid.png",
            ["#B71C1C", "#E53935", "#EF9A9A"]
        )

        plot_kde_multi(
            [grating_groups["Exc-Exc SST Grating"], grating_groups["Exc-Red SST Grating"], grating_groups["Red-Red SST Grating"]],
            ["Exc-Exc SST", "Exc-Red SST", "Red-Red SST"],
            "KDE: SST Groups (Grating)",
            "kde_sst_groups_grating.png",
            ["#B71C1C", "#E53935", "#EF9A9A"]
        )

        # **Plot KDEs Comparing Plaid vs Grating**
        for key in plaid_groups.keys():
            plot_kde_multi(
                [plaid_groups[key], grating_groups[key.replace("Plaid", "Grating")]],
                [key, key.replace("Plaid", "Grating")],
                f"KDE: {key} vs {key.replace('Plaid', 'Grating')}",
                f"kde_{key}_vs_{key.replace('Plaid', 'Grating')}.png",
                ["#007ACC", "#FF5733"]
            )

        # **Violin Plots**
        plot_bar_with_sem(
            list(plaid_groups.values()), list(plaid_groups.keys()),
            "Bar Plot: Plaid Groups", "bar_plaid_groups.png",
            ["#66BB6A", "#42A5F5", "#EF5350"]
        )

        plot_bar_with_sem(
            list(grating_groups.values()), list(grating_groups.keys()),
            "Bar Plot: Grating Groups", "bar_grating_groups.png",
            ["#66BB6A", "#42A5F5", "#EF5350"]
        )

        plot_bar_with_sem(
            [plaid_groups[key] for key in plaid_groups] + [grating_groups[key.replace("Plaid", "Grating")] for key in plaid_groups],
            list(plaid_groups.keys()) + [key.replace("Plaid", "Grating") for key in plaid_groups],
            "Bar Plot: Plaid vs Grating", "bar_plaid_vs_grating.png",
            ["#007ACC", "#FF5733"] * len(plaid_groups)
)
    """

#multivariate + ridge analysis
def noise_corr_analysis(save_dir):
    """Comprehensive multivariate regression and population-level analysis of noise correlations."""

    # **1️⃣ Load Data**
    file_path = os.path.join(save_dir, "orientation_segmented_noise_correlations.csv")
    if not os.path.exists(file_path):
        print(f"⚠ Missing file: {file_path}")
        return
    
    df = pd.read_csv(file_path)

    # **2️⃣ Subset Data for Key Angular Differences**
    df_0 = df[df["Angular Difference"] == 0]  # Co-tuned neurons
    df_90 = df[df["Angular Difference"] == 90]  # Orthogonal neurons

    for subset, name in zip([df_0, df_90], ["0°", "90°"]):
        print(f"\n🔍 Processing {name} neuron pairs ({len(subset)} rows)")

        # Compute average & difference for OSI and NI
        subset.loc[:, "OSI Average"] = subset[["OSI Neuron1", "OSI Neuron2"]].mean(axis=1)
        subset.loc[:, "OSI Difference"] = abs(subset["OSI Neuron1"] - subset["OSI Neuron2"])
        subset.loc[:, "NI Average"] = subset[["Normalization Index (NI) Neuron1", "Normalization Index (NI) Neuron2"]].mean(axis=1)
        subset.loc[:, "NI Difference"] = abs(subset["Normalization Index (NI) Neuron1"] - subset["Normalization Index (NI) Neuron2"])
        subset.loc[:, "OSI × NI"] = subset["OSI Average"] * subset["NI Average"]  # Interaction term
        subset.loc[:, "Fano Factor Average"] = subset[["Fano Factor Neuron1", "Fano Factor Neuron2"]].mean(axis=1)
        subset.loc[:, "Fano Factor Difference"] = abs(subset["Fano Factor Neuron1"] - subset["Fano Factor Neuron2"])
        subset.loc[:, "OSI × Fano"] = subset["OSI Average"] * subset["Fano Factor Average"]
        subset.loc[:, "NI × Fano"] = subset["NI Average"] * subset["Fano Factor Average"]

    # **Store these features in the main DataFrame**
    df = pd.concat([df_0, df_90])

    # **4️⃣ Regression Analysis & Population Segmentation**
    population_types = {
        "Exc-Exc Noise Correlation": "Exc-Exc",
        "Exc-Red Noise Correlation (PV)": df[(df["Inhibitory Population"] == "PV")],
        "Exc-Red Noise Correlation (SST)": df[(df["Inhibitory Population"] == "SST")],
        "Red-Red Noise Correlation (PV)": df[(df["Inhibitory Population"] == "PV")],
        "Red-Red Noise Correlation (SST)": df[(df["Inhibitory Population"] == "SST")],
    }

    for correlation_type, population_filter in population_types.items():
        for subset, name in zip([df_0, df_90], ["0°", "90°"]):
            for trial_type in ["Grating", "Plaid"]:
                trial_subset = subset[subset["Trial Type"] == trial_type]

                if isinstance(population_filter, pd.DataFrame):
                    trial_subset = trial_subset[trial_subset.index.isin(population_filter.index)]  # Apply filter for Exc-Red & Red-Red

                print(f"\n📊 Regression for {correlation_type} - {name} - {trial_type}")

                # Define predictors
                X = trial_subset[[
                    "OSI Average", "OSI Difference", "NI Average", "NI Difference", 
                    "Fano Factor Average", "Fano Factor Difference", "OSI × NI", 
                    "OSI × Fano", "NI × Fano"
                ]]
                X = sm.add_constant(X)  # Add intercept
                y = trial_subset[correlation_type.split(" (")[0]]  # Get the base correlation type

                # Remove NaNs
                valid_data = X.notnull().all(axis=1) & y.notnull()
                X, y = X[valid_data], y[valid_data]

                if len(X) > 10:
                    model = sm.OLS(y, X).fit()
                    print(model.summary())  # Print regression results
                else:
                    print("⚠ Not enough data for regression.")


    # **5️⃣ Compare Highest Noise Correlation Pairs (0° Grating)**
    df_0_grating = df_0[df_0["Trial Type"] == "Grating"]

    for correlation_type in population_types.keys():
        base_type = correlation_type.split(" (")[0]
        top_pairs = df_0_grating.nlargest(10, base_type)
        print(f"\n🔥 Top 10 {correlation_type} pairs for 0° Grating:")
        print(top_pairs[["Mouse", "Date", "Neuron1", "Neuron2", "OSI Average", "NI Average", base_type]])

    # **6️⃣ Statistical Testing: Compare Variance in Different Conditions**
    print("\n📊 Comparing Variance of Noise Correlations in Different Conditions")

    for correlation_type in population_types.keys():
        base_type = correlation_type.split(" (")[0]
        for trial_type in ["Grating", "Plaid"]:
            subset = df[df["Trial Type"] == trial_type]
            subset_var = subset[base_type].var()
            print(f"📈 Variance of {correlation_type} ({trial_type}): {subset_var:.4f}")

    print("\n✅ Comprehensive Analysis Complete!")

    def plot_pdp(df, save_dir, correlation_type):
        """Plot Partial Dependence of OSI and NI on Noise Correlation, split by PV and SST."""

        inhibitory_types = ["PV", "SST"]
        
        for inh in inhibitory_types:
            df_subset = df[df["Inhibitory Population"] == inh]
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f"Partial Dependence of OSI and NI on {correlation_type} ({inh})")
            
            # OSI Average PDP
            sns.lineplot(x=df_subset["OSI Average"], y=df_subset[correlation_type], ax=axes[0])
            axes[0].set_xlabel("OSI Average")
            axes[0].set_ylabel("Partial dependence")
            
            # NI Average PDP
            sns.lineplot(x=df_subset["NI Average"], y=df_subset[correlation_type], ax=axes[1])
            axes[1].set_xlabel("NI Average")
            axes[1].set_ylabel("Partial dependence")

            plt.savefig(os.path.join(save_dir, f"pdp_{correlation_type}_{inh}.png"))
            plt.close()
    # Run PDP for each noise correlation type
    for corr_type in ["Red-Red Noise Correlation", "Exc-Red Noise Correlation", "Exc-Exc Noise Correlation"]:
        plot_pdp(df, save_dir, corr_type)
    def plot_heatmap(df, save_dir, correlation_type):
        """Plot Heatmap of OSI and NI Effects on Noise Correlation, split by PV and SST."""

        inhibitory_types = ["PV", "SST"]

        for inh in inhibitory_types:
            df_subset = df[df["Inhibitory Population"] == inh]
            
            # Bin OSI and NI
            df_subset["OSI Bin"] = pd.qcut(df_subset["OSI Average"], 4, labels=["Low", "Medium", "High", "Very High"])
            df_subset["NI Bin"] = pd.qcut(df_subset["NI Average"], 4, labels=["Low", "Medium", "High", "Very High"])
            df_subset["Fano Bin"] = pd.qcut(df_subset["Fano Factor Average"], 4, labels=["Low", "Medium", "High", "Very High"])
            heatmap_data = df_subset.pivot_table(index="Fano Bin", columns="OSI Bin", values=correlation_type, aggfunc="mean")
            # Pivot table for heatmap

            plt.figure(figsize=(8, 6))
            sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", cbar=True)
            plt.title(f"Effect of OSI & NI on {correlation_type} ({inh})")
            plt.savefig(os.path.join(save_dir, f"heatmap_{correlation_type}_{inh}.png"))
            plt.close()
    # Run Heatmap for each noise correlation type
    for corr_type in ["Red-Red Noise Correlation", "Exc-Red Noise Correlation", "Exc-Exc Noise Correlation"]:
        plot_heatmap(df, save_dir, corr_type)
    def plot_regression_coefficients(df, save_dir):
        """Plot Regression Coefficients for Noise Correlations, split by PV and SST."""

        inhibitory_types = ["PV", "SST"]
        
        for inh in inhibitory_types:
            coefficient_data = []
            
            for correlation_type in ["Exc-Exc Noise Correlation", "Exc-Red Noise Correlation", "Red-Red Noise Correlation"]:
                df_subset = df[df["Inhibitory Population"] == inh]

                # Define predictors
                X = df_subset[[
                    "OSI Average", "OSI Difference", "NI Average", "NI Difference", 
                    "Fano Factor Average", "Fano Factor Difference", "OSI × NI", 
                    "OSI × Fano", "NI × Fano"
                ]]
                X = sm.add_constant(X)
                y = df_subset[correlation_type]

                valid_data = X.notnull().all(axis=1) & y.notnull()
                X, y = X[valid_data], y[valid_data]

                if len(X) > 10:
                    model = sm.OLS(y, X).fit()
                    for coef, pval, var in zip(model.params, model.pvalues, X.columns):
                        coefficient_data.append({
                            "Predictor": var,
                            "Coefficient": coef,
                            "P-value": pval,
                            "Noise Corr Type": correlation_type,
                            "Inhibitory Type": inh
                        })

            # Convert to DataFrame and plot
            coef_df = pd.DataFrame(coefficient_data)
            plt.figure(figsize=(12, 6))
            sns.barplot(data=coef_df, x="Coefficient", y="Predictor", hue="Noise Corr Type", dodge=True)
            plt.axvline(0, linestyle="--", color="black")
            plt.title(f"Regression Coefficient Importance Across Noise Correlations ({inh})")
            plt.savefig(os.path.join(save_dir, f"regression_coefficients_{inh}.png"))
            plt.close()
    # Run Regression Coefficients
    plot_regression_coefficients(df, save_dir)
    def plot_correlation_matrix(df, save_dir):
        """Plot Correlation Matrix with Hierarchical Clustering, split by PV and SST."""

        inhibitory_types = ["PV", "SST"]

        for inh in inhibitory_types:
            df_subset = df[df["Inhibitory Population"] == inh]

            correlation_matrix = df_subset[[
                "OSI Average", "OSI Difference", "NI Average", "NI Difference",
                "Fano Factor Average", "Fano Factor Difference", 
                "Exc-Exc Noise Correlation", "Exc-Red Noise Correlation", "Red-Red Noise Correlation"
            ]].corr()
            correlation_matrix = correlation_matrix.replace([np.inf, -np.inf], np.nan).dropna(how="all").dropna(axis=1, how="all")

            plt.figure(figsize=(10, 8))
            if not correlation_matrix.empty and correlation_matrix.isnull().sum().sum() == 0:
                sns.clustermap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
                plt.title(f"Hierarchical Clustering of Noise Correlations ({inh})")
                plt.savefig(os.path.join(save_dir, f"correlation_matrix_{inh}.png"))
                plt.close()
            else:
                print(f"⚠ Skipping correlation heatmap for {inh}: Correlation matrix contains NaNs or is empty.")
    # Run Correlation Matrix
    plot_correlation_matrix(df, save_dir)

    print("\n✅ Comprehensive Analysis & Visualization Complete!")

    """Performs ridge regression analysis on noise correlations, ensuring correct column references."""

    file_path = os.path.join(save_dir, "orientation_segmented_noise_correlations.csv")
    if not os.path.exists(file_path):
        print(f"⚠ Missing file: {file_path}")
        return
    
    df = pd.read_csv(file_path)

    # Define the predictor column names (matching `orientation_segmentation`)
    predictor_columns = [
        "OSI Average", "OSI Difference", "NI Average", "NI Difference",
        "Fano Factor Average", "Fano Factor Difference", "OSI × NI",
        "OSI × Fano", "NI × Fano"
    ]

    # Ensure the required predictor columns exist
    missing_cols = [col for col in predictor_columns if col not in df.columns]
    if missing_cols:
        print(f"⚠ Error: The following predictor columns are missing from the dataset: {missing_cols}")
        return

    # Define target variables (noise correlation types)
    target_variables = {
        "Exc-Exc Noise Correlation": "Exc-Exc",
        "Exc-Red Noise Correlation": "Exc-Red",
        "Red-Red Noise Correlation": "Red-Red"
    }

    # Define trial conditions (Grating vs Plaid)
    trial_types = ["Grating", "Plaid"]

    # Define inhibitory types
    inhibitory_types = ["PV", "SST"]

    results = []

    # Iterate over trial types, inhibitory types, and target variables
    for trial in trial_types:
        df_trial = df[df["Trial Type"] == trial]

        for inhibitory in inhibitory_types:
            df_inhibitory = df_trial[df_trial["Inhibitory Population"] == inhibitory]

            for target, target_name in target_variables.items():
                if target not in df_inhibitory.columns:
                    print(f"⚠ Skipping {target} (not in dataset) for {trial} - {inhibitory}")
                    continue

                # Prepare dataset for regression
                X = df_inhibitory[predictor_columns]
                y = df_inhibitory[target]

                # Remove NaNs
                valid_data = X.notnull().all(axis=1) & y.notnull()
                X, y = X[valid_data], y[valid_data]

                if len(X) < 10:
                    print(f"⚠ Not enough data for regression ({trial} - {inhibitory} - {target_name})")
                    continue

                # Standardize predictors
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Perform Ridge regression with cross-validation
                alphas = [0.01, 0.1, 0.33, 1.0, 10.0]
                ridge = RidgeCV(alphas=alphas, store_cv_values=True)
                ridge.fit(X_scaled, y)

                # Fit OLS for statistics
                X_ols = sm.add_constant(X_scaled)
                ols_model = sm.OLS(y, X_ols).fit()

                # Store results
                results.append({
                    "Trial Type": trial,
                    "Inhibitory Population": inhibitory,
                    "Noise Correlation Type": target_name,
                    "Best Alpha": ridge.alpha_,
                    "R²": ridge.score(X_scaled, y),
                    "p-values": ols_model.pvalues.to_dict(),
                    "Betas": ols_model.params.to_dict()
                })

                print(f"✅ Ridge Regression Completed: {trial} - {inhibitory} - {target_name}")
                print(f"   Best Alpha: {ridge.alpha_}, R²: {ridge.score(X_scaled, y):.3f}")
                print(f"   Betas: {ols_model.params.to_dict()}")

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_path = os.path.join(save_dir, "ridge_regression_results.csv")
    results_df.to_csv(results_path, index=False)

    print(f"✅ Ridge regression results saved to {results_path}")
def ridge_analysis(save_dir):
    """Performs ridge regression analysis on noise correlations, ensuring correct column references and feature creation."""

    file_path = os.path.join(save_dir, "orientation_segmented_noise_correlations.csv")
    if not os.path.exists(file_path):
        print(f"⚠ Missing file: {file_path}")
        return
    
    df = pd.read_csv(file_path)

    # **1️⃣ Create Predictor Columns**
    df["OSI Average"] = df[["OSI Neuron1", "OSI Neuron2"]].mean(axis=1)
    df["OSI Difference"] = abs(df["OSI Neuron1"] - df["OSI Neuron2"])
    df["NI Average"] = df[["Normalization Index (NI) Neuron1", "Normalization Index (NI) Neuron2"]].mean(axis=1)
    df["NI Difference"] = abs(df["Normalization Index (NI) Neuron1"] - df["Normalization Index (NI) Neuron2"])
    df["OSI × NI"] = df["OSI Average"] * df["NI Average"]  # Interaction term
    df["Fano Factor Average"] = df[["Fano Factor Neuron1", "Fano Factor Neuron2"]].mean(axis=1)
    df["Fano Factor Difference"] = abs(df["Fano Factor Neuron1"] - df["Fano Factor Neuron2"])
    df["OSI × Fano"] = df["OSI Average"] * df["Fano Factor Average"]
    df["NI × Fano"] = df["NI Average"] * df["Fano Factor Average"]

    # **2️⃣ Define Predictors and Targets**
    predictor_columns = [
        "OSI Average", "OSI Difference", "NI Average", "NI Difference",
        "Fano Factor Average", "Fano Factor Difference", "OSI × NI",
        "OSI × Fano", "NI × Fano"
    ]

    target_variables = {
        "Exc-Exc Noise Correlation": "Exc-Exc",
        "Exc-Red Noise Correlation": "Exc-Red",
        "Red-Red Noise Correlation": "Red-Red"
    }

    trial_types = ["Grating", "Plaid"]
    inhibitory_types = ["PV", "SST"]

    results = []
    adj_r2_values = {}

    # **3️⃣ Iterate Over Trial Types, Inhibitory Types, and Target Variables**
    for trial in trial_types:
        df_trial = df[df["Trial Type"] == trial]

        for inhibitory in inhibitory_types:
            df_inhibitory = df_trial[df_trial["Inhibitory Population"] == inhibitory]

            for target, target_name in target_variables.items():
                if target not in df_inhibitory.columns:
                    print(f"⚠ Skipping {target} (not in dataset) for {trial} - {inhibitory}")
                    continue

                # Prepare dataset for regression
                X = df_inhibitory[predictor_columns]
                y = df_inhibitory[target]

                # Remove NaNs
                valid_data = X.notnull().all(axis=1) & y.notnull()
                X, y = X[valid_data], y[valid_data]

                if len(X) < 10:
                    print(f"⚠ Not enough data for regression ({trial} - {inhibitory} - {target_name})")
                    continue

                # Standardize predictors
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                y_mean, y_std = np.mean(y), np.std(y)  # Compute mean and std of y
                y_stdized = (y - y_mean) / y_std  # Standardize y

                # Perform Ridge regression :D :D :D :D
                alphas = [0.01, 0.1, 0.33, 1.0, 10.0, 20.0, 30.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 2000.0]
                ridge = RidgeCV(alphas=alphas, store_cv_values=True)
                ridge.fit(X_scaled, y)

                # Keep predictors standardized
                X_ols_std = sm.add_constant(X_scaled)
                # Fit standardized OLS model
                ols_model = sm.OLS(y_stdized, X_ols_std).fit()

                #r^2 shit
                n, p = X_scaled.shape  # n = number of samples, p = number of predictors
                adj_r2 = 1 - (1 - ols_model.rsquared) * ((n - 1) / (n - p - 1))
                # Store adjusted R² values
                adj_r2_values[(trial, inhibitory, target_name)] = adj_r2

                # Compute Confidence Intervals
                conf_intervals = ols_model.conf_int()

                # Store results
                results.append({
                    "Trial Type": trial,
                    "Inhibitory Population": inhibitory,
                    "Noise Correlation Type": target_name,
                    "Best Alpha": ridge.alpha_,
                    "R²": ridge.score(X_scaled, y),
                    "p-values": ols_model.pvalues.to_dict(),
                    "Betas": ols_model.params.to_dict()
                })

                # Get Ridge beta coefficients
                ridge_betas = dict(zip(predictor_columns, ridge.coef_))

                # Sort features by absolute importance
                sorted_features = sorted(ridge_betas.items(), key=lambda x: abs(x[1]), reverse=True)

                # Store results with additional details
                results.append({
                    "Trial Type": trial,
                    "Inhibitory Population": inhibitory,
                    "Noise Correlation Type": target_name,
                    "Best Alpha": ridge.alpha_,
                    "R²": ridge.score(X_scaled, y),
                    "p-values": ols_model.pvalues.to_dict(),
                    "Betas (OLS)": ols_model.params.to_dict(),
                    "Betas (Ridge)": ridge_betas
                })

                print(f"✅ Ridge Regression Completed: {trial} - {inhibitory} - {target_name}")
                print(f"   Best Alpha: {ridge.alpha_}, R²: {ridge.score(X_scaled, y):.3f}, Adjusted R²: {adj_r2:.3f}")

                print("\n🔹 **Feature Importance Ranking (Ridge Coefficients)**")
                for feature, coef in sorted_features:
                    print(f"   {feature}: {coef:.5f}")

                print("\n🔹 **OLS Model Coefficients & P-values**")
                for feature, coef in ols_model.params.items():
                    pval = ols_model.pvalues[feature]
                    ci_low, ci_high = conf_intervals.loc[feature]
                    print(f"   {feature}: β = {coef:.5f}, p = {pval:.5f}, 95% CI = ({ci_low:.5f}, {ci_high:.5f})")

                print("\n🔹 **Standardized Coefficients (Comparability)**")
                for feature, coef in ols_model.params.items():
                    print(f"   {feature}: β = {coef:.5f}")

    # **4️⃣ Save Results to CSV**
    results_df = pd.DataFrame(results)
    results_path = os.path.join(save_dir, "ridge_regression_results.csv")
    results_df.to_csv(results_path, index=False)

    adj_r2_df = pd.DataFrame.from_dict(
        adj_r2_values, 
        orient="index", 
        columns=["Adjusted R²"]
    )
    # Set MultiIndex for better organization
    adj_r2_df.index = pd.MultiIndex.from_tuples(
        adj_r2_df.index, 
        names=["Trial Type", "Inhibitory Population", "Noise Correlation Type"]
    )
    # Pivot for plotting
    adj_r2_df = adj_r2_df.reset_index().pivot(
        index=["Inhibitory Population", "Noise Correlation Type"], 
        columns="Trial Type", 
        values="Adjusted R²"
    )

    # Bar chart for Adjusted R² values
    plt.figure(figsize=(8, 5))
    bar_width = 0.35
    index = np.arange(len(adj_r2_df))

    # Create bars for Grating and Plaid
    plt.bar(index, adj_r2_df["Grating"], bar_width, label="Grating", color="steelblue")
    plt.bar(index + bar_width, adj_r2_df["Plaid"], bar_width, label="Plaid", color="orange")

    # Labels and formatting
    plt.xlabel("Population")
    plt.ylabel("Adjusted R² Value")
    plt.title("Model Performance (Adjusted R²) Across Populations")
    plt.xticks(index + bar_width / 2, [f"{i[0]}-{i[1]}" for i in adj_r2_df.index], rotation=45, ha="right")
    plt.legend()
    plt.ylim(0, max(adj_r2_df.max()) * 1.1)  # Adjust y-limit dynamically

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bar_chart_adj_r2.png"))
    plt.show()

    # **5️⃣ Compute Summary Statistics**
    print("\n🔹 **Summary Statistics from Original CSV**")
    summary_stats = {}
    
    features_to_summarize = [
        "OSI Neuron1", "OSI Neuron2", 
        "Normalization Index (NI) Neuron1", "Normalization Index (NI) Neuron2",
        "Fano Factor Neuron1", "Fano Factor Neuron2"
    ]

    for feature in features_to_summarize:
        values = df[feature].dropna()  # Remove NaNs
        summary_stats[feature] = {
            "Mean": np.mean(values),
            "IQR": np.percentile(values, 75) - np.percentile(values, 25),
            "Min": np.min(values),
            "Max": np.max(values)
        }
    
    for feature, stats in summary_stats.items():
        print(f"   {feature}:")
        print(f"      Mean: {stats['Mean']:.5f}")
        print(f"      IQR: {stats['IQR']:.5f}")
        print(f"      Min: {stats['Min']:.5f}")
        print(f"      Max: {stats['Max']:.5f}")

    print("\n✅ Summary statistics printed successfully!")

    print(f"✅ Ridge regression results saved to {results_path}")

if __name__ == "__main__":
    #analyze_neurons()
    orientation_segmentation(save_dir)
    #analyze_orthogonal_noise_correlations()
    #summarize_all_results(save_dir)
    #noise_corr_analysis(save_dir)
    #ridge_analysis("/home/maclean/data_proc/noise_corr")