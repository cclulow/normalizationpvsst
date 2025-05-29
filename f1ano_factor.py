import numpy as np
import os
import pandas as pd
from neural_data_object import NeuralData
from fano_factor import calculate_fano_factor
import matplotlib.pyplot as plt
import pdb

save_dir = '/home/maclean/data_proc/fano'
os.makedirs(save_dir, exist_ok=True)


micedates = {
    '18S': ['071924', '072024'],
    '21N': ['022424', '022524'],
    '30G': ['082424', '082524'],
    '30N': ['062824', '062924', '070124', '070224'],
    '32L': ['042424', '042524'],
    '32L2': ['053024'],
    '37L': ['070424', '070524'],
    '37L2': ['070924', '071024'],
    '37R': ['070224', '070324'],
    '37R2': ['070924', '071024'],
    '42N': ['022524', '022624'],
    '44N': ['022524', '022624'],
    '42R2': ['072924', '073024', '073124'], 
    '42R': ['072224', '072324', '072424', '072524'],
}

OSI_THRESHOLD = 0.25  # minimum OSI for neuron selection -- change to 0.3?
std_f_before = 30
std_f_after = 60

def load_data(mouse, date, base_path='/media/maclean/3d4f021d-2b12-4203-943d-4f1052e2ccf0/1104_hal_data/'):
    proc_dir = os.path.join(base_path, mouse, date, 'proc')
    stim_info = np.load(os.path.join(proc_dir, 'stim_info.npy'))
    data = NeuralData(mouse, date, base_name=base_path, use_l='cascade')

    trial_idxs = data.trial_idxs
    if trial_idxs is None or len(trial_idxs) == 0:
        raise ValueError("No trial indices found.")

    spikes = np.array([
        data.spks[:, t - std_f_before:t + std_f_after]
        for t in trial_idxs
    ])
    spikes = np.transpose(spikes, (1, 0, 2))  # (neurons, trials, time_bins)

    vis_resp_cells = data.vis_responsive_cells() if hasattr(data, 'vis_responsive_cells') else None
    if vis_resp_cells is None or not vis_resp_cells.any():
        raise ValueError(f"No visually responsive cells for Mouse {mouse}, Date {date}.")

    if hasattr(data, 'red_labels') and data.red_labels is not None and len(data.red_labels) > 0:
        red_labels = data.red_labels
    else:
        print(f"No red labels found for Mouse {mouse}, Date {date}. Treating all spikes as excitatory.")
        red_labels = None
        
    orientations = stim_info[0]  # diviiding trials by orientation
    is_plaid = stim_info[1] == 1  # plaid is true, grating is false

    osis = data.get_osis() if hasattr(data, 'get_osis') else None

    return spikes, vis_resp_cells, stim_info, red_labels, orientations, is_plaid, osis

def analyze_neurons():
    all_results = []
    red_neuron_results = []
    for mouse, dates in micedates.items():
        for date in dates:
            try:
                # load data
                spikes, vis_resp_cells, stim_info, red_labels, orientations, is_plaid, osis = load_data(mouse, date)
                
                trial_groups_per_neuron = {}
                pref_ori = []

                for neuron_idx  in range(spikes.shape[0]): #only vis resp cells loop
                    mean_responses = []
                    for ori in np.unique(orientations):
                        ori_trials = orientations == ori
                        mean_responses.append(spikes[neuron_idx, ori_trials, :].mean())
                    pref_ori.append(np.unique(orientations)[np.argmax(mean_responses)])

                    # define trial groups
                    pref_mask = orientations == pref_ori[neuron_idx]
                    null_mask = orientations != pref_ori[neuron_idx]
                    plaid_mask = is_plaid
                    grating_mask = ~is_plaid
                    plaid_pref_or_null = plaid_mask & (pref_mask | null_mask)
                    #other_plaids = plaid_mask & ~plaid_pref_or_null
                    #other_gratings = grating_mask & ~pref_mask & ~null_mask

                    trial_groups_per_neuron[neuron_idx] = {
                        "Preferred": pref_mask,
                        "Null": null_mask,
                        "Plaid Pref/Null": plaid_pref_or_null,
                        #"Other Plaids": other_plaids,
                        #"Other Gratings": other_gratings,
                    }

                data = NeuralData(mouse, date, base_name='/media/maclean/3d4f021d-2b12-4203-943d-4f1052e2ccf0/1104_hal_data/')
                norm_indices = data.normalization_indices()
                all_noise_corrs = data.noise_correlations() #add trig things later
                noise_corrs = all_noise_corrs.mean(axis=0) # mean across stimuli
                mask = np.eye(noise_corrs.shape[0], dtype=bool)
                noise_corrs[mask] = np.nan
                # get which stimulus is which
                #_, stim = data.get_trial_response()
                #plaid_noise_corrs = all_noise_corrs[stim[:, 1] == 1].mean(0)
                #grate_noise_corrs = all_noise_corrs[stim[:, 1] == 0].mean(0)

#noise correlations section -- looking at 1) all noise corrs, 2) exc with inh, 3) inh-inh, 4) inh with exc

                avg_noise_corrs = np.nanmean(noise_corrs, axis=1)
                if red_labels is not None:
                    red_noise_corrs = noise_corrs[:, red_labels]
                    for idx in np.where(red_labels)[0]:
                        red_noise_corrs[idx, red_labels] = np.nan
                    avg_red_noise_corrs = np.nanmean(red_noise_corrs, axis=1)
                else:
                    avg_red_noise_corrs = None
                if red_labels is not None:
                    red_red_corrs = noise_corrs[np.ix_(red_labels, red_labels)]
                    np.fill_diagonal(red_red_corrs, np.nan)
                    avg_red_red_corrs = np.nanmean(red_red_corrs, axis=1)
                else:
                    avg_red_red_corrs = None
                if red_labels is not None:
                    avg_red_all_corrs = noise_corrs[red_labels, :] if red_labels is not None else None
                else:
                    avg_red_all_corrs = None
                if red_labels is not None:
                    exc_exc_corrs = noise_corrs[np.ix_(~red_labels, ~red_labels)]
                    np.fill_diagonal(exc_exc_corrs, np.nan)
                    avg_exc_exc_corrs = np.nanmean(exc_exc_corrs, axis=1)
                else:
                    avg_exc_exc_corrs = None

                #avg_noise_corrs = [np.mean(np.triu(noise_corr, k=1)) for noise_corr in noise_corrs]
                osi_values = data.get_osis() if hasattr(data, 'get_osis') else None

                """# Filter spikes and related metrics to visually responsive neurons
                filtered_indices = np.where(vis_resp_cells)[0]  # Indices of visually responsive neurons
                filtered_spikes = spikes[filtered_indices, :, :]  # Filtered spikes for visually responsive neurons
                filtered_trial_groups = {i: trial_groups_per_neuron[original_idx] for i, original_idx in enumerate(filtered_indices)}
                # Process each visually responsive neuron
                for neuron_idx in range(filtered_spikes.shape[0]):  # Iterate over filtered neurons
                    neuron_spikes = filtered_spikes[neuron_idx, :, :]  # Spikes for the current neuron
                    fano_factors = {}

                    # Use filtered_trial_groups for trial grouping
                    for group_name, condition_mask in filtered_trial_groups[neuron_idx].items():
                        if not condition_mask.any():
                            fano_factors[group_name] = np.nan
                            continue

                        # Extract spikes for trials in this group
                        group_spikes = neuron_spikes[condition_mask, :]  # Spikes for the condition
                        trial_summed_spikes = group_spikes.sum(axis=1)  # Summed spikes per trial

                        # Calculate Fano Factor
                        mean_spikes = np.mean(trial_summed_spikes)
                        variance_spikes = np.var(trial_summed_spikes)
                        fano_factors[group_name] = variance_spikes / mean_spikes if mean_spikes > 0 else np.nan"""
                
                #pdb.set_trace()
                # processing all neurons initially, postponing filtering
                for neuron_idx in range(spikes.shape[0]):  
                    if not vis_resp_cells[neuron_idx]:  # skip non-visually responsive neurons
                        continue

                    # original trial groups
                    neuron_spikes = spikes[neuron_idx, :, :]  
                    trial_groups = trial_groups_per_neuron[neuron_idx]  # og trial groups for this neuron
                    fano_factors = {}

                    # handling empty trial groups and calculate Fano factors
                    for group_name, condition_mask in trial_groups.items():
                        if not condition_mask.any():  #empty trial groups
                            print(f"Skipping group {group_name} for neuron {neuron_idx}, no trials.")
                            fano_factors[group_name] = np.nan
                            continue

                        assert condition_mask.shape[0] == neuron_spikes.shape[0], (
                            f"Condition mask size {condition_mask.shape[0]} does not match trials {neuron_spikes.shape[0]}"
                        )

                        #spikes for trials in this group
                        group_spikes = neuron_spikes[condition_mask, :]  # condition
                        if group_spikes.size == 0:  # edge case where no spikes are recorded
                            print(f"No spikes found for group {group_name}, neuron {neuron_idx}.")
                            fano_factors[group_name] = np.nan
                            continue

                        # trial-summed spikes per trial
                        trial_summed_spikes = group_spikes.sum(axis=1)

                        #Fano Factor
                        mean_spikes = np.mean(trial_summed_spikes)
                        variance_spikes = np.var(trial_summed_spikes)
                        fano_factors[group_name] = variance_spikes / mean_spikes if mean_spikes > 0 else np.nan

                    #results 
                    neuron_results = {
                        "Mouse": mouse,
                        "Date": date,
                        "Neuron": neuron_idx,  # Original neuron index
                        "Preferred Fano Factor": fano_factors.get("Preferred", np.nan),
                        "Null Fano Factor": fano_factors.get("Null", np.nan),
                        "Plaid Pref/Null Fano Factor": fano_factors.get("Plaid Pref/Null", np.nan),
                        "Normalization Index": norm_indices[neuron_idx] if neuron_idx < len(norm_indices) else None,
                        "Noise Correlation": avg_noise_corrs[neuron_idx] if neuron_idx < len(avg_noise_corrs) else None,
                        "All-Red Noise Correlation": avg_red_noise_corrs[neuron_idx] if avg_red_noise_corrs is not None else None,
                        "OSI": osi_values[neuron_idx] if osi_values is not None else np.nan,
                        "Red-Red Noise Correlation": avg_red_red_corrs[neuron_idx] if avg_red_red_corrs is not None else None,
                        "Red-All Noise Correlation": avg_red_all_corrs[neuron_idx] if avg_red_all_corrs is not None else None,
                        "Exc-Exc Noise Correlation": avg_exc_exc_corrs[neuron_idx] if avg_exc_exc_corrs is not None else None,
                    }
                    all_results.append(neuron_results)

                    if red_labels[neuron_idx]:
                        red_neuron_results.append(neuron_results)

            except Exception as e:
                print(f"Error processing Mouse {mouse}, Date {date}: {e}")
                print(f"Error processing Mouse {mouse}, Date {date}: {e}")
                print(f"Mouse: {mouse}, Date: {date}")
                print(f"Spikes shape: {spikes.shape if 'spikes' in locals() else 'Not loaded'}")
                print(f"Stim info shape: {stim_info.shape if 'stim_info' in locals() else 'Not loaded'}")
                print(f"Red labels: {red_labels.shape if red_labels is not None else 'None'}")
                vis_resp_cells = data.vis_responsive_cells() if hasattr(data, 'vis_responsive_cells') else None
                print(f"Vis resp cells shape: {vis_resp_cells.shape}")

    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("results_norm.csv", index=False)
        print("Results saved.")
    else:
        print("No results to save.")
    if red_neuron_results:
        red_results_df = pd.DataFrame(red_neuron_results)
        red_results_df.to_csv("neuron_results_red.csv", index=False)
        print("Red neuron results saved to neuron_results_red.csv.")
        print(f"Total red-labeled neurons saved: {len(red_neuron_results)}")
    else:
        print("No red-labeled neuron results to save.")

if __name__ == "__main__":
    analyze_neurons()
