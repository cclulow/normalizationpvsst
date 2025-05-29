#note need to fix
#vis responsive red cells (showing as 0 in the csv file)
#plots 3-6 are NOT working
#what orientation combos should i plot? how to compare etc etc etc etc

#segmented by sst and pv DONE
#segmented by orientation DONE
#set up all my keys and dicts and everything yay
#streamlined plotting process to be super boilerplate and whatev's
#need to figure out what to actually do about all of this


import numpy as np
import os
from neural_data_object import NeuralData
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

micedates = {
    '18S': ['071924','072024'],
    '21N': ['022424','022524'],
    '30G': ['082424','082524'],
    '30N': ['062824','062924','070124','070224'],
    '32L': ['042424','042524'],
    '32L2': ['053024'],
    '37L': ['070424','070524',],
    '37L2': ['070924','071024',],
    '37R': ['070224','070324',],
    '37R2': ['070924','071024',],
    '42N': ['022524','022624'],
    '44N': ['022524','022624'],
    '42R2': ['072924', '073024', '073124'], 
    '42R': ['072224', '072324', '072424', '072524'],
}

pv_mice = ['18S', '21N', '30G', '37L', '37L2','37R','37R2' '42N', '44N']
sst_mice = ['30N', '32L', '32L2', '42', '42R2']

#pv: 18S, 21N, 30G, 37L, 37R, 42N, 44N
#sst: 30N, 32L, 42R, 42R2 

#'
std_f_before = 30
std_f_after = 60
save_dir = '/home/maclean/data_proc/fano'

#load data
def load_data(mouse, date, base_path='/media/maclean/3d4f021d-2b12-4203-943d-4f1052e2ccf0/1104_hal_data/'):
    proc_dir = os.path.join(base_path, mouse, date, 'proc')
    stim_info = np.load(os.path.join(proc_dir, 'stim_info.npy'))
    data = NeuralData(mouse, date, base_name=base_path, use_l='cascade')

    vis_resp_cells = data.vis_responsive_cells() if hasattr(data, 'vis_responsive_cells') else None
    trial_idxs = data.trial_idxs
    spikes = np.array([
        data.spks[:, t - std_f_before:t + std_f_after]
        for t in trial_idxs
    ])
    spikes = np.transpose(spikes, (1, 0, 2))  # (rois, trials, time bins)
    spikes = spikes[vis_resp_cells]  # Filter for visually responsive spikes
    
    if hasattr(data, 'red_labels') and data.red_labels is not None and len(data.red_labels) > 0:
        red_labels = data.red_labels[vis_resp_cells]
    else:
        print(f"No red labels found for Mouse {mouse}, Date {date}. Treating all spikes as excitatory.")
        red_labels = None
    
    orientations = stim_info[0]  # orientation of each trial
    is_plaid = stim_info[1] == 1  # plaid (True) or grating (False) for each trial
    
    return spikes, stim_info, red_labels, orientations, is_plaid

def calculate_fano_factor(spike_counts):
    mean_spikes = np.mean(spike_counts, axis=1) #axis=1 accesses trials in spks.npy, this is correct i think
    variance_spikes = np.var(spike_counts, axis=1)

    fano_factor = variance_spikes / mean_spikes
    fano_factor = np.clip(variance_spikes / mean_spikes, None, 35) #don't do this later, like take this out
    return fano_factor

def analyze_fano_factor(mouse, date, pre_stim, post_stim, base_path='/media/maclean/3d4f021d-2b12-4203-943d-4f1052e2ccf0/1104_hal_data/'):
    spikes, stim_info, red_labels, orientations, is_plaid = load_data(mouse, date, base_path)

    unique_orientations = np.unique(orientations)  # Identify all unique orientations, need to add orientations var
    
    plaid_trials = stim_info[1] == 1
    grating_trials = stim_info[1] == 0
#segment by orientation ---------- not sure how but i will try
    if red_labels is not None:
        exc_spikes = spikes[~red_labels]  # Excitatory: not red-labeled ROIs
        inh_spikes = spikes[red_labels]  # Inhibitory: red-labeled ROIs
    else:
        print("No red labels found; treating all spikes as excitatory.")
        exc_spikes = spikes
        inh_spikes = None

    if exc_spikes.ndim == 2:  
        exc_spikes = exc_spikes[:, :, np.newaxis]
    if inh_spikes is not None and inh_spikes.ndim == 2:
        inh_spikes = inh_spikes[:, :, np.newaxis]

    def compute_fano(spikes, condition_mask, time_window):
        if spikes is None:
            return {}
        fano_factors = {}
        for ori in unique_orientations:
            ori_mask = orientations == ori
            combined_mask = np.logical_and(condition_mask, ori_mask)
            windowed_spikes = spikes[:, combined_mask, time_window[0]:time_window[1]].sum(axis=2)
            fano_factors[ori] = calculate_fano_factor(windowed_spikes)
        return fano_factors

    # Prepare results dictionary
    results = {}
    results = {
        "EXC_Plaid_Pre": compute_fano(exc_spikes, is_plaid, pre_stim),
        "EXC_Plaid_Post": compute_fano(exc_spikes, is_plaid, post_stim),
        "EXC_Grating_Pre": compute_fano(exc_spikes, ~is_plaid, pre_stim),
        "EXC_Grating_Post": compute_fano(exc_spikes, ~is_plaid, post_stim),
    }

    if inh_spikes is not None:
        results.update({
            "INH_Plaid_Pre": compute_fano(inh_spikes, is_plaid, pre_stim),
            "INH_Plaid_Post": compute_fano(inh_spikes, is_plaid, post_stim),
            "INH_Grating_Pre": compute_fano(inh_spikes, ~is_plaid, pre_stim),
            "INH_Grating_Post": compute_fano(inh_spikes, ~is_plaid, post_stim),
        })

    return results

pre_stim = (0,30)
post_stim = (30,60)

# Initialize dictionaries for aggregation
all_exc_pre = []
all_exc_post = []
all_exc_plaid = []
all_exc_grating = []
all_inh_pre_pv = []
all_inh_post_pv = []
all_inh_plaid_pv = []
all_inh_grating_pv = []
all_inh_pre_sst = []
all_inh_post_sst = []
all_inh_plaid_sst = []
all_inh_grating_sst = []

all_exc_plaid_pre = []
all_exc_plaid_post = []
all_exc_grating_pre = []
all_exc_grating_post = []
all_inh_plaid_pre_sst = []
all_inh_plaid_pre_pv = []
all_inh_plaid_post_sst = []
all_inh_plaid_post_pv = []
all_inh_grating_pre_sst = []
all_inh_grating_pre_pv = []
all_inh_grating_post_sst = []
all_inh_grating_post_pv = []

fano_factors_calculation = {}
vis_resp_red_cells_count = {}

#need to initialize a dictionary to actually save all of this

per_mouse_results = {
    "Mouse": [],
    "Date": [],
    "Condition": [],
    "Mean Fano Factor": [],
    "Number of Vis Resp Red Cells": []
    #orientation?
}

aggregated_means = {condition: {ori: [] for ori in range(0,180,45)} for condition in [
    "INH_Plaid_Pre", "INH_Plaid_Post", "INH_Grating_Pre", "INH_Grating_Post",
    "EXC_Plaid_Pre", "EXC_Plaid_Post", "EXC_Grating_Pre", "EXC_Grating_Post"
]}

for mouse, dates in micedates.items():
    fano_factors_calculation[mouse] = {}
    vis_resp_red_cells_count[mouse] = {}
    for date in dates:
        # Analyze Fano factors
        results = analyze_fano_factor(mouse, date, pre_stim, post_stim)
        fano_factors_calculation[mouse][date] = results
        for condition, orientation_data in results.items():
            for orientation, fano_values in orientation_data.items():
                valid_values = fano_values[~np.isnan(fano_values)] 
                if valid_values.size == 0:
                    continue
                key = f"{condition}_Ori{orientation}"
                if key not in aggregated_means:
                    aggregated_means[key] = [] #this might be the problem
                
                aggregated_means[key].extend(valid_values.tolist())

                per_mouse_results["Mouse"].append(mouse)
                per_mouse_results["Date"].append(date)
                per_mouse_results["Condition"].append(key)  
                per_mouse_results["Mean Fano Factor"].append(np.mean(valid_values))
                per_mouse_results["Number of Vis Resp Red Cells"].append(
                    vis_resp_red_cells_count.get(mouse, {}).get(date, 0)
                )
                #orientation add here

# worked before but now isn't working w ori additions
#apparently they are dictionaries now, not lists (whoops)
overall_means = {
    condition: np.mean(values) 
    for condition, values in aggregated_means.items() 
    if isinstance(values, list) and len(values) > 0  # checking for valid lists
}
print("\nAggregated Means Across All Mice and Dates (Segmented by Orientation):")
for condition, mean_value in overall_means.items():
    print(f"{condition}: Mean Fano Factor = {mean_value:.2f}")

results_df = pd.DataFrame(per_mouse_results)
results_file = os.path.join(save_dir, 'per_mouse_fano_factors_by_orientation.csv')
results_df.to_csv(results_file, index=False)
print(f"\nPer-mouse results saved to {results_file}.")

day_means = {key: [] for key in aggregated_means.keys()}

mice = ['18S', '21N', '30G', '30N', '32L', '32L2', '37L', '37L2', '37R', '37R2', '42N', '44N']

# Gather data
for mouse in mice:
    for date in micedates[mouse]:
        results = analyze_fano_factor(mouse, date, pre_stim, post_stim)

        # Process excitatory data by orientation
        if "EXC_Plaid_Pre" in results:
            for orientation, values in results["EXC_Plaid_Pre"].items():
                valid_values = values[~np.isnan(values)]  # Filter NaNs
                all_exc_pre.extend(valid_values)
                all_exc_plaid_pre.extend(valid_values)
                all_exc_plaid.extend(valid_values)

        if "EXC_Plaid_Post" in results:
            for orientation, values in results["EXC_Plaid_Post"].items():
                valid_values = values[~np.isnan(values)]  # Filter NaNs
                all_exc_post.extend(valid_values)
                all_exc_plaid_post.extend(valid_values)
                all_exc_plaid.extend(valid_values)

        if "EXC_Grating_Pre" in results:
            for orientation, values in results["EXC_Grating_Pre"].items():
                valid_values = values[~np.isnan(values)]  # Filter NaNs
                all_exc_grating.extend(valid_values)
                all_exc_grating_pre.extend(valid_values)

        if "EXC_Grating_Post" in results:
            for orientation, values in results["EXC_Grating_Post"].items():
                valid_values = values[~np.isnan(values)]  # Filter NaNs
                all_exc_grating.extend(valid_values)
                all_exc_grating_post.extend(valid_values)

        # Process inhibitory data by orientation
        if "INH_Plaid_Pre" in results:
            for orientation, values in results["INH_Plaid_Pre"].items():
                valid_values = values[~np.isnan(values)]  # Filter NaNs
                if mouse in pv_mice:
                    all_inh_pre_pv.extend(valid_values)
                    all_inh_plaid_pv.extend(valid_values)
                elif mouse in sst_mice:
                    all_inh_pre_sst.extend(valid_values)
                    all_inh_plaid_sst.extend(valid_values)

        if "INH_Plaid_Post" in results:
            for orientation, values in results["INH_Plaid_Post"].items():
                valid_values = values[~np.isnan(values)]  # Filter NaNs
                if mouse in pv_mice:
                    all_inh_post_pv.extend(valid_values)
                    all_inh_plaid_pv.extend(valid_values)
                elif mouse in sst_mice:
                    all_inh_post_sst.extend(valid_values)
                    all_inh_plaid_sst.extend(valid_values)

        if "INH_Grating_Pre" in results:
            for orientation, values in results["INH_Grating_Pre"].items():
                valid_values = values[~np.isnan(values)]  # Filter NaNs
                if mouse in pv_mice:
                    all_inh_grating_pv.extend(valid_values)
                elif mouse in sst_mice:
                    all_inh_grating_sst.extend(valid_values)

        if "INH_Grating_Post" in results:
            for orientation, values in results["INH_Grating_Post"].items():
                valid_values = values[~np.isnan(values)]  # Filter NaNs
                if mouse in pv_mice:
                    all_inh_grating_pv.extend(valid_values)
                elif mouse in sst_mice:
                    all_inh_grating_sst.extend(valid_values)

#trying to streamline (wayyyyyy too many plots, don't want to keep typing so much out)
#there was a problem with empty datasets idk which one is empty though...

def filter_datasets_and_labels(datasets, labels):
    filtered_datasets = [data for data in datasets if len(data) > 0]
    filtered_labels = [labels[i] for i, data in enumerate(datasets) if len(data) > 0]
    return filtered_datasets, filtered_labels
def filter_and_plot(datasets, labels, title, ylabel, save_path, figsize=(12, 6)):
    filtered_datasets, filtered_labels = filter_datasets_and_labels(datasets, labels)
    
    if filtered_datasets:  # Only plot if there's valid data
        plt.figure(figsize=figsize)
        plt.violinplot(filtered_datasets, showmeans=True)
        plt.xticks(range(1, len(filtered_labels) + 1), filtered_labels)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        print(f"Skipping plot: {title} - No valid data.")

plot_configs = [
    {
        "datasets": [all_exc_pre, all_exc_post],
        "labels": ['Exc Pre-Stimulus', 'Exc Post-Stimulus'],
        "title": "Aggregated Fano Factor: Excitatory Cells (Pre vs Post)",
        "ylabel": "Fano Factor",
        "save_path": os.path.join(save_dir, 'fano_factor_exc_pre_post.png')
    },
    {
        "datasets": [all_exc_plaid, all_exc_grating],
        "labels": ['Exc Plaid', 'Exc Grating'],
        "title": "Aggregated Fano Factor: Excitatory Cells (Plaid vs Grating)",
        "ylabel": "Fano Factor",
        "save_path": os.path.join(save_dir, 'fano_factor_plaid_grating.png')
    },
    {
        "datasets": [all_exc_plaid, all_exc_grating, all_exc_pre, all_exc_post],
        "labels": ['Exc Plaid', 'Exc Grating', 'Exc Pre', 'Exc Post'],
        "title": "Aggregated Fano Factor: Excitatory Cells",
        "ylabel": "Fano Factor",
        "save_path": os.path.join(save_dir, 'fano_factor_allexc.png')
    },
    {
        "datasets": [
            all_exc_pre, all_exc_post, all_inh_pre_pv, all_inh_post_pv, 
            all_inh_pre_sst, all_inh_post_sst
        ],
        "labels": [
            'Exc Pre-Stimulus', 'Exc Post-Stimulus', 'PV Inh Pre-Stimulus', 
            'PV Inh Post-Stimulus', 'SST Inh Pre-Stimulus', 'SST Inh Post-Stimulus'
        ],
        "title": "Aggregated Fano Factor: Pre vs Post Stimulus (PV vs SST)",
        "ylabel": "Fano Factor",
        "save_path": os.path.join(save_dir, 'fano_factor_pre_post_pv_sst.png')
    },
    {
        "datasets": [
            all_exc_plaid, all_exc_grating, all_inh_plaid_pv, all_inh_grating_pv, 
            all_inh_plaid_sst, all_inh_grating_sst
        ],
        "labels": [
            'Exc Plaid', 'Exc Grating', 'PV Inh Plaid', 'PV Inh Grating', 
            'SST Inh Plaid', 'SST Inh Grating'
        ],
        "title": "Aggregated Fano Factor: Plaid vs Grating (PV vs SST)",
        "ylabel": "Fano Factor",
        "save_path": os.path.join(save_dir, 'fano_factor_plaid_grating_pv_sst.png')
    }
]

# Generate plots using the unified function
for config in plot_configs:
    filter_and_plot(
        datasets=config["datasets"],
        labels=config["labels"],
        title=config["title"],
        ylabel=config["ylabel"],
        save_path=config["save_path"]
    )