import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ranksums, pearsonr
from neural_data_object import NeuralData

# Parameters
mice_dates = {
    '21N': ['022424', '022524'],
    '30G': ['082424'],
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
}

# Define PV and SST mice groups
pv_mice_list = ['18S', '21N', '30G', '37L', '37L2', '37R', '37R2', '42N', '44N']
sst_mice_list = ['30N', '32L', '32L2', '42R', '42R2']

# Base path
base_path = '/media/maclean/3d4f021d-2b12-4203-943d-4f1052e2ccf0/1104_hal_data/'

# Visualization parameters
plt.style.use('ggplot')
plt.rc('axes.spines', **{'bottom': True, 'left': True, 'right': False, 'top': False})

# Function to load data
def load_data(mouse, date, base_path=base_path):
    """Loads NeuralData and associated variables for a given mouse and date."""
    print(f"Loading data for {mouse}, {date}...")

    # Initialize NeuralData
    data = NeuralData(mouse, date, base_name=base_path, use_l='cascade')

    # Get visual responsiveness
    vis_resp_cells = data.vis_responsive_cells(easy=True) if hasattr(data, 'vis_responsive_cells') else None

    # Get normalization indices
    ni_values = data.normalization_indices() if hasattr(data, 'normalization_indices') else None

    # Get stimulus information
    stim_info = data.stim_info if hasattr(data, 'stim_info') else None
    orientations = stim_info[0] if stim_info is not None else None
    is_plaid = stim_info[1] == 1 if stim_info is not None else None

    # Get red labels
    proc_dir = os.path.join(base_path, mouse, date, 'proc')
    red_labels_path = os.path.join(proc_dir, 'red_labels.npy')
    red_labels = np.load(red_labels_path) if os.path.exists(red_labels_path) else None

    return {
        'data': data,
        'vis_resp_cells': vis_resp_cells,
        'ni_values': ni_values,
        'stim_info': stim_info,
        'orientations': orientations,
        'is_plaid': is_plaid,
        'red_labels': red_labels
    }

# Load data for all mice
all_data = {}
for mouse, dates in mice_dates.items():
    for date in dates:
        all_data[(mouse, date)] = load_data(mouse, date)

# Define PV and SST datasets
pv_data = {k: v for k, v in all_data.items() if k[0] in pv_mice_list}
sst_data = {k: v for k, v in all_data.items() if k[0] in sst_mice_list}

# Processing Loop for PV and SST
for name, data_dict in zip(["PV", "SST"], [pv_data, sst_data]):
    print(f'Processing {name} data...')

    # Extract values from pre-loaded data
    ni_values = np.concatenate([d['ni_values'] for d in data_dict.values() if d['ni_values'] is not None])
    vis_resp = [d['vis_resp_cells'] for d in data_dict.values() if d['vis_resp_cells'] is not None]
    red_labels = [d['red_labels'] for d in data_dict.values() if d['red_labels'] is not None]

    # Compute excitatory and inhibitory responses
    vis_resp_green_list = [np.logical_and(vis, ~r) for vis, r in zip(vis_resp, red_labels)]
    vis_resp_green = np.concatenate(vis_resp_green_list)
    vis_resp_red = np.concatenate([np.logical_and(vis, r) for vis, r in zip(vis_resp, red_labels)])

    # Extract indices for plotting
    green_indices = ni_values[vis_resp_green]
    red_indices = ni_values[vis_resp_red]

    # Plot violin plot
    fig, ax = plt.subplots(figsize=(8, 6))
    violins = plt.violinplot([green_indices, red_indices], showmeans=True)
    
    # Set colors
    violins['bodies'][0].set_facecolor('black')
    violins['bodies'][0].set_edgecolor('black')
    violins['bodies'][1].set_facecolor('red')
    violins['bodies'][1].set_edgecolor('black')

    ax.set_ylabel('Normalization Index')
    plt.title(f'{name}-Cre Mice')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Exc', name])
    ax.set_ylim([-1, 1])

    # Scatter points
    plt_scale = 0.03
    ax.scatter(np.ones(green_indices.shape) + np.random.normal(scale=plt_scale, size=green_indices.shape),
               green_indices, marker='.', color='black')
    ax.scatter(2 * np.ones(red_indices.shape) + np.random.normal(scale=plt_scale, size=red_indices.shape),
               red_indices, marker='.', color='black')

    plt.tick_params(axis='x', which='major', labelsize='40')
    fig.set_tight_layout(True)
    plt.savefig(f'{name}_idx_violin.png', dpi=300)

    print(f'Saved figure for {name}.')

print("Analysis complete.")


"""### 051624 main normalization analysis on pooled PV and SST mice
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ranksums, pearsonr
import neural_data_object
from neural_data_object import NeuralData

# params
mice_dates = {
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
}

# Define PV and SST mice groups
pv_mice_list = ['18S', '21N', '30G', '37L', '37L2', '37R', '37R2', '42N', '44N']
sst_mice_list = ['30N', '32L', '32L2', '42R', '42R2']

# Generate tuple pairs for PV and SST mice
pv_mice = [(mouse, date) for mouse in pv_mice_list for date in mice_dates.get(mouse, [])]
sst_mice = [(mouse, date) for mouse in sst_mice_list for date in mice_dates.get(mouse, [])]

type_names = ['PV', 'SST']
bases = ['/media/maclean/3d4f021d-2b12-4203-943d-4f1052e2ccf0/1104_hal_data/']


dF = False # for partials... could try the ones computed from dF/F too
lag_cutoff = 3 # greater peak cross corr lags set to zero
s = 2 # sigma used for partials, 2 or 4
use_l = 'cascade'
snr_cutoff = 0 # looked ok

plt.style.use('ggplot')
plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})

for name, dates, base in zip(type_names, [pv_mice, sst_mice], bases):
    print(f'running {name} mice: {dates}')
    print('loading data')
    base = '/media/maclean/3d4f021d-2b12-4203-943d-4f1052e2ccf0/1104_hal_data/'
    use_l == 'cascade'
    datas = [NeuralData(d[0], d[1], base, use_l=use_l, sure=False, smooth=2) for d in dates]
    data = datas[0]
    # first just look at distrs of indices
    # get labels and indices, pooling across days/mice
    vis_resp = [data.vis_responsive_cells(easy=True) for data in datas]
    snrs_list = [data.get_snrs() > snr_cutoff for data in datas]
    snrs = np.concatenate(snrs_list)
    red = [data.red_labels for data in datas]
    vis_resp_green_list = [np.logical_and(vis, ~r) for vis, r in zip(vis_resp, red)]
    vis_resp_green = np.concatenate(vis_resp_green_list)
    vis_resp_red = np.concatenate([np.logical_and(vis, r) for vis, r in zip(vis_resp, red)])
    #vis_resp_red = np.concatenate([r for vis, r in zip(vis_resp, red)])
    indices_list = [d.normalization_indices() for d in datas]
    indices = np.concatenate(indices_list)
    pref_oris = [data.normalization_indices(return_prefs=True)[1] for data in datas]
    pref_plaids = [data.normalization_indices(return_prefs=True)[2] for data in datas]
    #red = [np.logical_and(vis, idx > 0.5) for vis, idx in zip(vis_resp_green_list, indices_list)]
    # SNR restrictions
    #vis_resp_green_list = [np.logical_and(g, snr) for g, snr in zip(vis_resp_green_list, snrs_list)]
    #vis_resp_red = np.logical_and(vis_resp_red, snrs)
    #vis_resp_green = np.logical_and(vis_resp_green, snrs)# np.logical_and(indices < 0.75, indices > 0))
    #vis_resp_green_list = [np.logical_and(vis_g, np.logical_and(ind < 0.75, ind > 0)) for 
                           #vis_g, ind in zip(vis_resp_green_list, [data.normalization_indices() for data in datas])]
    #red = [np.logical_and(r, snr) for r, snr in zip(red, snrs_list)]

    # plot relevant indices
    green_indices = indices[vis_resp_green]
    red_indices = indices[vis_resp_red]
    print(f'plotting indices for {len(green_indices)} green and {len(red_indices)} red cells')
    plt.figure()
    _, bins, _ = plt.hist(green_indices, bins=np.linspace(-1, 1, 20), color='green', label='green', alpha=0.7, density=True)
    plt.hist(red_indices, bins=bins, color='red', label=name, alpha=0.7, density=True)
    plt.legend()
    plt.xlabel('normalization index')
    plt.xlim(-1, 1)
    plt.title(f'{name} mice normalization indices')
    plt.savefig(f'{name}_pooled_norm_indices.png')
    #plt.show()
    # violin plots for poster
    fig, ax = plt.subplots(figsize=(8, 6))
    violins = plt.violinplot([green_indices, red_indices], showmeans=True)
    # can set color apparently
    violins['bodies'][0].set_facecolor('black')
    violins['bodies'][0].set_edgecolor('black')
    violins['bodies'][1].set_facecolor('red')
    violins['bodies'][1].set_edgecolor('black')
    ax.set_ylabel('normalization index')
    plt.title(f'{name}-Cre mice')
    #plt.xticks([])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Exc', name])
    ax.set_ylim([-1, 1])
    # scatter points
    plt_scale = 0.03
    ax.scatter(np.ones(green_indices.shape) + np.random.normal(scale=plt_scale,size=green_indices.shape),
            green_indices, marker='.', color='black')
    ax.scatter(2 * np.ones(red_indices.shape) + np.random.normal(scale=plt_scale, size=red_indices.shape),
            red_indices, marker='.', color='black')
    plt.tick_params(axis='x', which='major', labelsize='40')
    fig.set_tight_layout(True)
    plt.savefig(f'{name}_idx_violin.png', dpi=300) # raise for final versions
    continue



    # statistically test difference
    stat, p = ranksums(green_indices, red_indices)
    print(f'p-value for wilcoxon rank sums test: {p}')

    # load in partials and lags now to look at those
    partials = [np.load(f'{d[0]}_{d[1]}_dF{dF}_s{s}_all_partials_l{use_l}.npy') for d in dates]
    mean_partials = [np.mean(part, 0) for part in partials] # mean over imaging blocks -- always want this I think
    lags = [np.load(f'{d[0]}_{d[1]}_s{s}_spk_window_corr_lags.npy') for d in dates]
    tril = [np.tril_indices(len(lgs), -1) for lgs in lags] # idxs for bottom triangle of array

    # get better partials -- small lags
    good_partials = [part.copy() for part in mean_partials]
    for part, lag in zip(good_partials, lags):
        part[np.abs(lag) > lag_cutoff] = 0 # bullshit corr if lag too large
    vis_red_partials = [part[vis_g][:, r] for part, vis_g, r in zip(good_partials, vis_resp_green_list, red)]
    vis_green_partials = [part[vis_g][:, ~r] for part, vis_g, r in zip(good_partials, vis_resp_green_list, red)]    
    # evaluate correlation between avg red-cell partial corr and normalization
    # without excluding outliers, for now...
    red_influences = np.concatenate([np.nanmean(day, 1) for day in vis_red_partials])
    nonzero = ~np.isnan(red_influences) # should I exclude these? they don't change much :p
    print(f'{np.sum(nonzero)} green vis resp cells with nonzero red influences')
    r, p = pearsonr(red_influences[nonzero], green_indices[nonzero])
    print(f'corr between avg {name} partial and norm index: {r}, p={p}')
    plt.figure()
    plt.scatter(red_influences[nonzero], green_indices[nonzero], marker='.', color='black')
    plt.xlabel(f'average partial corr with {name} cells')
    plt.ylabel('normalization index')
    plt.title(f'{name} cell influence--normalization index relationship, r={r:.3f}, p={p:.3f}')
    plt.savefig(f'{name}_influence_index_scatter.png')
    #plt.show()

    # do some work characterizing partials, comparing to noise corrs
    # but first process noise corrs
    full_noise = [data.noise_correlations(logs=False) for data in datas]
    avg_noise = [corrs.mean(0) for corrs in full_noise] # avg noise corrs across stimuli
    grand_noise = [data.noise_correlations(grand_corr=True)[1] for data in datas]
    for full, avg, grand in zip(full_noise, avg_noise, grand_noise):
        diag_idx = np.eye(len(avg), dtype=bool) # take out diagonals
        avg[diag_idx] = np.nan
        grand[diag_idx] = np.nan
        for stim in full:
            stim[diag_idx] = np.nan
    # during preferred plaid and preferred orientation for each cell
    plaid_noise = [corrs[plaid] for corrs, plaid in zip(full_noise, pref_plaids)]
    ori_noise = [corrs[ori] for corrs, ori in zip(full_noise, pref_oris)]

    # first redo influence analysis
    # I think these together get the indexing right...
    red_noise_influence = [plaid[vis_g][:, vis_g][:, :, np.logical_and(r, snr)] for 
                           plaid, vis_g, r, snr in zip(plaid_noise, vis_resp_green_list, red, snrs_list)]
    red_noise_influence = [grand[vis_g][:, np.logical_and(r, snr)] for 
                        grand, vis_g, r, snr in zip(grand_noise, vis_resp_green_list, red, snrs_list)]
    #red_noise_influence = np.concatenate([np.diag(np.nanmean(reds, 2)) for reds in red_noise_influence])

    red_noise_influence = np.concatenate([np.nanmean(reds, 1) for reds in red_noise_influence])
    r, p = pearsonr(red_noise_influence[~np.isnan(red_noise_influence)], green_indices[~np.isnan(red_noise_influence)])
    print(f'corr between avg {name} noise corr and norm index: {r}, p={p}')
    plt.figure()
    plt.scatter(red_noise_influence, green_indices, marker='.', color='black')
    plt.xlabel(f'average noise corr with {name} cells')
    plt.ylabel('normalization index')
    plt.title(f'{name} cell influence--normalization index relationship, r={r:.3f}, p={p:.3f}')
    plt.savefig(f'{name}_noise_influence_index_scatter.png')
    # do again with diff between plaid and ori noise corr
    diff_noise = [(plaid - ori) / (np.abs(plaid) + np.abs(ori)) for plaid, ori in zip(plaid_noise, ori_noise)]
    red_diff_influence = [diff[vis_g][:, vis_g][:, :, np.logical_and(r, snr)] for 
                           diff, vis_g, r, snr in zip(diff_noise, vis_resp_green_list, red, snrs_list)]
    red_diff_influence = np.concatenate([np.diag(np.nanmean(reds, 2)) for reds in red_diff_influence])
    r, p = pearsonr(red_diff_influence[~np.isnan(red_diff_influence)], green_indices[~np.isnan(red_diff_influence)])
    print(f'corr between avg {name} plaid-ori noise corr diff and norm index: {r}, p={p}')
    plt.figure()
    plt.scatter(red_diff_influence, green_indices, marker='.', color='black')
    plt.xlabel(f'average noise corr diff (plaid-ori) with {name} cells')
    plt.ylabel('normalization index')
    plt.title(f'{name} cell influence--normalization index relationship, r={r:.3f}, p={p:.3f}')
    plt.savefig(f'{name}_noise_diff_influence_index_scatter.png')

    # now compare noise corrs and partials
    flat_partials = np.concatenate([p[tri] for p, tri in zip(mean_partials, tril)])
    flat_noise = np.concatenate([n[tri] for n, tri in zip(avg_noise, tril)])
    flat_lags = np.concatenate([l[tri] for l, tri in zip(lags, tril)])
    nonnan_idxs = np.logical_and(~np.isnan(flat_partials), ~np.isnan(flat_noise))
    flat_partials = flat_partials[nonnan_idxs]
    flat_noise = flat_noise[nonnan_idxs]
    flat_lags = flat_lags[nonnan_idxs]

    good_partials = flat_partials[np.abs(flat_lags) <= lag_cutoff]
    good_noise = flat_noise[np.abs(flat_lags) <= lag_cutoff]
    # compare them, filtered and non-filtered
    r, p = pearsonr(flat_partials[~np.isnan(flat_noise)], flat_noise[~np.isnan(flat_noise)])
    print(f'noise and partials correlated at {r:.3f}, p={p:.5f}')
    plt.figure()
    plt.scatter(flat_partials, flat_noise, marker='.', color='black')
    plt.xlabel(f'partial corr for {name} mice')
    plt.ylabel(f'avg noise corr for {name} mice')
    plt.title(f'{name} mice all partial vs noise corrs, r={r:.3f}, p={p:.3f}')
    plt.savefig(f'{name}_noise_vs_partial_scatter.png')
    #plt.show()
    # same for lag <= 2
    r, p = pearsonr(good_partials[~np.isnan(good_noise)], good_noise[~np.isnan(good_noise)])
    print(f'low-lag noise and partials correlated at {r:.3f}, p={p:.5f}')
    plt.figure()
    plt.scatter(good_partials, good_noise, marker='.', color='black')
    plt.xlabel(f'partial corr for {name} mice')
    plt.ylabel(f'avg noise corr for {name} mice')
    plt.title(f'{name} mice low-lag partial vs noise corrs, r={r:.3f}, p={p:.3f}')
    plt.savefig(f'{name}_noise_vs_low_lagpartial_scatter.png')
    #plt.show()

"""