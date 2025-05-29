import numpy as np
from scipy.stats import pearsonr, spearmanr
import os
import matplotlib.pyplot as plt

# Load the previously computed matrices
save_dir = '/home/maclean/data_proc/figures/ni_noise_corr_analysis'
aligned_rois = np.load(os.path.join(save_dir, 'aligned_rois_matrix.npy'))
norm_indices_matrix = np.load(os.path.join(save_dir, 'norm_indices_matrix.npy'))
noise_corr_matrix_opposite_sst = np.load(os.path.join(save_dir, 'noise_corr_matrix_opposite_sst.npy'))
vis_resp_matrix = np.load(os.path.join(save_dir, 'vis_resp_matrix.npy'))  # Assuming you have this matrix for vis resp

# Define your dates
dates = ['072224', '072324', '072424', '072524']

# Function to calculate correlations
def determine_correlation(x, y, method='pearson'):
    valid_idx = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(valid_idx) > 0:
        if method == 'pearson':
            corr, p_value = pearsonr(x[valid_idx], y[valid_idx])
        elif method == 'spearman':
            corr, p_value = spearmanr(x[valid_idx], y[valid_idx])
        return corr, p_value
    else:
        return np.nan, np.nan

### Pooled Data Analysis ###
print("\n--- Pooled Data Across All Days ---")

# Pool all data for NI and noise correlations across all neurons and days
all_norm_indices = []
all_noise_corr_opposite_sst = []

for date_idx in range(len(dates)):
    norm_indices = norm_indices_matrix[:, date_idx]
    noise_corr_opposite = noise_corr_matrix_opposite_sst[:, date_idx]
    
    # All neurons
    valid_idx = ~np.isnan(norm_indices) & ~np.isnan(noise_corr_opposite)
    all_norm_indices.append(norm_indices[valid_idx])
    all_noise_corr_opposite_sst.append(noise_corr_opposite[valid_idx])

# Flatten the arrays
all_norm_indices = np.concatenate(all_norm_indices)
all_noise_corr_opposite_sst = np.concatenate(all_noise_corr_opposite_sst)

# Correlation for all neurons (pooled across all days)
corr_all_neurons_pooled, p_all_neurons_pooled = pearsonr(all_norm_indices, all_noise_corr_opposite_sst)
print(f"Pooled All Neurons: Pearson correlation = {corr_all_neurons_pooled}, p-value = {p_all_neurons_pooled}")

### 1. 50/50 Split of Neurons Based on NI ###

# Get median NI to split the neurons into high and low NI groups
median_ni = np.median(all_norm_indices)

# Split neurons into high NI and low NI groups
high_ni_idx = all_norm_indices > median_ni
low_ni_idx = all_norm_indices <= median_ni

# Calculate mean noise correlation for high NI neurons
high_ni_noise_corr_mean = np.nanmean(all_noise_corr_opposite_sst[high_ni_idx])
low_ni_noise_corr_mean = np.nanmean(all_noise_corr_opposite_sst[low_ni_idx])

print(f"\nMean noise correlation for High NI neurons (50%): {high_ni_noise_corr_mean}")
print(f"Mean noise correlation for Low NI neurons (50%): {low_ni_noise_corr_mean}")

# Optional: Plot the noise correlation distributions for high and low NI groups
plt.figure(figsize=(10, 6))
plt.hist(all_noise_corr_opposite_sst[high_ni_idx], bins=20, color='blue', alpha=0.7, label='High NI (50%)')
plt.hist(all_noise_corr_opposite_sst[low_ni_idx], bins=20, color='red', alpha=0.7, label='Low NI (50%)')
plt.xlabel('Noise Correlation with Opposite-Tuned SST Cells')
plt.ylabel('Number of Neurons')
plt.title('Noise Correlation Distribution: High vs. Low NI Neurons (50% Split)')
plt.legend()
plt.savefig(os.path.join(save_dir, 'noise_corr_distribution_high_vs_low_ni_50_split.png'))
plt.close()

### 2. Top 10% of NI Neurons ###

# Get the threshold for the top 10% of neurons based on NI
top_10_percent_threshold = np.percentile(all_norm_indices, 90)

# Split neurons into top 10% and others
top_10_percent_idx = all_norm_indices >= top_10_percent_threshold
bottom_90_percent_idx = all_norm_indices < top_10_percent_threshold

# Calculate mean noise correlation for top 10% NI neurons
top_10_noise_corr_mean = np.nanmean(all_noise_corr_opposite_sst[top_10_percent_idx])
bottom_90_noise_corr_mean = np.nanmean(all_noise_corr_opposite_sst[bottom_90_percent_idx])

print(f"\nMean noise correlation for Top 10% NI neurons: {top_10_noise_corr_mean}")
print(f"Mean noise correlation for Bottom 90% NI neurons: {bottom_90_noise_corr_mean}")

# Optional: Plot the noise correlation distributions for top 10% and bottom 90% NI groups
plt.figure(figsize=(10, 6))
plt.hist(all_noise_corr_opposite_sst[top_10_percent_idx], bins=20, color='green', alpha=0.7, label='Top 10% NI')
plt.hist(all_noise_corr_opposite_sst[bottom_90_percent_idx], bins=20, color='orange', alpha=0.7, label='Bottom 90% NI')
plt.xlabel('Noise Correlation with Opposite-Tuned SST Cells')
plt.ylabel('Number of Neurons')
plt.title('Noise Correlation Distribution: Top 10% vs. Bottom 90% NI Neurons')
plt.legend()
plt.savefig(os.path.join(save_dir, 'noise_corr_distribution_top_10_percent_ni.png'))
plt.close()

# Summary of Analysis
print(f"\nSummary of Pooled Data Analysis:")
print(f"Pooled All Neurons: Pearson correlation = {corr_all_neurons_pooled}, p-value = {p_all_neurons_pooled}")
print(f"High NI Neurons (50%): Mean noise correlation = {high_ni_noise_corr_mean}")
print(f"Low NI Neurons (50%): Mean noise correlation = {low_ni_noise_corr_mean}")
print(f"Top 10% NI Neurons: Mean noise correlation = {top_10_noise_corr_mean}")
print(f"Bottom 90% NI Neurons: Mean noise correlation = {bottom_90_noise_corr_mean}")