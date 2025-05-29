import numpy as np
import matplotlib.pyplot as plt
import os
from neural_data_object import NeuralData
import mat73
from scipy.stats import pearsonr, spearmanr

# Assuming I have already calculated these matrices:
# aligned_rois: the matrix with aligned ROIs across days
# norm_indices_matrix: the matrix with normalization indices across days
# noise_corr_matrix: the matrix with noise correlations across days

# Define your dates
mice = ['42R']
dates = ['072224', '072324', '072424', '072524']

# Create a directory to save plots
save_dir = '/home/maclean/data_proc/figures/ni_noise_corr_analysis'
os.makedirs(save_dir, exist_ok=True)

# Load the previously computed matrices (if necessary for subsequent analyses)
aligned_rois = np.load(os.path.join(save_dir, 'aligned_rois_matrix.npy'))
norm_indices_matrix = np.load(os.path.join(save_dir, 'norm_indices_matrix.npy'))
noise_corr_matrix_opposite_sst = np.load(os.path.join(save_dir, 'noise_corr_matrix_opposite_sst.npy'))
tuning_strength_matrix = np.load(os.path.join(save_dir, 'tuning_strength_matrix.npy'))

# 2. Cell Stability Analysis

def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Calculate variance of normalization indices and noise correlations
ni_variance = np.nanvar(norm_indices_matrix, axis=1)
noise_corr_variance = np.nanvar(noise_corr_matrix_opposite_sst, axis=1)

# Remove outliers from the variance data
ni_variance_no_outliers = remove_outliers(ni_variance)
noise_corr_variance_no_outliers = remove_outliers(noise_corr_variance)

# Plot the variance distributions
plt.figure(figsize=(10, 6))
plt.hist(ni_variance, bins=20, color='blue', edgecolor='black', alpha=0.7, label='Normalization Index Variance')
plt.hist(noise_corr_variance, bins=20, color='red', edgecolor='black', alpha=0.7, label='Noise Correlation Variance')
plt.xlabel('Variance')
plt.ylabel('Number of Cells')
plt.title('Variance in Normalization Index and Noise Correlations Across Days')
plt.legend()
plt.savefig(os.path.join(save_dir, 'ni_and_noise_corr_variance.png'))
plt.show()


# 3. Scatter plot of Normalization Index Variance vs. Noise Correlation Variance
plt.figure(figsize=(10, 6))
plt.scatter(ni_variance, noise_corr_variance, color='purple')
plt.xlabel('Normalization Index Variance')
plt.ylabel('Noise Correlation Variance')
plt.title('Normalization Index Variance vs. Noise Correlation Variance')
plt.savefig(os.path.join(save_dir, 'ni_vs_noise_corr_variance_scatter.png'))
plt.show()

### ADDITIONAL ANALYSIS AND PLOTTING ###

# Function to remove outliers
def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Calculate variance of normalization indices, noise correlations, and tuning strength
ni_variance = np.nanvar(norm_indices_matrix, axis=1)
noise_corr_variance = np.nanvar(noise_corr_matrix_opposite_sst, axis=1)
tuning_strength_variance = np.nanvar(tuning_strength_matrix, axis=1)

# Remove outliers from the variance data
ni_variance_no_outliers = remove_outliers(ni_variance)
noise_corr_variance_no_outliers = remove_outliers(noise_corr_variance)
tuning_strength_variance_no_outliers = remove_outliers(tuning_strength_variance)

# Plot the variance distributions
plt.figure(figsize=(10, 6))
plt.hist(ni_variance_no_outliers, bins=20, color='blue', edgecolor='black', alpha=0.7, label='Normalization Index Variance')
plt.hist(noise_corr_variance_no_outliers, bins=20, color='red', edgecolor='black', alpha=0.7, label='Noise Correlation Variance')
plt.hist(tuning_strength_variance_no_outliers, bins=20, color='green', edgecolor='black', alpha=0.7, label='Tuning Strength Variance')
plt.xlabel('Variance')
plt.ylabel('Number of Cells')
plt.title('Variance in Normalization Index, Noise Correlations, and Tuning Strength Across Days')
plt.legend()
plt.savefig(os.path.join(save_dir, 'ni_noise_corr_tuning_strength_variance.png'))
plt.show()

# Calculate change in noise correlation
change_in_noise_corr_day1_to_day2 = noise_corr_matrix_opposite_sst[:, 1] - noise_corr_matrix_opposite_sst[:, 0]
change_in_noise_corr_day1_to_day3 = noise_corr_matrix_opposite_sst[:, 2] - noise_corr_matrix_opposite_sst[:, 0]
change_in_noise_corr_day1_to_day4 = noise_corr_matrix_opposite_sst[:, 3] - noise_corr_matrix_opposite_sst[:, 0]

# Iterate through each date and create scatter plots for Tuning Strength vs. Normalization Index
for date_idx, date in enumerate(dates):
    # Filter out NaN values for tuning strength and normalization index for the current date
    valid_idx = ~np.isnan(tuning_strength_matrix[:, date_idx]) & ~np.isnan(norm_indices_matrix[:, date_idx])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(tuning_strength_matrix[valid_idx, date_idx], norm_indices_matrix[valid_idx, date_idx], color='blue')
    plt.xlabel('Tuning Strength')
    plt.ylabel('Normalization Index')
    plt.title(f'Tuning Strength vs. Normalization Index on {date}')
    plt.savefig(os.path.join(save_dir, f'tuning_strength_vs_ni_{date}.png'))
    plt.close()

# Iterate through each date and create scatter plots for Tuning Strength vs. Noise Correlations (Opposite-Tuned SST Cells)
for date_idx, date in enumerate(dates):
    # Filter out NaN values for tuning strength and noise correlation for the current date
    valid_idx = ~np.isnan(tuning_strength_matrix[:, date_idx]) & ~np.isnan(noise_corr_matrix_opposite_sst[:, date_idx])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(tuning_strength_matrix[valid_idx, date_idx], noise_corr_matrix_opposite_sst[valid_idx, date_idx], color='green')
    plt.xlabel('Tuning Strength')
    plt.ylabel('Noise Correlation with Opposite-Tuned SST Cells')
    plt.title(f'Tuning Strength vs. Noise Correlation on {date}')
    plt.savefig(os.path.join(save_dir, f'tuning_strength_vs_noise_corr_{date}.png'))
    plt.close()

# Change in Normalization Index vs. Change in Noise Correlation

# Calculate changes in normalization index and noise correlation between days
change_in_ni_day1_to_day2 = norm_indices_matrix[:, 1] - norm_indices_matrix[:, 0]
change_in_ni_day1_to_day3 = norm_indices_matrix[:, 2] - norm_indices_matrix[:, 0]
change_in_ni_day1_to_day4 = norm_indices_matrix[:, 3] - norm_indices_matrix[:, 0]

# Plotting Change in Normalization Index vs. Change in Noise Correlation for each pair of days
plt.figure(figsize=(10, 6))
plt.scatter(change_in_ni_day1_to_day2, change_in_noise_corr_day1_to_day2, color='blue', label='Day 1 to Day 2')
plt.scatter(change_in_ni_day1_to_day3, change_in_noise_corr_day1_to_day3, color='green', label='Day 1 to Day 3')
plt.scatter(change_in_ni_day1_to_day4, change_in_noise_corr_day1_to_day4, color='red', label='Day 1 to Day 4')
plt.xlabel('Change in Normalization Index')
plt.ylabel('Change in Noise Correlation')
plt.title('Change in Normalization Index vs. Change in Noise Correlation')
plt.legend()
plt.savefig(os.path.join(save_dir, 'change_in_ni_vs_change_in_noise_corr.png'))
plt.show()

# Cell Stability Analysis
plt.figure(figsize=(12, 8))
plt.boxplot([ni_variance_no_outliers, tuning_strength_variance_no_outliers, noise_corr_variance_no_outliers],
            labels=['NI Variance', 'Tuning Strength Variance', 'Noise Correlation Variance'])
plt.ylabel('Variance')
plt.title('Cell Stability Analysis Across Days')
plt.savefig(os.path.join(save_dir, 'cell_stability_analysis_no_outliers.png'))
plt.close()

# Additional Plots for Change in Normalization Index vs. Change in Noise Correlation
plt.figure(figsize=(10, 6))
plt.scatter(change_in_ni_day1_to_day2, change_in_noise_corr_day1_to_day2, color='blue')
plt.xlabel('Change in Normalization Index (072224 to 072324)')
plt.ylabel('Change in Noise Correlation (072224 to 072324)')
plt.title('Change in NI vs. Change in Noise Correlation (Day 1 to Day 2)')
plt.savefig(os.path.join(save_dir, 'change_in_ni_vs_change_in_noise_corr_day1_to_day2.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(change_in_ni_day1_to_day3, change_in_noise_corr_day1_to_day3, color='green')
plt.xlabel('Change in Normalization Index (072224 to 072424)')
plt.ylabel('Change in Noise Correlation (072224 to 072424)')
plt.title('Change in NI vs. Change in Noise Correlation (Day 1 to Day 3)')
plt.savefig(os.path.join(save_dir, 'change_in_ni_vs_change_in_noise_corr_day1_to_day3.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(change_in_ni_day1_to_day4, change_in_noise_corr_day1_to_day4, color='red')
plt.xlabel('Change in Normalization Index (072224 to 072524)')
plt.ylabel('Change in Noise Correlation (072224 to 072524)')
plt.title('Change in NI vs. Change in Noise Correlation (Day 1 to Day 4)')
plt.savefig(os.path.join(save_dir, 'change_in_ni_vs_change_in_noise_corr_day1_to_day4.png'))
plt.close()

# Save the aligned ROIs and matrices for further analysis
np.save(os.path.join(save_dir, 'aligned_rois_matrix.npy'), aligned_rois)
np.save(os.path.join(save_dir, 'norm_indices_matrix.npy'), norm_indices_matrix)
np.save(os.path.join(save_dir, 'noise_corr_matrix_opposite_sst.npy'), noise_corr_matrix_opposite_sst)
np.save(os.path.join(save_dir, 'tuning_strength_matrix.npy'), tuning_strength_matrix)