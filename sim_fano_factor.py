import numpy as np
import os

# Simulation parameters for Poisson data
num_rois = 505  # Number of neurons
num_trials = 1248  # Number of trials
num_time_bins = 90  # Time bins per trial
mean_spike_rate = 5  # Average spike rate for Poisson data

pre_stim = (0, 30)
post_stim = (30, 60)

def generate_poisson_data():
    poisson_spikes = np.random.poisson(mean_spike_rate, (num_rois, num_trials, num_time_bins))
    return poisson_spikes

def load_data():
    spikes = generate_poisson_data()
    print("Shape of simulated spikes:", spikes.shape)  # Expected: (neurons, trials, time)
    return spikes

def calculate_fano_factor(spike_counts):
    mean_spikes = np.mean(spike_counts, axis=1)
    variance_spikes = np.var(spike_counts, axis=1)
    mean_spikes[mean_spikes < 0.1] = np.nan  # Avoid division by zero or near-zero

    fano_factor = variance_spikes / mean_spikes
    return fano_factor

def analyze_fano_factor(pre_stim, post_stim):
    spikes = load_data()

    pre_stim_spikes = spikes[:, :, pre_stim[0]:pre_stim[1]].sum(axis=2)
    post_stim_spikes = spikes[:, :, post_stim[0]:post_stim[1]].sum(axis=2)

    results = {
        "Pre_Stimulus": calculate_fano_factor(pre_stim_spikes),
        "Post_Stimulus": calculate_fano_factor(post_stim_spikes)
    }
    return results

fano_factors = analyze_fano_factor(pre_stim, post_stim)

for condition, fano_values in fano_factors.items():
    fano_values = fano_values[~np.isnan(fano_values)]
    fano_range = (np.min(fano_values), np.max(fano_values))
    fano_variance = np.var(fano_values)
    fano_mean = np.mean(fano_values)
    print(f"{condition}: Range: {fano_range}, Variance: {fano_variance:.2f}, Mean: {fano_mean:.2f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.boxplot([fano_factors["Pre_Stimulus"], fano_factors["Post_Stimulus"]],
            labels=["Pre_Stimulus", "Post_Stimulus"], showmeans=True)
plt.title("Fano Factor for Simulated Poisson Data")
plt.ylabel("Fano Factor")
plt.show()