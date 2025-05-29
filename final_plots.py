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
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import ranksums, mannwhitneyu, ks_2samp, pearsonr, spearmanr, linregress, power_divergence, norm,f_oneway, ttest_ind, levene, ks_2samp, bartlett
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
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib
import joypy


save_dir = '/home/maclean/data_proc/noise_corr/finalplots'
os.makedirs(save_dir, exist_ok=True)

matplotlib.rcParams.update({
    "font.family": "Nimbus Sans",
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13
})

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
    "Exc-Exc Plaid": mcolors.to_rgba("#C0C0C0", alpha=0.7),  
    "Exc": mcolors.to_rgba("#C0C0C0", alpha=0.7),  
    "Excitatory NIs": mcolors.to_rgba("#C0C0C0", alpha=0.7), 
    "Excitatory OSIs": mcolors.to_rgba("#C0C0C0", alpha=0.7),   
    "Exc-Exc Grating": mcolors.to_rgba("#2e2e2e", alpha=0.9),  

    "PV-PV Plaid": mcolors.to_rgba("#ffadad", alpha=0.7),  # Light red
    "Exc-PV Plaid": mcolors.to_rgba("#ffb6c1", alpha=0.7),  # Light pink
    "PV": mcolors.to_rgba("#ffadad", alpha=0.7),  # Light red
    "PV NIs": mcolors.to_rgba("#ffadad", alpha=0.7),  # Light red
    "PV OSIs": mcolors.to_rgba("#ffadad", alpha=0.7),  # Light red
    "PV-PV Grating": mcolors.to_rgba("#b91c1c", alpha=0.9),  # Dark red
    "Exc-PV Grating": mcolors.to_rgba("#8B008B", alpha=0.9),  # Dark pink
    "Exc-PV": mcolors.to_rgba("#d81159", alpha=0.9),  # Dark Pink

    "SST-SST Plaid": mcolors.to_rgba("#a0c4ff", alpha=0.7),  # Light blue
    "SST": mcolors.to_rgba("#a0c4ff", alpha=0.7),  # Light blue
    "SST NIs": mcolors.to_rgba("#a0c4ff", alpha=0.7),  # Light blue
    "SST OSIs": mcolors.to_rgba("#a0c4ff", alpha=0.7),  # Light blue
    "Exc-SST Plaid": mcolors.to_rgba("#7fdbff", alpha=0.7),  # Light aqua
    "Exc-SST Grating": mcolors.to_rgba("#008B8B", alpha=0.9),  # Dark aqua
    "SST-SST Grating": mcolors.to_rgba("#1d4ed8", alpha=0.9)  # Dark blue
    
}
csv_path = os.path.join("/home/maclean/data_proc/noise_corr/orientation_segmented_noise_correlations.csv")
csv_path2 = os.path.join("/home/maclean/data_proc/noise_corr/ridge_regression_results.csv")
df = pd.read_csv(csv_path)
df2 = pd.read_csv(csv_path2)
def save_plot(fig, filename):
    """Save figure to disk."""
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

#FIGURE 2b and c (norm and OSI)
def extract_unique_normalization_indices(csv_path):
    """Extracts unique normalization indices from a CSV of neuron pairs and plots distributions."""
    
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Initialize sets for unique normalization indices
    excitatory_nis = set()
    pv_nis = set()
    sst_nis = set()
    excitatory_osis, pv_osis, sst_osis = set(), set(), set()

    # Iterate through rows to extract unique NIs
    for _, row in df.iterrows():
        ni1, ni2 = row["Normalization Index (NI) Neuron1"], row["Normalization Index (NI) Neuron2"]
        osi1, osi2 = row["OSI Neuron1"], row["OSI Neuron2"]

        # Neuron 1 classification
        if row["Neuron1 Type"] == "Excitatory":
            excitatory_nis.add(ni1)
            excitatory_osis.add(osi1)
        elif row["Inhibitory Population"] == "PV":
            pv_nis.add(ni1)
            pv_osis.add(osi1)
        elif row["Inhibitory Population"] == "SST":
            sst_nis.add(ni1)
            sst_osis.add(osi1)

        # Neuron 2 classification
        if row["Neuron2 Type"] == "Excitatory":
            excitatory_nis.add(ni2)
            excitatory_osis.add(osi2)
        elif row["Inhibitory Population"] == "PV":
            pv_nis.add(ni2)
            pv_osis.add(osi2)
        elif row["Inhibitory Population"] == "SST":
            sst_nis.add(ni2)
            sst_osis.add(osi2)

    # Convert sets to NumPy arrays
    excitatory_nis, pv_nis, sst_nis = map(np.array, [list(excitatory_nis), list(pv_nis), list(sst_nis)])
    excitatory_osis, pv_osis, sst_osis = map(np.array, [list(excitatory_osis), list(pv_osis), list(sst_osis)])

    def plot_distribution(data_dict, title, x_label,x_limits = None, y_limits=None, save_name=None):
        fig, axes = plt.subplots(len(data_dict), 1, figsize=(10, 10), sharex=True)
        fig.subplots_adjust(hspace=-0.6)  # 👈 Controls overlap between ridges

        fontname = 'Nimbus Sans'
        label_fontsize = 30
        tick_fontsize = 30
        title_fontsize = 36


        custom_ylabels = {
            "Excitatory NIs": "Excitatory Distribution (a.u.)",
            "PV NIs": "PV Distribution (a.u.)",
            "SST NIs": "SST Distribution (a.u.)",
            "Excitatory OSIs": "Excitatory Distribution (a.u.)",
            "PV OSIs": "PV Distribution (a.u.)",
            "SST OSIs": "SST Distribution (a.u.)"
        }

        for ax, (label, data) in zip(axes, data_dict.items()):
            sns.kdeplot(data, fill=True, alpha=0.7, linewidth=2,
                        color=color_palette[label], ax=ax, bw_adjust=1.2)

            if x_limits:
                ax.set_xlim(x_limits)
            if y_limits:
                ax.set_ylim(y_limits)

            ax.tick_params(axis='y', labelsize=tick_fontsize)
            ax.tick_params(axis='x', labelsize=tick_fontsize)
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)

        # Set shared x-label
        axes[-1].set_xlabel(x_label, fontsize=label_fontsize, fontname=fontname)
        # Set main title
        fig.suptitle(title, fontsize=title_fontsize, fontweight="bold", fontname=fontname)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{save_dir}/{save_name}", dpi=300)
        plt.show()


    # NI plot
    plot_distribution(
        {"Excitatory NIs": excitatory_nis, "PV NIs": pv_nis, "SST NIs": sst_nis},
        title="Normalization Index Distributions", 
        x_label="Normalization Index",
        save_name="ni_joyplot.png",
        x_limits=(-1.0, 1.2),
        y_limits=(0, 2.5)
    )

    # OSI plot
    plot_distribution(
        {"Excitatory OSIs": excitatory_osis, "PV OSIs": pv_osis, "SST OSIs": sst_osis},
        title="Orientation Selectivity Index Distributions", 
        x_label="OSI",
        save_name="osi_joyplot.png",
        x_limits=(-0.4, 1.0),
        y_limits=(0, 6.5)
    )

    # Compute statistics
    def compute_statistics(data1, data2, label1, label2):
        print(f"\n### {label1} vs {label2} ###")
        print(f"Number of data points - {label1}: {len(data1)}, {label2}: {len(data2)}")
        print(f"{label1}: Mean = {np.mean(data1):.4f}, Variance = {np.var(data1, ddof=1):.4f}")
        print(f"{label2}: Mean = {np.mean(data2):.4f}, Variance = {np.var(data2, ddof=1):.4f}")

        # Mann-Whitney U Test
        mwu_stat, mwu_p = mannwhitneyu(data1, data2, alternative='two-sided')
        print(f"Mann-Whitney U Test: Stat = {mwu_stat:.4f}, p = {mwu_p:.4e}")

        # Kolmogorov-Smirnov Test
        ks_stat, ks_p = ks_2samp(data1, data2)
        print(f"Kolmogorov-Smirnov Test: Stat = {ks_stat:.4f}, p = {ks_p:.4e}")

        # Levene's Test
        levene_stat, levene_p = levene(data1, data2)
        print(f"Levene’s Test: Stat = {levene_stat:.4f}, p = {levene_p:.4e}")

        # Bartlett’s Test
        bartlett_stat, bartlett_p = bartlett(data1, data2)
        print(f"Bartlett’s Test: Stat = {bartlett_stat:.4f}, p = {bartlett_p:.4e}")

    # Statistical Comparisons
    print("\n### Normalization Index (NI) Distributions ###")
    compute_statistics(excitatory_nis, pv_nis, "Excitatory NI", "PV NI")
    compute_statistics(excitatory_nis, sst_nis, "Excitatory NI", "SST NI")
    compute_statistics(pv_nis, sst_nis, "PV NI", "SST NI")

    print("\n### Orientation Selectivity Index (OSI) Distributions ###")
    compute_statistics(excitatory_osis, pv_osis, "Excitatory OSI", "PV OSI")
    compute_statistics(excitatory_osis, sst_osis, "Excitatory OSI", "SST OSI")
    compute_statistics(pv_osis, sst_osis, "PV OSI", "SST OSI")

    return excitatory_nis, pv_nis, sst_nis, excitatory_osis, pv_osis, sst_osis
#FIGURE 2d but trying an overlay strat
def extract_fano_factors_with_overlay(csv_path):
    """Extracts unique Fano factors for Grating and Plaid conditions separately and plots distributions."""

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Initialize sets for unique Fano factors
    excitatory_fano_grating = set()
    excitatory_fano_plaid = set()
    pv_fano_grating = set()
    pv_fano_plaid = set()
    sst_fano_grating = set()
    sst_fano_plaid = set()

    # Iterate through rows to extract unique Fano factors
    for _, row in df.iterrows():
        fano1_grating = row["Fano Factor Neuron1"] if row["Trial Type"] == "Grating" else None
        fano2_grating = row["Fano Factor Neuron2"] if row["Trial Type"] == "Grating" else None
        fano1_plaid = row["Fano Factor Neuron1"] if row["Trial Type"] == "Plaid" else None
        fano2_plaid = row["Fano Factor Neuron2"] if row["Trial Type"] == "Plaid" else None

        # Neuron 1 classification
        if row["Neuron1 Type"] == "Excitatory":
            if fano1_grating is not None:
                excitatory_fano_grating.add(fano1_grating)
            if fano1_plaid is not None:
                excitatory_fano_plaid.add(fano1_plaid)
        elif row["Inhibitory Population"] == "PV":
            if fano1_grating is not None:
                pv_fano_grating.add(fano1_grating)
            if fano1_plaid is not None:
                pv_fano_plaid.add(fano1_plaid)
        elif row["Inhibitory Population"] == "SST":
            if fano1_grating is not None:
                sst_fano_grating.add(fano1_grating)
            if fano1_plaid is not None:
                sst_fano_plaid.add(fano1_plaid)

        # Neuron 2 classification
        if row["Neuron2 Type"] == "Excitatory":
            if fano2_grating is not None:
                excitatory_fano_grating.add(fano2_grating)
            if fano2_plaid is not None:
                excitatory_fano_plaid.add(fano2_plaid)
        elif row["Inhibitory Population"] == "PV":
            if fano2_grating is not None:
                pv_fano_grating.add(fano2_grating)
            if fano2_plaid is not None:
                pv_fano_plaid.add(fano2_plaid)
        elif row["Inhibitory Population"] == "SST":
            if fano2_grating is not None:
                sst_fano_grating.add(fano2_grating)
            if fano2_plaid is not None:
                sst_fano_plaid.add(fano2_plaid)
    
    # Initialize sets for unique normalization indices
    excitatory_nis = set()
    pv_nis = set()
    sst_nis = set()

    # Iterate through rows to extract unique NIs
    for _, row in df.iterrows():
        ni1 = row["Normalization Index (NI) Neuron1"]
        ni2 = row["Normalization Index (NI) Neuron2"]

        # Neuron 1 classification
        if row["Neuron1 Type"] == "Excitatory":
            excitatory_nis.add(ni1)
        elif row["Inhibitory Population"] == "PV":
            pv_nis.add(ni1)
        elif row["Inhibitory Population"] == "SST":
            sst_nis.add(ni1)

        # Neuron 2 classification
        if row["Neuron2 Type"] == "Excitatory":
            excitatory_nis.add(ni2)
        elif row["Inhibitory Population"] == "PV":
            pv_nis.add(ni2)
        elif row["Inhibitory Population"] == "SST":
            sst_nis.add(ni2)

    # Convert sets to NumPy arrays
    excitatory_nis = np.array(list(excitatory_nis))
    pv_nis = np.array(list(pv_nis))
    sst_nis = np.array(list(sst_nis))

    # Convert sets to NumPy arrays
    excitatory_fano_grating = np.array(list(excitatory_fano_grating))
    excitatory_fano_plaid = np.array(list(excitatory_fano_plaid))
    pv_fano_grating = np.array(list(pv_fano_grating))
    pv_fano_plaid = np.array(list(pv_fano_plaid))
    sst_fano_grating = np.array(list(sst_fano_grating))
    sst_fano_plaid = np.array(list(sst_fano_plaid))
    exc_fano_diff = np.array(excitatory_fano_plaid) - np.array(excitatory_fano_grating)
    pv_fano_diff = np.array(pv_fano_plaid) - np.array(pv_fano_grating)
    sst_fano_diff = np.array(sst_fano_plaid) - np.array(sst_fano_grating)
    
    """Extracts and overlays Fano factor distributions for Grating and Plaid conditions separately."""
    def plot_fano_distribution(title, x_label, save_name=None):
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.subplots_adjust(hspace=0.3) 
        
        fontname = 'Nimbus Sans'
        label_fontsize = 18
        tick_fontsize = 30
        title_fontsize = 24
        
        x_limits = (-2, 8)
        y_limits = (0, 0.6)  # Standard y-limits across all plots

        populations = {
            "Excitatory": (excitatory_fano_grating, excitatory_fano_plaid, "Exc-Exc Grating", "Exc-Exc Plaid"),
            "PV": (pv_fano_grating, pv_fano_plaid, "PV-PV Grating", "PV-PV Plaid"),
            "SST": (sst_fano_grating, sst_fano_plaid, "SST-SST Grating", "SST-SST Plaid")
        }

        custom_ylabels = {
            "Excitatory": "Exc ",
            "PV": "PV",
            "SST": "SST"
        }

        for ax, (label, (grating_data, plaid_data, grating_key, plaid_key)) in zip(axes, populations.items()):
            sns.kdeplot(grating_data, fill=True, alpha=0.8, linewidth=2, color=color_palette[grating_key],
                        ax=ax, bw_adjust=1.2, label="Grating")
            sns.kdeplot(plaid_data, fill=True, alpha=0.6, linewidth=2, color=color_palette[plaid_key],
                        ax=ax, bw_adjust=1.2, label="Plaid")

            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.set_ylabel(custom_ylabels[label], rotation=0, labelpad=60, fontsize=30, va='center', fontname=fontname)
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontname(fontname)

            ax.set_xlabel(x_label, fontsize=30, fontname=fontname)
            ax.legend()

        fig.suptitle(title, fontsize=36, fontname=fontname, fontweight="bold", y=0.98)
        plt.savefig(f"{save_dir}/{save_name}", dpi=300)
        plt.show()

    plot_fano_distribution(
        title="Fano Factor Distributions (Grating vs. Plaid)",
        x_label="Fano Factor",
        save_name="fano_factor_ridges_combined.png"
    )

    # **Run Statistics**
    def compute_statistics(data1, data2, label1, label2):
        print(f"\n### {label1} vs {label2} ###")
        print(f"Number of data points - {label1}: {len(data1)}, {label2}: {len(data2)}")  # New line

        print(f"{label1}: Mean = {np.mean(data1):.4f}, Variance = {np.var(data1, ddof=1):.4f}")
        print(f"{label2}: Mean = {np.mean(data2):.4f}, Variance = {np.var(data2, ddof=1):.4f}")

        # Mann-Whitney U Test (Non-parametric)
        mwu_stat, mwu_p = mannwhitneyu(data1, data2, alternative='two-sided')
        print(f"Mann-Whitney U Test: Stat = {mwu_stat:.4f}, p = {mwu_p:.4e}")

        # Kolmogorov-Smirnov Test (Distribution Shape)
        ks_stat, ks_p = ks_2samp(data1, data2)
        print(f"Kolmogorov-Smirnov Test: Stat = {ks_stat:.4f}, p = {ks_p:.4e}")

        # Levene's Test (Variance Comparison)
        levene_stat, levene_p = levene(data1, data2)
        print(f"Levene’s Test: Stat = {levene_stat:.4f}, p = {levene_p:.4e}")

        # Bartlett’s Test (Variance Homogeneity)
        bartlett_stat, bartlett_p = bartlett(data1, data2)
        print(f"Bartlett’s Test: Stat = {bartlett_stat:.4f}, p = {bartlett_p:.4e}")

    print("\n### Fano Factor Distributions (Plaid vs Grating) ###")
    compute_statistics(excitatory_fano_plaid, excitatory_fano_grating, "Excitatory Plaid", "Excitatory Grating")
    compute_statistics(pv_fano_plaid, pv_fano_grating, "PV Plaid", "PV Grating")
    compute_statistics(sst_fano_plaid, sst_fano_grating, "SST Plaid", "SST Grating")

    print("\n### Fano Factor Distributions (Across Cell Types) ###")
    compute_statistics(excitatory_fano_plaid, pv_fano_plaid, "Excitatory Plaid", "PV Plaid")
    compute_statistics(excitatory_fano_plaid, sst_fano_plaid, "Excitatory Plaid", "SST Plaid")
    compute_statistics(pv_fano_plaid, sst_fano_plaid, "PV Plaid", "SST Plaid")
    compute_statistics(excitatory_fano_grating, pv_fano_grating, "Excitatory Grating", "PV Grating")
    compute_statistics(excitatory_fano_grating, sst_fano_grating, "Excitatory Grating", "SST Grating")
    compute_statistics(pv_fano_grating, sst_fano_grating, "PV Grating", "SST Grating")

    return excitatory_fano_grating, excitatory_fano_plaid, pv_fano_grating, pv_fano_plaid, sst_fano_grating, sst_fano_plaid

#FIGURE 3 but exc-exc only -- absorbed this into the next loop
def plot_exc_exc_noise_ridgeline(csv_path):
    """Plots a ridgeline KDE of Exc-Exc noise correlations for Plaid vs. Grating with Nature-style formatting."""

    # Load CSV
    df = pd.read_csv(csv_path)

    # Extract Exc-Exc Noise Correlations
    grating_corrs = df[df["Trial Type"] == "Grating"]["Exc-Exc Noise Correlation"].dropna()
    plaid_corrs = df[df["Trial Type"] == "Plaid"]["Exc-Exc Noise Correlation"].dropna()

    # Check for missing data
    if grating_corrs.empty or plaid_corrs.empty:
        print("⚠ Warning: Missing data for Grating or Plaid!")
        return

    # Prepare data for seaborn ridgeline plot
    data = pd.DataFrame({
        "Noise Correlation": np.concatenate([grating_corrs, plaid_corrs]),
        "Condition": ["Grating"] * len(grating_corrs) + ["Plaid"] * len(plaid_corrs)
    })
    
    # Define custom colors
    plaid_color = "#006400"  # Dark Green
    grating_color = "#90EE90"  # Light Green

    # Ridgeline plot setup
    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        data=data, x="Noise Correlation", hue="Condition", fill=True, 
        alpha=0.7, common_norm=False, linewidth=1.5,
        palette={"Grating": grating_color, "Plaid": plaid_color}  # Assign colors
    )

    # Formatting
    plt.xlabel("Exc-Exc Noise Correlation", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title("Comparison of Exc-Exc Noise Correlations: Grating vs. Plaid", fontsize=16, fontweight="bold")
    plt.axvline(0, color='black', linestyle='--', alpha=0.7, linewidth=1)  # Reference line at 0

    # Custom Legend with Colored Lines
    plaid_patch = mpatches.Patch(color=plaid_color, label="Plaid")
    grating_patch = mpatches.Patch(color=grating_color, label="Grating")
    plt.legend(handles=[plaid_patch, grating_patch], title="Condition", title_fontsize=14, loc="upper left", frameon=False)

    # Remove unnecessary spines (Nature-style formatting)
    sns.despine()

    # Show the plot
    plt.show()
#FIGURE 3 all (a,b,c,d,e) baller awesome works
def plot_joyplot_noise_correlations(csv_path, save_dir):
    """Plots ridgeline KDEs (joyplots) for PV-PV, Exc-PV, SST-SST, and Exc-SST comparing Plaid vs. Grating."""
    
    # Load CSV
    df = pd.read_csv(csv_path)

    # Separate into plaid and grating dataframes
    plaid_df = df[df["Trial Type"] == "Plaid"]
    grating_df = df[df["Trial Type"] == "Grating"]



    # Extract noise correlation groups
    noise_corr_groups = {
        "Exc-Exc Plaid": plaid_df.loc[plaid_df["Neuron1 Type"] == "Excitatory", "Exc-Exc Noise Correlation"].dropna(),
        "Exc-Exc Grating": grating_df.loc[grating_df["Neuron1 Type"] == "Excitatory", "Exc-Exc Noise Correlation"].dropna(),
        "PV-PV Plaid": plaid_df.loc[plaid_df["Inhibitory Population"] == "PV", "Red-Red Noise Correlation"].dropna(),
        "Exc-PV Plaid": plaid_df.loc[plaid_df["Inhibitory Population"] == "PV", "Exc-Red Noise Correlation"].dropna(),
        "PV-PV Grating": grating_df.loc[grating_df["Inhibitory Population"] == "PV", "Red-Red Noise Correlation"].dropna(),
        "Exc-PV Grating": grating_df.loc[grating_df["Inhibitory Population"] == "PV", "Exc-Red Noise Correlation"].dropna(),
        "SST-SST Plaid": plaid_df.loc[plaid_df["Inhibitory Population"] == "SST", "Red-Red Noise Correlation"].dropna(),
        "Exc-SST Plaid": plaid_df.loc[plaid_df["Inhibitory Population"] == "SST", "Exc-Red Noise Correlation"].dropna(),
        "SST-SST Grating": grating_df.loc[grating_df["Inhibitory Population"] == "SST", "Red-Red Noise Correlation"].dropna(),
        "Exc-SST Grating": grating_df.loc[grating_df["Inhibitory Population"] == "SST", "Exc-Red Noise Correlation"].dropna()
    }



    # Compute average Fano Factor per neuron pair
    df["Avg Fano Factor"] = (df["Fano Factor Neuron1"] + df["Fano Factor Neuron2"]) / 2

    # Combine into a single dataframe for JoyPy
    joy_df = pd.DataFrame([(key, val) for key, values in noise_corr_groups.items() for val in values], columns=["Condition", "Noise Correlation"])


    # Define color mapping
    kde_colors = {
  
        "PV-PV Grating": "#b91c1c",  # Dark magenta
        "PV-PV Plaid": "#ffadad",  # Light pink
        "Exc-PV Grating": "#7f1d1d",  # Dark red
        "Exc-PV Plaid": "#fca5a5",  # Light red


        "Exc-SST Grating": "#000080", 
        "Exc-SST Plaid": "#6495ED",  
        "SST-SST Grating": "#4169E1",   
        "SST-SST Plaid": "#CDEDF6",  
        "Exc-Exc Grating": "#D3D3D3",
        "Exc-Exc Plaid": "#4B4B4B"
    }
    # Extract Fano Factor vs Noise Correlation Data
    fano_corr_groups = {
        "E-I PV": {"data": df[df["Inhibitory Population"] == "PV"], "x": "Exc-Red Noise Correlation", "color": kde_colors["Exc-PV Plaid"]},
        "I-I PV": {"data": df[df["Inhibitory Population"] == "PV"], "x": "Red-Red Noise Correlation", "color": kde_colors["PV-PV Plaid"]},
        "E-I SST": {"data": df[df["Inhibitory Population"] == "SST"], "x": "Exc-Red Noise Correlation", "color": kde_colors["Exc-SST Plaid"]},
        "I-I SST": {"data": df[df["Inhibitory Population"] == "SST"], "x": "Red-Red Noise Correlation", "color": kde_colors["SST-SST Plaid"]}
    }

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Nimbus Sans'],  # 👈 Will fall back if Nimbus not installed
        'font.size': 18,                     # 👈 Default font size for ticks, labels, legend
        'axes.titlesize': 24,               # 👈 Title font size
        'axes.labelsize': 18,               # 👈 X and Y axis label font size
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'legend.title_fontsize': 18
    })

    # Create joyplot for PV-related connections
    plt.figure(figsize=(10, 8))
    fig, axes = joypy.joyplot(
        data=joy_df[joy_df["Condition"].str.contains("PV")], 
        by="Condition", 
        column="Noise Correlation",
        overlap=2, 
        linewidth=1.2, 
        color=[kde_colors[label] for label in noise_corr_groups.keys() if "PV" in label],
        fade=True
    )
    plt.xlabel("Noise Correlation")
    plt.title("PV-PV & Exc-PV Noise Correlations (Plaid vs. Grating)", fontsize=14, fontweight="bold")
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.subplots_adjust(top=0.9, bottom=0.15)
    plt.savefig(f"{save_dir}/pv_noise_correlation_joyplot.png", dpi=300)
    plt.show()

    # Create joyplot for SST-related connections
    plt.figure(figsize=(10, 8))
    fig, axes = joypy.joyplot(
        data=joy_df[joy_df["Condition"].str.contains("SST")], 
        by="Condition", 
        column="Noise Correlation",
        overlap=2, 
        linewidth=1.2, 
        color=[kde_colors[label] for label in noise_corr_groups.keys() if "SST" in label],
        fade=True
    )
    plt.xlabel("Noise Correlation")
    plt.title("SST-SST & Exc-SST Noise Correlations (Plaid vs. Grating)", fontsize=14, fontweight="bold")
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.subplots_adjust(top=0.9, bottom=0.15, right=0.95)
    plt.savefig(f"{save_dir}/sst_noise_correlation_joyplot.png", dpi=300)
    plt.show()

    # Create joyplot for Exc-Exc Noise Correlations
    plt.figure(figsize=(10, 8))
    fig, axes = joypy.joyplot(
        data=joy_df[joy_df["Condition"].str.contains("Exc-Exc")], 
        by="Condition", 
        column="Noise Correlation",
        overlap=4, 
        linewidth=1.2, 
        color=[kde_colors["Exc-Exc Plaid"], kde_colors["Exc-Exc Grating"]],
        fade=True
    )
    plt.xlabel("Noise Correlation")
    plt.title("Exc-Exc Noise Correlations (Plaid vs. Grating)", fontsize=14, fontweight="bold")
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.subplots_adjust(top=0.9, bottom=0.15)
    plt.savefig(f"{save_dir}/exc_exc_noise_correlation_joyplot.png", dpi=300)
    plt.show()

    print("✅ Joyplots for Noise Correlations Saved.")

    # Create scatter plot for Fano Factor vs Noise Correlation (Grating)
    regression_colors = {
        "I-I PV": "#8B0000",     # Dark red
        "E-I PV": "#FF9999",     # Light pink
        "I-I SST": "#000080",    # Navy
        "E-I SST": "#ADD8E6",    # Light blue
    }

    # Create scatter plot for Fano Factor vs Noise Correlation (Grating)
    plt.rcParams["font.family"] = "Nimbus Sans"

    # Define trial and population mapping
    plot_order = [
        ("Plaid", "PV"),
        ("Plaid", "SST"),
        ("Grating", "PV"),
        ("Grating", "SST")
    ]

    for trial_type, population in plot_order:
        plt.figure(figsize=(8, 6))
        
        for label, info in fano_corr_groups.items():
            if population in label:
                subset = info["data"][info["data"]["Trial Type"] == trial_type].copy()
                subset = subset[[info["x"], "Avg Fano Factor"]].dropna()
                subset = subset[np.isfinite(subset[info["x"]]) & np.isfinite(subset["Avg Fano Factor"])]
                
                if not subset.empty:
                    # Regression
                    X = sm.add_constant(subset[info["x"]])
                    y = subset["Avg Fano Factor"]
                    model = sm.OLS(y, X).fit()
                    
                    beta = model.params[1]
                    r_squared = model.rsquared
                    p_value = model.pvalues[1]
                    
                    print(f"\n--- Regression Results: {label} ({trial_type}) ---")
                    print(f"β (Slope): {beta:.4f}")
                    print(f"R²: {r_squared:.4f}")
                    print(f"P-value: {p_value:.4e}")

                    sns.regplot(
                        x=subset[info["x"]],
                        y=subset["Avg Fano Factor"],
                        scatter=False,
                        color="black",
                        line_kws={"linewidth": 5.5, "alpha": 1.0},
                        truncate=False,
                        ci=None
                    )
                    # Plot
                    sns.regplot(
                        x=subset[info["x"]],
                        y=subset["Avg Fano Factor"],
                        scatter=True,
                        color=regression_colors[label],
                        label=f"{label} (β={beta:.2f}, P-value: {p_value:.4e})",
                        scatter_kws={'alpha': 0.4},
                        line_kws={"color": regression_colors[label], "linewidth": 4, "alpha": 1.0, "linestyle": "solid"},
                        truncate=False
                    )
        plt.xlim(-0.5, 1.1)
        plt.ylim(0, 5)
        plt.xlabel("Noise Correlation", fontsize=14)
        plt.ylabel("Average Fano Factor", fontsize=14)
        plt.title(f"Fano Factor vs. Noise Correlation ({trial_type} - {population})", fontsize=16, fontweight="bold")
        plt.legend(title="Connection Type", fontsize=12)
        plt.axvline(0, color="gray", linestyle="--", linewidth=1)
        plt.grid(False)
        plt.savefig(f"{save_dir}/fano_vs_noise_{trial_type.lower()}_{population.lower()}.png", dpi=300)
        plt.show()



    """sns.violinplot(
        data=joy_df, x="Condition", y="Noise Correlation", 
        palette=kde_colors, inner=None, alpha=0.7
    )
    sns.stripplot(
        data=joy_df, x="Condition", y="Noise Correlation", 
        jitter=True, alpha=0.5, color="black", size=2
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Noise Correlation")
    plt.xlabel("Condition")
    plt.title("Distribution of Noise Correlations with Individual Data Points")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(f"{save_dir}/violin_scatter_noise.png", dpi=300)
    plt.show()"""

    def compute_statistics(data1, data2, label1, label2):
        """Computes and prints statistical comparisons between two datasets."""
        print(f"\n### {label1} vs {label2} ###")
        print(f"{label1}: Mean = {np.mean(data1):.4f}, Variance = {np.var(data1, ddof=1):.4f}, Sample Size = {len(data1)}")
        print(f"{label2}: Mean = {np.mean(data2):.4f}, Variance = {np.var(data2, ddof=1):.4f}, Sample Size = {len(data2)}")

        # Mann-Whitney U Test
        mwu_stat, mwu_p = mannwhitneyu(data1, data2, alternative='two-sided')
        print(f"Mann-Whitney U Test: Stat = {mwu_stat:.4f}, p = {mwu_p:.4e}")

        # Kolmogorov-Smirnov Test
        ks_stat, ks_p = ks_2samp(data1, data2)
        print(f"Kolmogorov-Smirnov Test: Stat = {ks_stat:.4f}, p = {ks_p:.4e}")

        # Levene's Test (Variance Comparison)
        levene_stat, levene_p = levene(data1, data2)
        print(f"Levene’s Test: Stat = {levene_stat:.4f}, p = {levene_p:.4e}")

        # Bartlett’s Test (Variance Homogeneity)
        bartlett_stat, bartlett_p = bartlett(data1, data2)
        print(f"Bartlett’s Test: Stat = {bartlett_stat:.4f}, p = {bartlett_p:.4e}")

        # Compute Cohen’s d (Effect Size)
        pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
        cohen_d = (np.mean(data1) - np.mean(data2)) / pooled_std
        print(f"Effect Size (Cohen’s d): {cohen_d:.4f}")

        # Power Analysis
        es = abs(cohen_d)  # Effect size for power calculation
        power_analysis = TTestIndPower()
        power = power_analysis.solve_power(effect_size=es, nobs1=len(data1), ratio=len(data2)/len(data1), alpha=0.05)
        print(f"Power Analysis: {power:.3f} (1-β)")


    ### **Statistical Comparisons for Plaid Conditions**
    print("\n### Noise Correlations (Plaid Conditions) ###")
    compute_statistics(noise_corr_groups["Exc-Exc Plaid"], noise_corr_groups["Exc-PV Plaid"], "E-E Plaid", "E-PV Plaid")
    compute_statistics(noise_corr_groups["Exc-Exc Plaid"], noise_corr_groups["Exc-SST Plaid"], "E-E Plaid", "E-SST Plaid")
    compute_statistics(noise_corr_groups["Exc-PV Plaid"], noise_corr_groups["Exc-SST Plaid"], "E-PV Plaid", "E-SST Plaid")
    compute_statistics(noise_corr_groups["Exc-Exc Plaid"], noise_corr_groups["PV-PV Plaid"], "E-E Plaid", "PV-PV Plaid")
    compute_statistics(noise_corr_groups["Exc-Exc Plaid"], noise_corr_groups["SST-SST Plaid"], "E-E Plaid", "SST-SST Plaid")
    compute_statistics(noise_corr_groups["PV-PV Plaid"], noise_corr_groups["SST-SST Plaid"], "PV-PV Plaid", "SST-SST Plaid")

    ### **Statistical Comparisons for Grating Conditions**
    print("\n### Noise Correlations (Grating Conditions) ###")
    compute_statistics(noise_corr_groups["Exc-Exc Grating"], noise_corr_groups["Exc-PV Grating"], "E-E Grating", "E-PV Grating")
    compute_statistics(noise_corr_groups["Exc-Exc Grating"], noise_corr_groups["Exc-SST Grating"], "E-E Grating", "E-SST Grating")
    compute_statistics(noise_corr_groups["Exc-PV Grating"], noise_corr_groups["Exc-SST Grating"], "E-PV Grating", "E-SST Grating")
    compute_statistics(noise_corr_groups["Exc-Exc Grating"], noise_corr_groups["PV-PV Grating"], "E-E Grating", "PV-PV Grating")
    compute_statistics(noise_corr_groups["Exc-Exc Grating"], noise_corr_groups["SST-SST Grating"], "E-E Grating", "SST-SST Grating")
    compute_statistics(noise_corr_groups["PV-PV Grating"], noise_corr_groups["SST-SST Grating"], "PV-PV Grating", "SST-SST Grating")

    ### **Population-Level Comparisons (Plaid vs. Grating)**
    print("\n### Population-Level Noise Correlation Comparisons (Plaid vs. Grating) ###")
    compute_statistics(noise_corr_groups["Exc-Exc Plaid"], noise_corr_groups["Exc-Exc Grating"], "E-E Plaid", "E-E Grating")
    compute_statistics(noise_corr_groups["Exc-PV Plaid"], noise_corr_groups["Exc-PV Grating"], "E-PV Plaid", "E-PV Grating")
    compute_statistics(noise_corr_groups["Exc-SST Plaid"], noise_corr_groups["Exc-SST Grating"], "E-SST Plaid", "E-SST Grating")
    compute_statistics(noise_corr_groups["PV-PV Plaid"], noise_corr_groups["PV-PV Grating"], "PV-PV Plaid", "PV-PV Grating")
    compute_statistics(noise_corr_groups["SST-SST Plaid"], noise_corr_groups["SST-SST Grating"], "SST-SST Plaid", "SST-SST Grating")

    print("\n✅ Statistical comparisons complete.")

    connection_types = ["E-E", "E-PV", "E-SST", "PV-PV", "SST-SST"]
    grating_means = [
        noise_corr_groups["Exc-Exc Grating"].mean(),
        noise_corr_groups["Exc-PV Grating"].mean(),
        noise_corr_groups["Exc-SST Grating"].mean(),
        noise_corr_groups["PV-PV Grating"].mean(),
        noise_corr_groups["SST-SST Grating"].mean()
    ]
    plaid_means = [
        noise_corr_groups["Exc-Exc Plaid"].mean(),
        noise_corr_groups["Exc-PV Plaid"].mean(),
        noise_corr_groups["Exc-SST Plaid"].mean(),
        noise_corr_groups["PV-PV Plaid"].mean(),
        noise_corr_groups["SST-SST Plaid"].mean()
    ]

    x = np.arange(len(connection_types))  # label locations
    width = 0.35  # bar width

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, grating_means, width, label='Grating', color='white', edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, plaid_means, width, label='Plaid', color='black')

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Nimbus Sans']})

    # Axis labels and formatting
    ax.set_ylabel('Mean Noise Correlation', fontsize=16)
    ax.set_xlabel('Connection Type', fontsize=16)
    ax.set_title('Mean Noise Correlations by Connection Type and Stimulus', fontsize=24, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(connection_types, fontsize=16)
    ax.legend(fontsize=16)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/barplot_mean_noise_correlations.png", dpi=300)
    plt.show()

#FIGURE 4
def plot_bar_charts_by_type():
    # Load the results from the CSV
    results_path = csv_path
    results_df = pd.read_csv(results_path)

    # Define the noise correlation types you want to analyze
    noise_corr_types = {
        "Exc-Exc": "Exc-Exc Noise Correlation",
        "Exc-Red PV": "Exc-Red Noise Correlation",
        "Exc-Red SST": "Exc-Red Noise Correlation",
        "Red-Red PV": "Red-Red Noise Correlation",
        "Red-Red SST": "Red-Red Noise Correlation",
    }

    # Define mapping of angular difference groups
    group_mapping = {
        "1": [0.0],
        "2/8": [22.5, 157.5],
        "3/7": [45.0, 135.0],
        "4/6": [67.5, 112.5],
        "5": [90.0],
    }

    group_angles = {
        "1": 0.0,
        "2/8": 22.5,
        "3/7": 45.0,
        "4/6": 67.5,
        "5": 90.0,
    }

    # Iterate over noise correlation types and generate bar plots
    for nc_type, column in noise_corr_types.items():
        print(f"Processing: {nc_type}")


        # Filter dataframe based on inhibitory population if needed
        if "PV" in nc_type:
            df_filtered = results_df[results_df["Inhibitory Population"] == "PV"]
            df_filtered = df_filtered[df_filtered[column] != 1]  # Exclude noise correlations == 1
        elif "SST" in nc_type:
            df_filtered = results_df[results_df["Inhibitory Population"] == "SST"]
            df_filtered = df_filtered[df_filtered[column] != 1]  # Exclude noise correlations == 1
        else:
            df_filtered = results_df.copy()  # Exc-Exc case, take all
            df_filtered = df_filtered[df_filtered[column] != 1]  # Exclude noise correlations == 1

        # Split by Trial Type
        plaid_grouped_data = {group: [] for group in group_mapping.keys()}
        grating_grouped_data = {group: [] for group in group_mapping.keys()}

        for group, angles in group_mapping.items():
            plaid_grouped_data[group] = df_filtered.loc[
                (df_filtered["Angular Difference"].isin(angles)) & (df_filtered["Trial Type"] == "Plaid"),
                column
            ].dropna().values.tolist()

            grating_grouped_data[group] = df_filtered.loc[
                (df_filtered["Angular Difference"].isin(angles)) & (df_filtered["Trial Type"] == "Grating"),
                column
            ].dropna().values.tolist()

        # Compute means and SEMs
        plaid_means, plaid_errors, plaid_angles = [], [], []
        grating_means, grating_errors, grating_angles = [], [], []

        for group, data in plaid_grouped_data.items():
            if len(data) > 0:
                plaid_angles.append(group_angles[group])
                plaid_means.append(np.mean(data))
                plaid_errors.append(sem(data))

        for group, data in grating_grouped_data.items():
            if len(data) > 0:
                grating_angles.append(group_angles[group])
                grating_means.append(np.mean(data))
                grating_errors.append(sem(data))

        # Create the bar plot
        plt.figure(figsize=(8, 6))

        x_labels = [group_angles[key] for key in group_mapping.keys()]

        bar_width = 4  # Adjust bar width
        x_positions = np.array(x_labels)  # Use actual degree values for x-axis
        spacing = 3  # Adjust spacing between bars

        plaid_positions = x_positions - spacing / 2  # Shift plaid left
        grating_positions = x_positions + spacing / 2  # Shift grating right

        color_mapping = {
            "Exc-Exc": ("#99d98c", "#386641"),  # Light green (Plaid), Dark green (Grating)
            "Exc-Red PV": ("#ffb6c1", "#8B008B"),  # Light pink (Plaid), Dark pink (Grating)
            "Red-Red PV": ("#ffadad", "#b91c1c"),  # Light red (Plaid), Dark red (Grating)
            "Exc-Red SST": ("#7fdbff", "#008B8B"),  # Light aqua (Plaid), Dark aqua (Grating)
            "Red-Red SST": ("#a0c4ff", "#1d4ed8")  # Light blue (Plaid), Dark blue (Grating)
        }

        plaid_color, grating_color = color_mapping.get(nc_type, ("gray", "black"))  # Default to gray/black


        plt.bar(plaid_positions, plaid_means, yerr=plaid_errors, capsize=5, 
                color=plaid_color, alpha=0.7, width=bar_width, label="Plaid")
        plt.bar(grating_positions, grating_means, yerr=grating_errors, capsize=5, 
                color=grating_color, alpha=0.7, width=bar_width, label="Grating")

        plt.xticks(ticks=x_labels, labels=[str(int(angle)) for angle in x_labels])
        plt.xlabel("Angular Difference in Tuning")
        plt.ylabel("Mean Noise Correlation")
        plt.title(f"Bar Plot: {nc_type} Noise Correlations (Plaid vs Grating)")
        plt.legend()
        plt.grid(True, axis="y")
        plt.ylim(-0.04, 0.2)  # Set Y-axis range for all plots

        # Save the plot
        plt.savefig(os.path.join(save_dir, f"bar_plot_{nc_type.replace(' ', '_').lower()}.png"))
        plt.close()

        # Define the five conditions
        bar_labels = ["Exc-Exc", "Exc-Red PV", "Exc-Red SST", "Red-Red PV", "Red-Red SST"]
        bar_colors = ["#4B4B4B", "#ffb6c1", "#7fdbff", "#ffadad", "#a0c4ff"]
        bar_width = 2  # Narrower bars for spacing
        spacing = 2  # Adjust for better visualization

        # Create PLAID bar plot
        plt.figure(figsize=(8, 6))

        x_labels = [group_angles[key] for key in group_mapping.keys()]
        x_positions = np.array(x_labels)

        for i, (label, color) in enumerate(zip(bar_labels, bar_colors)):
            bar_means = [
                np.mean(plaid_grouped_data["1"]),  # Exc-Exc (use PV dataset)
                np.mean(plaid_grouped_data["Exc-Red PV"]),
                np.mean(plaid_grouped_data["Exc-Red SST"]),
                np.mean(plaid_grouped_data["Red-Red PV"]),
                np.mean(plaid_grouped_data["Red-Red SST"]),
            ]

            bar_errors = [
                sem(plaid_grouped_data["1"]) if len(plaid_grouped_data["1"]) > 1 else 0,
                sem(plaid_grouped_data["Exc-Red PV"]) if len(plaid_grouped_data["Exc-Red PV"]) > 1 else 0,
                sem(plaid_grouped_data["Exc-Red SST"]) if len(plaid_grouped_data["Exc-Red SST"]) > 1 else 0,
                sem(plaid_grouped_data["Red-Red PV"]) if len(plaid_grouped_data["Red-Red PV"]) > 1 else 0,
                sem(plaid_grouped_data["Red-Red SST"]) if len(plaid_grouped_data["Red-Red SST"]) > 1 else 0,
            ]
            plt.bar(x_positions + (i - 2) * spacing, bar_means, yerr=bar_errors, capsize=5, 
                    color=color, alpha=0.7, width=bar_width, label=label)

        plt.xticks(ticks=x_labels, labels=[str(int(angle)) for angle in x_labels])
        plt.xlabel("Angular Difference in Tuning")
        plt.ylabel("Mean Noise Correlation")
        plt.title("Bar Plot: Noise Correlations (Plaid)")
        plt.legend()
        plt.grid(True, axis="y")
        plt.ylim(-0.04, 0.2)  # Standardized y-axis range
        plt.savefig(os.path.join(save_dir, "bar_plot_noise_correlations_plaid.png"))
        plt.close()

        # Create GRATING bar plot
        plt.figure(figsize=(10, 6))

        for i, (label, color) in enumerate(zip(bar_labels, bar_colors)):
            bar_means = [
                np.mean(grating_grouped_data["1"]),  # Exc-Exc (use PV dataset)
                np.mean(grating_grouped_data["Exc-Red PV"]),
                np.mean(grating_grouped_data["Exc-Red SST"]),
                np.mean(grating_grouped_data["Red-Red PV"]),
                np.mean(grating_grouped_data["Red-Red SST"]),
            ]

            bar_errors = [
                sem(grating_grouped_data["1"]) if len(grating_grouped_data["1"]) > 1 else 0,
                sem(grating_grouped_data["Exc-Red PV"]) if len(grating_grouped_data["Exc-Red PV"]) > 1 else 0,
                sem(grating_grouped_data["Exc-Red SST"]) if len(grating_grouped_data["Exc-Red SST"]) > 1 else 0,
                sem(grating_grouped_data["Red-Red PV"]) if len(grating_grouped_data["Red-Red PV"]) > 1 else 0,
                sem(grating_grouped_data["Red-Red SST"]) if len(grating_grouped_data["Red-Red SST"]) > 1 else 0,
            ]
            plt.bar(x_positions + (i - 2) * spacing, bar_means, yerr=bar_errors, capsize=5, 
                    color=color, alpha=0.7, width=bar_width, label=label)

        plt.xticks(ticks=x_labels, labels=[str(int(angle)) for angle in x_labels])
        plt.xlabel("Angular Difference in Tuning")
        plt.ylabel("Mean Noise Correlation")
        plt.title("Bar Plot: Noise Correlations (Grating)")
        plt.legend()
        plt.grid(True, axis="y")
        plt.ylim(-0.04, 0.2)  # Standardized y-axis range
        plt.savefig(os.path.join(save_dir, "bar_plot_noise_correlations_grating.png"))
        plt.close()

    print("\n✅ All bar plots generated and saved.")

    def fit_models(x, y, label):
        """Fits linear, quadratic, and exponential models and compares fits."""
        # Linear Fit
        X_lin = sm.add_constant(x)
        model_lin = sm.OLS(y, X_lin).fit()
    
        # Quadratic Fit
        X_quad = np.column_stack((x, x**2))
        X_quad = sm.add_constant(X_quad)
        model_quad = sm.OLS(y, X_quad).fit()

        # Exponential Fit (Convert to log scale)
        y_log = np.log(y + 1e-8)  # Avoid log(0)
        model_exp = sm.OLS(y_log, X_lin).fit()

        # Print results
        print(f"\n=== Model Fits for {label} ===")
        print(f"Linear: R² = {model_lin.rsquared:.4f}, p = {model_lin.pvalues[1]:.4e}")
        print(f"Quadratic: R² = {model_quad.rsquared:.4f}, p = {model_quad.pvalues[2]:.4e}")
        print(f"Exponential: R² = {model_exp.rsquared:.4f}, p = {model_exp.pvalues[1]:.4e}")

        # Compare fits
        best_fit = "Linear"
        best_r2 = model_lin.rsquared
        if model_quad.rsquared > best_r2:
            best_fit = "Quadratic"
            best_r2 = model_quad.rsquared
        if model_exp.rsquared > best_r2:
            best_fit = "Exponential"

        print(f"Best Fit: {best_fit} (R² = {best_r2:.4f})")

        return best_fit
    def compute_statistics(data1, data2, label1, label2):
        """Computes and prints statistical comparisons between two datasets."""
        print(f"\n### {label1} vs {label2} ###")
        print(f"{label1}: Mean = {np.mean(data1):.4f}, SEM = {np.std(data1, ddof=1)/np.sqrt(len(data1)):.4f}, Variance = {np.var(data1, ddof=1):.4f}, Sample Size = {len(data1)}")
        print(f"{label2}: Mean = {np.mean(data2):.4f}, SEM = {np.std(data2, ddof=1)/np.sqrt(len(data2)):.4f}, Variance = {np.var(data2, ddof=1):.4f}, Sample Size = {len(data2)}")

        # Mann-Whitney U Test (non-parametric comparison)
        mwu_stat, mwu_p = mannwhitneyu(data1, data2, alternative='two-sided')
        print(f"Mann-Whitney U Test: Stat = {mwu_stat:.4f}, p = {mwu_p:.4e}")

        # Kolmogorov-Smirnov Test (distribution comparison)
        ks_stat, ks_p = ks_2samp(data1, data2)
        print(f"Kolmogorov-Smirnov Test: Stat = {ks_stat:.4f}, p = {ks_p:.4e}")

        # Levene's Test (Variance Comparison)
        levene_stat, levene_p = levene(data1, data2)
        print(f"Levene’s Test: Stat = {levene_stat:.4f}, p = {levene_p:.4e}")

        # Bartlett’s Test (Variance Homogeneity)
        bartlett_stat, bartlett_p = bartlett(data1, data2)
        print(f"Bartlett’s Test: Stat = {bartlett_stat:.4f}, p = {bartlett_p:.4e}")

        # Compute Cohen’s d (Effect Size)
        pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
        cohen_d = (np.mean(data1) - np.mean(data2)) / pooled_std
        print(f"Effect Size (Cohen’s d): {cohen_d:.4f}")

        # Power Analysis
        es = abs(cohen_d)  # Effect size for power calculation
        power_analysis = TTestIndPower()
        power = power_analysis.solve_power(effect_size=es, nobs1=len(data1), ratio=len(data2)/len(data1), alpha=0.05)
        print(f"Power Analysis: {power:.3f} (1-β)")

        # Spearman Correlation (checks monotonic relationships)
        spearman_corr, spearman_p = spearmanr(data1, data2)
        print(f"Spearman Correlation: R = {spearman_corr:.4f}, p = {spearman_p:.4e}")
    
    # List of connection types
    connections = ["Exc-Exc", "Exc-PV", "Exc-SST", "PV-PV", "SST-SST"]

    for conn in connections:
        print(f"\n=== Running Stats for {conn} ===")

        # Extract data for each condition
        plaid_0 = results_df[(results_df["Angular Difference"] == 0.0) & 
                            (results_df["Trial Type"] == "Plaid")][conn].dropna().values
        plaid_90 = results_df[(results_df["Angular Difference"] == 90.0) & 
                            (results_df["Trial Type"] == "Plaid")][conn].dropna().values
        grating_0 = results_df[(results_df["Angular Difference"] == 0.0) & 
                            (results_df["Trial Type"] == "Grating")][conn].dropna().values
        grating_90 = results_df[(results_df["Angular Difference"] == 90.0) & 
                                (results_df["Trial Type"] == "Grating")][conn].dropna().values

        # Compute statistics for each comparison
        compute_statistics(plaid_0, plaid_90, f"{conn} Plaid 0°", f"{conn} Plaid 90°")
        compute_statistics(grating_0, grating_90, f"{conn} Grating 0°", f"{conn} Grating 90°")
        compute_statistics(plaid_0, grating_0, f"{conn} Plaid 0°", f"{conn} Grating 0°")
        compute_statistics(plaid_90, grating_90, f"{conn} Plaid 90°", f"{conn} Grating 90°")
    
    # Run on Exc-Exc Plaid and Grating
    x_vals = results_df["Angular Difference"].dropna().values
    y_vals_plaid = results_df[(results_df["Trial Type"] == "Plaid")]["Exc-Exc"].dropna().values
    y_vals_grating = results_df[(results_df["Trial Type"] == "Grating")]["Exc-Exc"].dropna().values

    fit_models(x_vals, y_vals_plaid, "Exc-Exc Plaid")
    fit_models(x_vals, y_vals_grating, "Exc-Exc Grating")

    for conn in ["Exc-PV", "Exc-SST", "PV-PV", "SST-SST"]:
        y_vals_plaid = results_df[(results_df["Trial Type"] == "Plaid")][conn].dropna().values
        y_vals_grating = results_df[(results_df["Trial Type"] == "Grating")][conn].dropna().values

        fit_models(x_vals, y_vals_plaid, f"{conn} Plaid")
        fit_models(x_vals, y_vals_grating, f"{conn} Grating")
def plot_bar_charts_by_type2():
    # Load the results from the CSV
    results_path = csv_path
    results_df = pd.read_csv(results_path)

    # Define the noise correlation types you want to analyze
    noise_corr_types = {
        "Exc-Exc": "Exc-Exc Noise Correlation",
        "Exc-Red PV": "Exc-Red Noise Correlation",
        "Exc-Red SST": "Exc-Red Noise Correlation",
        "Red-Red PV": "Red-Red Noise Correlation",
        "Red-Red SST": "Red-Red Noise Correlation",
    }

    # Define mapping of angular difference groups
    group_mapping = {
        "1": [0.0],
        "2/8": [22.5, 157.5],
        "3/7": [45.0, 135.0],
        "4/6": [67.5, 112.5],
        "5": [90.0],
    }

    group_angles = {
        "1": 0.0,
        "2/8": 22.5,
        "3/7": 45.0,
        "4/6": 67.5,
        "5": 90.0,
    }

    # Iterate over noise correlation types and generate bar plots
    for nc_type, column in noise_corr_types.items():
        print(f"Processing: {nc_type}")

        # Filter dataframe based on inhibitory population if needed
        if "PV" in nc_type:
            df_filtered = results_df[results_df["Inhibitory Population"] == "PV"]
        elif "SST" in nc_type:
            df_filtered = results_df[results_df["Inhibitory Population"] == "SST"]
        else:
            df_filtered = results_df.copy()  # Exc-Exc case, take all
        df_filtered = df_filtered[df_filtered[column] != 1]  # Exclude noise correlations == 1

        # Split by Trial Type
        plaid_grouped_data = {group: [] for group in group_mapping.keys()}
        grating_grouped_data = {group: [] for group in group_mapping.keys()}

        for group, angles in group_mapping.items():
            plaid_grouped_data[group] = df_filtered.loc[
                (df_filtered["Angular Difference"].isin(angles)) & (df_filtered["Trial Type"] == "Plaid"),
                column
            ].dropna().values.tolist()

            grating_grouped_data[group] = df_filtered.loc[
                (df_filtered["Angular Difference"].isin(angles)) & (df_filtered["Trial Type"] == "Grating"),
                column
            ].dropna().values.tolist()

        # Compute means and SEMs
        plaid_means, plaid_errors, plaid_angles = [], [], []
        grating_means, grating_errors, grating_angles = [], [], []

        for group, data in plaid_grouped_data.items():
            if len(data) > 0:
                plaid_angles.append(group_angles[group])
                plaid_means.append(np.mean(data))
                plaid_errors.append(sem(data))

        for group, data in grating_grouped_data.items():
            if len(data) > 0:
                grating_angles.append(group_angles[group])
                grating_means.append(np.mean(data))
                grating_errors.append(sem(data))
        
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Nimbus Sans'],
            'font.size': 18,
            'axes.labelsize': 22,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'axes.titlesize': 24,
            'legend.fontsize': 18,
            'legend.title_fontsize': 18
        })

        # Create the bar plot
        plt.figure(figsize=(10, 8))

        x_labels = [group_angles[key] for key in group_mapping.keys()]

        bar_width = 4  # Adjust bar width
        x_positions = np.array(x_labels)  # Use actual degree values for x-axis
        spacing = 3.9  # Adjust spacing between bars

        grating_positions = x_positions - spacing / 2  # Shift plaid left
        plaid_positions = x_positions + spacing / 2  # Shift grating right

        color_mapping = {
            "Exc-Exc": ("#4B4B4B", "#D3D3D3"),
            "Exc-Red PV": ("#ffadad", "#b91c1c"),
            "Red-Red PV": ("#ffadad", "#b91c1c"),
            "Exc-Red SST": ("#a0c4ff", "#1d4ed8"),
            "Red-Red SST": ("#a0c4ff", "#1d4ed8")
        }

        plaid_color, grating_color = color_mapping.get(nc_type, ("gray", "black"))  # Default to gray/black

        pretty_names = {
            "Exc-Red PV": "Exc-PV",
            "Red-Red PV": "PV-PV",
            "Exc-Red SST": "Exc-SST",
            "Red-Red SST": "SST-SST"
        }
        display_nc_type = pretty_names.get(nc_type, nc_type)

        plt.bar(grating_positions, grating_means, yerr=grating_errors, capsize=5, 
                color=grating_color, alpha=0.7, width=bar_width, label="Grating", edgecolor="black", linewidth=0.8)
        plt.bar(plaid_positions, plaid_means, yerr=plaid_errors, capsize=5, 
                color=plaid_color, alpha=0.7, width=bar_width, label="Plaid", edgecolor="black", linewidth=0.8)
        

        plt.xticks(ticks=x_labels, labels=[str(int(angle)) for angle in x_labels])
        plt.xlabel("Angular Difference in Preferred Orientation")
        plt.ylim(-0.06, 0.32)
        plt.ylabel("Mean Noise Correlation")
        plt.title(f"{display_nc_type} Noise Correlations Segmented by Tuning Difference")
        plt.legend()
        plt.grid(False)
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)

        # Save the plot
        plt.savefig(os.path.join(save_dir, f"bar_plot_{nc_type.replace(' ', '_').lower()}.png"))
        plt.close()
    

    print("\n✅ All bar plots generated and saved.")


    def fit_models(x, y, label):
        """Fits linear, quadratic, and exponential models and compares fits."""
        # Linear Fit
        X_lin = sm.add_constant(x)
        model_lin = sm.OLS(y, X_lin).fit()
    
        # Quadratic Fit
        # Shift x values so that the model is symmetric around 45°
        x_symm = np.abs(x - 45)  # Ensures symmetry about 45°
        X_symm_quad = np.column_stack((x_symm, x_symm**2))
        X_symm_quad = sm.add_constant(X_symm_quad)
        model_quad = sm.OLS(y, X_symm_quad).fit()

        # Exponential Fit (Convert to log scale)
        y_log = np.log(np.where(y < 0, 0, y) + 1e-8)  # Replace negatives with 0 before log transformation
        model_exp = sm.OLS(y_log, X_lin).fit()

        # Print results
        print(f"\n=== Model Fits for {label} ===")
        print(f"Linear: R² = {model_lin.rsquared:.4f}, p = {model_lin.pvalues[1]:.4e}")
        print(f"Linear Slope (β): {model_lin.params[1]:.6f}")
        print(f"Quadratic: R² = {model_quad.rsquared:.4f}, p = {model_quad.pvalues[2]:.4e}")
        print(f"Exponential: R² = {model_exp.rsquared:.4f}, p = {model_exp.pvalues[1]:.4e}")

        # Compare fits
        best_fit = "Linear"
        best_r2 = model_lin.rsquared
        if model_quad.rsquared > best_r2:
            best_fit = "Quadratic"
            best_r2 = model_quad.rsquared
        if model_exp.rsquared > best_r2:
            best_fit = "Exponential"

        print(f"Best Fit: {best_fit} (R² = {best_r2:.4f})")

        return best_fit
    def compute_statistics(data1, data2, label1, label2):
        """Computes and prints statistical comparisons between two datasets."""
        print(f"\n### {label1} vs {label2} ###")
        print(f"{label1}: Mean = {np.mean(data1):.4f}, SEM = {np.std(data1, ddof=1)/np.sqrt(len(data1)):.4f}, Variance = {np.var(data1, ddof=1):.4f}, Sample Size = {len(data1)}")
        print(f"{label2}: Mean = {np.mean(data2):.4f}, SEM = {np.std(data2, ddof=1)/np.sqrt(len(data2)):.4f}, Variance = {np.var(data2, ddof=1):.4f}, Sample Size = {len(data2)}")

        # Mann-Whitney U Test (non-parametric comparison)
        mwu_stat, mwu_p = mannwhitneyu(data1, data2, alternative='two-sided')
        print(f"Mann-Whitney U Test: Stat = {mwu_stat:.4f}, p = {mwu_p:.4e}")

        # Kolmogorov-Smirnov Test (distribution comparison)
        ks_stat, ks_p = ks_2samp(data1, data2)
        print(f"Kolmogorov-Smirnov Test: Stat = {ks_stat:.4f}, p = {ks_p:.4e}")

        # Levene's Test (Variance Comparison)
        levene_stat, levene_p = levene(data1, data2)
        print(f"Levene’s Test: Stat = {levene_stat:.4f}, p = {levene_p:.4e}")

        # Bartlett’s Test (Variance Homogeneity)
        bartlett_stat, bartlett_p = bartlett(data1, data2)
        print(f"Bartlett’s Test: Stat = {bartlett_stat:.4f}, p = {bartlett_p:.4e}")

        # Compute Cohen’s d (Effect Size)
        pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
        cohen_d = (np.mean(data1) - np.mean(data2)) / pooled_std
        print(f"Effect Size (Cohen’s d): {cohen_d:.4f}")

        # Power Analysis
        es = abs(cohen_d)  # Effect size for power calculation
        power_analysis = TTestIndPower()
        power = power_analysis.solve_power(effect_size=es, nobs1=len(data1), ratio=len(data2)/len(data1), alpha=0.05)
        print(f"Power Analysis: {power:.3f} (1-β)")

    # List of connection types
    connections = ["Exc-Exc", "Exc-Red PV", "Exc-Red SST", "Red-Red PV", "Red-Red SST"]

    for conn in connections:
        print(f"\n=== Running Stats for {conn} ===")

        # Filter for PV or SST pairs in Red-Red comparisons
        if conn == "Red-Red PV":
            filtered_df = results_df[results_df["Inhibitory Population"] == "PV"]
        elif conn == "Red-Red SST":
            filtered_df = results_df[results_df["Inhibitory Population"] == "SST"]
        elif conn == "Exc-Red PV":
            filtered_df = results_df[results_df["Inhibitory Population"] == "PV"]
        elif conn == "Exc-Red SST":
            filtered_df = results_df[results_df["Inhibitory Population"] == "SST"]
        else:
            filtered_df = results_df

        plaid_0 = filtered_df[(filtered_df["Angular Difference"] == 0.0) & 
                            (filtered_df["Trial Type"] == "Plaid")][noise_corr_types[conn]].dropna().values
        plaid_90 = filtered_df[(filtered_df["Angular Difference"] == 90.0) & 
                            (filtered_df["Trial Type"] == "Plaid")][noise_corr_types[conn]].dropna().values
        grating_0 = filtered_df[(filtered_df["Angular Difference"] == 0.0) & 
                                (filtered_df["Trial Type"] == "Grating")][noise_corr_types[conn]].dropna().values
        grating_90 = filtered_df[(filtered_df["Angular Difference"] == 90.0) & 
                                (filtered_df["Trial Type"] == "Grating")][noise_corr_types[conn]].dropna().values

        compute_statistics(plaid_0, plaid_90, f"{conn} Plaid 0°", f"{conn} Plaid 90°")
        compute_statistics(grating_0, grating_90, f"{conn} Grating 0°", f"{conn} Grating 90°")
        compute_statistics(plaid_0, grating_0, f"{conn} Plaid 0°", f"{conn} Grating 0°")
        compute_statistics(plaid_90, grating_90, f"{conn} Plaid 90°", f"{conn} Grating 90°")
    
    """for conn, col in noise_corr_types.items():
        df_plaid = results_df.loc[(results_df["Trial Type"] == "Plaid")].dropna(subset=["Angular Difference", col])
        df_grating = results_df.loc[(results_df["Trial Type"] == "Grating")].dropna(subset=["Angular Difference", col])

        if not df_plaid.empty and not df_grating.empty:  # Ensure we have data
            x_vals_plaid, y_vals_plaid = df_plaid["Angular Difference"].values, df_plaid[col].values
            x_vals_grating, y_vals_grating = df_grating["Angular Difference"].values, df_grating[col].values

            fit_models(x_vals_plaid, y_vals_plaid, f"{conn} Plaid")
            fit_models(x_vals_grating, y_vals_grating, f"{conn} Grating")"""

#FIGURE 5
def plot_regression_heatmaps():
    # Load regression results
    csv_path2 = "/home/maclean/data_proc/noise_corr/ridge_regression_results.csv"
    results_df = pd.read_csv(csv_path2)

    # Define the populations of interest (PV & SST-based pairs)
    populations = [
        ("PV", "Exc-Exc"),
        ("PV", "Exc-Red"),
        ("SST", "Exc-Red")
    ]

    # Define the predictors in order
    predictors = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]


    # Initialize matrices for storing scaled β-values and p-values
    plaid_betas = np.zeros((len(populations), len(predictors)))
    grating_betas = np.zeros((len(populations), len(predictors)))

    plaid_pvalues = np.zeros((len(populations), len(predictors)))
    grating_pvalues = np.zeros((len(populations), len(predictors)))


    r2_values = {}  # Store R² values for potential future plots

    # Iterate over each population and extract values
    for i, (inh_pop, noise_corr_type) in enumerate(populations):
        for trial in ["Plaid", "Grating"]:
            # Filter the dataset for this combination
            subset = results_df[(results_df["Trial Type"] == trial) & 
                                (results_df["Inhibitory Population"] == inh_pop) & 
                                (results_df["Noise Correlation Type"] == noise_corr_type)]

            if subset.empty:
                continue  # Skip if no data

            # Extract β-coefficients and p-values (convert from string to dictionary)
            beta_dict = eval(subset.iloc[0]["p-values"])  # Convert string to dictionary
            
            # Store R² value
            r2_values[f"{trial} {inh_pop} {noise_corr_type}"] = subset.iloc[0]["R²"]

            # Extract β-values, normalize by IQR, and store them
            for j, predictor in enumerate(predictors):
                raw_beta = beta_dict.get(predictor, 0)  # Default to 0 if missing
                scaled_beta = raw_beta  # Scale using IQR

                # Store scaled β-values and raw p-values
                if trial == "Plaid":
                    plaid_betas[i, j] = scaled_beta
                    plaid_pvalues[i, j] = beta_dict.get(predictor, 1)  # Default p-value to 1 if missing
                else:
                    grating_betas[i, j] = scaled_beta
                    grating_pvalues[i, j] = beta_dict.get(predictor, 1)  # Default p-value to 1 if missing

    # Convert to DataFrames for visualization
    pair_labels = ["E-E", "E-PV", "E-SST"]
    plaid_betas_df = pd.DataFrame(plaid_betas, index=pair_labels, columns=predictors)
    grating_betas_df = pd.DataFrame(grating_betas, index=pair_labels, columns=predictors)
    plaid_pvalues_df = pd.DataFrame(plaid_pvalues, index=[f"{inh} {nc}" for inh, nc in populations], columns=predictors)
    grating_pvalues_df = pd.DataFrame(grating_pvalues, index=[f"{inh} {nc}" for inh, nc in populations], columns=predictors)

    # Define colormap
    cmap_beta = "coolwarm"  # Diverging colormap for β-values
    cmap_pval = "cividis"  # Sequential colormap for p-values (lower = stronger significance)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Nimbus Sans'],
        'font.size': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'axes.titlesize': 24,
        'legend.fontsize': 18,
        'legend.title_fontsize': 18
    })

    # Plot β-values heatmaps
    plt.figure(figsize=(10, 6))
    sns.heatmap(plaid_betas_df, annot=True, cmap=cmap_beta, center=0, linewidths=0.5, fmt=".2f")
    plt.title("Scaled β-Coefficients (Plaid)")
    plt.xlabel("Predictors")
    plt.ylabel("Pair Type")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"heatmap_plaid_betas.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.heatmap(grating_betas_df, annot=True, cmap=cmap_beta, center=0, linewidths=0.5, fmt=".2f")
    plt.title("Scaled β-Coefficients (Grating)")
    plt.xlabel("Predictors")
    plt.ylabel("Pair Type")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"heatmap_grating_betas.png"))
    plt.close()

    # Create clean, styled broken y-axis bar plot (matched to orientation diff bar plots)

    r2_df = pd.DataFrame.from_dict(r2_values, orient="index", columns=["R²"])
    r2_df.index = pd.MultiIndex.from_tuples(
        [idx.split(" ", 2) for idx in r2_df.index], 
        names=["Trial Type", "Inhibitory Population", "Noise Correlation Type"]
)
    r2_df = r2_df.reset_index().pivot(index=["Inhibitory Population", "Noise Correlation Type"], columns="Trial Type", values="R²")

    x_labels = ["E-E", "E-PV", "E-SST"]
    
    x_positions = np.arange(len(x_labels)) * 10  # fake "angle" spacing for visual vibe
    r2_df = r2_df.loc[[("PV", "Exc-Exc"), ("PV", "Exc-Red"), ("SST", "Exc-Red")]]

    bar_width = 2.5
    spacing = 3
    grating_positions = x_positions - spacing / 2
    plaid_positions = x_positions + spacing / 2

    # Create broken y-axis plot
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,8), gridspec_kw={'height_ratios': [1, 3]})

    # Top panel (high outlier SST values)
    ax1.bar(grating_positions, r2_df["Grating"], bar_width, label="Grating", color="#2B2B2B", edgecolor="black", linewidth=0.8)
    ax1.bar(plaid_positions, r2_df["Plaid"], bar_width, label="Plaid", color="#E5E5E5", edgecolor="black", linewidth=0.8)
    
    ax1.set_ylim(0.05, 0.16)

    # Bottom panel (normal values)
    ax2.bar(grating_positions, r2_df["Grating"], bar_width, color="#2B2B2B", edgecolor="black", linewidth=0.8)
    ax2.bar(plaid_positions, r2_df["Plaid"], bar_width, color="#E5E5E5", edgecolor="black", linewidth=0.8)
    
    ax2.set_ylim(0, 0.042)

    # Formatting
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(bottom=False)
    ax2.xaxis.tick_bottom()

    # Break marks
    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # X-ticks and labels
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels, rotation=45, ha="right")

    # Axis and title
    fig.text(0.04, 0.5, 'R² Value', va='center', rotation='vertical', fontsize=18)
    fig.suptitle("Model Performance (R²) Across Populations", fontsize=24, fontweight='bold')
    fig.legend(loc='upper right')

    plt.tight_layout(rect=[0.08, 0, 1, 1])
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(os.path.join(save_dir, "bar_chart_r2_broken_axis.png"))
    plt.show()


    print("\n✅ All regression heatmaps generated and saved.")

if __name__ == "__main__":
    #extract_unique_normalization_indices(csv_path=csv_path)
    #plot_exc_exc_noise_ridgeline(csv_path=csv_path)
    #extract_fano_factors_with_overlay(csv_path=csv_path)
    #plot_joyplot_noise_correlations(csv_path=csv_path, save_dir=save_dir)
    #plot_bar_charts_by_type2()
    plot_regression_heatmaps()
