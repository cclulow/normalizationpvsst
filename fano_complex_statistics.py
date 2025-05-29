import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ranksums, pearsonr, spearmanr
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from statsmodels.formula.api import mixedlm

results_file = "/home/maclean/data_proc/results_norm.csv"
save_dir = "/home/maclean/data_proc/fano/cluster_stimulus_analysis"
os.makedirs(save_dir, exist_ok=True)

results_df = pd.read_csv(results_file)
results_df.columns = results_df.columns.str.strip()

# Mapping of metric names to shorter, file-safe names
metric_name_map = {
    "Preferred Fano Factor": "Preferred",
    "Null Fano Factor": "Null",
    "Plaid Pref/Null Fano Factor": "Plaid",
    "All-Red Noise Correlation": "Red_Noise",
    "Noise Correlation": "Noise",
    "Normalization Index": "Normalization",
    "Pref to Null % Change": "Pref_to_Null",
    "Pref to Plaid % Change": "Pref_to_Plaid",
    "OSI": "OSI"
}

def sanitize_name(name):
    return metric_name_map.get(name, name.replace("/", "-").replace(" ", "_"))

def compute_percent_change(metric1, metric2):
    return 100 * (results_df[metric2] - results_df[metric1]) / results_df[metric1]

if "Pref to Null % Change" not in results_df.columns:
    results_df["Pref to Null % Change"] = compute_percent_change("Preferred Fano Factor", "Null Fano Factor")

if "Pref to Plaid % Change" not in results_df.columns:
    results_df["Pref to Plaid % Change"] = compute_percent_change("Preferred Fano Factor", "Plaid Pref/Null Fano Factor")

def fit_sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-c * (x - d))) + b

def fit_quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def plot_violin(data, group_col, value_col):
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=group_col, y=value_col, data=data, inner="point", scale="width", cut=0)
    plt.title(f"{value_col} by {group_col}")
    plt.ylabel(value_col)
    plt.xlabel(group_col)
    plot_path = os.path.join(save_dir, f"{sanitize_name(value_col)}_by_{sanitize_name(group_col)}.png")
    plt.savefig(plot_path)
    plt.close()

def analyze_clusters():
    cluster_features = results_df[["Preferred Fano Factor", "Null Fano Factor", "Plaid Pref/Null Fano Factor", "Normalization Index"]].dropna()
    kmeans = KMeans(n_clusters=4, random_state=42).fit(cluster_features)
    results_df["Cluster"] = -1
    results_df.loc[cluster_features.index, "Cluster"] = kmeans.labels_

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(cluster_features["Preferred Fano Factor"], cluster_features["Null Fano Factor"], cluster_features["Normalization Index"], c=kmeans.labels_, cmap="viridis", alpha=0.8)
    ax.set_title("Clusters Based on Fano Factors and Normalization Index")
    ax.set_xlabel("Preferred Fano Factor")
    ax.set_ylabel("Null Fano Factor")
    ax.set_zlabel("Normalization Index")
    plt.colorbar(scatter, label="Cluster")
    cluster_plot_path = os.path.join(save_dir, sanitize_name("Clusters_3D_Fano_NI.png"))
    plt.savefig(cluster_plot_path)
    plt.close()

    metrics = ["Noise Correlation", "All-Red Noise Correlation", "OSI"]
    for metric in metrics:
        valid_data = results_df[["Cluster", metric]].dropna()
        stat, p_value = ranksums(valid_data[valid_data["Cluster"] == 0][metric], valid_data[valid_data["Cluster"] == 1][metric])
        print(f"Cluster Analysis for {metric}: Statistic={stat:.3f}, P-value={p_value:.3e}")
        if p_value < 0.05:
            plot_violin(results_df, "Cluster", metric)

def analyze_stimulus_specific_variability():
    correlation_pairs = [
        ("Normalization Index", "Pref to Null % Change"),
        ("Normalization Index", "Pref to Plaid % Change"),
        ("OSI", "Pref to Null % Change"),
        ("OSI", "Pref to Plaid % Change"),
    ]

    for x_metric, y_metric in correlation_pairs:
        valid_data = results_df.dropna(subset=[x_metric, y_metric])
        x = valid_data[x_metric]
        y = valid_data[y_metric]
        pearson_corr, pearson_p = pearsonr(x, y)
        spearman_corr, spearman_p = spearmanr(x, y)

        print(f"{x_metric} vs. {y_metric}: Pearson Corr={pearson_corr:.3f}, P-value={pearson_p:.3e}")
        print(f"{x_metric} vs. {y_metric}: Spearman Corr={spearman_corr:.3f}, P-value={spearman_p:.3e}")

        if pearson_p < 0.05 or spearman_p < 0.05:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=x, y=y, alpha=0.7, label="Data")
            sns.regplot(x=x, y=y, scatter=False, label="Linear Fit", color="blue")
            quadratic_params, _ = curve_fit(fit_quadratic, x, y)
            quadratic_y = fit_quadratic(np.array(x), *quadratic_params)
            plt.plot(x, quadratic_y, label="Quadratic Fit", color="green")
            try:
                sigmoid_params, _ = curve_fit(fit_sigmoid, x, y, maxfev=10000)
                sigmoid_y = fit_sigmoid(np.array(x), *sigmoid_params)
                plt.plot(x, sigmoid_y, label="Sigmoid Fit", color="red")
            except RuntimeError:
                pass
            plt.title(f"{x_metric} vs. {y_metric}")
            plt.xlabel(x_metric)
            plt.ylabel(y_metric)
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(save_dir, sanitize_name(f"{x_metric}_vs_{y_metric}_stimulus_specific.png"))
            plt.savefig(plot_path)
            plt.close()

def analyze_tuned_vs_untuned():
    results_df["Tuned"] = (results_df["OSI"] > results_df["OSI"].median()).astype(int)
    metrics = ["Normalization Index", "Noise Correlation", "All-Red Noise Correlation"]

    for metric in metrics:
        valid_data = results_df[["Tuned", metric]].dropna()
        stat, p_value = ranksums(valid_data[valid_data["Tuned"] == 1][metric], valid_data[valid_data["Tuned"] == 0][metric])
        print(f"Tuned vs Untuned for {metric}: Statistic={stat:.3f}, P-value={p_value:.3e}")
        if p_value < 0.05:
            plot_violin(results_df, "Tuned", metric)

def analyze_red_noise_correlation():
    pv_mice = ['18S', '21N', '30G', '37L', '37L2', '37R', '37R2', '42N', '44N']
    sst_mice = ['30N', '32L', '32L2']

    results_df["Inhibitory Population"] = results_df["Mouse"].apply(
        lambda mouse: "PV" if mouse in pv_mice else "SST" if mouse in sst_mice else None
    )
    results_df["Inhibitory Population"].fillna("Other", inplace=True)

    correlation_pairs = [
        ("All-Red Noise Correlation", "Preferred Fano Factor"),
        ("All-Red Noise Correlation", "Null Fano Factor"),
        ("All-Red Noise Correlation", "Plaid Pref/Null Fano Factor"),
        ("All-Red Noise Correlation", "Pref to Null % Change"),
        ("All-Red Noise Correlation", "Pref to Plaid % Change"),
    ]

    for x_metric, y_metric in correlation_pairs:
        valid_data = results_df.dropna(subset=[x_metric, y_metric, "Inhibitory Population"])
        x = valid_data[x_metric]
        y = valid_data[y_metric]
        pearson_corr, pearson_p = pearsonr(x, y)
        spearman_corr, spearman_p = spearmanr(x, y)
        print(f"{x_metric} vs. {y_metric}: Pearson Corr={pearson_corr:.3f}, P-value={pearson_p:.3e}")
        print(f"{x_metric} vs. {y_metric}: Spearman Corr={spearman_corr:.3f}, P-value={spearman_p:.3e}")
        if pearson_p < 0.05 or spearman_p < 0.05:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=x, y=y, hue=valid_data["Inhibitory Population"], alpha=0.7)
            sns.regplot(x=x, y=y, scatter=False, color="blue")
            plt.title(f"{x_metric} vs. {y_metric}")
            plt.xlabel(x_metric)
            plt.ylabel(y_metric)
            scatter_path = os.path.join(save_dir, sanitize_name(f"{x_metric}_vs_{y_metric}_by_population.png"))
            plt.savefig(scatter_path)
            plt.close()

    # Prepare lme_data and ensure column names match formula
    lme_data = results_df.rename(columns={
        "All-Red Noise Correlation": "Red_Noise_Correlation",
        "Preferred Fano Factor": "Preferred_Fano_Factor"
    })

    # Filter for necessary columns and drop missing values
    lme_data = lme_data[["Red_Noise_Correlation", "Preferred_Fano_Factor", "Mouse", "Date", "Inhibitory Population"]].dropna()

    if lme_data.empty:
        print("Not enough data for linear mixed-effects model.")
        return

    try:
        for population in ["PV", "SST"]:
            pop_data = lme_data[lme_data["Inhibitory Population"] == population]
            if pop_data.empty:
                print(f"Not enough data for {population} population.")
                continue

            lme_model = mixedlm("Red_Noise_Correlation ~ Preferred_Fano_Factor", pop_data, groups=pop_data["Mouse"])
            lme_result = lme_model.fit()
            print(f"{population} Population - Mixed Effects Model:")
            print(lme_result.summary())
    except Exception as e:
        print(f"Linear mixed-effects model failed: {e}")


if __name__ == "__main__":
    analyze_clusters()
    analyze_stimulus_specific_variability()
    analyze_tuned_vs_untuned()
    analyze_red_noise_correlation()
    import pandas as pd


results_file = "/home/maclean/data_proc/results_norm.csv"
save_dir = "/home/maclean/data_proc/fano/cluster_stimulus_analysis"
os.makedirs(save_dir, exist_ok=True)


pv_mice = ['18S', '21N', '30G', '37L', '37L2', '37R', '37R2', '42N', '44N']
sst_mice = ['30N', '32L', '32L2']
results_df["Inhibitory Population"] = results_df["Mouse"].apply(
    lambda mouse: "PV" if mouse in pv_mice else "SST" if mouse in sst_mice else None
)
results_df["Inhibitory Population"].fillna("Other", inplace=True)
def sanitize_name(name):
    return name.replace("/", "-").replace(" ", "_")

def compute_percent_change(metric1, metric2):
    return 100 * (results_df[metric2] - results_df[metric1]) / results_df[metric1]
if "Pref to Null % Change" not in results_df.columns:
    results_df["Pref to Null % Change"] = compute_percent_change("Preferred Fano Factor", "Null Fano Factor")
if "Pref to Plaid % Change" not in results_df.columns:
    results_df["Pref to Plaid % Change"] = compute_percent_change("Preferred Fano Factor", "Plaid Pref/Null Fano Factor")

def plot_violin(data, group_col, value_col, plot_name):
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=group_col, y=value_col, data=data, inner="point", scale="width", cut=0)
    plt.title(f"{value_col} by {group_col}")
    plt.ylabel(value_col)
    plt.xlabel(group_col)
    plot_path = os.path.join(save_dir, sanitize_name(plot_name))
    plt.savefig(plot_path)
    plt.close()

# Statistical tests for PV vs SST (go back to this)
def analyze_pv_sst_differences():
    print("\nAnalyzing Differences Between PV and SST Populations...")
    metrics = [
        "All-Red Noise Correlation",
        "Red-Red Noise Correlation",
        "Red-All Noise Correlation",
        "Exc-Exc Noise Correlation",
        "Normalization Index",
        "OSI",
    ]

    for metric in metrics:
        pv_data = results_df[results_df["Inhibitory Population"] == "PV"][metric].dropna()
        sst_data = results_df[results_df["Inhibitory Population"] == "SST"][metric].dropna()
        stat, p_value = ranksums(pv_data, sst_data)
        print(f"{metric}: Statistic={stat:.3f}, P-value={p_value:.3e}")
        if p_value < 0.05:
            combined_data = results_df[results_df["Inhibitory Population"].isin(["PV", "SST"])]
            plot_violin(combined_data, "Inhibitory Population", metric, f"{metric}_PV_vs_SST.png")

# Segment by stimulus conditions (pref, null, plaid)
def analyze_stimulus_conditions():
    print("\nAnalyzing Red Noise Correlations by Stimulus Condition and Population...")
    conditions = ["Preferred", "Null", "Plaid"]
    condition_cols = {
        "Preferred": "Preferred Fano Factor",
        "Null": "Null Fano Factor",
        "Plaid": "Plaid Pref/Null Fano Factor"
    }

    for condition, col in condition_cols.items():
        for population in ["PV", "SST"]:
            data = results_df[(results_df["Inhibitory Population"] == population) & (results_df[col].notna())]
            if data.empty:
                continue
            plt.figure(figsize=(8, 6))
            sns.violinplot(x="Inhibitory Population", y=col, data=data, inner="point", scale="width", cut=0)
            plt.title(f"{condition} Fano Factor by Population ({population})")
            plot_path = os.path.join(save_dir, sanitize_name(f"{condition}_Fano_Factor_by_{population}.png"))
            plt.savefig(plot_path)
            plt.close()

def analyze_context_roles():
    print("\nAnalyzing Context-Dependent Roles of Red Noise Correlations...")
    combined_data = results_df[results_df["Inhibitory Population"].isin(["PV", "SST"])]
        #data = results_df[results_df["Inhibitory Population"] == population]
    for metric in ["OSI", "Normalization Index"]:
            # Ensure valid data for both x and y
        valid_data = combined_data.dropna(subset=[metric, "All-Red Noise Correlation", "Inhibitory Population"])
        pv_data = valid_data[valid_data["Inhibitory Population"] == "PV"]
        sst_data = valid_data[valid_data["Inhibitory Population"] == "SST"]

        # Compute statistics for each population
        if not pv_data.empty:
            pv_pearson_corr, pv_pearson_p = pearsonr(pv_data[metric], pv_data["All-Red Noise Correlation"])
            pv_spearman_corr, pv_spearman_p = spearmanr(pv_data[metric], pv_data["All-Red Noise Correlation"])
            print(f"PV - {metric} vs. Red Noise Correlation:")
            print(f"    Pearson Corr={pv_pearson_corr:.3f}, P-value={pv_pearson_p:.3e}")
            print(f"    Spearman Corr={pv_spearman_corr:.3f}, P-value={pv_spearman_p:.3e}")

        if not sst_data.empty:
            sst_pearson_corr, sst_pearson_p = pearsonr(sst_data[metric], sst_data["All-Red Noise Correlation"])
            sst_spearman_corr, sst_spearman_p = spearmanr(sst_data[metric], sst_data["All-Red Noise Correlation"])
            print(f"SST - {metric} vs. Red Noise Correlation:")
            print(f"    Pearson Corr={sst_pearson_corr:.3f}, P-value={sst_pearson_p:.3e}")
            print(f"    Spearman Corr={sst_spearman_corr:.3f}, P-value={sst_spearman_p:.3e}")
        
        # Plot scatterplot with two regression lines
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=valid_data,
            x=metric,
            y="All-Red Noise Correlation",
            hue="Inhibitory Population",
            alpha=0.7
        )
        sns.regplot(
            data=pv_data,
            x=metric,
            y="All-Red Noise Correlation",
            scatter=False,
            label="PV Regression",
            color="blue"
        )
        sns.regplot(
            data=sst_data,
            x=metric,
            y="All-Red Noise Correlation",
            scatter=False,
            label="SST Regression",
            color="orange"
        )
        
        plt.title(f"{metric} vs. Red Noise Correlation (PV and SST Populations)")
        plt.xlabel(metric)
        plt.ylabel("All-Red Noise Correlation")
        plt.legend(title="Population / Regression")
        plt.grid(True)

        # Save the plot
        plot_path = os.path.join(
            save_dir,
            sanitize_name(f"{metric}_vs_Red_Noise_Correlation_Comparison.png")
        )
        plt.savefig(plot_path)
        plt.close()
           
            
            
"""x = valid_data[metric]
            y = valid_data["All-Red Noise Correlation"]
            hue = valid_data["Inhibitory Population"]
            
            if len(valid_data) > 0  # Check if there is valid data to analyze
                pearson_corr, pearson_p = pearsonr(x, y)
                spearman_corr, spearman_p = spearmanr(x, y)
                print(f"{population} - {metric} vs. Red Noise Correlation:")
                print(f"    Pearson Corr={pearson_corr:.3f}, P-value={pearson_p:.3e}")
                print(f"    Spearman Corr={spearman_corr:.3f}, P-value={spearman_p:.3e}")
                
                # Scatter plot
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=x, y=y, alpha=0.7, label=f"{population} Data")
                sns.regplot(x=x, y=y, scatter=False, label="Linear Fit", color="blue")
                plt.title(f"{metric} vs. Red Noise Correlation ({population})")
                plt.xlabel(metric)
                plt.ylabel("All-Red Noise Correlation")
                plt.legend()
                plt.grid(True)
                plot_path = os.path.join(
                    save_dir,
                    sanitize_name(f"{metric}_vs_Red_Noise_Correlation_{population}.png")
                )
                plt.savefig(plot_path)
                plt.close()"""

# Network-level dynamics (averaging)
def analyze_network_dynamics_kde():
    print("\nAnalyzing Network-Level Dynamics in Red Noise Correlations...")
    
    # Filter data for PV and SST populations
    pv_data = results_df[results_df["Inhibitory Population"] == "PV"]["All-Red Noise Correlation"].dropna()
    sst_data = results_df[results_df["Inhibitory Population"] == "SST"]["All-Red Noise Correlation"].dropna()
    
    # Calculate summary statistics
    pv_avg = pv_data.mean()
    pv_std = pv_data.std()
    sst_avg = sst_data.mean()
    sst_std = sst_data.std()

    for metric in ["All-Red Noise Correlation", "Red-Red Noise Correlation", "Red-All Noise Correlation", "Exc-Exc Noise Correlation"]:
        pv_data = results_df[results_df["Inhibitory Population"] == "PV"][metric].dropna()
        sst_data = results_df[results_df["Inhibitory Population"] == "SST"][metric].dropna()

        # Print summary statistics
        print(f"{metric} - PV Avg: {pv_data.mean():.3f}, Std: {pv_data.std():.3f}")
        print(f"{metric} - SST Avg: {sst_data.mean():.3f}, Std: {sst_data.std():.3f}")

        # Plot KDE
        plt.figure(figsize=(10, 7))
        sns.kdeplot(pv_data, shade=True, label=f"PV ({metric})", bw_adjust=0.5)
        sns.kdeplot(sst_data, shade=True, label=f"SST ({metric})", bw_adjust=0.5)
        plt.title(f"KDE of {metric} (PV vs SST)")
        plt.xlabel(metric)
        plt.ylabel("Density")
        plt.legend(title="Population")
        plt.grid(True)

        plot_path = os.path.join(save_dir, f"KDE_{sanitize_name(metric)}_PV_vs_SST.png")
        plt.savefig(plot_path)
        plt.show()

if __name__ == "__main__":
    analyze_pv_sst_differences()
    analyze_stimulus_conditions()
    analyze_context_roles()
    analyze_network_dynamics_kde()