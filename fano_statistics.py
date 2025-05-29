import pandas as pd
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from scipy.optimize import curve_fit


results_file = "/home/maclean/data_proc/results_norm.csv"
save_dir = "/home/maclean/data_proc/fano/comparison"
os.makedirs(save_dir, exist_ok=True)

results_df = pd.read_csv(results_file)
results_df.columns = results_df.columns.str.strip()

# Compute percentage changes if missing
def compute_percent_change(metric1, metric2):
    return 100 * (results_df[metric2] - results_df[metric1]) / results_df[metric1]

if "Pref to Null % Change" not in results_df.columns:
    results_df["Pref to Null % Change"] = compute_percent_change("Preferred Fano Factor", "Null Fano Factor")

if "Pref to Plaid % Change" not in results_df.columns:
    results_df["Pref to Plaid % Change"] = compute_percent_change("Preferred Fano Factor", "Plaid Pref/Null Fano Factor")

# Function to plot violin plots
def plot_violin(data, group_col, value_col):
    plt.figure(figsize=(8, 6))
    sns.violinplot(
        x=group_col,
        y=value_col,
        data=data,
        inner="point",
        scale="width",
        cut=0
    )
    plt.title(f"{value_col} by {group_col}")
    plt.ylabel(value_col)
    plot_path = os.path.join(save_dir, f"{value_col.replace(' ', '_')}_by_{group_col.replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Violin plot saved: {plot_path}")

# Function to perform rank-sum test
def perform_ranksum(data, col2, group_col):
    groups = data[group_col].unique()
    if len(groups) == 2:  # Rank-sum test
        group1 = data[data[group_col] == 0][col2].dropna()
        group2 = data[data[group_col] == 1][col2].dropna()
        stat, p_value = stats.ranksums(group1, group2)
    else:  # Kruskal-Wallis test for multiple groups
        group_values = [data[data[group_col] == g][col2].dropna() for g in groups]
        stat, p_value = stats.kruskal(*group_values)
    return stat, p_value

# Cluster-based segmentation
def analyze_cluster_segments():
    print("\nRunning Cluster-Based Segmentation...")
    cluster_data = results_df[["Normalization Index", "Noise Correlation", "OSI"]].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42).fit(cluster_data)
    results_df["Cluster"] = -1
    results_df.loc[cluster_data.index, "Cluster"] = kmeans.labels_

    metrics = [
        "Red Noise Correlation",
        "Pref to Null % Change",
        "Pref to Plaid % Change",
    ]

    for metric in metrics:
        stat, p_value = perform_ranksum(results_df, metric, "Cluster")
        print(f"Cluster Analysis for {metric}: Statistic={stat:.3f}, P-value={p_value:.3e}")
        if p_value < 0.05:
            plot_violin(results_df, "Cluster", metric)

# Percentile-based segmentation with finer granularity
def analyze_percentile_segments():
    print("\nRunning Percentile-Based Segmentation...")
    results_df["NI Percentile Group"] = pd.qcut(
        results_df["Normalization Index"], q=[0, 0.05, 0.5, 0.95, 1], 
        labels=["Bottom 5%", "Lower Middle 45%", "Upper Middle 45%", "Top 5%"]
    )

    metrics = [
        "Noise Correlation",
        "OSI",
        "Pref to Null % Change",
        "Pref to Plaid % Change",
    ]

    for metric in metrics:
        stat, p_value = perform_ranksum(results_df, metric, "NI Percentile Group")
        print(f"Percentile Analysis for {metric}: Statistic={stat:.3f}, P-value={p_value:.3e}")
        if p_value < 0.05:
            plot_violin(results_df, "NI Percentile Group", metric)

# Combined metric analysis
def analyze_combined_metrics():
    print("\nRunning Combined Metric Analysis...")
    results_df["Combined Metric"] = results_df["Normalization Index"] * results_df["Noise Correlation"]
    median_combined = results_df["Combined Metric"].median()
    results_df["Combined Metric Group"] = (results_df["Combined Metric"] > median_combined).astype(int)

    metrics = [
        "Red Noise Correlation",
        "OSI",
        "Pref to Null % Change",
        "Pref to Plaid % Change",
    ]

    for metric in metrics:
        stat, p_value = perform_ranksum(results_df, metric, "Combined Metric Group")
        print(f"Combined Metric Analysis for {metric}: Statistic={stat:.3f}, P-value={p_value:.3e}")
        if p_value < 0.05:
            plot_violin(results_df, "Combined Metric Group", metric)

# Difference-based segmentation
def analyze_difference_segments():
    print("\nRunning Difference-Based Segmentation...")
    if "Difference Group" not in results_df.columns:
        results_df["Fano Change Difference"] = results_df["Pref to Null % Change"] - results_df["Pref to Plaid % Change"]
        results_df["Difference Group"] = pd.cut(
            results_df["Fano Change Difference"], bins=[-np.inf, -10, 10, np.inf], 
            labels=["Large Decrease", "Neutral", "Large Increase"]
        )

    metrics = [
        "Normalization Index",
        "Noise Correlation",
        "OSI",
    ]

    for metric in metrics:
        stat, p_value = perform_ranksum(results_df, metric, "Difference Group")
        print(f"Difference-Based Analysis for {metric}: Statistic={stat:.3f}, P-value={p_value:.3e}")
        if p_value < 0.05:
            plot_violin(results_df, "Difference Group", metric)

def analyze_combined_metric_in_detail():
    print("\nExploring Combined Metric in Detail...")

    # Extract high and low combined metric groups
    median_combined = results_df["Combined Metric"].median()
    high_combined = results_df[results_df["Combined Metric"] > median_combined]
    low_combined = results_df[results_df["Combined Metric"] <= median_combined]

    # Detailed statistics for high and low groups
    for group, group_data in [("High", high_combined), ("Low", low_combined)]:
        for metric in ["Red Noise Correlation", "OSI"]:
            mean_value = group_data[metric].mean()
            std_value = group_data[metric].std()
            count = len(group_data)
            print(f"{group} Combined Metric Group - {metric}: "
                  f"Mean={mean_value:.3f}, Std={std_value:.3f}, Count={count}")

    # Segment Combined Metric into quartiles
    results_df["Combined Metric Quartile"] = pd.qcut(
        results_df["Combined Metric"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )

    # Violin plot for Red Noise Correlation and OSI by Combined Metric quartile
    for metric in ["Red Noise Correlation", "OSI"]:
        plot_violin(results_df, "Combined Metric Quartile", metric)

    # Correlation heatmap for selected metrics including OSI
    correlation_metrics = ["Normalization Index", "Noise Correlation", "Red Noise Correlation", "OSI"]
    correlation_data = results_df[correlation_metrics].dropna()
    corr_matrix = correlation_data.corr(method="spearman")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        xticklabels=correlation_metrics,
        yticklabels=correlation_metrics,
        square=True
    )
    plt.title("Correlation Matrix: Normalization Index, Noise Correlation, Red Noise Correlation, OSI")
    correlation_path = os.path.join(save_dir, "Correlation_Matrix_Combined_Metric_with_OSI.png")
    plt.savefig(correlation_path)
    plt.close()
    print(f"Correlation matrix plot saved: {correlation_path}")

    # Scatter plot of Combined Metric vs Red Noise Correlation
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="Combined Metric",
        y="Red Noise Correlation",
        hue="Combined Metric Quartile",
        data=results_df,
        palette="viridis",
        alpha=0.7
    )
    plt.title("Red Noise Correlation vs Combined Metric")
    plt.xlabel("Combined Metric")
    plt.ylabel("Red Noise Correlation")
    scatter_path = os.path.join(save_dir, "Red_Noise_Correlation_vs_Combined_Metric.png")
    plt.grid(True)
    plt.savefig(scatter_path)
    plt.close()
    print(f"Scatter plot saved: {scatter_path}")

    # Scatter plot of Combined Metric vs OSI
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="Combined Metric",
        y="OSI",
        hue="Combined Metric Quartile",
        data=results_df,
        palette="viridis",
        alpha=0.7
    )
    plt.title("OSI vs Combined Metric")
    plt.xlabel("Combined Metric")
    plt.ylabel("OSI")
    scatter_path = os.path.join(save_dir, "OSI_vs_Combined_Metric.png")
    plt.grid(True)
    plt.savefig(scatter_path)
    plt.close()
    print(f"Scatter plot saved: {scatter_path}")

def segment_neurons_by_variability():
    print("\nRunning Segmentation of Neurons by Variability Patterns...")

    # Select features for clustering
    cluster_features = results_df[["Preferred Fano Factor", "Null Fano Factor", "Plaid Pref/Null Fano Factor", "Normalization Index"]].dropna()

    # Perform Gaussian Mixture Model clustering
    gmm = GaussianMixture(n_components=3, random_state=42)
    cluster_labels = gmm.fit_predict(cluster_features)

    # Assign cluster labels to the results DataFrame
    results_df["Variability Cluster"] = -1
    results_df.loc[cluster_features.index, "Variability Cluster"] = cluster_labels

    # 3D Scatter Plot of Clusters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        cluster_features["Preferred Fano Factor"],
        cluster_features["Null Fano Factor"],
        cluster_features["Normalization Index"],
        c=cluster_labels, cmap="viridis", alpha=0.8
    )
    ax.set_title("3D Clusters: Variability Patterns")
    ax.set_xlabel("Preferred Fano Factor")
    ax.set_ylabel("Null Fano Factor")
    ax.set_zlabel("Normalization Index")
    plt.colorbar(scatter, label="Cluster ID")
    cluster_path = os.path.join(save_dir, "Variability_Clusters_3D.png")
    plt.savefig(cluster_path)
    plt.close()
    print(f"3D scatter plot saved: {cluster_path}")

    # Violin plots for Noise Correlation, OSI, and Red Noise Correlation
    cluster_metrics = ["Noise Correlation", "OSI", "Red Noise Correlation"]
    for metric in cluster_metrics:
        stat, p_value = perform_ranksum(results_df, metric, "Variability Cluster")
        print(f"Cluster Analysis for {metric}: Statistic={stat:.3f}, P-value={p_value:.3e}")
        if p_value < 0.05:
            plot_violin(results_df, "Variability Cluster", metric)

def fit_sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-c * (x - d))) + b

def fit_quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def analyze_stimulus_specific_variability():
    print("\nRunning Stimulus-Specific Variability Analysis...")

    # Metrics to analyze
    correlation_pairs = [
        ("Normalization Index", "Pref to Null % Change"),
        ("Normalization Index", "Pref to Plaid % Change"),
        ("OSI", "Pref to Null % Change"),
        ("OSI", "Pref to Plaid % Change"),
    ]

    for x_metric, y_metric in correlation_pairs:
        # Drop NaN values
        valid_data = results_df.dropna(subset=[x_metric, y_metric])
        x = valid_data[x_metric]
        y = valid_data[y_metric]

        # Calculate correlations
        pearson_corr, pearson_p = stats.pearsonr(x, y)
        spearman_corr, spearman_p = stats.spearmanr(x, y)

        print(f"{x_metric} vs. {y_metric}: Pearson Corr={pearson_corr:.3f}, P-value={pearson_p:.3e}")
        print(f"{x_metric} vs. {y_metric}: Spearman Corr={spearman_corr:.3f}, P-value={spearman_p:.3e}")

        # Scatter plot with linear fit
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x, y=y, alpha=0.7, label="Data")
        sns.regplot(x=x, y=y, scatter=False, label="Linear Fit", color="blue")

        # Quadratic fit
        quadratic_params, _ = curve_fit(fit_quadratic, x, y)
        quadratic_y = fit_quadratic(np.array(x), *quadratic_params)
        plt.plot(x, quadratic_y, label="Quadratic Fit", color="green")

        # Sigmoid fit
        try:
            sigmoid_params, _ = curve_fit(fit_sigmoid, x, y, maxfev=10000)
            sigmoid_y = fit_sigmoid(np.array(x), *sigmoid_params)
            plt.plot(x, sigmoid_y, label="Sigmoid Fit", color="red")
        except RuntimeError:
            print(f"Could not fit sigmoid to {x_metric} vs. {y_metric}.")

        # Add labels and legend
        plt.title(f"{x_metric} vs. {y_metric}")
        plt.xlabel(x_metric)
        plt.ylabel(y_metric)
        plt.legend()
        plt.grid(True)

        # Save plot
        plot_path = os.path.join(save_dir, f"{x_metric.replace(' ', '_')}_vs_{y_metric.replace(' ', '_')}_stimulus_specific.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Scatter plot saved: {plot_path}")

if __name__ == "__main__":
    analyze_cluster_segments()
    analyze_percentile_segments()
    analyze_combined_metrics()
    analyze_difference_segments()
    analyze_combined_metric_in_detail()
    segment_neurons_by_variability()
    fit_sigmoid()
    fit_quadratic()
    analyze_stimulus_specific_variability()
