import seaborn as sns
import matplotlib.pyplot as plt


def plot_error_distribution(df):

    sns.countplot(x="error_type", data=df)
    plt.title("Error Type Distribution")
    plt.savefig("results/error_distribution.png")
    plt.close()


def plot_noise_vs_error(df):

    sns.boxplot(x="error_type", y="noise_strength", data=df)
    plt.title("Noise Strength vs Error Type")
    plt.savefig("results/noise_vs_error.png")
    plt.close()


def plot_depth_vs_error_rate(df):

    sns.scatterplot(x="circuit_depth", y="error_rate", data=df)
    plt.title("Circuit Depth vs Error Rate")
    plt.savefig("results/depth_vs_error_rate.png")
    plt.close()