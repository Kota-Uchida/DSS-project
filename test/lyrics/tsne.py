import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import seaborn as sns
import os


def tsne_from_csv(df,feature_cols, perplexity=30, random_state=42):
    plt.rcParams['font.family'] = 'IPAexGothic'  
    X = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_embedded = tsne.fit_transform(X_scaled)
    df["tsne_x"] = X_embedded[:, 0]
    df["tsne_y"] = X_embedded[:, 1]
    if "cluster" in df.columns:
        clusters = sorted(df["cluster"].unique())
        n_clusters = len(clusters)
        colors = plt.get_cmap('tab20', n_clusters) if n_clusters <= 20 else plt.get_cmap('hsv', n_clusters)
        color_list = [colors(i) for i in range(n_clusters)]
        plt.figure(figsize=(10, 8))
        for idx, cluster_id in enumerate(clusters):
            subset = df[df["cluster"] == cluster_id]
            plt.scatter(subset["tsne_x"], subset["tsne_y"], label=f"Cluster {cluster_id}", alpha=0.7, color=color_list[idx])
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        plt.figure(figsize=(10, 8))
        plt.scatter(df["tsne_x"], df["tsne_y"], alpha=0.7, color="blue")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/tsne_songs_re.png")
    return df


feature_cols = [
        "token_count", "type_count", "type_token_ratio",
        "ratio_名詞", "ratio_動詞", "ratio_形容詞", "ratio_副詞"
    ]
df = pd.read_csv("../../data/songs/popular_songs_analysis_results.csv")

# t-SNE feature ablation plot
all_feature_sets = []
all_titles = []
# 全て含めた場合
all_feature_sets.append(feature_cols)
all_titles.append("All features")
# 1つずつ除いた場合
for i, col in enumerate(feature_cols):
    ablated = [c for j, c in enumerate(feature_cols) if j != i]
    all_feature_sets.append(ablated)
    all_titles.append(f"w/o {col}")

n_plots = len(all_feature_sets)
ncols = 4
nrows = (n_plots + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
axes = axes.flatten()

# 各パターンごとに個別に出力
for idx, (cols, title) in enumerate(zip(all_feature_sets, all_titles)):
    df_tmp = df.copy()
    df_tmp = tsne_from_csv(df_tmp, cols, perplexity=30, random_state=42)
    plt.figure(figsize=(8, 6))
    if "cluster" in df_tmp.columns:
        clusters = sorted(df_tmp["cluster"].unique())
        n_clusters = len(clusters)
        colors = plt.get_cmap('tab20', n_clusters) if n_clusters <= 20 else plt.get_cmap('hsv', n_clusters)
        color_list = [colors(i) for i in range(n_clusters)]
        for cidx, cluster_id in enumerate(clusters):
            subset = df_tmp[df_tmp["cluster"] == cluster_id]
            plt.scatter(subset["tsne_x"], subset["tsne_y"], label=f"Cluster {cluster_id}", alpha=0.7, color=color_list[cidx], s=10)
        plt.legend(fontsize=10, loc='best')
    else:
        plt.scatter(df_tmp["tsne_x"], df_tmp["tsne_y"], alpha=0.7, color="blue", s=10)
    plt.title(title, fontsize=13)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    if idx == 0:
        fname = "tsne_all_features.png"
    else:
        fname = f"tsne_wo_{feature_cols[idx-1]}.png"
    plt.savefig(f"figures/{fname}", dpi=150)
    plt.close()

