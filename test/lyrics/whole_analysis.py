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

def whole_analysis_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # cluster_from_csv(df, n_clusters=4)
    cluster_with_optimal_k(df, k_range=(2, 10))
    tsne_from_csv(df)
    return df

def cluster_from_csv(df, n_clusters=4):
    feature_cols = [
        "type_token_ratio",
        "ratio_名詞", "ratio_動詞", "ratio_形容詞", "ratio_副詞"
    ]
    df_features = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df["pc1"] = components[:, 0]
    df["pc2"] = components[:, 1]
    plt.figure(figsize=(8, 6))
    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id]
        plt.scatter(subset["pc1"], subset["pc2"], label=f"Cluster {cluster_id}", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Clusters (Songs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/cluster_plot_songs.png")
    return df

def cluster_with_optimal_k(df, k_range=(2, 10)):
    feature_cols = [
        "token_count", "type_count", "type_token_ratio",
        "ratio_名詞", "ratio_動詞", "ratio_形容詞", "ratio_副詞"
    ]
    X = df[feature_cols].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    sse = []
    sil_scores = []
    k_values = list(range(k_range[0], k_range[1]+1))

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        sse.append(kmeans.inertia_)
        sil = silhouette_score(X_scaled, labels)
        sil_scores.append(sil)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(k_values, sse, marker='o')
    ax[0].set_title("Elbow method（SSE）")
    ax[0].set_xlabel("Number of Clusters (k)")
    ax[0].set_ylabel("SSE (Sum of Squared Errors)")

    ax[1].plot(k_values, sil_scores, marker='o', color="green")
    ax[1].set_title("Silhouette score")
    ax[1].set_xlabel("Number of Clusters (k)")
    ax[1].set_ylabel("Score（-1〜1）")

    plt.tight_layout()
    plt.show()

    best_k = k_values[sil_scores.index(max(sil_scores))]
    print(f"Estimated number of optial cluster: {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df["pc1"] = components[:, 0]
    df["pc2"] = components[:, 1]

    plt.figure(figsize=(8, 6))
    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id]
        plt.scatter(subset["pc1"], subset["pc2"], label=f"Cluster {cluster_id}", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Semantic Clustering with Optimal k")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/cluster_optimal_k.png")

    return df

def tsne_from_csv(df, perplexity=30, random_state=42):
    plt.rcParams['font.family'] = 'IPAexGothic'  
    feature_cols = [
        "token_count", "type_count", "type_token_ratio",
        "ratio_名詞", "ratio_動詞", "ratio_形容詞", "ratio_副詞"
    ]
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
        colors = cm.get_cmap('tab20', n_clusters) if n_clusters <= 20 else cm.get_cmap('hsv', n_clusters)
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
    plt.savefig("figures/tsne_songs.png")
    return df

if __name__ == "__main__":
    result_df = whole_analysis_from_csv("../../data/songs/popular_songs_results.csv")
    result_df.to_csv("../../data/songs/popular_songs_analysis_results.csv", index=False)
