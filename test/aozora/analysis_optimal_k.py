import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def cluster_with_optimal_k(csv_path, k_range=(2, 10), output_csv="clustered_results.csv"):
    # 1. 読み込み & 特徴量抽出
    df = pd.read_csv(csv_path)
    feature_cols = [
        "token_count", "type_count", "type_token_ratio",
        "ratio_名詞", "ratio_動詞", "ratio_形容詞", "ratio_副詞"
    ]
    X = df[feature_cols].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    # 2. クラスタ数候補の評価
    sse = []
    sil_scores = []
    k_values = list(range(k_range[0], k_range[1]+1))

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        sse.append(kmeans.inertia_)
        sil = silhouette_score(X_scaled, labels)
        sil_scores.append(sil)

    # 3. Elbow法・Silhouette法のプロット
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

    # 4. 最適クラスタ数（最大Silhouette）を選択
    best_k = k_values[sil_scores.index(max(sil_scores))]
    print(f"Estimated number of optial cluster: {best_k}")

    # 5. 最終クラスタリング + PCA可視化
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
    plt.show()

    # 6. 結果保存
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

    return df

if __name__ == "__main__":
    clustered_df = cluster_with_optimal_k("aozora_results.csv", k_range=(2, 10), output_csv="clustered_results.csv")