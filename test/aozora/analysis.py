import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def cluster_from_csv(csv_path, n_clusters=4, output_csv="clustered_results.csv"):
    # 1. データ読み込み
    df = pd.read_csv(csv_path)

    # 2. 特徴量列の選択
    feature_cols = [
        "token_count", "type_count", "type_token_ratio",
        "ratio_名詞", "ratio_動詞", "ratio_形容詞", "ratio_副詞"
    ]
    df_features = df[feature_cols].fillna(0)

    # 3. スケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    # 4. クラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # 5. 次元削減（PCA）
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df["pc1"] = components[:, 0]
    df["pc2"] = components[:, 1]

    # 6. 可視化
    plt.figure(figsize=(8, 6))
    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id]
        plt.scatter(subset["pc1"], subset["pc2"], label=f"Cluster {cluster_id}", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Clusters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 7. 保存
    df.to_csv(output_csv, index=False)
    print(f"結果を保存しました: {output_csv}")

    return df

if __name__ == "__main__":
    # クラスタリングを実行
    clustered_df = cluster_from_csv("aozora_results.csv", n_clusters=4, output_csv="clustered_results.csv")
    print(clustered_df.head())