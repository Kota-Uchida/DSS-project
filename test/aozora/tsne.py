import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_from_csv(csv_path, output_csv="tsne_results.csv", perplexity=30, random_state=42):
    # 1. CSV読み込み
    df = pd.read_csv(csv_path)

    # 2. 特徴量抽出
    feature_cols = [
        "token_count", "type_count", "type_token_ratio",
        "ratio_名詞", "ratio_動詞", "ratio_形容詞", "ratio_副詞"
    ]
    X = df[feature_cols].fillna(0)

    # 3. スケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. t-SNEによる2次元圧縮
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_embedded = tsne.fit_transform(X_scaled)

    # 5. 結果をDataFrameに追加
    df["tsne_x"] = X_embedded[:, 0]
    df["tsne_y"] = X_embedded[:, 1]

    # 6. プロット
    plt.figure(figsize=(10, 8))
    for author in df["author"].unique():
        subset = df[df["author"] == author]
        plt.scatter(subset["tsne_x"], subset["tsne_y"], label=author, alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("t-SNE Visualization of Authors")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 7. 保存
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

    return df

if __name__ == "__main__":
    # t-SNEを実行
    tsne_df = tsne_from_csv("aozora_results.csv", output_csv="tsne_results.csv", perplexity=30, random_state=42)