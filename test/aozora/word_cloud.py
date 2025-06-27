import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

def plot_keyword_clouds_by_cluster(csv_path, cluster_column="cluster", keyword_column="keywords", delimiter=",", save_dir="keyword_clouds"):
    # Load CSV
    df = pd.read_csv(csv_path)
    if cluster_column not in df.columns:
        raise ValueError(f"Cluster column '{cluster_column}' not found.")
    if keyword_column not in df.columns:
        raise ValueError(f"Keyword column '{keyword_column}' not found.")

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # Aggregate keywords by cluster
    clusters = sorted(df[cluster_column].dropna().unique())
    for cluster_id in clusters:
        df_cluster = df[df[cluster_column] == cluster_id]
        all_keywords = []

        for kw_str in df_cluster[keyword_column].dropna():
            all_keywords.extend([kw.strip() for kw in kw_str.split(delimiter) if kw.strip()])

        counter = Counter(all_keywords)
        if not counter:
            continue

        # Draw word cloud
        wordcloud = WordCloud(
            font_path="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",  # Specify Japanese font if needed
            width=800, height=600, background_color='white',
            max_words=100, collocations=False
        ).generate_from_frequencies(counter)

        plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Cluster {cluster_id} Keywords")
        plt.tight_layout()

        # Save image
        save_path = os.path.join(save_dir, f"cluster_{cluster_id}_keywords.png")
        plt.savefig(save_path, dpi=150)
        print(f"Saved keyword cloud for cluster {cluster_id}: {save_path}")
        plt.close()

if __name__ == "__main__":
    plot_keyword_clouds_by_cluster("aozora_analysis_results.csv")