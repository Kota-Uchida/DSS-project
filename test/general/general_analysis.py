import pandas as pd
import numpy as np
from scipy.special import rel_entr
import matplotlib.pyplot as plt
import os

df_lyrics = pd.read_csv("../../data/songs/popular_songs_analysis_results.csv")
df_literature = pd.read_csv("../../data/aozora_results/aozora_analysis_results.csv")


def get_cluster_distributions(df, feature_cols, cluster_col="cluster"):
    distributions = {}
    for cluster_id in sorted(df[cluster_col].dropna().unique()):
        cluster_df = df[df[cluster_col] == cluster_id]
        cluster_dist = {}
        for col in feature_cols:
            values = cluster_df[col].dropna().values
            value_counts = pd.Series(values).value_counts(normalize=True).sort_index()
            cluster_dist[col] = value_counts
        distributions[cluster_id] = cluster_dist
    return distributions


def js_divergence(p: pd.Series, q: pd.Series) -> float:
    """
    2つの離散確率分布p, q（pd.Series）に対し、Jensen-Shannon Divergence (JSD) を計算する。
    JSD(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m), m = 0.5*(p+q)
    事象空間が異なる場合は、両方のインデックスの和集合で補完し、確率0を明示的に与える。
    """
    all_index = p.index.union(q.index)
    p_full = p.reindex(all_index, fill_value=0)
    q_full = q.reindex(all_index, fill_value=0)
    eps = 1e-12
    p_full = p_full.clip(lower=eps)
    q_full = q_full.clip(lower=eps)
    m = 0.5 * (p_full + q_full)
    # KL(p||m)
    kl_pm = np.sum(p_full * np.log(p_full / m))
    # KL(q||m)
    kl_qm = np.sum(q_full * np.log(q_full / m))
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd


feature_cols = [
    "token_count","type_count", "lexical_richness", "ratio_名詞", "ratio_動詞", "ratio_形容詞", "ratio_副詞"
]

figure_name_map = {
    "token_count": "Token Count",
    "type_count": "Type Count",
    "lexical_richness": "Lexical Richness",
    "ratio_名詞": "Noun Ratio",
    "ratio_動詞": "Verb Ratio",
    "ratio_形容詞": "Adjective Ratio",
    "ratio_副詞": "Adverb Ratio"
}

lyrics_distributions = get_cluster_distributions(df_lyrics, feature_cols, cluster_col="cluster")
literature_distributions = get_cluster_distributions(df_literature, feature_cols, cluster_col="cluster")

# Example usage:
# print(lyrics_distributions[0]["token_count"])

# jsd = js_divergence(lyrics_distributions[0]["token_count"], literature_distributions[0]["token_count"])
# print(f"Jensen-Shannon Divergence: {jsd}")

# 歌詞3クラスタ＋文学3クラスタ＝計6クラスタでJSDを計算し6×6ヒートマップを作成
all_distributions = []
all_labels = []
for i in range(3):
    all_distributions.append(lyrics_distributions[i])
    all_labels.append(f"lyrics_{i}")
for j in range(3):
    all_distributions.append(literature_distributions[j])
    all_labels.append(f"literature_{j}")

cluster_num_all = 6
all_jsd_values = []
heatmap_matrices = []
for feature in feature_cols:
    heatmap_matrix = pd.DataFrame(
        np.zeros((cluster_num_all, cluster_num_all)),
        index=all_labels,
        columns=all_labels
    )
    for i in range(cluster_num_all):
        for j in range(cluster_num_all):
            if feature in all_distributions[i] and feature in all_distributions[j]:
                jsd = js_divergence(all_distributions[i][feature], all_distributions[j][feature])
                heatmap_matrix.iloc[i, j] = jsd
                all_jsd_values.append(jsd)
    heatmap_matrices.append(heatmap_matrix)

vmin = min(all_jsd_values)
vmax = max(all_jsd_values)

n_features = len(feature_cols)
ncols = 4 if n_features > 4 else n_features
nrows = (n_features + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
axes = axes.flatten() if n_features > 1 else [axes]
for idx, (feature, heatmap_matrix) in enumerate(zip(feature_cols, heatmap_matrices)):
    im = axes[idx].imshow(heatmap_matrix, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[idx].set_title(f"{figure_name_map[feature]}", fontsize=14)
    axes[idx].set_xticks(np.arange(cluster_num_all))
    axes[idx].set_yticks(np.arange(cluster_num_all))
    axes[idx].set_xticklabels(all_labels, rotation=45, ha='right')
    axes[idx].set_yticklabels(all_labels)
    axes[idx].set_xlabel("Cluster")
    axes[idx].set_ylabel("Cluster")
for idx in range(len(feature_cols), len(axes)):
    axes[idx].axis('off')
fig.subplots_adjust(right=0.88, wspace=0.4, hspace=0.4)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Jensen-Shannon Divergence')
plt.suptitle("Jensen-Shannon Divergence Heatmaps", fontsize=18)
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/js_divergence_heatmaps_6x6_all_features.png")
plt.close()





