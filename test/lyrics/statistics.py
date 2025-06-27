import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

csv_path = "../../data/songs/popular_songs_analysis_results.csv"
df = pd.read_csv(csv_path)

# 可視化する特徴量リスト
feature_cols = [
    "token_count", "type_count", "lexical_richness",
    "ratio_名詞", "ratio_動詞", "ratio_形容詞", "ratio_副詞"
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

# 年ごとに平均値を算出
mean_by_year = df.groupby("year")[feature_cols].mean().reset_index()
# 年ごとのサンプル数
count_by_year = df.groupby("year").size().reset_index(name="count")

# パーセント表示にする特徴量
percent_cols = ["lexical_richness","ratio_名詞", "ratio_動詞", "ratio_形容詞", "ratio_副詞"]
mean_by_year_percent = mean_by_year.copy()
for col in percent_cols:
    if col in mean_by_year_percent:
        mean_by_year_percent[col] = mean_by_year_percent[col] * 100

def get_ylabel(col):
    return "Mean Value (%)" if col in percent_cols else "Mean Value"

n_features = len(feature_cols)
ncols = 2
nrows = (n_features + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 4*nrows), sharex=True)
axes = axes.flatten()

for idx, col in enumerate(feature_cols):
    ax = axes[idx]
    # 折れ線グラフ
    ax.plot(mean_by_year_percent["year"], mean_by_year_percent[col], marker='o', label="Mean")
    # 回帰直線
    x = mean_by_year_percent["year"].values
    y = mean_by_year_percent[col].values
    if len(x) > 1:
        coef = np.polyfit(x, y, 1)
        y_fit = np.polyval(coef, x)
        ax.plot(x, y_fit, color='red', linestyle='--', label="Regression")
        slope = coef[0]
    ax.set_title(f"{figure_name_map.get(col, col)} (m={slope:.3f})", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel(get_ylabel(col))
    ax.grid(True, linestyle='--', alpha=0.5)
    # サンプル数の棒グラフ（右y軸）
    ax2 = ax.twinx()
    ax2.bar(count_by_year["year"], count_by_year["count"], color='gray', alpha=0.2, width=1.0, label="Sample Count")
    ax2.set_ylabel("Sample Count")
    ax2.set_ylim(bottom=0)
    # 凡例
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=9)

for idx in range(len(feature_cols), len(axes)):
    axes[idx].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.suptitle("Yearly Trends of Linguistic Features (Mean, Regression, Sample Count)", fontsize=16)
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/yearly_trends_linguistic_features_regression_samplecount_with_slope.png", dpi=150)
plt.close()
