import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

def plot_keyword_clouds_by_decade(csv_path, year_column="year", keyword_column="keywords", delimiter=",", save_dir="keyword_clouds_by_decade"):
    # Load CSV
    df = pd.read_csv(csv_path)
    if year_column not in df.columns:
        raise ValueError(f"Year column '{year_column}' not found.")
    if keyword_column not in df.columns:
        raise ValueError(f"Keyword column '{keyword_column}' not found.")

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # 年代区分リスト
    decade_bins = [(1968, 1975), (1976, 1980), (1981, 1985),(1986, 1990), (1991, 1995), (1996, 2000), (2001, 2005), (2006, 2010)]
    for start, end in decade_bins:
        df_decade = df[(df[year_column] >= start) & (df[year_column] <= end)]
        all_keywords = []
        for kw_str in df_decade[keyword_column].dropna():
            all_keywords.extend([kw.strip() for kw in kw_str.split(delimiter) if kw.strip()])
        counter = Counter(all_keywords)
        if not counter:
            continue
        # Draw word cloud
        wordcloud = WordCloud(
            font_path="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
            width=800, height=600, background_color='white',
            max_words=100, collocations=False
        ).generate_from_frequencies(counter)
        plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Keywords {start}-{end}")
        plt.tight_layout()
        # Save image
        save_path = os.path.join(save_dir, f"keywords_{start}_{end}.png")
        plt.savefig(save_path, dpi=150)
        print(f"Saved keyword cloud for {start}-{end}: {save_path}")
        plt.close()

plot_keyword_clouds_by_decade("../../data/songs/popular_songs_analysis_results.csv")
