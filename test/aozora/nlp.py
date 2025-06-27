import os
from pathlib import Path
from aozora_nlp import AozoraNLP 
import csv
from collections import defaultdict
import re

def collect_texts(base_dir="../../data/aozora_cleaned") -> dict[str, dict[str, str]]:
    '''
    Load all works by each author under base_dir.
    Returns: {author_name: {title: text}}
    '''
    all_texts = {}
    for author_dir in Path(base_dir).iterdir():
        if author_dir.is_dir():
            author = author_dir.name
            all_texts[author] = {}
            for txt_file in author_dir.glob("*.txt"):
                title = txt_file.stem
                with open(txt_file, encoding="utf-8") as f:
                    all_texts[author][title] = f.read()
    return all_texts

def analyze_all_texts(all_texts: dict[str, dict[str, str]], nlp: AozoraNLP) -> dict:
    '''
    Perform morphological analysis and stopword removal on all texts.
    Returns: {author: {title: {"tokens": [...], "clean_tokens": [...]}}}
    '''
    result = {}
    for author, works in all_texts.items():
        result[author] = {}
        for title, text in works.items():
            tokens = nlp.safe_tokenize(text)
            clean_tokens = nlp.remove_stopwords(tokens)
            result[author][title] = {
                "tokens": tokens,
                "clean_tokens": clean_tokens
            }
    return result

def extract_keywords_all(parsed_data: dict, nlp: AozoraNLP, top_n=20) -> dict:
    '''
    Extract keywords based on clean_tokens.
    Returns: {author: {title: [keyword1, keyword2, ...]}}
    '''
    keywords = {}
    for author in parsed_data:
        keywords[author] = {}
        for title in parsed_data[author]:
            clean_tokens = parsed_data[author][title]["clean_tokens"]
            keywords[author][title] = nlp.extract_keywords(clean_tokens, top_n=top_n)
    return keywords

def compute_style_metrics_all(parsed_data: dict, nlp: AozoraNLP) -> dict:
    '''
    Calculate style metrics such as POS ratio and lexical diversity.
    Returns: {author: {title: {metric_name: value}}}
    '''
    style_metrics = {}
    for author in parsed_data:
        style_metrics[author] = {}
        for title in parsed_data[author]:
            tokens = parsed_data[author][title]["tokens"]
            style_metrics[author][title] = nlp.compute_style_metrics(tokens)
    return style_metrics

def estimate_topics_per_author(parsed_data: dict, nlp: AozoraNLP, num_topics=3) -> dict:
    '''
    Estimate topic models by grouping works for each author.
    Returns: {author: {topic_id: [keywords]}}
    '''
    topic_results = {}
    for author in parsed_data:
        docs = []
        for title in parsed_data[author]:
            tokens = parsed_data[author][title]["clean_tokens"]
            docs.append(" ".join(tokens))
        topic_results[author] = nlp.estimate_topics(docs, num_topics=num_topics)
    return topic_results


def export_results_to_csv(style_metrics: dict, keywords: dict, topic_results: dict, output_path="aozora_results.csv", top_k=10):
    '''
    style_metrics: {author: {title: {metric: value}}}
    keywords: {author: {title: [keyword1, keyword2, ...]}}
    '''
    rows = []
    for author in style_metrics:
        for title in style_metrics[author]:
            metrics = style_metrics[author][title]
            word_count = sum(metrics.get(f"ratio_{pos}", 0) for pos in ["名詞", "動詞", "形容詞", "副詞"])
            tokens = parsed_data[author][title]["clean_tokens"]
            token_count = len(tokens)
            type_count = len(set(tokens))
            topics_str = ""
            if author in topic_results:
                topics_str = " | ".join(
                    [f"T{tid}: {', '.join(words[:5])}" for tid, words in topic_results[author].items()]
                )
            row = {
                "author": author,
                "title": title,
                "token_count": token_count,
                "type_count": type_count,
                "lexical_richness": metrics.get("lexical_richness", 0),
                "type_token_ratio": metrics.get("type_token_ratio", 0),
                "ratio_名詞": metrics.get("ratio_名詞", 0),
                "ratio_動詞": metrics.get("ratio_動詞", 0),
                "ratio_形容詞": metrics.get("ratio_形容詞", 0),
                "ratio_副詞": metrics.get("ratio_副詞", 0),
                "keywords": ", ".join(keywords[author][title][:top_k]) if author in keywords and title in keywords[author] else ""  ,
                "topics": topics_str
            }
            rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else [
        "author", "title", "token_count", "type_count", "lecical_richness", "type_token_ratio",
        "ratio_名詞", "ratio_動詞", "ratio_形容詞", "ratio_副詞",
        "keywords", "topics"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Export completed: {output_path}")


if __name__ == "__main__":
    # Create an instance of AozoraNLP
    nlp = AozoraNLP(stopwords_path="../../src/stopwords.txt")

    # Collect text data
    all_texts = collect_texts()

    # Morphological analysis and stopword removal
    parsed_data = analyze_all_texts(all_texts, nlp)

    # Keyword extraction
    keywords = extract_keywords_all(parsed_data, nlp)

    # Calculate style metrics
    style_metrics = compute_style_metrics_all(parsed_data, nlp)

    # Estimate topic models
    topic_results = estimate_topics_per_author(parsed_data, nlp)

    export_results_to_csv(style_metrics, keywords, topic_results, output_path="aozora_results.csv")