import pandas as pd
from collections import Counter
from sudachipy import dictionary, SplitMode


tokenizer = dictionary.Dictionary().create()

def load_keywords_from_csv(csv_path: str, keyword_column: str = "keywords", delimiter: str = ",") -> list[str]:
    df = pd.read_csv(csv_path)
    keywords = []
    for kw_str in df[keyword_column].dropna():
        kws = [kw.strip() for kw in kw_str.split(delimiter)]
        keywords.extend(kws)
    return keywords

def extract_frequent_keywords(keywords: list[str], min_count=5) -> list[str]:
    counter = Counter(keywords)
    return [word for word, count in counter.items() if count >= min_count]

def save_stopword_candidates(words: list[str], output_path="stopwords_candidate.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        for word in sorted(words):
            f.write(word + "\n")
    print(f"Saved stopwords candidates {output_path}")

def is_proper_noun(word: str) -> bool:
    """
    SudachiPyで語を解析し、固有名詞かどうかを判定する
    """
    tokens = tokenizer.tokenize(word, SplitMode.C)
    for m in tokens:
        pos = m.part_of_speech()
        if "名詞" in pos[0] and "固有名詞" in pos[1]:
            return True
    return False

def extract_proper_nouns_from_keywords(csv_path: str, keyword_column="keywords", delimiter=",", min_count=2) -> list[str]:
    df = pd.read_csv(csv_path)
    keywords = []
    for kw_str in df[keyword_column].dropna():
        kws = [kw.strip() for kw in kw_str.split(delimiter)]
        keywords.extend(kws)

    counter = Counter(keywords)
    proper_nouns = []

    for word, count in counter.items():
        if count >= min_count and is_proper_noun(word):
            proper_nouns.append(word)

    return sorted(set(proper_nouns))

def save_stopwords(words: list[str], output_path="stopwords_proper_nouns.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        for word in words:
            f.write(word + "\n")
    print(f"登場人物・固有名詞候補 {len(words)} 件を保存: {output_path}")

if __name__ == "__main__":
    keywords = load_keywords_from_csv("aozora_results.csv", keyword_column="keywords")

    frequent_words = extract_frequent_keywords(keywords, min_count=1)

    save_stopword_candidates(frequent_words, "stopwords_candidate.txt")

    proper_nouns = extract_proper_nouns_from_keywords("aozora_results.csv")
    
    save_stopwords(proper_nouns, "stopwords_proper_nouns.txt")