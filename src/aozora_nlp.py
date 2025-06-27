from sudachipy import dictionary
from sudachipy import SplitMode
import os
import json
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import requests
import re


class AozoraNLP:
    def __init__(self, tokenizer_mode="C", stopwords_path="stopwords.txt"):
        self.tokenizer = dictionary.Dictionary().create()
        self.mode = getattr(SplitMode, tokenizer_mode)
        self.stopwords = set()
        if stopwords_path is None:
            try:
                url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ja/master/stopwords-ja.txt"
                res = requests.get(url)
                res.encoding = "utf-8"
                self.stopwords = set([line.strip() for line in res.text.splitlines() if line.strip()])
            except Exception as e:
                print("Stopwords download failed:", e)
        elif os.path.exists(stopwords_path):
            with open(stopwords_path, encoding="utf-8") as f:
                self.stopwords = set([line.strip() for line in f if line.strip()])

    def tokenize(self, text: str) -> list[tuple[str, str, str]]:
        results = []
        for m in self.tokenizer.tokenize(text, self.mode):
            surface = m.surface()
            pos = m.part_of_speech()[0]
            lemma = m.dictionary_form()
            results.append((surface, pos, lemma))
        return results
    
    def safe_tokenize(self, text: str, chunk_size: int = 40000) -> list[tuple[str, str, str]]:
        tokens = []
        # 句点（。）や改行で分割
        import re
        sentences = re.split(r'(?<=[。！？\n])', text)

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) > chunk_size:
                tokens.extend(self.tokenize(current_chunk))
                current_chunk = sentence
            else:
                current_chunk += sentence

        if current_chunk:
            tokens.extend(self.tokenize(current_chunk))
        
        return tokens

    
    def remove_stopwords(self, tokens: list[tuple[str, str, str]]) -> list[str]:
        clean = []
        for surface, pos, lemma in tokens:
            if lemma in self.stopwords:
                continue
            if pos in ["助詞", "助動詞", "記号", "補助記号"]:
                continue
            if len(lemma) < 2:
                continue
            clean.append(lemma)
        return clean



    def extract_keywords(self, tokens: list[str], top_n=20) -> list[str]:
        counter = Counter(tokens)
        return [word for word, _ in counter.most_common(top_n)]



    def estimate_topics(self, docs: list[str], num_topics=5) -> dict[int, list[str]]:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(docs)
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
        lda.fit(X)
        terms = vectorizer.get_feature_names_out()
        topics = {}
        for i, comp in enumerate(lda.components_):
            top_words = [terms[t] for t in comp.argsort()[-10:][::-1]]
            topics[i] = top_words
        return topics

    def compute_style_metrics(self, tokens: list[tuple[str, str, str]]) -> dict[str, float]:
        total = len(tokens)
        if total == 0:
            return {}

        # type/token比率
        type_token_ratio = len(set([lemma for _, _, lemma in tokens])) / total

        # 品詞カウント
        pos_counts = {"名詞": 0, "動詞": 0, "形容詞": 0, "副詞": 0}
        for _, pos, _ in tokens:
            if pos in pos_counts:
                pos_counts[pos] += 1

        # 文の切れ目推定（簡易：句点と改行）
        surface_text = "".join([surface for surface, _, _ in tokens])
        sentences = re.split(r"[。．！？\n]", surface_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences) if sentences else 1  # ゼロ除算防止

        avg_sentence_length = round(total / sentence_count, 2)

        # 読点（、）の頻度
        comma_count = surface_text.count("、")
        avg_commas_per_sentence = round(comma_count / sentence_count, 3)

        # 語彙の豊富さ（ユニークなlemma数 / トークン数）
        lemmas = [lemma for _, _, lemma in tokens]
        lexical_richness = round(len(set(lemmas)) / len(lemmas), 4)

        return {
            "type_token_ratio": round(type_token_ratio, 4),
            "avg_sentence_length": avg_sentence_length,
            "avg_commas_per_sentence": avg_commas_per_sentence,
            "lexical_richness": lexical_richness,
            **{f"ratio_{pos}": round(count / total, 4) for pos, count in pos_counts.items()}
        }