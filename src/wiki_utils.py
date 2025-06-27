import os
import wikipediaapi
import csv
import re
import random


class WikiLib:
    def __init__(self, language: str = 'ja', base_dir: str = 'data'):
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            user_agent="DSS-project/0.1 (contact: kametty.138@gmail.com)"
        )
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def sanitize_filename(self, name: str) -> str:
        """ファイル名として使えない文字を除去"""
        return name.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")

    def save_page_text(self, page, dir_path: str):
        """ページ本文をUTF-8で保存"""
        if not page.exists():
            return
        text = page.text
        if not text.strip():
            return
        title = self.sanitize_filename(page.title)
        filepath = os.path.join(dir_path, f"{title}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)

    def fetch_author_pages(self, authors: list[str], depth: int = 1):
        """
        作家ごとにWikipediaページを取得し、深さ1までのリンクを保存
        """
        for author in authors:
            print(f"処理中: {author}")
            author_dir = os.path.join(self.base_dir, author)
            os.makedirs(author_dir, exist_ok=True)

            # 作家ページ
            author_page = self.wiki.page(author)
            if not author_page.exists():
                print(f"  → ページが存在しません: {author}")
                continue

            self.save_page_text(author_page, author_dir)

            if depth >= 1:
                # 深さ1リンクページも保存
                links = author_page.links
                print(f"  → リンク数: {len(links)}")
                for linked_title in links:
                    linked_page = self.wiki.page(linked_title)
                    self.save_page_text(linked_page, author_dir)
    def show_num_of_links(self, authors: list[str]):
        """
        指定した作家のWikipediaページのリンク数を表示
        """
        for author in authors:
            author_page = self.wiki.page(author)
            if not author_page.exists():
                print(f"ページが存在しません: {author}")
                return
            links = author_page.links
            print(f"{author} のリンク数: {len(links)}")
    
    def save_link_titles_to_csv(self, authors: list[str], csv_path: str):
        """
        各作家のWikipediaページからリンクされているページタイトルをCSVに保存
        CSV形式: 作家名, リンクタイトル
        """
        rows = []

        for author in authors:
            author_page = self.wiki.page(author)
            if not author_page.exists():
                print(f"ページが存在しません: {author}")
                continue

            links = author_page.links
            print(f"{author} のリンク数: {len(links)}")
            for title in links.keys():
                rows.append([author, title])

        # CSV書き出し
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['author', 'linked_title'])  # ヘッダ
            writer.writerows(rows)

        print(f"リンクタイトルを保存しました: {csv_path}")
    
    def filter_unuseful_links(self, csv_path: str='../../data/wiki/wiki_links.csv', output_path: str='../../data/wiki/wiki_links_filtered.csv'):
        """
        CSVから有用なリンクのみを抽出し、別のCSVに保存
        """
        rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                author, title = row
                if self.filter_by_title(title):
                    rows.append(row)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['author', 'linked_title'])
            writer.writerows(rows)

        print(f"有用なリンクを保存しました: {output_path}")
    
    def filter_by_title(self, title: str) -> bool:
        if ":" in title or title.endswith('(曖昧さ回避)'):
            return False
        if re.search(r'[A-Za-z0-9]', title):  # 英字または数字を含む場合は除外
            return False
        else:
            return True
    
    def fetch_random_long_articles(self, csv_path: str="../../data/wiki/wiki_links_filtered.csv", min_length: int = 3000, sample_size: int = 15):
        """
        各作家ごとに、長さ min_length 以上の記事をランダムに sample_size 件ダウンロードして保存
        """
        # 作家ごとにリンクを分類
        author_to_titles = {}

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                author = row['author']
                title = row['linked_title']
                author_to_titles.setdefault(author, []).append(title)

        for author, titles in author_to_titles.items():
            print(f"\n 作家: {author}")
            author_dir = os.path.join(self.base_dir, author)
            os.makedirs(author_dir, exist_ok=True)

            long_articles = []

            for title in titles:
                page = self.wiki.page(title)
                if page.exists() and len(page.text) >= min_length:
                    long_articles.append(page)
                    print(f"  - 記事: {page.title} (長さ: {len(page.text)})")

            print(f"  - 対象記事数（長さ ≥ {min_length}）: {len(long_articles)}")

            selected_pages = random.sample(long_articles, min(sample_size, len(long_articles)))

            for page in selected_pages:
                saved = self.save_page_text(page, author_dir)
                if saved:
                    print(f"保存: {page.title}")
                else:
                    print(f"スキップ: {page.title}")