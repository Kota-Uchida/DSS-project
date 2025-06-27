import requests
from bs4 import BeautifulSoup
import re
import os
import zipfile
import io
import urllib.parse
import html
import pandas as pd
import threading

'''
Note:
- There are 2 relative paths used in this code:
  1. "../../data/aozora" - This is the directory where the downloaded works will be saved.
  2. "../../utils/list_person_all_utf8.csv" - This is the CSV file containing author names and IDs.
'''

class Aozora:
    def __init__(self, author_names: list[str], limit: int = 5, min_length: int = 25000, save_dir: str = "../../data/aozora"):
        self.author_names = author_names
        self.limit = limit
        self.min_length = min_length
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.downloaded_titles = set()

    '''
    A set of functions to fetch works by authors from Aozora Bunko.

    get_work_links(self, author_url)
    download_work_per_author(self, author_name, author_url)
    get_id_from_authorname(self, author_name)
    generate_author_url(self, author_name)
    download_all_authors(self)
    '''
    def get_work_links(self, author_url):
        '''
        Fetches works by an author from Aozora Bunko.
        Args: 
            author_url (str): The URL of the author's page on Aozora Bunko.
            limit (int): Maximum number of works to fetch.
            min_length (int): Minimum length of the text to consider.
        Returns:
            list of tuples: Each tuple contains (title, zip_url, text) for each work
        '''
        limit = self.limit
        min_length = self.min_length
        res = requests.get(author_url)
        soup = BeautifulSoup(res.content, "html.parser")
        ols = soup.find_all("ol")
        print(f"Found {len(ols)} ol elements.")
        result = []
        for ol in ols:
            works = ol.find_all("li")
            for work in works:
                if len(result) >= limit:
                    break
                a_tag = work.find("a", href=True)
                if not a_tag:
                    continue
                work_url = "https://www.aozora.gr.jp" + a_tag["href"].replace("..", "")
                # print(f"Processing work URL: {work_url}")
                work_res = requests.get(work_url)
                work_soup = BeautifulSoup(work_res.content, "html.parser")
                link_tag = work_soup.find("a", href=re.compile(r".*\.zip$"))
                if not link_tag:
                    print("No zip file link found.")
                    continue
                zip_url = urllib.parse.urljoin(work_url, link_tag["href"])
                # print(f"ZIP URL: {zip_url}")
                zip_res = requests.get(zip_url)
                with zipfile.ZipFile(io.BytesIO(zip_res.content)) as zf:
                    txt_name = [name for name in zf.namelist() if name.endswith('.txt')][0]
                    text = zf.read(txt_name).decode('shift_jis', errors='ignore')
                if len(text) >= min_length:
                    result.append((html.unescape(a_tag.text.strip()), zip_url, text))
            if len(result) >= limit:
                break
        return result

    def download_work_per_author(self, author_name, author_url):
        '''
        Downloads works for a specific author and saves them to files.
        Args: 
            author_name (str): The name of the author.
            author_url (str): The URL of the author's page on Aozora Bunko.
        Returns:
            None
        '''
        works = self.get_work_links(author_url)
        print(f"Found {len(works)} works for {author_name}.")
        os.makedirs(f"{self.save_dir}/{author_name}", exist_ok=True)
        for fname in os.listdir(f"{self.save_dir}/{author_name}"):
            if fname.endswith(".txt"):
                # ファイル名からタイトル部分を抽出
                try:
                    title = fname.split("_", 1)[1][:-4]  # 1_{title}.txt → {title}
                    self.downloaded_titles.add(title)
                except IndexError:
                    continue
        for i, (title, url, text) in enumerate(works):
            if title in self.downloaded_titles:
                continue
            filename = f"{self.save_dir}/{author_name}/{i+1}_{title}.txt"
            with self.lock:
                if os.path.exists(filename):
                    print(f"File {filename} already exists, skipping.")
                    continue
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(text)
            self.downloaded_titles.add(title)
            

    def get_id_from_authorname(self, author_name):
        '''
        Retrieves the author ID from the author's name.
        Args: 
            author_name (str): The name of the author.
        Returns:
            int: The ID of the author.
        '''
        df = pd.read_csv("../../utils/list_person_all_utf8.csv")
        if author_name not in df["著者名"].values:
            raise ValueError(f"Author name '{author_name}' not found in the list.")
        author_id = df[df["著者名"] == author_name]["人物ID"].values[0]
        return author_id

    def generate_author_url(self, author_name):
        '''
        Generates the URL for the author's page on Aozora Bunko.
        Args: 
            author_name (str): The name of the author.
        Returns:
            str: The URL of the author's page.
        '''
        author_id = self.get_id_from_authorname(author_name)
        return f"https://www.aozora.gr.jp/index_pages/person{author_id}.html"

    def download_all_authors(self, max_workers: int = 8):
        '''
        Downloads works for all authors specified in the author_names list using parallel threads.
        '''
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def task(author_name):
            try:
                author_url = self.generate_author_url(author_name)
                self.download_work_per_author(author_name, author_url)
                return (author_name, "Success")
            except ValueError as e:
                print(e)
                return (author_name, "Failed")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(task, name): name for name in self.author_names}
            for future in as_completed(futures):
                author_name = futures[future]
                try:
                    result = future.result()
                    print(f"[{result[1]}] {author_name}")
                except Exception as e:
                    print(f"[Error] {author_name}: {e}")
    
    '''
    A set of functions to clean and process the downloaded works.
    '''
    