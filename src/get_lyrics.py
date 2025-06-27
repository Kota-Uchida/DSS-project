import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import urllib.parse
import csv
import os

class PopularSongFetcher:
    def __init__(self, start_year: int, end_year: int):
        self.start_year = start_year
        self.end_year = end_year

    def fetch_songs_from_amigo_ranking(self, output_path="../../data/songs/popular_songs_all.csv"):
        all_songs = []
        for year in range(self.start_year, self.end_year + 1):
            print(f"Fetching songs for year: {year}")
            try:
                songs = self.fetch_songs_from_amigo_ranking_of_the_year(year)
                if not songs:
                    print(f"No songs found for year {year}.")
                    continue
                all_songs.extend(songs)
            except Exception as e:
                print(f"Error fetching songs for year {year}: {e}")
            time.sleep(1)
        # 統合して1つのCSVに保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["year", "title", "artist"])
            writer.writeheader()
            for song in all_songs:
                writer.writerow(song)

    def fetch_songs_from_amigo_ranking_of_the_year(self, year:int, max_count=100) -> list[dict]:
        url = f"https://amigo.lovepop.jp/yearly/ranking.cgi?year={year}"
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'html.parser', from_encoding='EUC-JP')
        results = []
        rows = soup.select("table tr")[1:max_count+1]
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 3:
                continue
            rank = cols[0].text.strip()
            title = cols[1].text.strip()
            artist = cols[2].text.strip()
            # タイトルやアーティストに「／」や改行が含まれる場合は分割して複数行にする
            # 例: "NEO UNIVERSE\n／finale" → ["NEO UNIVERSE", "finale"]
            # アーティストも同様に分割
            # まずタイトルで分割
            titles = [t.strip() for t in title.replace('\r', '').split('\n／') if t.strip()]
            # さらに「／」で分割（念のため）
            titles = [subt for t in titles for subt in t.split('／') if subt.strip()]
            # アーティストも同様
            artists = [a.strip() for a in artist.replace('\r', '').split('\n／') if a.strip()]
            artists = [suba for a in artists for suba in a.split('／') if suba.strip()]
            # タイトルとアーティストの組み合わせで全て出力
            for t in titles:
                for a in artists:
                    results.append({
                        "year": year,
                        "title": t,
                        "artist": a
                    })
        return results

class LyricsFetcher:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def search_jlyric_url(self, title: str, artist: str) -> str | None:
        def clean(name: str) -> str:
            import re
            name = re.sub(r"\[.*?\]|\(.*?\)|（.*?）", "", name)
            name = re.sub(r"[【】『』「」]", "", name)
            name = name.replace("　", " ").strip()
            return name

        title = clean(title)
        artist = clean(artist)
        params = {
            "kt": title,
            "ka": artist,
            "ct": "2",
            "ca": "2",
            "cl": "2",
            "ex": "on"
        }
        search_url = "https://j-lyric.net/search.php?" + urllib.parse.urlencode(params)
        print(f"Searching J-Lyric: {search_url}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        for a in soup.select("div.bdy > p.mid > a"):
            href = a.get("href")
            if href:
                if href.startswith("http"):
                    return href
                elif href.startswith("/artist/"):
                    return "https://j-lyric.net" + href
        return None

    def fetch_lyrics_from_jlyric(self, url: str) -> str | None:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        lyric_div = soup.find('p', id='Lyric')
        return lyric_div.get_text(separator='\n').strip() if lyric_div else None

    def fill_lyrics(self):
        df = pd.read_csv(self.csv_path)
        if 'lyrics' not in df.columns:
            df['lyrics'] = ""
        else:
            df['lyrics'] = df['lyrics'].astype(str)
        drop_indices = []
        for idx, row in df.iterrows():
            if pd.isna(row['lyrics']) or str(row['lyrics']).strip() == "":
                title, artist = row['title'], row['artist']
                print(f"Searching: {title} / {artist}")
                try:
                    url = self.search_jlyric_url(title, artist)
                    print(f"Found URL: {url}")
                    if url:
                        lyrics = self.fetch_lyrics_from_jlyric(url)
                        if lyrics:
                            # 歌詞内の改行をスペースに変換
                            lyrics = lyrics.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                            df.at[idx, 'lyrics'] = lyrics
                            print(f"Fetched lyrics for {title} by {artist}.")
                        else:
                            print(f"No lyrics found for {title} by {artist}. 削除します。")
                            drop_indices.append(idx)
                    else:
                        print(f"No URL found for {title} by {artist}. 削除します。")
                        drop_indices.append(idx)
                except Exception as e:
                    print(f"Error fetching lyrics for {title} by {artist}: {e} 削除します。")
                    drop_indices.append(idx)
                time.sleep(0.2)
        if drop_indices:
            df = df.drop(drop_indices).reset_index(drop=True)
        df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    fetcher = PopularSongFetcher(1968, 2010)
    fetcher.fetch_songs_from_amigo_ranking("../data/songs/popular_songs_all.csv")
    # 歌詞付与処理（必要なら）
    lyrics_fetcher = LyricsFetcher("../data/songs/popular_songs_all.csv")
    lyrics_fetcher.fill_lyrics()