import requests
from bs4 import BeautifulSoup
import time
import os
import csv

class LyricsScraper:
    def __init__(self, start_year: int, end_year: int):
        self.start_year = start_year
        self.end_year = end_year

    def fetch_songs_from_billboard(self, year: int, month: int):
        for day in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            url = f"https://www.billboard-japan.com/charts/detail?a=hot100&year={year}&month={month:02d}&day={day:02d}"
            res = requests.get(url)
            if res.status_code != 200:
                print(f"Page not found for {year}-{month:02d}-{day:02d}. Skipping.")
                continue
            soup = BeautifulSoup(res.text, "html.parser")
            entries = []
            # 各順位のtrを取得
            for row in soup.find_all("tr", class_=lambda x: x and x.startswith("rank")):
                title_tag = row.find("p", class_="musuc_title")
                artist_tag = row.find("p", class_="artist_name")
                if title_tag and artist_tag:
                    title = title_tag.get_text(strip=True)
                    # アーティスト名はaタグがあればそちら、なければpタグのテキスト
                    artist_a = artist_tag.find("a")
                    artist = artist_a.get_text(strip=True) if artist_a else artist_tag.get_text(strip=True)
                    entries.append((title, artist))
            print(f"Fetched {len(entries)} entries for {year}-{month:02d}-{day:02d}.")
            if entries:
                print(f"Fetched {len(entries)} entries for {year}-{month:02d}-{day:02d}.")
                time.sleep(1)  # Respectful scraping delay
                return entries
            else:
                print(f"No entries found for {year}-{month:02d}-{day:02d}. Trying next date.")
        print(f"No valid chart found for {year}-{month:02d}.")
        return []

    def fetch_lyrics(self, title: str, artist: str):
        q = f"{title} {artist}"
        url = f"https://search.yahoo.co.jp/search?p={q}+site:j-lyric.net"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = [a['href'] for a in soup.select('a') if 'j-lyric.net/artist/' in a.get('href', '')]
        print(f"Found {len(links)} links for '{title}' by '{artist}'.")
        if not links:
            return None
        time.sleep(1)
        song_page = requests.get(links[0], headers=headers)
        soup = BeautifulSoup(song_page.text, 'html.parser')
        lyric_div = soup.find("div", {"id": "Lyric"})
        return lyric_div.get_text(separator="\n").strip() if lyric_div else None

    def save_lyrics_csv(self, lyrics_data):
        os.makedirs("../../data/lyrics", exist_ok=True)
        filename = "lyrics_all.csv"
        with open(f"../../data/lyrics/{filename}", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["year", "month", "title", "artist", "lyrics"])
            writer.writeheader()
            for row in lyrics_data:
                writer.writerow(row)
    
    def run_scraping(self):
        all_lyrics_data = []
        for year in range(self.start_year, self.end_year + 1):
            for month in [1, 4, 7, 10]:  # 四半期ごとに抽出
                songs = self.fetch_songs_from_billboard(year, month)
                for title, artist in songs[:10]:  # 上位10曲だけ
                    lyrics = self.fetch_lyrics(title, artist)
                    all_lyrics_data.append({
                        "year": year,
                        "month": month,
                        "title": title,
                        "artist": artist,
                        "lyrics": lyrics if lyrics else ""
                    })
        self.save_lyrics_csv(all_lyrics_data)

if __name__ == "__main__":
    scraper = LyricsScraper(start_year=2010, end_year=2023)
    scraper.run_scraping()
    print("Lyrics scraping completed.")