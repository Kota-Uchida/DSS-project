import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import urllib.parse

def search_jlyric_url(title: str, artist: str) -> str | None:
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
        "ct": "2",  # 曲名:中間一致
        "ca": "2",  # 歌手:中間一致
        "cl": "2",  # 歌詞:中間一致
        "ex": "on"
    }
    search_url = "https://j-lyric.net/search.php?" + urllib.parse.urlencode(params)
    print(f"Searching J-Lyric: {search_url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    # ここを修正
    for a in soup.select("div.bdy > p.mid > a"):
        href = a.get("href")
        if href:
            if href.startswith("http"):
                return href
            elif href.startswith("/artist/"):
                return "https://j-lyric.net" + href
    return None

def fetch_lyrics_from_jlyric(url: str) -> str | None:
    """J-Lyricの曲ページから歌詞を抽出"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    lyric_div = soup.find('p', id='Lyric')
    return lyric_div.get_text(separator='\n').strip() if lyric_div else None

def fill_lyrics(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        if pd.isna(row['lyrics']) or str(row['lyrics']).strip() == "":
            title, artist = row['title'], row['artist']
            print(f"Searching: {title} / {artist}")

            try:
                url = search_jlyric_url(title, artist)
                print(f"Found URL: {url}")
                if url:
                    lyrics = fetch_lyrics_from_jlyric(url)
                    if lyrics:
                        df.at[idx, 'lyrics'] = lyrics
                        print(f"歌詞取得成功: {title}")
                    else:
                        print(f"歌詞が見つかりませんでした: {title}")
                else:
                    print(f"URLが見つかりませんでした: {title}")
            except Exception as e:
                print(f"エラー発生: {title} / {e}")
            time.sleep(1.0)  # 過負荷防止のためのウェイト

    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n保存完了: {output_path}")

# 使用例（ファイル名を必要に応じて変更してください）
if __name__ == "__main__":
    fill_lyrics("../data/lyrics/lyrics_all.csv", "../data/lyrics/lyrics_filled.csv")
