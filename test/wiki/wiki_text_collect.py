import wikipediaapi
import csv
import re
import os

def clean_title(raw_title: str) -> str:
    # 先頭の数字とアンダースコアを削除（例：1_）
    title = re.sub(r'^\d+_', '', raw_title)
    # _cleaned や _something を削除（末尾のアンダースコア以降を取り除く）
    title = re.sub(r'_.*$', '', title)
    return title

def sanitize(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")


wiki = wikipediaapi.Wikipedia(
    language='ja',
    user_agent='Aozora-Analyzer/0.1 (contact: your_email@example.com)'
)

output_dir = "../../data/wiki"
os.makedirs(output_dir, exist_ok=True)

with open('../aozora/aozora_analysis_results.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)

    for row in reader:
        raw_title = row['title']
        title = clean_title(raw_title)
        author = row['author']

        for name in [title, author]:
            page = wiki.page(name)
            if page.exists():
                text = page.text
                if text.strip():
                    filename = os.path.join(output_dir, f"{sanitize(name)}.txt")
                    if not os.path.exists(filename):
                        with open(filename, 'w', encoding='utf-8') as wf:
                            wf.write(text)
                        print(f"保存: {name}")
                    else:
                        print(f"スキップ（既存）: {name}")
                else:
                    print(f"空ページ: {name}")
            else:
                print(f"ページなし: {name}")
