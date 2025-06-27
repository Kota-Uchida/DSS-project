import re

class Clean:
    def __init__(self):
        pass

    def preprocess_text(text):
    # 1. 前処理（ヘッダ・フッタ除去）
        delimiter_matches = list(re.finditer(r"-{5,}", text))
        if "テキスト中に現れる記号について" in text and len(delimiter_matches) >= 2:
            text = text[delimiter_matches[1].end():].lstrip()
        elif match := re.search(r"(［＃.*?］)?\s*　", text):
            text = text[match.start():].lstrip()

        end_keywords = [
            r"底本：", r"青空文庫：", r"公開日：", r"入力：", r"校正：", r"作成ファイル：", r"修正："
        ]
        end_pattern = re.compile("|".join(end_keywords))
        end_match = list(end_pattern.finditer(text))
        if end_match:
            text = text[:end_match[0].start()].rstrip()

        # 2. 注記・挿絵注の削除
        text = re.sub(r"［＃.*?挿絵.*?］", "", text)
        text = re.sub(r"［＃.*?］", "", text)

        # 3. 傍点・ルビの削除
        text = re.sub(r"［＃「.*?」の傍点］", "", text)
        text = re.sub(r"［＃「.*?」の傍点終わり］", "", text)
        text = re.sub(r"《[^《》]+》", "", text)

        # 4. 句読点・記号の除去
        text = re.sub(r"[・「」『』（）【】《》〈〉〔〕［］｛｝【】ー〜―…：；？！“”‘’]", "", text)
        text = re.sub(r"[!-/:-@[-`{-~]", "", text)

        # 5. 空白・英数字の除去
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[0-9A-Za-zＡ-Ｚａ-ｚ０-９]", "", text)
        text = text.strip()

        return text