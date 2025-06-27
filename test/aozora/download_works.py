from aozora import Aozora


author_names = [
    "谷崎 潤一郎", "田山 花袋", "寺田 寅彦", "永井 荷風", "中島 敦", "夏目 漱石", "樋口 一葉", "堀 辰雄",
    "宮沢 賢治", "森 鴎外", "夢野 久作", "横光 利一", "西田 幾多郎", "福沢 諭吉", "三木 清", "和辻 哲郎",
    "芥川 竜之介", "有島 武郎", "石川 啄木", "泉 鏡花", "伊藤 左千夫", "江戸川 乱歩", "梶井 基次郎",
    "国木田 独歩", "小林 多喜二", "坂口 安吾", "島崎 藤村", "太宰 治", "戸坂 潤"
]

aozora = Aozora(
    author_names=author_names,
    save_dir="../../data/aozora"
)

aozora.download_all_authors()
