"""Generate ~300 dummy reviews for pipeline testing."""

import os
import random

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(ROOT, "data", "dummy", "reviews.csv")

RESTAURANTS = [
    "鼎泰豐", "阿宗麵線", "度小月", "永康牛肉麵", "林東芳牛肉麵",
    "饒河夜市小吃", "士林夜市雞排", "逢甲夜市", "六合夜市", "花園夜市",
    "春水堂", "50嵐", "路易莎", "鬍鬚張", "三商巧福",
]

AREAS = ["台北", "台中", "高雄", "台南", "新竹"]

# ─── Positive templates (label=2, rating 4-5) ───

POSITIVE = [
    "這家的牛肉麵真的超好吃，湯頭濃郁，肉質軟嫩，CP值超高",
    "排隊等了半小時完全值得，滷肉飯配半熟蛋簡直絕配",
    "份量十足價格實惠，難怪每次來都大排長龍",
    "老闆人很好，服務態度親切，餐點也很美味",
    "湯頭清甜不油膩，麵條Q彈有嚼勁，大推",
    "環境乾淨整潔，餐點擺盤精美，非常適合約會",
    "招牌小籠包皮薄餡多汁鮮，一籠不夠再來一籠",
    "這間早午餐CP值爆表，飲料還可以無限續杯",
    "臭豆腐外酥內嫩，泡菜爽口，台灣味十足必吃",
    "珍珠奶茶珍珠QQ的很好喝，甜度剛剛好",
    "鹹酥雞炸得酥脆不油膩，蒜頭九層塔超香",
    "蚵仔煎外皮酥脆蚵仔新鮮飽滿，醬料調得很棒",
    "雞排又大又厚實，外皮酥脆肉質多汁，必點",
    "吃到飽的品質意外地好，食材新鮮選擇多",
    "甜點超級好吃，提拉米蘇綿密細緻，大推薦",
    "用餐環境舒適寬敞，服務生態度很好很專業",
    "這家的滷味入味又不會太鹹，每種都想點",
    "道地的台灣味，每次回國必來報到的店",
    "食材非常新鮮，海鮮類完全不腥，處理得很好",
    "整體用餐體驗非常棒，會想再回訪的好店",
    "招牌菜色每道都很有水準，廚師手藝了得",
    "朋友推薦來的果然沒失望，下次還要再來",
    "位置很好找交通方便，餐點好吃服務好",
    "炒飯粒粒分明香氣十足，配上酸辣湯絕了",
    "冰品消暑解渴超好吃，芒果冰料多又實在",
    "深夜食堂的感覺，宵夜來這裡吃最對了",
    "麻辣鍋湯頭香辣夠味，食材新鮮品質好",
    "便當菜色豐富配菜用心，米飯也煮得很好",
    "蛋餅皮酥脆內餡飽滿，早餐的最佳選擇",
    "這家的水餃皮薄Q彈餡料鮮美，吃了會上癮",
]

POSITIVE_SUFFIXES = [
    "，下次還會再來！", "，強力推薦！", "，五顆星！", "，必吃！",
    "，回訪率百分百！", "。", "，真的很讚！", "，大滿足！",
    "，非常滿意！", "，值得專程來吃！",
]

# ─── Neutral templates (label=1, rating 3) ───

NEUTRAL = [
    "味道普通，沒有特別驚艷但也不差",
    "環境還可以，東西就一般般",
    "價格跟味道成正比，不算貴但也不特別好吃",
    "普普通通的一餐，不會特別想再來也不會排斥",
    "餐點中規中矩，服務態度也還行",
    "份量適中價格合理，但口味沒什麼特色",
    "路過就進來吃吃看，感覺就是一般的小吃店",
    "朋友很推但我覺得還好，可能口味不太合",
    "裝潢不錯但餐點沒有到驚豔的程度",
    "等了蠻久的，味道只能說過得去",
    "不難吃但也不會特別想推薦給朋友",
    "有些菜色不錯有些普通，整體算可以",
    "食材算新鮮但調味太平淡了一點",
    "位置不太好找，餐點水準一般",
    "第一次來覺得新鮮，但回訪的動力不大",
    "網路評價很高但實際吃起來覺得還好而已",
    "服務態度不錯但食物沒有很突出",
    "菜單選擇很多但每道都差不多的味道",
    "同事推薦來的，個人覺得普通",
    "外帶回家吃口感有點打折扣",
    "早餐選擇之一，不會踩雷但也不會驚喜",
    "價位偏中等，品質也是中等水平",
    "小菜不錯但主餐表現普通",
    "網美店裝潢好看但食物一般",
    "份量有點少，但味道還可以接受",
    "出餐速度快這點不錯，但口味較單調",
    "來過兩三次了，感覺品質起伏不定",
    "整體來說算是及格的水平吧",
    "沒有特別的優點也沒有明顯的缺點",
    "方便快速解決一餐的選擇",
]

# ─── Negative templates (label=0, rating 1-2) ───

NEGATIVE = [
    "完全踩雷，服務態度差等了一個小時才上菜",
    "太鹹了而且份量很少，價格又貴完全不值得",
    "食物冷掉了還端上來，衛生堪慮不敢再來",
    "雷店一家，東西難吃到不行浪費錢",
    "服務生態度很差，叫了好幾次都不理人",
    "份量縮水價格漲了，CP值越來越低",
    "等太久了催了也沒用，餐點還出錯",
    "環境吵雜髒亂，桌子油膩膩的很不舒服",
    "看到蟑螂直接走人，衛生太差了吧",
    "網路上的照片跟實際差太多了根本詐騙",
    "難吃到想哭，完全浪費一餐的扣打",
    "油耗味很重食材不新鮮，吃完肚子不舒服",
    "結帳金額跟菜單不一樣多收錢",
    "吃完拉肚子超級不舒服，食安有問題",
    "態度差又貴又難吃，踩到大雷",
    "冷氣不涼座位又擠，用餐體驗很差",
    "調味料放太多化學味很重不健康的感覺",
    "老闆態度很兇服務零分",
    "菜色跟照片完全不同嚴重不符",
    "排了一小時結果超失望根本不值得排",
    "上菜慢就算了東西還是冷的真的傻眼",
    "價格虛高不合理，附近隨便一家都比這好吃",
    "廁所超級髒，影響食慾完全吃不下",
    "飲料喝起來有怪味道，不知道放了什麼",
    "麵條煮太爛沒口感，湯頭也沒味道",
    "雞排肉質乾柴又沒味道，失望透頂",
    "外帶等了四十分鐘結果餐還少一份",
    "不推薦不會再來第二次了",
    "食物份量跟兒童餐一樣少價格卻不便宜",
    "整體體驗非常糟糕一顆星都嫌多",
]

NEGATIVE_SUFFIXES = [
    "，絕對不會再來。", "，踩雷！", "，不推薦。", "，浪費錢！",
    "，超失望。", "，一顆星。", "，差評！", "。",
    "，拜託不要來。", "，千萬別來。",
]


def generate():
    random.seed(42)
    rows = []

    # Positive: 100 reviews
    for i in range(100):
        text = random.choice(POSITIVE) + random.choice(POSITIVE_SUFFIXES)
        rows.append({
            "review_text": text,
            "rating": random.choice([4, 5]),
            "restaurant": random.choice(RESTAURANTS),
            "area": random.choice(AREAS),
            "label": 2,
            "likes": random.randint(0, 30),
        })

    # Neutral: 100 reviews
    for i in range(100):
        text = random.choice(NEUTRAL)
        rows.append({
            "review_text": text,
            "rating": 3,
            "restaurant": random.choice(RESTAURANTS),
            "area": random.choice(AREAS),
            "label": 1,
            "likes": random.randint(0, 10),
        })

    # Negative: 100 reviews
    for i in range(100):
        text = random.choice(NEGATIVE) + random.choice(NEGATIVE_SUFFIXES)
        rows.append({
            "review_text": text,
            "rating": random.choice([1, 2]),
            "restaurant": random.choice(RESTAURANTS),
            "area": random.choice(AREAS),
            "label": 0,
            "likes": random.randint(0, 20),
        })

    random.shuffle(rows)
    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Generated {len(df)} dummy reviews → {OUTPUT}")
    print(f"  Label distribution: {df['label'].value_counts().sort_index().to_dict()}")
    return df


if __name__ == "__main__":
    generate()
