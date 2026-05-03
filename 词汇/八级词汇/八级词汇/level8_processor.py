import pandas as pd
import json
import time
import os
import requests

# ================= 配置 =================
NEW_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gemini-3.1-pro-preview").strip()
INPUT_TXT = "日本語__八級語彙.txt" 
OUTPUT_CSV = "日本語_八级语汇_全中文版.csv"

TAGS = ["交通", "便利店", "餐厅", "购物", "医院", "家里", "学校", "面试", "会议", "邮件", "商务礼仪", "电话", "公司日常", "SNS", "游戏", "弹幕", "寒暄", "请求", "委婉拒绝", "赞美", "吐槽/调侃", "情绪发泄", "职场潜规则"]
# ========================================

def try_request(url, word):
    target_url = f"{url.rstrip('/')}/v1/chat/completions" if "/v1" not in url else url
    
    # 极简指令：加入 (zh) 强制中文
    prompt = (
        f"Analyze:{word}. JSON ONLY:\n"
        f"1.pos:词性(zh).\n"
        f"2.freq:1-5.\n"
        f"3.ex:3 exs(ja & zh).\n"
        f"4.social(reason use zh):{{\"casual\":{{label_ja:\"カジュアル\",label_zh:\"友人・同僚\",allowed:bool,reason:\"\"}},\"business\":{{label_ja:\"ビジネス\",label_zh:\"上司・目上\",allowed:bool,reason:\"\"}},\"formal\":{{label_ja:\"フォーマル\",label_zh:\"店員・陌生人\",allowed:bool,reason:\"\"}}}}\n"
        f"5.heat:select 3 from {TAGS}, sum 100.\n"
        f"6.insight:linguistic analysis(zh, >120 chars)."
    )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }
    headers = {"Authorization": f"Bearer {NEW_KEY}", "Content-Type": "application/json"}
    
    try:
        response = requests.post(target_url, headers=headers, json=payload, timeout=30)
        return response.json()['choices'][0]['message']['content'] if response.status_code == 200 else None
    except:
        return None

def process():
    print(f"🚀 Flash Lite 启动 | 全中文强制模式")
    
    raw_entries = []
    if not os.path.exists(INPUT_TXT): 
        print(f"❌ 找不到文件: {INPUT_TXT}")
        return
        
    with open(INPUT_TXT, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            # 顺序修正：0单词, 1音频, 2假名, 3意思
            if len(parts) >= 4:
                raw_entries.append({
                    "word": parts[0],
                    "audio": parts[1], 
                    "reading": parts[2],
                    "meaning": parts[3]
                })

    done_words = pd.read_csv(OUTPUT_CSV)['word'].astype(str).tolist() if os.path.exists(OUTPUT_CSV) else []

    for index, entry in enumerate(raw_entries):
        word = entry['word']
        if word in done_words: continue
        
        print(f"⚡ {index+1}/{len(raw_entries)}: {word}")
        raw_json = try_request("https://api.vectorengine.ai/v1/chat/completions", word)
        
        if raw_json:
            try:
                d = json.loads(raw_json)
                new_row = {
                    "word": word,
                    "reading": entry['reading'],
                    "mp3": entry['audio'],
                    "meaning": entry['meaning'],
                    "pos": d.get('pos', ''), # 词性
                    "frequency": d.get('freq', 3),
                    "examples": json.dumps(d.get('ex', []), ensure_ascii=False),
                    "social_context": json.dumps(d.get('social', {}), ensure_ascii=False),
                    "heatmap_data": json.dumps(d.get('heat', {}), ensure_ascii=False),
                    "insight_text": d.get('insight', ''), # 分析内容
                    "level": "Level 8"
                }
                pd.DataFrame([new_row]).to_csv(OUTPUT_CSV, mode='a', index=False, header=not os.path.exists(OUTPUT_CSV), encoding='utf-8-sig')
                done_words.append(word)
            except:
                print(f"⚠️ {word} 解析失败")
        time.sleep(0.3)

if __name__ == "__main__":
    process()
