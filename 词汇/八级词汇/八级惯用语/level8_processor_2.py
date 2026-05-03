import pandas as pd
import json
import time
import os
import requests
import re
import asyncio
import edge_tts
from pykakasi import kakasi

# ================= 配置 =================
NEW_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gemini-3.1-pro-preview").strip()
INPUT_TXT = "日本語__八級慣用語.txt" 
OUTPUT_CSV = "日本語_八级语汇_全中文版.csv"
AUDIO_DIR = "level8_audios"
LEVEL_TAG = "level8"
VOICE = "ja-JP-NanamiNeural"

TAGS = ["交通", "便利店", "餐厅", "购物", "医院", "家里", "学校", "面试", "会议", "邮件", "商务礼仪", "电话", "公司日常", "SNS", "游戏", "弹幕", "寒暄", "请求", "委婉拒绝", "赞美", "吐槽/调侃", "情绪发泄", "职场潜规则"]

kks = kakasi()

def get_romaji(text):
    result = kks.convert(text)
    romaji = "".join([item['hepburn'] for item in result])
    return re.sub(r'[^a-zA-Z]', '', romaji).lower()

def clean_html(text):
    if not text: return ""
    clean = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', clean).strip()

async def generate_audio(text, filename):
    path = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(path): return
    try:
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(path)
    except: pass

def try_api_request(word, meaning):
    target_url = "https://api.vectorengine.ai/v1/chat/completions"
    
    # 极度强化的专家级 Prompt
    prompt = (
        f"作为顶级日语翻译专家，深度解析惯用语:【{word}】。参考释义:【{meaning}】。\n"
        f"请返回纯 JSON。核心要求：\n\n"
        f"1. ex (地道例句): 必须提供3个日本生活中极度地道的例句（包含口语、职场套话等真实语境）。每个例句包含 ja (日语) 和 zh (中文翻译)。\n\n"
        f"2. social (社交真实性): 严禁照抄模板。根据该词的贬义/敬语属性，真实判断在 友人(casual)、上司(business)、陌生人(formal) 场合是否 allowed。允许中性/褒义惯用语在职场为 true。必须给出具体的社交逻辑作为 reason。\n\n"
        f"3. heat (真实概率): 从 {TAGS} 选前3个，给出 1-100 的真实概率整数，不强制总和 100。\n\n"
        f"JSON 结构要求：\n"
        f"{{\n"
        f"  \"pos\": \"词性\",\n"
        f"  \"freq\": 1-5,\n"
        f"  \"ex\": [ {{\"ja\": \"例句1\", \"zh\": \"翻译1\"}}, {{\"ja\": \"例句2\", \"zh\": \"翻译2\"}}, {{\"ja\": \"例句3\", \"zh\": \"翻译3\"}} ],\n"
        f"  \"social\": {{\n"
        f"    \"casual\": {{\"label_ja\": \"カジュアル\", \"label_zh\": \"友人・同僚\", \"allowed\": bool, \"reason\": \"\"}},\n"
        f"    \"business\": {{\"label_ja\": \"ビジネス\", \"label_zh\": \"上司・目上\", \"allowed\": bool, \"reason\": \"\"}},\n"
        f"    \"formal\": {{\"label_ja\": \"フォーマル\", \"label_zh\": \"店員・陌生人\", \"allowed\": bool, \"reason\": \"\"}}\n"
        f"  }},\n"
        f"  \"heat\": {{ \"标签1\": 整数, \"标签2\": 整数, \"标签3\": 整数 }},\n"
        f"  \"insight\": \"包含词源、文化背景及使用禁忌的深度解析(>120字)\"\n"
        f"}}"
    )
    
    headers = {"Authorization": f"Bearer {NEW_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME, 
        "messages": [{"role": "user", "content": prompt}], 
        "response_format": {"type": "json_object"}, 
        "temperature": 0.5 # 适度提高温度，让例句更有灵气
    }
    
    try:
        response = requests.post(target_url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except: pass
    return None

async def main():
    print(f"🚀 启动终极版解析任务... 正在生成地道例句与真实社交分析")
    if not os.path.exists(AUDIO_DIR): os.makedirs(AUDIO_DIR)
    
    raw_entries = []
    with open(INPUT_TXT, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                romaji = get_romaji(parts[1])
                raw_entries.append({
                    "word": parts[0].strip(), "reading": parts[1].strip(), 
                    "audio_filename": f"{LEVEL_TAG}__{romaji}__.mp3", "meaning": clean_html(parts[3])
                })

    # 读取 CSV 实现续传 (建议先删除旧 CSV 以保证格式统一)
    done_words = pd.read_csv(OUTPUT_CSV)['word'].astype(str).tolist() if os.path.exists(OUTPUT_CSV) else []

    for index, entry in enumerate(raw_entries):
        word = entry['word']
        if word in done_words: continue
        
        print(f"⚡ [{index+1}/{len(raw_entries)}] 深度处理: {word}")
        await generate_audio(entry['reading'], entry['audio_filename'])
        
        raw_json = try_api_request(word, entry['meaning'])
        if raw_json:
            try:
                d = json.loads(raw_json)
                new_row = {
                    "word": word, "reading": entry['reading'], "mp3": entry['audio_filename'], 
                    "meaning": entry['meaning'], "pos": d.get('pos', ''), "frequency": d.get('freq', 3), 
                    "examples": json.dumps(d.get('ex', []), ensure_ascii=False), 
                    "social_context": json.dumps(d.get('social', {}), ensure_ascii=False), 
                    "heatmap_data": json.dumps(d.get('heat', {}), ensure_ascii=False), 
                    "insight_text": d.get('insight', ''), "level": "Level 8"
                }
                pd.DataFrame([new_row]).to_csv(OUTPUT_CSV, mode='a', index=False, header=not os.path.exists(OUTPUT_CSV), encoding='utf-8-sig')
            except: pass
        
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(main())
