from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from ai_service import AIService
from dictionary_service import DictionaryService
import uvicorn
import os
import time

KNOWLEDGE_BASE = {
    "コンビニ": {
        "word": "コンビニ",
        "kana": "こんびに",
        "romaji": "konbini",
        "pos": "名词",
        "pitch": "0",
        "origin": "Convenience Store",
        "meaning": "便利店",
        "frequency": 5,
        "safety": 98,
        "tone": "中性/极其安全",
        "image": "https://images.unsplash.com/photo-1580828343064-fde4fc206bc6?w=400&h=250&fit=crop&auto=format",
        "social_targets": [
            {"label": "店员/店员", "allowed": True},
            {"label": "友人/朋友", "allowed": True},
            {"label": "上司/老板", "allowed": True},
            {"label": "初对面/初次见面", "allowed": True}
        ],
        "contexts": [
            {"scene": "自助结账/人工柜台", "usage": 85},
            {"scene": "打印/多媒体机器使用", "usage": 10},
            {"scene": "收发快递/缴费", "usage": 5}
        ],
        "examples": [
            {"jp": "お弁当の温めはどうされますか？", "cn": "您的便当需要加热吗？（店员常用语）"},
            {"jp": "袋は大丈夫です、ありがとうございます。", "cn": "不用袋子了，谢谢。（顾客地道回应）"},
            {"jp": "このコンビニにはイートインコーナーがありますか？", "cn": "这家便利店有店内用餐区吗？"}
        ],
        "explanation": "‘コンビニ’是日本现代生活的缩影。在这里，社交规则极其简单：保持礼貌即可。作为顾客，即使不使用高难度的敬语（Keigo），只要在句尾加上‘です’或‘ます’，或者简单的‘お願いします’，就会显得非常得体。"
    },
    "居酒屋": {
        "word": "居酒屋",
        "kana": "いざかや",
        "romaji": "izakaya",
        "pos": "名词",
        "pitch": "0",
        "meaning": "居酒屋",
        "frequency": 4,
        "safety": 82,
        "tone": "欢快/社交活跃",
        "image": "https://images.unsplash.com/photo-1514933651103-005eec06c04b?w=400&h=250&fit=crop&auto=format",
        "social_targets": [
            {"label": "同僚/同事", "allowed": True},
            {"label": "友人/朋友", "allowed": True},
            {"label": "上司/上司", "allowed": True},
            {"label": "初对面/初次见面", "allowed": False}
        ],
        "contexts": [
            {"scene": "下班后的‘饮会’ (Nomikai)", "usage": 75},
            {"scene": "深夜二次会/女子会", "usage": 25}
        ],
        "examples": [
            {"jp": "とりあえず、生でお願いします！", "cn": "先来杯生啤！（居酒屋点餐万能句）"},
            {"jp": "お会計、割り勘にしましょうか？", "cn": "结账的话，我们要平摊吗？"},
            {"jp": "この店、お通し代はいくらですか？", "cn": "这家店的‘餐前小菜’费用是多少？"}
        ],
        "explanation": "居酒屋是日本职场社交的延伸。虽然气氛随性，但依然存在微妙的潜规则：比如‘とりあえず生’（总之先上啤酒）是一种默认的开场礼仪。与上司同行时，要注意由下级负责斟酒，但在言语上可以比在办公室内稍微放松一些。"
    },
    "お疲れ様": {
        "word": "お疲れ様",
        "kana": "おつかれさま",
        "romaji": "otsukaresama",
        "pos": "常用语",
        "pitch": "0",
        "meaning": "辛苦了 (You must be tired)",
        "frequency": 5,
        "safety": 90,
        "tone": "职场/正式",
        "image": "https://picsum.photos/id/1060/800/450",
        "social_targets": [
            {"label": "朋友", "allowed": True},
            {"label": "同事", "allowed": True},
            {"label": "上司", "allowed": True},
            {"label": "下级", "allowed": True}
        ],
        "contexts": [
            {"scene": "职场交际", "usage": 95},
            {"scene": "朋友道别", "usage": 5}
        ],
        "examples": [
            {"jp": "お疲れ様です、お先に失礼します。", "cn": "辛苦了，我先失礼了（下班先走）。"},
            {"jp": "皆さん、今日はお疲れ様でした。", "cn": "各位，今天辛苦了。"}
        ],
        "explanation": "日本职场最重要的“万能钥匙”。不仅用于慰劳，还常作为打招呼、道别、甚至邮件的开头。对上司或平级都适用，但对下级可以省略'です'。"
    }
}

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock In-memory database for feedback
FEEDBACKS = []

# Initialize Services
ai = AIService()
dictionary = DictionaryService()

# --- Page Routes ---

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("web.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    with open("admin.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/forum", response_class=HTMLResponse)
async def forum_page():
    with open("forum.html", "r", encoding="utf-8") as f:
        return f.read()

# --- API Routes ---

@app.get("/api/analyze")
async def analyze(word: str = Query(...)):
    # 1. Lookup in Dictionary (Absolute Authority)
    dict_info = dictionary.lookup(word)
    
    # 2. Use AI to augment (This will now use your OpenAI Key)
    result = await ai.analyze_word(word, dict_info)
    return result

@app.get("/api/chat")
async def chat(q: str = Query(...)):
    answer = await ai.chat(q)
    return {"answer": answer}

@app.post("/api/feedback")
async def submit_feedback(data: dict = Body(...)):
    feedback_item = {
        "content": data.get("content"),
        "timestamp": time.time()
    }
    FEEDBACKS.insert(0, feedback_item) # Newest first
    return {"status": "success"}

@app.get("/api/feedbacks")
async def get_feedbacks():
    return FEEDBACKS

app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    print("Starting Japanese Scene Lab Production Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
