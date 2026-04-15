import os
import json
import google.generativeai as genai
from openai import AsyncOpenAI
from typing import Optional, Dict, Any

# Configure API Keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

class AIService:
    def __init__(self):
        self.provider = None
        
        # Priority 1: OpenAI
        if OPENAI_API_KEY:
            self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            self.provider = "openai"
            print("AI Service: Using OpenAI (ChatGPT).")
        
        # Priority 2: Gemini
        elif GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            # Use the cutting-edge preview model available for this key
            self.gemini_model = genai.GenerativeModel('gemini-3-flash-preview')
            self.provider = "gemini"
            print("AI Service: Using Google Gemini (gemini-3-flash-preview).")
        
        else:
            self.provider = "mock"
            print("Warning: No AI API Key set. Running in Mock mode.")

    async def analyze_word(self, word: str, dict_info: Optional[Dict] = None) -> Dict[str, Any]:
        if self.provider == "mock":
            return self._get_mock_analysis(word, dict_info)

        prompt = f"""
        你是一个专业的日语专家。请分析日语单词或短语: "{word}"。
        参考词典基础信息: {json.dumps(dict_info) if dict_info else "无"}
        
        请严格按照以下 JSON 格式返回分析结果，不要包含任何其他文字：
        {{
            "word": "{word}",
            "kana": "对应的假名",
            "romaji": "对应的罗马字",
            "pos": "词性(如: 名词, 动词-五段等)",
            "pitch": "语调编号(如: 0, 1, 2)",
            "origin": "如果是外来语，提供原始外语单词，否则为 null",
            "meaning": "准确的中文含义",
            "frequency": 1-5之间的整数(5代表最高频),
            "safety": 0-100之间的整数(代表社交安全性),
            "tone": "语气描述(如: 中性, 礼貌, 粗鲁)",
            "contexts": [
                {{"scene": "场景名", "usage": 占比百分比}}
            ],
            "social_targets": [
                {{"label": "对象(如: 上司, 朋友, 店员)", "allowed": true/false}}
            ],
            "image": "一个与场景相关的 Unsplash 图片关键词 URL，格式如: https://images.unsplash.com/photo-1580828343064-fde4fc206bc6?w=400&h=250&fit=crop",
            "examples": [
                {{"jp": "日语例句", "cn": "中文翻译"}}
            ],
            "explanation": "深度场景化解释，包括在日本社会中的使用习惯和文化背景。"
        }}
        """
        
        try:
            if self.provider == "openai":
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo", # Switch to a more widely available model for compatibility
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                text = response.choices[0].message.content
            else:
                response = self.gemini_model.generate_content(prompt)
                text = response.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
            
            return json.loads(text)
        except Exception as e:
            print(f"!!! AI Error ({self.provider}): {type(e).__name__} - {str(e)}")
            return self._get_mock_analysis(word, dict_info)

    async def chat(self, question: str) -> str:
        if self.provider == "mock":
            return f"AI 助手模拟回复: 关于 '{question}'，这是一个非常地道的问题。在真实接入 API (Gemini 或 ChatGPT) 后，我会为您分析更复杂的文化背景。"

        try:
            if self.provider == "openai":
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": f"你是一个日语学习助手。请简短地回答用户的日语相关问题: {question}"}]
                )
                return response.choices[0].message.content
            else:
                response = self.gemini_model.generate_content(f"你是一个日语学习助手。请简短地回答用户的日语相关问题: {question}")
                return response.text
        except Exception as e:
            print(f"!!! Chat Error ({self.provider}): {str(e)}")
            return f"抱歉，AI 暂时遇到一点问题: {str(e)}"

    def _get_mock_analysis(self, word: str, dict_info: Optional[Dict] = None) -> Dict[str, Any]:
        kana = dict_info.get("kana", "読み方") if dict_info else "読み方"
        meaning = dict_info.get("meaning", f"词典中关于 '{word}' 的解释") if dict_info else f"对 '{word}' 的模拟解析"
        pos = dict_info.get("pos", "名词") if dict_info else "名词"
        
        origin = None
        if all('\u30a0' <= c <= '\u30ff' for c in word):
            origin = "Loanword"

        base = {
            "word": word,
            "kana": kana,
            "romaji": "romaji",
            "pos": pos,
            "pitch": "0",
            "origin": origin,
            "meaning": meaning,
            "frequency": 3,
            "safety": 80,
            "tone": "中性",
            "image": f"https://images.unsplash.com/photo-1528164344705-47542687000d?w=400&h=250&fit=crop&auto=format&q=80",
            "social_targets": [
                {"label": "友人/朋友", "allowed": True},
                {"label": "上司/上司", "allowed": True},
                {"label": "店员/店员", "allowed": True}
            ],
            "contexts": [{"scene": "一般生活", "usage": 100}],
            "examples": [{"jp": f"{word}を勉強しています。", "cn": f"正在学习'{word}'。"}],
            "explanation": "由于未配置 API Key，当前显示的是由本地词典 jamdict 提供的基础信息。为了获得 AI 驱动的深度场景分析、社交建议和动态图片，请在环境变量中配置 GEMINI_API_KEY 或 OPENAI_API_KEY。"
        }
        return base
