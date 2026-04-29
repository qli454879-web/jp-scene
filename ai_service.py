import os
import json
import asyncio
import google.generativeai as genai
from openai import AsyncOpenAI
from typing import Optional, Dict, Any, List

# Configure API Keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").strip()
# 默认用更聪明也更快的模型；如需更省钱/更快，可在环境变量覆盖 OPENAI_MODEL
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview").strip() or "gemini-3-flash-preview"

class AIService:
    def __init__(self):
        self.provider = None
        
        # Priority 1: OpenAI
        if OPENAI_API_KEY:
            # 支持 OpenAI 兼容中转（通过环境变量配置，避免硬编码凭证/URL）
            if OPENAI_BASE_URL:
                self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            else:
                self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            self.provider = "openai"
            print("AI Service: Using OpenAI (ChatGPT).")
        
        # Priority 2: Gemini
        elif GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            # Use the cutting-edge preview model available for this key
            self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
            self.provider = "gemini"
            print(f"AI Service: Using Google Gemini ({GEMINI_MODEL}).")
        
        else:
            self.provider = "mock"
            print("Warning: No AI API Key set. Running in Mock mode.")

    async def analyze_word(self, word: str, dict_info: Optional[Dict] = None) -> Dict[str, Any]:
        if self.provider == "mock":
            return self._get_mock_analysis(word, dict_info)

        prompt = f"""
        Role: Japanese Language Expert.
        Word: "{word}"
        Dict Info: {json.dumps(dict_info) if dict_info else "N/A"}
        
        Output JSON only:
        {{
            "word": "{word}",
            "kana": "假名",
            "romaji": "romaji",
            "pos": "词性",
            "pitch": "语调0-9",
            "origin": "Loanword original or null",
            "meaning": "准确中文",
            "frequency": 1-5,
            "safety": 0-100,
            "tone": "语气描述",
            "contexts": [{{"scene": "场景", "usage": %}}],
            "social_targets": [{{"label": "上司/朋友/店员/初对面", "allowed": bool}}],
            "image": "https://images.unsplash.com/photo-1580828343064-fde4fc206bc6?w=400&h=250&fit=crop",
            "examples": [{{"jp": "地道例句", "cn": "中文"}}],
            "explanation": "深度场景化解释及文化背景。"
        }}
        """
        
        try:
            if self.provider == "openai":
                response = await self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "system", "content": "You are a Japanese expert outputting JSON."}, {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    timeout=20
                )
                text = response.choices[0].message.content
            else:
                # Use a more stable model or list of fallback models
                response = self.gemini_model.generate_content(prompt)
                text = response.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
            
            return json.loads(text)
        except Exception as e:
            print(f"!!! AI Error ({self.provider}): {type(e).__name__} - {str(e)}")
            # If quota exceeded, return mock but with a better message
            mock_data = self._get_mock_analysis(word, dict_info)
            if "quota" in str(e).lower() or "429" in str(e):
                mock_data["explanation"] = "【系统提示】AI 额度已达今日上限，当前为您展示基础词典解析。请稍后再试或更换 API Key。"
            return mock_data

    def _persona_system_prompt(self) -> str:
        # 更自然：允许追问、允许根据上下文续聊；保留人设但不限制死 3-8 句
        return (
            "你是“小雪梨”，也是“哈基米日语博士”。\n"
            "要求：\n"
            "- 不要自称“日语学习助手”。\n"
            "- 每次回答开头用“【小雪梨】”或“【哈基米日语博士】”二选一（尽量轮换）。\n"
            "- 语气像聪明、靠谱、会聊天的真人老师，不要机械模板。\n"
            "- 能读上下文：如果用户在追问/承接上一句，请直接接着答；必要时可以反问 1 个澄清问题。\n"
            "- 日语问题：优先给结论 + 例句 + 常见坑；动词变形/语法纠错要明确指出哪里不自然并给更自然版本。\n"
            "- 闲聊：也要像正常 AI 一样回应，不要一直把话题拉回背词。\n"
        )

    def _messages_to_transcript(self, messages: List[Dict[str, str]]) -> str:
        lines: List[str] = []
        for m in messages:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                lines.append(f"用户：{content}")
            elif role == "assistant":
                lines.append(f"小雪梨：{content}")
        return "\n".join(lines).strip()

    async def chat(self, question: str, messages: Optional[List[Dict[str, str]]] = None) -> str:
        if self.provider == "mock":
            return f"AI 助手模拟回复: 关于 '{question}'，这是一个非常地道的问题。在真实接入 API (Gemini 或 ChatGPT) 后，我会为您分析更复杂的文化背景。"

        sys_prompt = self._persona_system_prompt()
        hist = messages if isinstance(messages, list) else []
        # 限制长度，避免吞吐过大导致变慢
        hist = hist[-12:]

        try:
            if self.provider == "openai":
                msgs: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}]
                for m in hist:
                    r = (m.get("role") or "").strip()
                    c = (m.get("content") or "").strip()
                    if not c:
                        continue
                    if r in ("user", "assistant"):
                        msgs.append({"role": r, "content": c[:1200]})
                if question and (not msgs or msgs[-1]["role"] != "user"):
                    msgs.append({"role": "user", "content": question})
                response = await self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=msgs,
                    max_tokens=800,
                    temperature=0.7,
                    timeout=18,
                )
                return response.choices[0].message.content
            else:
                transcript = self._messages_to_transcript(hist)
                prompt = (
                    f"{sys_prompt}\n"
                    "下面是对话历史（可能为空）：\n"
                    f"{transcript}\n\n"
                    f"用户：{question}\n"
                    "小雪梨："
                )
                # google.generativeai 的参数在不同版本略有差异，尽量兼容
                try:
                    response = self.gemini_model.generate_content(
                        prompt,
                        generation_config={"temperature": 0.7, "max_output_tokens": 800},
                    )
                except Exception:
                    response = self.gemini_model.generate_content(prompt)
                return response.text
        except Exception as e:
            print(f"!!! Chat Error ({self.provider}): {str(e)}")
            msg = str(e)
            low = msg.lower()
            # 上游额度/限流：不要把原始报错回显给用户
            if "quota" in low or "429" in low or "rate" in low and "limit" in low:
                raise RuntimeError("AI_QUOTA") from e
            raise RuntimeError("AI_ERROR") from e

    async def get_quick_tip(self, word: str) -> str:
        if self.provider == "mock":
            return f"这是一个非常有趣的词！在日常生活中，{word} 经常出现在某些特定的语境里，记得多结合例句来记忆哦。"

        prompt = f"你是一个幽默、接地气的日语老师。请针对日语单词 '{word}'，用中文给出一句非常简短、人性化的记忆点或地道用法建议（50字以内）。不要说废话，直接像朋友聊天一样给出见解。"
        
        try:
            if self.provider == "openai":
                response = await self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100
                )
                return response.choices[0].message.content.strip()
            else:
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
        except Exception as e:
            msg = str(e).lower()
            if "quota" in msg or "429" in msg:
                raise RuntimeError("AI_QUOTA") from e
            return "这个词挺有意思的，建议结合场景深度分析来看看它的具体用法！"

    async def get_vocab_meta(self, word: str, kana: str = "", meaning: str = "") -> Dict[str, Any]:
        if self.provider == "mock":
            origin = None
            if all('\u30a0' <= c <= '\u30ff' or c == 'ー' for c in word):
                origin = meaning.split(",")[0].strip() if meaning else None
            return {"meaning_zh": meaning, "origin": origin}

        prompt = f"""
你是面向中国用户的日语词汇老师。请为下面词条输出严格 JSON（不要解释，不要 markdown）。

词条: {word}
读音: {kana}
释义(可能是英文或不完整): {meaning}

要求：
- meaning_zh 必须是简短准确的中文释义（尽量 2~12 字）。
- 如果该词是外来语/片假名借词，origin 给出来源原词（通常为英文小写，如 good）。不是外来语则 origin 为 null。

输出 JSON:
{{"meaning_zh": "...", "origin": null}}
"""

        try:
            if self.provider == "openai":
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": "Return JSON only."}, {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    timeout=20
                )
                text = response.choices[0].message.content
            else:
                response = self.gemini_model.generate_content(prompt)
                text = response.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()

            data = json.loads(text)
            meaning_zh = (data.get("meaning_zh") or "").strip()
            origin = data.get("origin")
            if origin is not None:
                origin = str(origin).strip()
                if origin.lower() in ["null", "none", ""]:
                    origin = None
            return {"meaning_zh": meaning_zh, "origin": origin}
        except Exception:
            origin = None
            if all('\u30a0' <= c <= '\u30ff' or c == 'ー' for c in word):
                origin = meaning.split(",")[0].strip() if meaning else None
            return {"meaning_zh": meaning, "origin": origin}

    async def enrich_word_profile(
        self,
        word: str,
        kana: str = "",
        meaning_zh: str = "",
        level: str = "N1",
    ) -> Optional[Dict[str, Any]]:
        """
        为词库批量补全：社交语体、频次、场景标签、深度解读、例句等。
        mock 模式返回 None（调用方应中止写库）。
        成功返回与 Postgres words 表扩展列对齐的 dict；解析/校验失败返回 None。
        """
        if self.provider == "mock":
            return None

        prompt = f"""
你是日语词汇专家，面向中文母语者。词条等级: {level}。

词条: {word}
读音: {kana}
已有释义(中文，仅供参考): {meaning_zh}

只输出一个 JSON 对象，不要用 markdown 代码块，不要其它说明文字。字段与类型:
- register_social: string，中文 2~4 句：语体(书面/口语)、适合对谁说、不当使用可能生硬或失礼的情况。
- usage_frequency: integer，1~5（5=日常口语极高频，1=罕用或极书面专语）
- usage_frequency_note: string，中文 4~12 字，如「书面报告常见」
- scene_tags: string 数组，3~6 个极短中文场景标签，如「职场」「邮件」「新闻」
- social_targets: string 数组，2~5 个听话人角色中文词，如「同事」「客户」「朋友」
- offense_risk: integer，0~100，对陌生人/上级用错时的冒犯或失礼风险估计
- scene_deep_dive: string，中文一段 80~200 字：典型场景与文化语感
- example_ja: string，一句自然日语例句（可使用该词的活用形）
- example_zh: string，例句中文翻译
"""

        def _parse_json_text(text: str) -> Dict[str, Any]:
            t = (text or "").strip()
            if "```json" in t:
                t = t.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in t:
                t = t.split("```", 1)[1].split("```", 1)[0].strip()
            return json.loads(t)

        def _norm_str_list(v: Any, max_n: int) -> List[str]:
            if v is None:
                return []
            if isinstance(v, str):
                parts = [x.strip() for x in v.replace("，", ",").split(",") if x.strip()]
                return parts[:max_n]
            if isinstance(v, list):
                out: List[str] = []
                for it in v:
                    if isinstance(it, str) and it.strip():
                        out.append(it.strip())
                    elif isinstance(it, dict):
                        lab = it.get("label") or it.get("name") or it.get("target")
                        if lab and str(lab).strip():
                            out.append(str(lab).strip())
                    if len(out) >= max_n:
                        break
                return out
            return []

        def _clamp_int(v: Any, lo: int, hi: int, default: int) -> int:
            try:
                x = int(v)
            except (TypeError, ValueError):
                return default
            return max(lo, min(hi, x))

        try:
            if self.provider == "openai":
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You output JSON only for Japanese CJK learners."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    timeout=45,
                )
                text = response.choices[0].message.content or ""
            else:
                def _run_g() -> str:
                    r = self.gemini_model.generate_content(prompt)
                    return (getattr(r, "text", None) or "").strip()

                text = await asyncio.to_thread(_run_g)

            data = _parse_json_text(text)
            register_social = (data.get("register_social") or "").strip()
            scene_deep_dive = (data.get("scene_deep_dive") or "").strip()
            example_ja = (data.get("example_ja") or "").strip()
            example_zh = (data.get("example_zh") or "").strip()
            usage_frequency_note = (data.get("usage_frequency_note") or "").strip()
            if not register_social or not example_ja:
                return None

            return {
                "register_social": register_social,
                "usage_frequency": _clamp_int(data.get("usage_frequency"), 1, 5, 3),
                "usage_frequency_note": usage_frequency_note or "—",
                "scene_tags": _norm_str_list(data.get("scene_tags"), 8) or ["一般"],
                "social_targets": _norm_str_list(data.get("social_targets"), 8) or ["通用"],
                "offense_risk": _clamp_int(data.get("offense_risk"), 0, 100, 0),
                "scene_deep_dive": scene_deep_dive or register_social,
                "example_ja": example_ja,
                "example_zh": example_zh or "",
            }
        except Exception as e:
            print(f"!!! enrich_word_profile ({self.provider}): {type(e).__name__} - {e}")
            return None

    async def enrich_library_entry(
        self,
        word: str,
        reading: str = "",
        meaning: str = "",
        level: str = "N2",
    ) -> Optional[Dict[str, Any]]:
        """
        For vocab_library: generate social_context (JSON), heatmap_data (JSON), insight_text (TEXT).
        image_url is intentionally handled outside and kept empty.
        """
        if self.provider == "mock":
            return None

        prompt = f"""
你是日语词汇老师，面向中文学习者。请只输出一个 JSON 对象，不要解释，不要 markdown。

词条等级: {level}
单词: {word}
读音: {reading}
释义: {meaning}

输出字段（严格 JSON）：
1) social_context: 对应 3 个社交场景块，结构如下（allowed 为布尔，reason 为 8~30 字中文简述）：
{{
  "casual": {{"label_ja":"カジュアル","label_zh":"友人・同僚","allowed":true,"reason":"..."}},
  "business": {{"label_ja":"ビジネス","label_zh":"上司・目上","allowed":false,"reason":"..."}},
  "formal": {{"label_ja":"フォーマル","label_zh":"店員・陌生人","allowed":false,"reason":"..."}}
}}
2) heatmap_data: 高频出现场景百分比（整数 0~100），3~6 项，按高到低，例如：
{{"SNS/ネット": 90, "一般生活": 60, "正式场合": 5}}
3) insight_text: 中文 60~140 字，像图中那样一段总结（可包含少量日语注记，如【俗】等）。
"""

        def _parse_json_text(text: str) -> Dict[str, Any]:
            t = (text or "").strip()
            if "```json" in t:
                t = t.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in t:
                t = t.split("```", 1)[1].split("```", 1)[0].strip()
            return json.loads(t)

        def _clamp_pct(v: Any) -> int:
            try:
                n = int(round(float(v)))
            except Exception:
                return 0
            return max(0, min(100, n))

        try:
            if self.provider == "openai":
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Return JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    timeout=45,
                )
                text = response.choices[0].message.content or ""
            else:
                def _run_g() -> str:
                    r = self.gemini_model.generate_content(prompt)
                    return (getattr(r, "text", None) or "").strip()

                text = await asyncio.to_thread(_run_g)

            data = _parse_json_text(text)
            social_context = data.get("social_context") or {}
            heatmap_data = data.get("heatmap_data") or {}
            insight_text = (data.get("insight_text") or "").strip()

            if not isinstance(social_context, dict) or not isinstance(heatmap_data, dict) or not insight_text:
                return None

            # Normalize expected keys for social_context
            for k, ja, zh in (
                ("casual", "カジュアル", "友人・同僚"),
                ("business", "ビジネス", "上司・目上"),
                ("formal", "フォーマル", "店員・陌生人"),
            ):
                v = social_context.get(k) if isinstance(social_context, dict) else None
                if not isinstance(v, dict):
                    v = {}
                allowed = bool(v.get("allowed"))
                reason = str(v.get("reason") or "").strip()
                social_context[k] = {
                    "label_ja": str(v.get("label_ja") or ja),
                    "label_zh": str(v.get("label_zh") or zh),
                    "allowed": allowed,
                    "reason": reason[:60],
                }

            # Normalize heatmap to int percent mapping (top 6)
            hm: Dict[str, int] = {}
            for sk, sv in heatmap_data.items():
                key = str(sk).strip()
                if not key:
                    continue
                hm[key] = _clamp_pct(sv)
            hm = dict(sorted(hm.items(), key=lambda kv: kv[1], reverse=True)[:6])

            return {"social_context": social_context, "heatmap_data": hm, "insight_text": insight_text}
        except Exception as e:
            print(f"!!! enrich_library_entry ({self.provider}): {type(e).__name__} - {e}")
            return None

    def _get_mock_analysis(self, word: str, dict_info: Optional[Dict] = None) -> Dict[str, Any]:
        kana = dict_info.get("kana", "読み方") if dict_info else "読み方"
        meaning = (
            dict_info.get("meaning", f"词典中关于 '{word}' 的解释")
            if dict_info
            else f"词典中暂无关于 '{word}' 的解释"
        )
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
